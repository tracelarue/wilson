import argparse
import asyncio
import base64
import io
import json
import os
import sys
import traceback

import cv2
import mss
import numpy as np
import PIL.Image
import pyaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

if sys.version_info < (3, 11, 0):
    import exceptiongroup
    import taskgroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# Audio configuration constants
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
# Reduced chunk size for lower latency (was 11520)
# 512 samples = 32ms at 16kHz, good balance of latency and efficiency
CHUNK_SIZE = 512
OUTPUT_CHUNK_SIZE = 768  # 32ms at 24kHz

# Gemini Live model and default settings
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
DEFAULT_VIDEO_MODE = "none"  # Options: "camera", "screen", "none"
DEFAULT_RESPONSE_MODALITY = "AUDIO"  # Options: "TEXT", "AUDIO"
INITIAL_CONNECT_MESSAGE = "connect"
ENABLE_SYNTHETIC_KEEPALIVE = False  # Can trigger 1008 policy errors on some Live API states
RECONNECT_BASE_DELAY_SECONDS = 1.0
RECONNECT_MAX_DELAY_SECONDS = 8.0

# System instructions are loaded from separate files based on mode
# See system_instructions_sim.txt and system_instructions_real.txt
TOOL_CALL_PACING_INSTRUCTIONS = """
Tool-calling pace requirements:
- For multi-step workflows, call the next required tool immediately after receiving the previous tool result.
- Do not pause between tool calls for extra narration unless the user explicitly asks for detailed play-by-play.
- Keep status updates short (one sentence) and only at major milestones.
""".strip()


# Simple buffer pool to reduce memory allocations
class BufferPool:
    """Simple object pool for numpy arrays to reduce garbage collection pressure"""

    def __init__(self, buffer_size, max_buffers=20):
        self.buffer_size = buffer_size
        self.max_buffers = max_buffers
        self._pool = []

    def get(self):
        """Get a buffer from the pool or create a new one"""
        if self._pool:
            return self._pool.pop()
        return np.empty(self.buffer_size, dtype=np.int16)

    def put(self, buffer):
        """Return a buffer to the pool"""
        if len(self._pool) < self.max_buffers:
            self._pool.append(buffer)


# Audio resampling functions
def resample_audio(audio_data, original_rate, target_rate):
    """Resample audio data from original_rate to target_rate using fast linear interpolation"""
    if original_rate == target_rate:
        return audio_data

    # Convert bytes to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Calculate resampling ratio
    ratio = target_rate / original_rate

    # Calculate new length
    new_length = int(len(audio_array) * ratio)

    # Fast linear interpolation instead of FFT-based resampling
    # Create indices for the new sample positions
    old_indices = np.arange(len(audio_array))
    new_indices = np.linspace(0, len(audio_array) - 1, new_length)

    # Use numpy's interp for fast linear interpolation
    resampled = np.interp(new_indices, old_indices, audio_array)

    # Convert back to int16
    resampled = resampled.astype(np.int16)

    # Convert back to bytes
    return resampled.tobytes()


# Load MCP server configuration from mcp_config.json
def load_mcp_config():
    """Load MCP server configuration from mcp_config.json file."""
    config_path = os.path.join(os.path.dirname(__file__), "mcp_config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"mcp_config.json not found at {config_path}. "
            "Please create it following the README instructions."
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract ros-mcp-server configuration
    if "mcpServers" not in config or "ros-mcp-server" not in config["mcpServers"]:
        raise ValueError(
            "Invalid mcp_config.json: missing 'mcpServers.ros-mcp-server' configuration"
        )

    server_config = config["mcpServers"]["ros-mcp-server"]
    return server_config


def load_system_instructions(mode):
    """Load system instructions from file based on mode."""
    if mode == "sim":
        instructions_file = "system_instructions_sim.txt"
    elif mode == "robot":
        instructions_file = "system_instructions_real.txt"
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'sim' or 'robot'")

    instructions_path = os.path.join(os.path.dirname(__file__), instructions_file)

    if not os.path.exists(instructions_path):
        raise FileNotFoundError(
            f"System instructions file not found at {instructions_path}. "
            f"Please ensure {instructions_file} exists."
        )

    with open(instructions_path, "r") as f:
        return f.read()


def list_audio_devices():
    """
    List all available PyAudio devices for debugging.

    Returns:
        tuple: (default_input_index, default_output_index, all_devices_info)
    """
    pya_temp = pyaudio.PyAudio()
    devices_info = []

    try:
        default_input = pya_temp.get_default_input_device_info()
        default_input_index = default_input['index']
    except Exception:
        default_input_index = None

    try:
        default_output = pya_temp.get_default_output_device_info()
        default_output_index = default_output['index']
    except Exception:
        default_output_index = None

    print("\n" + "="*60)
    print("Available Audio Devices:")
    print("="*60)

    for i in range(pya_temp.get_device_count()):
        try:
            info = pya_temp.get_device_info_by_index(i)
            devices_info.append(info)

            device_type = []
            if info['maxInputChannels'] > 0:
                device_type.append("INPUT")
                if i == default_input_index:
                    device_type.append("(DEFAULT INPUT)")
            if info['maxOutputChannels'] > 0:
                device_type.append("OUTPUT")
                if i == default_output_index:
                    device_type.append("(DEFAULT OUTPUT)")

            type_str = " | ".join(device_type) if device_type else "UNAVAILABLE"

            print(f"[{i}] {info['name']}")
            print(f"    Type: {type_str}")
            print(f"    Channels: In={info['maxInputChannels']}, Out={info['maxOutputChannels']}")
            print(f"    Sample Rate: {int(info['defaultSampleRate'])} Hz")
            print()
        except Exception as e:
            print(f"[{i}] Error reading device: {e}")
            print()

    print("="*60 + "\n")
    pya_temp.terminate()

    return default_input_index, default_output_index, devices_info


def find_audio_device(device_index, device_type="input"):
    """
    Validate and find an audio device, with fallback to default.

    Args:
        device_index: Specific device index to use, or None for default
        device_type: "input" or "output"

    Returns:
        int or None: Valid device index
    """
    pya_temp = pyaudio.PyAudio()

    try:
        # If specific index is requested, validate it
        if device_index is not None:
            try:
                info = pya_temp.get_device_info_by_index(device_index)
                if device_type == "input" and info['maxInputChannels'] > 0:
                    pya_temp.terminate()
                    return device_index
                elif device_type == "output" and info['maxOutputChannels'] > 0:
                    pya_temp.terminate()
                    return device_index
                else:
                    print(f"⚠️  Device {device_index} doesn't support {device_type}, using default instead")
            except Exception as e:
                print(f"⚠️  Device {device_index} not available: {e}")
                print(f"   Falling back to default {device_type} device")

        # Fall back to default device
        if device_type == "input":
            default_info = pya_temp.get_default_input_device_info()
        else:
            default_info = pya_temp.get_default_output_device_info()

        pya_temp.terminate()
        return default_info['index']

    except Exception as e:
        print(f"🔴 Error finding {device_type} device: {e}")
        pya_temp.terminate()
        return None


# Load server configuration
mcp_config = load_mcp_config()

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command=mcp_config["command"],
    args=mcp_config["args"],
    env=mcp_config.get("env"),
)

# Load Google API key from environment files in priority order
# 1. First try /wilson/.env
# 2. Then try /home/trace/wilson/.env
env_paths = [
    "/wilson/.env",
    "/home/trace/wilson/.env",
]

api_key = None
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            print(f"Loaded Google API key from {env_path}")
            break

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found. Please ensure it exists in either "
        "/wilson/.env or /home/trace/wilson/.env"
    )

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=api_key,
)


pya = pyaudio.PyAudio()


class AudioLoop:
    """
    Main class for handling Gemini Live audio/video interaction with MCP server integration.

    Manages real-time audio streaming, video capture, and tool calls through MCP
    """

    def __init__(
        self,
        mode="sim",
        video_mode=DEFAULT_VIDEO_MODE,
        response_modality=DEFAULT_RESPONSE_MODALITY,
        active_muting=True,
        mute_mic=False,
    ):
        """
        Initialize the AudioLoop with specified mode, video mode and response modality.

        Args:
            mode: Operating mode - "sim" for simulation (default) or "robot" for real robot
            video_mode: Video input source - "camera", "screen", or "none"
            response_modality: Response format - "TEXT" or "AUDIO"
            active_muting: Whether to mute mic during audio playback
            mute_mic: Whether to start with the microphone muted and keep it muted
        """
        self.mode = mode
        self.video_mode = video_mode
        self.response_modality = response_modality
        self.active_muting = active_muting
        self.startup_muted = mute_mic

        # Load system instructions based on mode
        base_system_instructions = load_system_instructions(mode).rstrip()
        self.system_instructions = (
            f"{base_system_instructions}\n\n{TOOL_CALL_PACING_INSTRUCTIONS}"
        )

        # Audio format constants
        self.format = pyaudio.paInt16
        self.chunk_size = CHUNK_SIZE
        self.received_audio_buffer = OUTPUT_CHUNK_SIZE
        self.api_sample_rate = SEND_SAMPLE_RATE  # Gemini API input rate
        self.api_output_sample_rate = RECEIVE_SAMPLE_RATE  # Gemini API output rate

        # Mode-specific audio configuration
        if self.mode == "sim":
            # Simulation mode - current working settings
            self.mic_channels = 1
            self.speaker_channels = 1
            self.mic_index = None  # Use default microphone
            self.speaker_index = None  # Use default speaker
            self.mic_sample_rate = 16000  # Hardware sample rate matches API
            self.speaker_sample_rate = 24000  # Hardware sample rate matches API output
        elif self.mode == "robot":
            # Robot mode - hardware-specific settings
            self.mic_channels = 1
            self.speaker_channels = 1
            # Validate device indices, fall back to defaults if unavailable
            self.mic_index = find_audio_device(2, "input")  # Try device 3, fall back to default
            self.speaker_index = find_audio_device(0, "output")  # Try device 2, fall back to default
            self.mic_sample_rate = 48000  # Hardware sample rate (needs resampling)
            self.speaker_sample_rate = 48000  # Hardware sample rate (needs resampling)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'sim' or 'robot'")

        # Communication queues
        self.audio_in_queue = None  # Queue for incoming audio from Gemini
        self.out_queue = None  # Queue for outgoing data to Gemini

        # Session and task management
        self.session = None  # Gemini Live session
        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

        # Control flags for audio management
        self.mic_active = not self.startup_muted
        self.mic_lock = asyncio.Lock()

        # Audio streaming state tracking
        self.audio_stream_active = False
        self.audio_stream_lock = asyncio.Lock()
        self.last_audio_chunk_time = None

        # Buffer pool for resampling (reduces memory allocations)
        # Calculate max buffer size needed for resampling
        max_input_samples = max(self.chunk_size, self.received_audio_buffer) * 2
        self.buffer_pool = BufferPool(buffer_size=max_input_samples)

    async def _set_mic_active(self, active, message=None):
        """Update microphone capture state while honoring persistent startup mute."""
        if self.startup_muted:
            active = False

        async with self.mic_lock:
            state_changed = self.mic_active != active
            self.mic_active = active

        if message and state_changed:
            print(message)

    async def send_text(self):
        """
        Handle text input from user and send to Gemini Live session.

        Continuously prompts for user input and sends it to the session.
        Breaks the loop when user types 'q' to quit.
        """
        while True:
            text = await asyncio.to_thread(
                input,
                "🎤 message > ",
            )
            if text.lower() == "q":
                break

            await self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": text or "."}]}, turn_complete=True
            )

    async def handle_tool_call(self, tool_call):
        """
        Process tool calls from Gemini and execute them via MCP session.

        Args:
            tool_call: Tool call request from Gemini containing function calls
        """
        import time

        for function_call in tool_call.function_calls:
            start_time = time.perf_counter()
            print(f"\n🔧 Calling tool: {function_call.name}")
            print(f"   Arguments: {function_call.args}")

            # Execute tool call through MCP server
            # Add custom timeout for navigation actions (5 minutes instead of default 2 minutes)
            tool_args = dict(function_call.args or {})
            if function_call.name == "navigate_to_location":
                tool_args["timeout"] = 300.0  # 5 minutes for navigation

            try:
                result = await self.mcp_session.call_tool(
                    name=function_call.name,
                    arguments=tool_args,
                )
                response_data = self._parse_tool_result(result)
            except Exception as e:
                response_data = {"error": str(e)}
                print(f"🔴 Tool execution failed for {function_call.name}: {e}")

            # Send final response to Gemini
            # IMPORTANT: Never use will_continue=False as it signals failure/retry
            # Always omit will_continue for final successful responses
            function_responses = [
                types.FunctionResponse(
                    name=function_call.name,
                    id=function_call.id,
                    response=response_data
                    # Always omit will_continue - it defaults to None which signals completion
                )
            ]
            try:
                await self.session.send_tool_response(function_responses=function_responses)
                elapsed = time.perf_counter() - start_time
                print(f"✅ Sent tool result for {function_call.name} in {elapsed:.2f}s")
            except Exception as e:
                print(f"🔴 Error sending tool response: {e}")

    def _content_item_to_text(self, content_item):
        """Convert a single MCP content item into a plain string."""
        if hasattr(content_item, "text"):
            return content_item.text
        if hasattr(content_item, "model_dump"):
            dumped = content_item.model_dump()
            return json.dumps(dumped, separators=(",", ":"))
        if isinstance(content_item, dict):
            return json.dumps(content_item, separators=(",", ":"))
        return str(content_item)

    def _parse_tool_result(self, result):
        """
        Parse MCP tool result into a Gemini function response payload.

        Returns:
            dict: JSON-serializable payload for FunctionResponse.response
        """
        if hasattr(result, "structuredContent") and result.structuredContent is not None:
            structured = result.structuredContent
            if isinstance(structured, dict):
                return structured
            return {"result": structured}

        if hasattr(result, "content"):
            result_text = "\n".join(self._content_item_to_text(item) for item in result.content)
        else:
            result_text = str(result)

        if not result_text:
            return {"result": ""}

        try:
            parsed = json.loads(result_text)
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        except (json.JSONDecodeError, ValueError):
            return {"result": result_text}

    def _get_frame(self, cap):
        """
        Capture and process a single frame from camera.

        Args:
            cap: OpenCV VideoCapture object

        Returns:
            dict: Frame data with mime_type and base64-encoded image data, or None if failed
        """
        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            return None

        # Convert BGR (OpenCV) to RGB (PIL) to prevent blue tint
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create PIL image and resize for efficiency
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])

        # Convert to JPEG format
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        # Return as base64-encoded data
        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        """
        Continuously capture frames from camera and add them to output queue.

        Uses asyncio.to_thread to prevent blocking the audio pipeline.
        Captures frames at 1 second intervals.
        """
        # Initialize camera (0 = default camera)
        # Run in thread to prevent blocking audio pipeline
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)

        # Set lower resolution for faster capture and less bandwidth
        await asyncio.to_thread(cap.set, cv2.CAP_PROP_FRAME_WIDTH, 640)
        await asyncio.to_thread(cap.set, cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # Capture frame in separate thread
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            # Try to add frame without blocking if queue is full (skip frame)
            try:
                self.out_queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass  # Skip this frame if queue is full

            # Send frame at 1 second intervals
            await asyncio.sleep(1.0)

        # Clean up camera resource
        cap.release()

    def _get_screen(self):
        """
        Capture and process a screenshot from the primary monitor.

        Returns:
            dict: Screen data with mime_type and base64-encoded image data
        """
        # Initialize screen capture
        screen_capture = mss.mss()
        primary_monitor = screen_capture.monitors[0]

        # Capture screenshot
        screenshot = screen_capture.grab(primary_monitor)

        # Convert to PIL Image
        image_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        # Convert to JPEG format
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        # Return as base64-encoded data
        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):
        """
        Continuously capture screenshots and add them to output queue.

        Captures screenshots at 1 second intervals.
        """
        while True:
            # Capture screenshot in separate thread
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            # Try to add frame without blocking if queue is full (skip frame)
            try:
                self.out_queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass  # Skip this frame if queue is full

            # Send screenshot at 1 second intervals
            await asyncio.sleep(1.0)

    async def send_realtime(self):
        """
        Send real-time data (audio/video) from output queue to Gemini session.

        Continuously processes messages from the output queue and sends them to Gemini.
        """
        while True:
            message = await self.out_queue.get()
            await self.session.send_realtime_input(media=message)

    async def websocket_keepalive(self):
        """
        Send periodic keepalive pings to prevent WebSocket timeout during long operations.

        Runs independently of audio flow to ensure connection stays alive even when
        event loop is busy with tool execution.
        """
        while True:
            try:
                # Wait 15 seconds between keepalives (well under 20s ping timeout)
                await asyncio.sleep(15.0)

                # Send minimal silence packet as keepalive
                keepalive_audio = b'\x00\x00' * 8  # 16 bytes of silence
                await self.session.send_realtime_input(
                    media={"data": keepalive_audio, "mime_type": "audio/pcm"}
                )
            except asyncio.CancelledError:
                # Task was cancelled, exit gracefully
                break
            except Exception as e:
                # Keepalive failures are non-critical
                print(f"⚠️ Keepalive failed (non-fatal): {e}")

    def _iter_nested_exceptions(self, exc):
        """Yield leaf exceptions from nested ExceptionGroup-like objects."""
        nested = getattr(exc, "exceptions", None)
        if nested:
            for sub_exc in nested:
                yield from self._iter_nested_exceptions(sub_exc)
            return
        yield exc

    def _is_retryable_policy_violation(self, exc):
        """
        Detect 1008 policy-violation Live API closures that are safe to reconnect.
        """
        for leaf in self._iter_nested_exceptions(exc):
            message = str(leaf)
            if "1008" in message and "Operation is not implemented" in message:
                return True
        return False

    async def listen_audio(self):
        """
        Continuously capture audio from microphone and add to output queue.

        Sets up microphone input stream and reads audio data in chunks.
        Resamples audio if hardware rate differs from API rate (robot mode).
        """
        # Get microphone info
        if self.mic_index is not None:
            mic_info = pya.get_device_info_by_index(self.mic_index)
        else:
            mic_info = pya.get_default_input_device_info()
        print("Microphone:", mic_info["name"])

        # Initialize audio input stream
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=self.format,
            channels=self.mic_channels,
            rate=self.mic_sample_rate,
            input=True,
            input_device_index=self.mic_index if self.mic_index is not None else mic_info["index"],
            frames_per_buffer=self.chunk_size,
        )

        if self.startup_muted:
            print("🔇 Microphone muted at startup and will remain muted")

        # Configure overflow handling for debug vs release
        overflow_kwargs = {"exception_on_overflow": False} if __debug__ else {}

        stream_active = True

        # Continuously read audio data
        while True:
            # Check if mic should be active
            async with self.mic_lock:
                mic_currently_active = self.mic_active

            if mic_currently_active:
                # If stream was stopped, restart it
                if not stream_active:
                    await asyncio.to_thread(self.audio_stream.start_stream)
                    stream_active = True

                # Read audio data (blocking call, no need for sleep)
                audio_data = await asyncio.to_thread(
                    self.audio_stream.read, self.chunk_size, **overflow_kwargs
                )

                # Resample if hardware rate differs from API rate (robot mode)
                if self.mic_sample_rate != self.api_sample_rate:
                    audio_data = resample_audio(audio_data, self.mic_sample_rate, self.api_sample_rate)

                await self.out_queue.put({"data": audio_data, "mime_type": "audio/pcm"})
            else:
                # Stop the stream completely to prevent any audio capture
                if stream_active:
                    await asyncio.to_thread(self.audio_stream.stop_stream)
                    stream_active = False

                # Just sleep while muted - no audio is being captured
                await asyncio.sleep(0.1)

    async def receive_audio(self):
        """
        Background task to receive responses from Gemini session.

        Processes audio data, text responses, and tool calls from Gemini.
        Handles interruptions by clearing the audio queue.

        PHASE 2 FIX: Tool calls run as async tasks to avoid blocking the receive loop.
        This ensures WebSocket messages (including server pings) are processed continuously,
        preventing "keepalive ping timeout" errors during long operations.
        """
        # Track active tool call tasks
        tool_call_tasks = set()

        while True:
            turn = self.session.receive()
            turn_text = ""
            first_text = True
            has_audio_in_turn = False

            async for response in turn:
                # Handle server content with model turn
                server_content = response.server_content
                if server_content and server_content.model_turn:
                    for part in server_content.model_turn.parts:
                        # Handle audio data from inline_data parts
                        if part.inline_data:
                            # Signal that audio streaming has started
                            if not has_audio_in_turn:
                                async with self.audio_stream_lock:
                                    self.audio_stream_active = True
                                has_audio_in_turn = True
                            try:
                                self.audio_in_queue.put_nowait(part.inline_data.data)
                            except asyncio.QueueFull:
                                # Drop audio chunk if queue is full to prevent crash
                                # This happens when audio arrives faster than it can be played
                                pass

                        # Handle text responses
#                        if part.text:
#                            text_content = part.text
#                            if first_text:
#                                print(f"\n🤖 > {text_content}", end="", flush=True)
#                                first_text = False
#                            else:
#                                print(text_content, end="", flush=True)
#                            turn_text += text_content
#                    continue
#
#                # Fallback: Handle text responses from Gemini (for backward compatibility)
#                if text_content := response.text:
#                    if first_text:
#                        print(f"\n🤖 > {text_content}", end="", flush=True)
#                        first_text = False
#                    else:
#                        print(text_content, end="", flush=True)
#                    turn_text += text_content

                # Handle server content (currently disabled)
                """
                server_content = response.server_content
                if server_content is not None:
                    self.handle_server_content(server_content)
                    continue
                """

                # Handle tool calls from Gemini - NON-BLOCKING
                # Create async task instead of awaiting inline to keep receive loop active
                tool_call = response.tool_call
                if tool_call is not None:
                    # Wait for any existing tool calls to complete first (sequential execution)
                    # This prevents duplicate/parallel tool calls for the same action
                    if tool_call_tasks:
                        await asyncio.wait(tool_call_tasks)
                        tool_call_tasks.clear()

                    # Now create new task for this tool execution
                    task = asyncio.create_task(self.handle_tool_call(tool_call))
                    tool_call_tasks.add(task)
                    # Remove from set when task completes
                    task.add_done_callback(lambda t: tool_call_tasks.discard(t))

            # Turn is complete - signal end of audio stream
            async with self.audio_stream_lock:
                self.audio_stream_active = False

            # Complete the response display
            if turn_text:
                print()  # Add newline after response
                print("🎤 message > ", end="", flush=True)  # Show next prompt

            # Handle interruptions by clearing queued audio
            # This prevents audio backlog when user interrupts the model
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        """
        Play audio responses from Gemini through speakers.

        Continuously reads audio data from input queue and plays it.
        Mutes microphone during playback to prevent feedback.
        Resamples audio if hardware rate differs from API output rate (robot mode).
        """
        # Initialize audio output stream
        audio_stream = await asyncio.to_thread(
            pya.open,
            format=self.format,
            channels=self.speaker_channels,
            rate=self.speaker_sample_rate,
            output=True,
            output_device_index=self.speaker_index,
            frames_per_buffer=self.received_audio_buffer,
        )

        audio_playing = False

        # Continuously play audio from queue
        while True:
            try:
                # Wait for audio with a reasonable timeout
                try:
                    audio_bytes = await asyncio.wait_for(self.audio_in_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check if we were playing audio and the stream is now complete
                    if audio_playing:
                        async with self.audio_stream_lock:
                            stream_still_active = self.audio_stream_active

                        # If stream is complete and queue is empty, we're done
                        if not stream_still_active and self.audio_in_queue.empty():
                            if self.active_muting:
                                await self._set_mic_active(
                                    True,
                                    "🎤 Microphone unmuted - audio playback complete",
                                )
                            audio_playing = False
                    continue

                # Update last audio time
                self.last_audio_chunk_time = asyncio.get_event_loop().time()

                # If this is the first audio chunk in a sequence, mute the microphone (if enabled)
                if not audio_playing:
                    if self.active_muting:
                        await self._set_mic_active(
                            False,
                            "🔇 Microphone muted while audio is playing",
                        )
                        audio_playing = True

                        # Small delay to ensure mic is fully muted
                        await asyncio.sleep(0.1)
                    else:
                        audio_playing = True

                # Resample if hardware rate differs from API output rate (robot mode)
                if self.speaker_sample_rate != self.api_output_sample_rate:
                    audio_bytes = resample_audio(audio_bytes, self.api_output_sample_rate, self.speaker_sample_rate)

                # Play the audio
                await asyncio.to_thread(audio_stream.write, audio_bytes)

            except Exception as e:
                print(f"🔴 Audio playback error: {str(e)}")
                # Re-enable microphone in case of error (if muting is enabled)
                if self.active_muting:
                    await self._set_mic_active(
                        True,
                        "🎤 Microphone unmuted after audio error",
                    )
                audio_playing = False
                await asyncio.sleep(0.1)

    async def run(self):
        """
        Main execution method that sets up and runs the Gemini Live session.

        Connects to MCP server, configures tools, and starts all async tasks
        for audio/video processing and communication.
        """

        # PHASE 1 FIX: Monkey-patch websockets library to disable client-side keepalive
        # This prevents "keepalive ping timeout" errors during long-running operations
        # The websockets library's default 20s ping timeout is too aggressive for
        # navigation actions that can take 40+ seconds
        import websockets.asyncio.client
        original_connect = websockets.asyncio.client.connect

        def patched_connect(*args, **kwargs):
            # Disable client-side ping/pong keepalive mechanism
            # Let Gemini's server handle keepalive instead
            kwargs.setdefault('ping_interval', None)  # No automatic pings from client
            kwargs.setdefault('ping_timeout', None)   # No timeout on pong responses
            return original_connect(*args, **kwargs)

        websockets.asyncio.client.connect = patched_connect
        print("🔧 Patched websockets library: disabled client-side keepalive")

        # Define logging callback to receive log messages from MCP server
        async def logging_handler(params):
            """Handle log messages (info, debug, warning, error) from MCP server"""
            level_emoji = {
                "debug": "🔍",
                "info": "ℹ️",
                "notice": "📢",
                "warning": "⚠️",
                "error": "🔴",
                "critical": "🚨",
                "alert": "🆘",
                "emergency": "🔥"
            }
            emoji = level_emoji.get(params.level, "📝")
            print(f"   {emoji} [{params.level.upper()}] {params.data}")

        reconnect_delay = RECONNECT_BASE_DELAY_SECONDS
        while True:
            # Connect to MCP server using stdio
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write, logging_callback=logging_handler) as mcp_session:
                    # Initialize the connection between client and server
                    await mcp_session.initialize()

                    # Store MCP session for tool calling
                    self.mcp_session = mcp_session

                    # Get available tools from MCP server
                    available_tools = await mcp_session.list_tools()

                    # Convert MCP tools to Gemini-compatible format
                    # The Live API does NOT support automatic MCP tool calling
                    # So we must manually convert tools and handle execution
                    functional_tools = []

                    for tool in available_tools.tools:
                        tool_description = {"name": tool.name, "description": tool.description}

                        # Process tool parameters if they exist
                        if tool.inputSchema["properties"]:
                            tool_description["parameters"] = {
                                "type": tool.inputSchema["type"],
                                "properties": {},
                            }

                            # Convert each parameter to Gemini format
                            for param_name in tool.inputSchema["properties"]:
                                param_schema = tool.inputSchema["properties"][param_name]

                                # Handle direct type or anyOf union types
                                if "type" in param_schema:
                                    param_type = param_schema["type"]
                                elif "anyOf" in param_schema:
                                    # For anyOf, use the first non-null type
                                    param_type = "string"  # default fallback
                                    for type_option in param_schema["anyOf"]:
                                        if type_option.get("type") != "null":
                                            param_type = type_option["type"]
                                            break
                                else:
                                    param_type = "string"  # Fallback default

                                # Build parameter definition
                                param_definition = {
                                    "type": param_type,
                                    "description": "",
                                }

                                # Handle array types that need items specification
                                if param_type == "array" and "items" in param_schema:
                                    items_schema = param_schema["items"]
                                    if "type" in items_schema:
                                        param_definition["items"] = {"type": items_schema["type"]}
                                    else:
                                        # Default to object for complex array items
                                        param_definition["items"] = {"type": "object"}

                                tool_description["parameters"]["properties"][param_name] = (
                                    param_definition
                                )

                            # Add required parameters list if specified
                            if "required" in tool.inputSchema:
                                tool_description["parameters"]["required"] = tool.inputSchema[
                                    "required"
                                ]

                        functional_tools.append(tool_description)

                    # Configure Gemini Live tools (MCP tools + built-in capabilities)
                    tools = [
                        {
                            "function_declarations": functional_tools,
                            #"code_execution": {},  # Enable code execution
                            #"google_search": {},  # Enable web search
                        },
                    ]

                    # Configure Gemini Live session
                    live_config = types.LiveConnectConfig(
                        response_modalities=[
                            self.response_modality
                        ],  # "Enable text or audio responses based on configuration"
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
                            )
                        ),
                        system_instruction=types.Content(parts=[types.Part(text=self.system_instructions)]),
                        tools=tools,
                    )

                    try:
                        # Start Gemini Live session and create task group
                        async with (
                            client.aio.live.connect(model=MODEL, config=live_config) as session,
                            asyncio.TaskGroup() as task_group,
                        ):
                            self.session = session

                            # Initialize communication queues
                            self.audio_in_queue = asyncio.Queue(maxsize=400)  # Audio from Gemini (buffer for smooth playback)
                            self.out_queue = asyncio.Queue(maxsize=3)  # Data to Gemini (small buffer for low latency)

                            # Send an initial one-time turn to establish context/state.
                            await self.session.send_client_content(
                                turns={"role": "user", "parts": [{"text": INITIAL_CONNECT_MESSAGE}]},
                                turn_complete=True,
                            )

                            # Start all async tasks
                            send_text_task = task_group.create_task(self.send_text())
                            task_group.create_task(self.send_realtime())
                            if ENABLE_SYNTHETIC_KEEPALIVE:
                                task_group.create_task(self.websocket_keepalive())
                            task_group.create_task(self.listen_audio())

                            # Start video capture based on selected mode
                            if self.video_mode == "camera":
                                task_group.create_task(self.get_frames())
                            elif self.video_mode == "screen":
                                task_group.create_task(self.get_screen())

                            # Start audio processing tasks
                            task_group.create_task(self.receive_audio())
                            task_group.create_task(self.play_audio())

                            # Wait for user to quit (send_text_task completes when user types 'q')
                            await send_text_task
                            raise asyncio.CancelledError("User requested exit")

                    except asyncio.CancelledError:
                        # Normal exit when user types 'q'
                        return
                    except asyncio.ExceptionGroup as exception_group:
                        # Handle any errors that occurred in the task group
                        if hasattr(self, 'audio_stream') and self.audio_stream is not None:
                            self.audio_stream.close()

                        if self._is_retryable_policy_violation(exception_group):
                            print(
                                f"⚠️ Live API closed with policy violation (1008). "
                                f"Reconnecting in {reconnect_delay:.1f}s..."
                            )
                            await asyncio.sleep(reconnect_delay)
                            reconnect_delay = min(reconnect_delay * 2.0, RECONNECT_MAX_DELAY_SECONDS)
                            continue

                        traceback.print_exception(exception_group)
                        return


if __name__ == "__main__":
    # Parse command line arguments for video mode selection
    parser = argparse.ArgumentParser(
        description="Gemini Live integration with MCP server for robot control"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sim",
        help="Operating mode: 'sim' for simulation (default) or 'robot' for real robot",
        choices=["sim", "robot"],
    )
    parser.add_argument(
        "--video",
        type=str,
        default=DEFAULT_VIDEO_MODE,
        help="Video input source for visual context",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--responses",
        type=str,
        default=DEFAULT_RESPONSE_MODALITY,
        help="Response format from Gemini",
        choices=["TEXT", "AUDIO"],
    )
    parser.add_argument(
        "--active-muting",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Mute microphone during audio playback (true/false, default: true)",
    )
    parser.add_argument(
        "--mute-mic",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Initialize with the microphone muted and keep it muted (true/false, default: false)",
    )
    args = parser.parse_args()

    # List available audio devices for debugging
    print(
        f"\n🔧 Initializing in '{args.mode}' mode with video='{args.video}', "
        f"responses='{args.responses}' and mute_mic='{args.mute_mic}'"
    )
    list_audio_devices()

    # Initialize and run the audio loop
    audio_loop = AudioLoop(
        mode=args.mode,
        video_mode=args.video,
        response_modality=args.responses,
        active_muting=args.active_muting,
        mute_mic=args.mute_mic,
    )
    asyncio.run(audio_loop.run())
