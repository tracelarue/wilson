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
MODEL = "models/gemini-2.5-flash-live-preview"
DEFAULT_VIDEO_MODE = "none"  # Options: "camera", "screen", "none"
DEFAULT_RESPONSE_MODALITY = "AUDIO"  # Options: "TEXT", "AUDIO"

# System instructions are loaded from separate files based on mode
# See system_instructions_sim.txt and system_instructions_real.txt


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
    ):
        """
        Initialize the AudioLoop with specified mode, video mode and response modality.

        Args:
            mode: Operating mode - "sim" for simulation (default) or "robot" for real robot
            video_mode: Video input source - "camera", "screen", or "none"
            response_modality: Response format - "TEXT" or "AUDIO"
            active_muting: Whether to mute mic during audio playback
        """
        self.mode = mode
        self.video_mode = video_mode
        self.response_modality = response_modality
        self.active_muting = active_muting

        # Load system instructions based on mode
        self.system_instructions = load_system_instructions(mode)

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
        self.mic_active = True
        self.mic_lock = asyncio.Lock()

        # Audio streaming state tracking
        self.audio_stream_active = False
        self.audio_stream_lock = asyncio.Lock()
        self.last_audio_chunk_time = None

        # Buffer pool for resampling (reduces memory allocations)
        # Calculate max buffer size needed for resampling
        max_input_samples = max(self.chunk_size, self.received_audio_buffer) * 2
        self.buffer_pool = BufferPool(buffer_size=max_input_samples)

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
        import re

        for function_call in tool_call.function_calls:
            print(f"\n🔧 Calling tool: {function_call.name}")
            print(f"   Arguments: {function_call.args}")

            # Track last distance and time for detecting progress and throttling
            last_distance = None
            last_sent_time = 0.0

            # Action tools that should stream progress to Gemini
            action_tools = ['send_action_goal', 'navigate_to_location']
            is_action_tool = function_call.name in action_tools

            # Define progress callback to receive real-time progress notifications
            async def progress_handler(progress: float, total: float | None, message: str | None):
                """Handle progress notifications from MCP server and forward to Gemini"""
                nonlocal last_distance, last_sent_time, is_action_tool

                # Print RAW feedback for debugging
                print(f"\n📊 RAW FEEDBACK - progress: {progress}, total: {total}, message: {message}")

                if message:
                    # Parse feedback data from the message
                    progress_data = {
                        "status": "in_progress",
                        "progress": progress,
                        "message": message[:500]  # Truncate long messages
                    }

                    # Extract distance_remaining for Nav2 actions
                    distance = None
                    if "distance_remaining" in message:
                        try:
                            # Look for 'distance_remaining': 2.84 or "distance_remaining": 2.84
                            match = re.search(r"['\"]distance_remaining['\"]:\s*([0-9.]+)", message)
                            if match:
                                distance = float(match.group(1))
                                progress_data["distance_remaining"] = distance
                                print(f"   Distance remaining: {distance:.2f} m")
                        except (ValueError, AttributeError):
                            pass

                    # Also check for old format with just 'remaining'
                    elif "remaining" in message.lower() and "distance_remaining" not in message:
                        try:
                            match = re.search(r"['\"]remaining['\"]:\s*([0-9.]+)", message)
                            if match:
                                remaining = float(match.group(1))
                                progress_data["distance_remaining"] = remaining
                                print(f"   Distance remaining: {remaining:.2f}")
                        except (ValueError, AttributeError):
                            pass

                    # Check for completion/status messages
                    if "completed" in message.lower():
                        print(f"   {message}")
                        progress_data["status"] = "completed"
                    elif "timed out" in message.lower():
                        print(f"   {message}")
                        progress_data["status"] = "timeout"
                    elif "failed" in message.lower() or "error" in message.lower():
                        print(f"   {message}")
                        progress_data["status"] = "error"
                    elif "feedback #" in message.lower():
                        if progress % 100 == 0:  # Show every 100th feedback
                            print(f"   Progress update #{int(progress)}")
                    else:
                        print(f"   Status: {message}")

                    # Send progress update to Gemini for action tools
                    if is_action_tool:
                        current_time = time.time()

                        # Throttle updates: send every 5 seconds OR on significant distance change
                        # More conservative to avoid overwhelming WebSocket connection
                        should_send = False
                        if current_time - last_sent_time > 5.0:
                            should_send = True
                        elif distance is not None and last_distance is not None:
                            if abs(last_distance - distance) > 1.0:  # >1.0m change
                                should_send = True
                        elif last_distance is None and distance is not None:
                            should_send = True  # First distance update

                        if should_send:
                            try:
                                # Send intermediate progress response to Gemini
                                progress_response = types.FunctionResponse(
                                    name=function_call.name,
                                    id=function_call.id,
                                    response=progress_data,
                                    will_continue=True  # More responses coming
                                )
                                await self.session.send_tool_response(
                                    function_responses=[progress_response]
                                )
                                print(f"📤 Sent progress to Gemini: {progress_data.get('distance_remaining', 'N/A')}m")
                                last_sent_time = current_time
                                if distance is not None:
                                    last_distance = distance
                            except Exception as e:
                                # WebSocket errors during progress updates are non-fatal
                                # The final result will still be sent when action completes
                                print(f"⚠️ Failed to send progress to Gemini (non-fatal): {e}")
                                # Stop trying to send more progress after connection issues
                                is_action_tool = False

            # Execute the tool call through MCP server with progress callback
            # Add custom timeout for navigation actions (5 minutes instead of default 2 minutes)
            tool_args = function_call.args.copy()
            if function_call.name == "navigate_to_location":
                tool_args["timeout"] = 300.0  # 5 minutes for navigation

            result = await self.mcp_session.call_tool(
                name=function_call.name,
                arguments=tool_args,
                progress_callback=progress_handler,  # Capture progress notifications and forward to Gemini
            )

            # Check for structured content first (preferred for complex data)
            if hasattr(result, "structuredContent") and result.structuredContent:
                import json
                result_text = json.dumps(result.structuredContent, indent=2)
            else:
                # Convert MCP result to JSON-serializable format
                result_content = []
                if hasattr(result, "content"):
                    for content_item in result.content:
                        # Handle TextContent objects with 'text' attribute
                        if hasattr(content_item, "text"):
                            result_content.append(content_item.text)
                        # Handle objects that can be dumped to dict
                        elif hasattr(content_item, "model_dump"):
                            dumped = content_item.model_dump()
                            if isinstance(dumped, dict):
                                result_content.append(str(dumped))
                            else:
                                result_content.append(str(dumped))
                        # Handle plain strings
                        elif isinstance(content_item, str):
                            result_content.append(content_item)
                        # Handle dicts
                        elif isinstance(content_item, dict):
                            result_content.append(str(content_item))
                        # Fallback to string conversion
                        else:
                            result_content.append(str(content_item))

                # Join content items into a single result string
                result_text = "\n".join(result_content) if result_content else str(result)

            print(f"\n📥 Raw result_text:\n{result_text}\n")

            # Format response for Gemini - extract status if available
            try:
                import json
                result_data = json.loads(result_text)
                response_data = result_data
            except (json.JSONDecodeError, ValueError):
                response_data = {"result": result_text}  # Fallback to string wrapper
                
            #print(response_data)

            # Send final response to Gemini
            # For action tools, use will_continue=False to signal completion
            function_responses = [
                types.FunctionResponse(
                    name=function_call.name,
                    id=function_call.id,
                    response=response_data,
                    will_continue=False if is_action_tool else None  # Signal final response for action tools
                )
            ]
            print(function_responses)
            try:
                await self.session.send_tool_response(function_responses=function_responses)
                print(f"✅ Sent final result to Gemini for {function_call.name}")
            except Exception as e:
                print(f"🔴 Error sending tool response: {e}")

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
                print("🔄 Sent WebSocket keepalive")
            except asyncio.CancelledError:
                # Task was cancelled, exit gracefully
                break
            except Exception as e:
                # Keepalive failures are non-critical
                print(f"⚠️ Keepalive failed (non-fatal): {e}")

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
        """
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
                            self.audio_in_queue.put_nowait(part.inline_data.data)

                        # Handle text responses
                        if part.text:
                            text_content = part.text
                            if first_text:
                                print(f"\n🤖 > {text_content}", end="", flush=True)
                                first_text = False
                            else:
                                print(text_content, end="", flush=True)
                            turn_text += text_content
                    continue

                # Fallback: Handle text responses from Gemini (for backward compatibility)
                if text_content := response.text:
                    if first_text:
                        print(f"\n🤖 > {text_content}", end="", flush=True)
                        first_text = False
                    else:
                        print(text_content, end="", flush=True)
                    turn_text += text_content

                # Handle server content (currently disabled)
                """
                server_content = response.server_content
                if server_content is not None:
                    self.handle_server_content(server_content)
                    continue
                """

                # Handle tool calls from Gemini
                tool_call = response.tool_call
                if tool_call is not None:
                    await self.handle_tool_call(tool_call)

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
                                async with self.mic_lock:
                                    self.mic_active = True
                                    audio_playing = False
                                    print("🎤 Microphone unmuted - audio playback complete")
                            else:
                                audio_playing = False
                    continue

                # Update last audio time
                self.last_audio_chunk_time = asyncio.get_event_loop().time()

                # If this is the first audio chunk in a sequence, mute the microphone (if enabled)
                if not audio_playing:
                    if self.active_muting:
                        async with self.mic_lock:
                            self.mic_active = False
                            audio_playing = True
                            print("🔇 Microphone muted while audio is playing")

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
                    async with self.mic_lock:
                        self.mic_active = True
                        audio_playing = False
                        print("🎤 Microphone unmuted after audio error")
                else:
                    audio_playing = False
                await asyncio.sleep(0.1)

    async def run(self):
        """
        Main execution method that sets up and runs the Gemini Live session.

        Connects to MCP server, configures tools, and starts all async tasks
        for audio/video processing and communication.
        """

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

                # Action tools that need NON_BLOCKING behavior for streaming progress
                action_tools = ['send_action_goal', 'navigate_to_location']

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

                    # Set behavior to NON_BLOCKING for action tools to enable streaming progress
                    if tool.name in action_tools:
                        tool_description["behavior"] = "NON_BLOCKING"

                    functional_tools.append(tool_description)

                # Configure Gemini Live tools (MCP tools + built-in capabilities)
                tools = [
                    {
                        "function_declarations": functional_tools,
                        "code_execution": {},  # Enable code execution
                        "google_search": {},  # Enable web search
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
                        self.audio_in_queue = asyncio.Queue(maxsize=50)  # Audio from Gemini (buffer for smooth playback)
                        self.out_queue = asyncio.Queue(maxsize=3)  # Data to Gemini (small buffer for low latency)

                        # Start all async tasks
                        send_text_task = task_group.create_task(self.send_text())
                        task_group.create_task(self.send_realtime())
                        task_group.create_task(self.websocket_keepalive())  # Prevent timeout during long operations
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
                    pass
                except asyncio.ExceptionGroup as exception_group:
                    # Handle any errors that occurred in the task group
                    if hasattr(self, 'audio_stream') and self.audio_stream is not None:
                        self.audio_stream.close()
                    traceback.print_exception(exception_group)


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
    args = parser.parse_args()

    # List available audio devices for debugging
    print(f"\n🔧 Initializing in '{args.mode}' mode with video='{args.video}' and responses='{args.responses}'")
    list_audio_devices()

    # Initialize and run the audio loop
    audio_loop = AudioLoop(
        mode=args.mode,
        video_mode=args.video,
        response_modality=args.responses,
        active_muting=args.active_muting,
    )
    asyncio.run(audio_loop.run())
