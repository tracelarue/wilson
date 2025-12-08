"""Core session management and orchestration for Gemini Live."""

import asyncio
import sys
import traceback
import pyaudio
from google.genai import types
from mcp import ClientSession
from mcp.client.stdio import stdio_client

try:
    from .config import (
        MODEL,
        DEFAULT_VIDEO_MODE,
        DEFAULT_RESPONSE_MODALITY,
        CHUNK_SIZE,
        OUTPUT_CHUNK_SIZE,
        SEND_SAMPLE_RATE,
        RECEIVE_SAMPLE_RATE,
        load_system_instructions,
        server_params,
        client,
    )
    from .audio_utils import BufferPool, find_audio_device
    from .video_capture import VideoCaptureHandler
    from .audio_streams import AudioStreamHandler
    from .tool_handler import ToolHandler
except ImportError:
    from config import (
        MODEL,
        DEFAULT_VIDEO_MODE,
        DEFAULT_RESPONSE_MODALITY,
        CHUNK_SIZE,
        OUTPUT_CHUNK_SIZE,
        SEND_SAMPLE_RATE,
        RECEIVE_SAMPLE_RATE,
        load_system_instructions,
        server_params,
        client,
    )
    from audio_utils import BufferPool, find_audio_device
    from video_capture import VideoCaptureHandler
    from audio_streams import AudioStreamHandler
    from tool_handler import ToolHandler

if sys.version_info < (3, 11, 0):
    import exceptiongroup
    import taskgroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup


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
        self.mcp_session = None  # MCP client session
        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

        # Control flags for audio management
        self.mic_active = True
        self.mic_lock = asyncio.Lock()

        # Audio streaming state tracking
        self.audio_stream_active = False
        self.audio_stream_lock = asyncio.Lock()

        # Handler instances (initialized later)
        self.video_handler = None
        self.audio_handler = None
        self.tool_handler = None

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

    async def send_realtime(self):
        """
        Send real-time data (audio/video) from output queue to Gemini session.

        Continuously processes messages from the output queue and sends them to Gemini.
        """
        while True:
            message = await self.out_queue.get()
            await self.session.send_realtime_input(media=message)

    async def receive_audio(self):
        """
        Background task to receive responses from Gemini session.

        Processes audio data, text responses, and tool calls from Gemini.
        Handles interruptions by clearing the audio queue.

        Tool calls run as async tasks to avoid blocking the receive loop.
        This ensures WebSocket messages are processed continuously.
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

                # Handle tool calls from Gemini - NON-BLOCKING
                # Create async task instead of awaiting inline to keep receive loop active
                tool_call = response.tool_call
                if tool_call is not None:
                    # Wait for any existing tool calls to complete first (sequential execution)
                    # This prevents duplicate/parallel tool calls for the same action
                    # NON_BLOCKING behavior still allows conversation to continue during execution
                    if tool_call_tasks:
                        await asyncio.wait(tool_call_tasks)
                        tool_call_tasks.clear()

                    # Now create new task for this tool execution
                    task = asyncio.create_task(self.tool_handler.handle_tool_call(tool_call))
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

                for tool in available_tools.tools:
                    tool_description = {"name": tool.name, "description": tool.description}

                    # Add NON_BLOCKING behavior for long-running action tools
                    if tool.name in ["send_action_goal"]:
                        tool_description["behavior"] = "NON_BLOCKING"

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
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zubenelgenubi")
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

                        # Initialize handler instances
                        self.video_handler = VideoCaptureHandler(self.out_queue)
                        self.audio_handler = AudioStreamHandler(
                            mode=self.mode,
                            format=self.format,
                            mic_channels=self.mic_channels,
                            speaker_channels=self.speaker_channels,
                            mic_index=self.mic_index,
                            speaker_index=self.speaker_index,
                            mic_sample_rate=self.mic_sample_rate,
                            speaker_sample_rate=self.speaker_sample_rate,
                            audio_in_queue=self.audio_in_queue,
                            out_queue=self.out_queue,
                            mic_lock=self.mic_lock,
                            audio_stream_lock=self.audio_stream_lock,
                            active_muting=self.active_muting,
                        )
                        self.tool_handler = ToolHandler(mcp_session, session)

                        # Start all async tasks
                        send_text_task = task_group.create_task(self.send_text())
                        task_group.create_task(self.send_realtime())
                        task_group.create_task(self.audio_handler.listen_audio())

                        # Start video capture based on selected mode
                        if self.video_mode == "camera":
                            task_group.create_task(self.video_handler.get_frames())
                        elif self.video_mode == "screen":
                            task_group.create_task(self.video_handler.get_screen())

                        # Start audio processing tasks
                        task_group.create_task(self.receive_audio())
                        task_group.create_task(self.audio_handler.play_audio())

                        # Wait for user to quit (send_text_task completes when user types 'q')
                        await send_text_task
                        raise asyncio.CancelledError("User requested exit")

                except asyncio.CancelledError:
                    # Normal exit when user types 'q'
                    pass
                except asyncio.ExceptionGroup as exception_group:
                    # Handle any errors that occurred in the task group
                    if hasattr(self.audio_handler, 'audio_stream') and self.audio_handler.audio_stream is not None:
                        self.audio_handler.audio_stream.close()
                    traceback.print_exception(exception_group)
