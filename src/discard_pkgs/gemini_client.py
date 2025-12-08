import asyncio
import base64
import io
import os
import sys
import traceback

import cv2
import pyaudio
import PIL.Image
import mss

import argparse
from dotenv import load_dotenv

from google import genai
from google.genai import types

from mcp_handler import MCPClient

load_dotenv()

if sys.version_info < (3, 11, 0):
    import taskgroup
    import exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# Audio configuration constants
AUDIO_FORMAT = pyaudio.paInt16  # 16-bit audio format
AUDIO_CHANNELS = 1  # Mono audio
SEND_SAMPLE_RATE = 16000  # Sample rate for sending audio to Gemini
RECEIVE_SAMPLE_RATE = 24000  # Sample rate for receiving audio from Gemini
CHUNK_SIZE = 1024  # Audio buffer size

# Gemini Live model and default settings
MODEL = "models/gemini-2.5-flash-live-preview"
DEFAULT_VIDEO_MODE = "none"  # Options: "camera", "screen", "none"
DEFAULT_RESPONSE_MODALITY = "AUDIO"  # Options: "TEXT", "AUDIO"

# System instructions to guide Gemini's behavior and tool usage.
system_instructions = """You are Wilson, a mobile robot.
                        You have access to the tools provided by ros_mcp_server.
                        To connect to the robot, use ip 192.168.52.129, port 9090. When successfuly connected, reply just "Succesfully connected".
                        When told to go to coordinates, use the publish_once tool to publish to /goal_pose.
                        - {'position': {'x': -1.0, 'y': 3.0, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}
                        When told to go to a location, use the publish_once tool to publish to /goal_pose.
                        Locations for /goal_pose are:
                        - kitchen = {'position': {'x': -1.0, 'y': 3.0, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}
                        - living_room = {'position': {'x': 0.0, 'y': 3.0, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}
                        - mini_fridge = {'position': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}
                        
                        
                        """

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GOOGLE_API_KEY"),
)


pya = pyaudio.PyAudio()


class AudioLoop:
    """
    Main class for handling Gemini Live audio/video interaction with MCP server integration.

    Manages real-time audio streaming, video capture, and tool calls through MCP
    """

    def __init__(self, video_mode=DEFAULT_VIDEO_MODE, response_modality=DEFAULT_RESPONSE_MODALITY):
        """
        Initialize the AudioLoop with specified video mode and response modality.

        Args:
            video_mode (str): Video input mode - "camera", "screen", or "none"
            response_modality (str): Response format - "TEXT" or "AUDIO"
        """
        self.video_mode = video_mode
        self.response_modality = response_modality

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

        # MCP client for robot control
        self.mcp_client = MCPClient()

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
                turns={"role": "user", "parts": [{"text": text or "."}]},
                turn_complete=True
            )

    def handle_server_content(self, server_content):
        """
        Process and display server content including code execution results.

        Args:
            server_content: Server response content from Gemini
        """
        model_turn = server_content.model_turn

        # Display executable code and execution results
        if model_turn:
            for part in model_turn.parts:
                # Show executable code blocks
                executable_code = part.executable_code
                if executable_code is not None:
                    print("-------------------------------")
                    print(f"``` python\n{executable_code.code}\n```")
                    print("-------------------------------")

                # Show code execution output
                code_execution_result = part.code_execution_result
                if code_execution_result is not None:
                    print("-------------------------------")
                    print(f"```\n{code_execution_result.output}\n```")
                    print("-------------------------------")

        # Display grounding metadata if available
        grounding_metadata = getattr(server_content, "grounding_metadata", None)
        if grounding_metadata is not None:
            print(grounding_metadata.search_entry_point.rendered_content)

        return

    async def handle_tool_call(self, tool_call):
        """
        Process tool calls from Gemini and execute them via MCP client.

        Args:
            tool_call: Tool call request from Gemini containing function calls
        """
        for function_call in tool_call.function_calls:
            # Execute the tool call through MCP server
            result = await self.mcp_client.session.call_tool(
                name=function_call.name,
                arguments=function_call.args,
            )
            # print(result)  # Uncomment to debug raw tool call results

            # Convert MCP result to JSON-serializable format
            # The result is a CallToolResult object with 'content' list
            result_content = []
            if hasattr(result, 'content'):
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        result_content.append(content_item.text)
                    elif hasattr(content_item, 'model_dump'):
                        result_content.append(content_item.model_dump())
                    else:
                        result_content.append(str(content_item))
            
            # Join content items into a single result string
            result_text = "\n".join(result_content) if result_content else str(result)

            # Format response for Gemini
            function_responses = [
                types.FunctionResponse(
                    name=function_call.name,
                    id=function_call.id,
                    response={"result": result_text},
                )
            ]

            # print('\n>>> ', function_responses)  # Uncomment to debug formatted responses
            await self.session.send_tool_response(function_responses=function_responses)

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

        while True:
            # Capture frame in separate thread
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            # Send frame at 1 second intervals
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

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

            # Send screenshot at 1 second intervals
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

    async def send_realtime(self):
        """
        Send real-time data (audio/video) from output queue to Gemini session.

        Continuously processes messages from the output queue and sends them to Gemini.
        """
        while True:
            message = await self.out_queue.get()
            await self.session.send_realtime_input(media=message)

    async def listen_audio(self):
        """
        Continuously capture audio from microphone and add to output queue.

        Sets up microphone input stream and reads audio data in chunks.
        """
        # Get default microphone info
        mic_info = pya.get_default_input_device_info()
        print("Microphone:", mic_info["name"])

        # Initialize audio input stream
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        # Configure overflow handling for debug vs release
        overflow_kwargs = {"exception_on_overflow": False} if __debug__ else {}

        # Continuously read audio data
        while True:
            # Check if mic should be active
            async with self.mic_lock:
                mic_currently_active = self.mic_active
            
            if mic_currently_active:
                # Small sleep to prevent CPU overuse
                await asyncio.sleep(0.01)
                # Read audio data
                audio_data = await asyncio.to_thread(
                    self.audio_stream.read, CHUNK_SIZE, **overflow_kwargs
                )
                await self.out_queue.put({"data": audio_data, "mime_type": "audio/pcm"})
            else:
                # When muted, don't let data accumulate in buffers
                await asyncio.sleep(0.1)
                # Read and discard data while muted to prevent buffer buildup
                await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **overflow_kwargs)

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

            async for response in turn:
                # Handle server content with model turn
                server_content = response.server_content
                if server_content and server_content.model_turn:
                    for part in server_content.model_turn.parts:
                        # Handle audio data from inline_data parts
                        if part.inline_data:
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
        """
        # Initialize audio output stream
        audio_stream = await asyncio.to_thread(
            pya.open,
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )

        audio_playing = False
        last_audio_time = asyncio.get_event_loop().time()

        # Continuously play audio from queue
        while True:
            try:
                # Add a timeout to detect stalled audio
                try:
                    audio_bytes = await asyncio.wait_for(self.audio_in_queue.get(), timeout=5.0)
                    last_audio_time = asyncio.get_event_loop().time()
                except asyncio.TimeoutError:
                    # No audio for 5 seconds, check if we need to ensure mic is unmuted
                    current_time = asyncio.get_event_loop().time()
                    if audio_playing and (current_time - last_audio_time) > 3.0:
                        print("⚠️ Audio timeout detected - resetting microphone state")
                        async with self.mic_lock:
                            self.mic_active = True
                            audio_playing = False
                            print("🎤 Microphone re-enabled after audio timeout")
                    continue
                
                # If this is the first audio chunk in a sequence, mute the microphone
                if not audio_playing:
                    async with self.mic_lock:
                        self.mic_active = False
                        audio_playing = True
                        print("🔇 Microphone muted while audio is playing")
                    
                    # Add a delay to ensure the mic is fully muted before audio starts
                    await asyncio.sleep(0.25)
                
                # Play the audio
                await asyncio.to_thread(audio_stream.write, audio_bytes)
                
                # Check if the queue is empty (reached end of audio)
                if self.audio_in_queue.qsize() == 0:
                    # Wait briefly to make sure no more chunks are coming
                    await asyncio.sleep(2)
                    if self.audio_in_queue.qsize() == 0:
                        # No more audio chunks, re-enable microphone
                        async with self.mic_lock:
                            if not self.mic_active:
                                self.mic_active = True
                                audio_playing = False
                                print("🎤 Microphone unmuted after audio playback")
            
            except Exception as e:
                print(f"🔴 Audio playback error: {str(e)}")
                # Re-enable microphone in case of error
                async with self.mic_lock:
                    self.mic_active = True
                    audio_playing = False
                    print("🎤 Microphone unmuted after audio error")
                await asyncio.sleep(0.1)

    async def run(self):
        """
        Main execution method that sets up and runs the Gemini Live session.

        Connects to MCP server, configures tools, and starts all async tasks
        for audio/video processing and communication.
        """
        # Connect to MCP server and get available tools
        await self.mcp_client.connect_to_server()
        available_tools = await self.mcp_client.session.list_tools()

        # Convert MCP tools to Gemini-compatible format
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
            system_instruction=types.Content(
                parts=[types.Part(text=system_instructions)]
            ),
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
                self.audio_in_queue = asyncio.Queue()  # Audio from Gemini
                self.out_queue = asyncio.Queue(
                    maxsize=5
                )  # Data to Gemini (limited size)

                # Start all async tasks
                send_text_task = task_group.create_task(self.send_text())
                task_group.create_task(self.send_realtime())
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
        default=DEFAULT_VIDEO_MODE,
        help="Video input source for visual context",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--response-modality",
        type=str,
        default=DEFAULT_RESPONSE_MODALITY,
        help="Response format from Gemini",
        choices=["TEXT", "AUDIO"],
    )
    args = parser.parse_args()

    # Initialize and run the audio loop
    audio_loop = AudioLoop(video_mode=args.mode, response_modality=args.response_modality)
    asyncio.run(audio_loop.run())
