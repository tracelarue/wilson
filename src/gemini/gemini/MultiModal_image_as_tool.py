#!/usr/bin/env python3.11

"""
MultiModal Gemini Node for ROS2

Handles audio/video communication with Gemini API and navigation commands.

Setup: pip install google-genai opencv-python pyaudio pillow mss
"""

import os
import asyncio
import base64
import io
import json
import traceback
import cv2
import pyaudio
import PIL.Image
import mss
import argparse
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import RealtimeInputConfig, ActivityHandling
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math
import numpy as np
from scipy import signal
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg      import Point
from visualization_msgs.msg import Marker

pya = pyaudio.PyAudio()

# Audio resampling functions
def resample_audio(audio_data, original_rate, target_rate):
    """Resample audio data from original_rate to target_rate"""
    if original_rate == target_rate:
        return audio_data
    else:       
    # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate resampling ratio
        ratio = target_rate / original_rate
        
        # Calculate new length
        new_length = int(len(audio_array) * ratio)
        
        # Resample the audio
        resampled = signal.resample(audio_array, new_length)
        
        # Convert back to int16
        resampled = resampled.astype(np.int16)
        
        # Convert back to bytes
        return resampled.tobytes()



# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables from .env file in wilson directory
wilson_dir = os.path.join(os.path.expanduser('~'), 'wilson')
env_path = os.path.join(wilson_dir, '.env')
load_dotenv(dotenv_path=env_path)

# API configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
MODEL = "models/gemini-2.5-flash-live-preview"
DEFAULT_MODE = "none"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=GOOGLE_API_KEY,
)

# Define navigation tool
navigate_to_location = {
    "name": "navigate_to_location",
    "description": "Navigate the robot to a specific location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location name to navigate to (e.g., 'kitchen', 'living room', 'bedroom')"
            },
            "x": {
                "type": "number",
                "description": "The x-coordinate on the map"
            },
            "y": {
                "type": "number",
                "description": "The y-coordinate on the map"
            }
        },
        "required": ["location", "x", "y"]
    },
}

image_input = {
    "name": "image_input",
    "description": "Process an image from the camera",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "What the user wants to know about the image"
            }
        },
        "required": ["prompt"]
    },
}

get_distance = {
    "name": "get_distance",
    "description": "Calculate the distance to an object in the image",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Return a bounding box for the user-requested object. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."
            }
        },
        "required": ["prompt"]
    },
}

# Tools configuration
tools = [{'google_search': {}}, {'code_execution': {}}, {"function_declarations": [navigate_to_location, image_input, get_distance]}]

# Gemini API configuration
CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    context_window_compression=(
        types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow(),
        )
    ),
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
        )
    ),
    system_instruction = types.Content(
        parts=[
            types.Part(
                text="""Your name is Wilson.   
                        You can use the google search tool to find information on the internet.
                        You have access to the camera and images through calling the image_input tool.
                        You can use the get_distance tool to find the distance to objects in the image.
                        You can also navigate to specific locations in the house.
                        When asked to navigate somewhere, use the navigate_to_location tool.
                        Available locations:
                        - Mini fridge (x=0.5, y=0)
                        - Living room (x=-2, y=-2.5)
                        - Kitchen (x=-0.5, y=3)
                        """
            )
        ]
    ),
    tools=tools,
    realtime_input_config = RealtimeInputConfig(
        activity_handling = types.ActivityHandling.NO_INTERRUPTION,
    ),
)



class MultiModalGeminiNode(Node):
    def __init__(self, mode, video_mode=DEFAULT_MODE):
        super().__init__('MultiModalGeminiNode')
        self.video_mode = video_mode
        self.mode = mode
        self.format = pyaudio.paInt16
        self.chunk_size = 11520
        self.received_audio_buffer = 11520
        
        if self.mode == "sim":
            # Audio and Video constants
            self.mic_channels = 1
            self.speaker_channels = 1
            mic_info = pya.get_default_input_device_info()
            self.mic_index = mic_info['index']  # Using default microphone
            self.mic_sample_rate = 16000  # Hardware sample rate for microphone
            self.api_sample_rate = 16000  # Sample rate required by Gemini API
            self.speaker_sample_rate = 24000  # Hardware sample rate for speakers
            self.api_output_sample_rate = 24000  # Adjusted to match the actual API output rate
            self.declare_parameter("h_fov",1.089)
            self.declare_parameter("aspect_ratio",4.0/3.0)
            self.declare_parameter("camera_frame",'camera_link_optical')
            self.image_height = 480
            self.image_width = 640
            self.h_fov = self.get_parameter('h_fov').get_parameter_value().double_value
            self.v_fov = self.h_fov/self.get_parameter('aspect_ratio').get_parameter_value().double_value
            self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        
        if self.mode == "robot":
            self.mic_channels = 1
            self.speaker_channels = 2
            self.mic_index = 2 # Using 2 for microphone
            self.speaker_index = 1  # Using 1 for speakers
            self.mic_sample_rate = 48000  # Hardware sample rate for microphone
            self.api_sample_rate = 16000  # Sample rate required by Gemini API
            self.speaker_sample_rate = 48000  # Hardware sample rate for speakers
            self.api_output_sample_rate = 24000  # Adjusted to match the actual API output rate


        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.send_text_task = None
        self.receive_audio_task = None
        self.play_task = None
        
        # Control flags for audio management
        self.mic_active = True
        self.mic_lock = asyncio.Lock()
        
        # Logger setup
        self.logger = logging.getLogger('gemini_node')

        # Create image subscription
        self.latest_image = None
        self.bridge = CvBridge()
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        self.cam_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            image_qos
        )
        self.depth_cam_subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_image_callback,
            image_qos
        )
        self.object_3d_pub = self.create_publisher(Point, "/object_3d_position", 1)
        self.object_marker_pub = self.create_publisher(Marker, "/object_marker", 1)




        # Create the action client for navigation
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.wait_for_nav_server()

    # Text input processing
    async def send_text(self):
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)
    
    # Message sending to Gemini API
    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            if msg.get('mime_type') == 'image/jpeg':
                print(f"📸 Sent image to Gemini")
            await self.session.send(input=msg)

    # Audio capture
    async def listen_audio(self):
        try:
            print("🎤 Initializing microphone...")
            
            mic_info = pya.get_default_input_device_info()
            print(f"🎤 Using microphone: {mic_info['name']}")
            
            self.audio_stream = await asyncio.to_thread(
                pya.open,
                format=self.format,
                channels=self.mic_channels,
                rate=self.mic_sample_rate,  # Use hardware sample rate (48kHz)
                input=True,
                input_device_index=self.mic_index,
                frames_per_buffer=self.chunk_size,
            )
            print("🎤 Microphone initialized successfully")
        
            while True:
                try:
                    # Check if mic should be active
                    async with self.mic_lock:
                        mic_currently_active = self.mic_active
                    
                    if mic_currently_active:
                        # Small sleep to prevent CPU overuse
                        await asyncio.sleep(0.01)
                        # Read audio at hardware rate (48kHz)
                        data = await asyncio.to_thread(self.audio_stream.read, self.chunk_size, exception_on_overflow=False)
                        
                        # Resample from hardware rate (48kHz) to API rate (16kHz)
                        resampled_data = resample_audio(data, self.mic_sample_rate, self.api_sample_rate)
                        
                        # Send resampled data to API
                        await self.out_queue.put({"data": resampled_data, "mime_type": "audio/pcm"})
                    else:
                        # When muted, don't let data accumulate in buffers
                        await asyncio.sleep(0.1)
                        # Read and discard data while muted to prevent buffer buildup
                        await asyncio.to_thread(self.audio_stream.read, self.chunk_size, exception_on_overflow=False)
                except Exception as e:
                    print(f"🔴 Microphone error: {str(e)}")
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"🔴 Failed to initialize microphone: {str(e)}")
            traceback.print_exc()
        finally:
            if hasattr(self, 'audio_stream') and self.audio_stream:
                self.audio_stream.close()

    # Receive and process responses from Gemini API
    async def receive_audio(self):
        try:
            print("🤖 Initializing Gemini response handler")
            
            msg_counter = 0
            tool_call_counter = 0
            
            while True:
                try:
                    turn = self.session.receive()
                    
                    async for response in turn:
                        msg_counter += 1
                        if data := response.data:
                            self.audio_in_queue.put_nowait(data)
                            continue
                            
                        if text := response.text:
                            print(text, end="")
                            
                        # Process tool calls
                        if hasattr(response, 'tool_call') and response.tool_call:
                            tool_call_counter += 1
                            print(f"\n🛠️ Tool call received (#{tool_call_counter})")
                            
                            function_responses = []
                            for fc in response.tool_call.function_calls:
                                print(f"📋 Function: {fc.name}, ID: {fc.id}")
                                print(f"📋 Arguments: {fc.args}")
                                
                                # Process the tool call and get result
                                result = self.get_tool_response(fc.name, fc.args)
                                
                                function_response = types.FunctionResponse(
                                    id=fc.id,
                                    name=fc.name,
                                    response={"result": result}
                                )
                                function_responses.append(function_response)

                            # Send all function responses back to the model
                            if function_responses:
                                print(f"📤 Sending tool responses back to Gemini")
                                await self.session.send_tool_response(function_responses=function_responses)
                                
                except Exception as e:
                    print(f"🔴 Error receiving from Gemini: {str(e)}")
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print("🛑 Receive audio task cancelled")
            raise
        except Exception as e:
            print(f"🔴 Fatal error in receive_audio: {str(e)}")
            traceback.print_exc()

    # Play audio responses from Gemini
    async def play_audio(self):
        try:
            print("🔊 Initializing audio output...")
            
            output_info = pya.get_default_output_device_info()
            print(f"🔊 Using speaker: {output_info['name']}")
            
            output_device_index = self.speaker_index if hasattr(self, 'speaker_index') else None
            stream = await asyncio.to_thread(
                pya.open,
                format=self.format,
                channels=self.speaker_channels,
                rate=self.speaker_sample_rate,  # Use hardware sample rate (48kHz)
                output=True,
                output_device_index=output_device_index,
                frames_per_buffer=self.received_audio_buffer,
            )
            print("🔊 Audio output initialized successfully")
            
            audio_chunks_played = 0
            audio_playing = False
            last_audio_time = asyncio.get_event_loop().time()
            
            while True:
                try:
                    # Add a timeout to detect stalled audio
                    try:
                        bytestream = await asyncio.wait_for(self.audio_in_queue.get(), timeout=5.0)
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
                        await asyncio.sleep(.25)  # Reduced from 1.0 to 0.5 for better responsiveness
                    
                    try:
                        # Resample from API output rate to speaker rate
                        resampled_output = resample_audio(bytestream, self.api_output_sample_rate, self.speaker_sample_rate)
                        
                        # Play the resampled audio
                        await asyncio.to_thread(stream.write, resampled_output)
                        audio_chunks_played += 1
                    except Exception as audio_err:
                        print(f"🔴 Audio playback error: {str(audio_err)}")
                        # Continue to next chunk rather than breaking completely
                    
                    # Check if the queue is empty (reached end of audio)
                    if self.audio_in_queue.qsize() == 0:
                        # Wait briefly to make sure no more chunks are coming
                        await asyncio.sleep(3)  # Reduced from 1.5 to 0.5 for better responsiveness
                        if self.audio_in_queue.qsize() == 0:
                            # No more audio chunks, re-enable microphone
                            async with self.mic_lock:
                                if not self.mic_active:
                                    self.mic_active = True
                                    audio_playing = False
                                    print("🎤 Microphone unmuted after audio playback")
                    
                except Exception as e:
                    print(f"🔴 Audio output error: {str(e)}")
                    traceback.print_exc()  # Add stack trace for better debugging
                    
                    # Re-enable microphone in case of error
                    async with self.mic_lock:
                        self.mic_active = True
                        audio_playing = False
                        print("🎤 Microphone unmuted after audio error")
                    
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"🔴 Failed to initialize audio output: {str(e)}")
            traceback.print_exc()
        finally:
            # Make sure to re-enable microphone before closing
            async with self.mic_lock:
                self.mic_active = True
                print("🎤 Microphone unmuted during cleanup")
                
            if 'stream' in locals() and stream:
                stream.stop_stream()
                stream.close()
    
    # Navigation Functions
    def wait_for_nav_server(self):
        server_ready = self.nav_to_pose_client.wait_for_server(timeout_sec=2.0)
        if server_ready:
            self.get_logger().info('Nav2 action server is available!')
        else:
            self.get_logger().warn('Nav2 action server not available after waiting. Will try again when needed.')
        return server_ready
            
    def send_nav_goal(self, x, y, yaw=0.0):
        goal_msg = NavigateToPose.Goal()
        
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0
        
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(yaw/2)
        goal_msg.pose.pose.orientation.w = math.cos(yaw/2)
        
        self.get_logger().info(f'Sending navigation goal to x={x}, y={y}, yaw={yaw}')
        
        self.nav_to_pose_client.wait_for_server()
        self._send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.nav_feedback_callback
        )
        self._send_goal_future.add_done_callback(self.nav_goal_response_callback)
        return True
        
    def nav_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by Nav2 action server')
            return
        
        self.get_logger().info('Goal accepted by Nav2 action server')
        
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.nav_result_callback)
    
    def nav_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        if status == 4:  # SUCCEEDED
            self.get_logger().info('Navigation succeeded')
        else:
            self.get_logger().info(f'Navigation finished with status {status}')
    
    def nav_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Distance remaining: {feedback.distance_remaining}')

    # Image Functions
    def image_callback(self,msg):
        self.latest_image = msg

    def depth_image_callback(self, msg):
        self.latest_depth_image = msg

    def gemini_image_understanding(self, prompt):
        if self.latest_image is None:
            self.image_response = "No image available from the camera. Please make sure the camera is publishing images."
            print(self.image_response)
            return
        client = genai.Client(api_key=GOOGLE_API_KEY)

        cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
        frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        image_bytes = image_io.read()

        response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
            ),
            prompt
        ]
        )

        self.image_response = response.text

    def get_distance(self):
        height = 480
        width = 640
        self.json_output = self.image_response
        lines = self.json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                self.json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
                self.json_output = self.json_output.split("```")[0]  # Remove everything after the closing "```"
                break  # Exit the loop once "```json" is found
        self.bounding_boxes = self.json_output
        for i, bounding_box in enumerate(json.loads(self.bounding_boxes)):
            # Convert normalized coordinates to absolute coordinates
            norm_y1 = int(bounding_box["box_2d"][0]/1000 * height)
            norm_x1 = int(bounding_box["box_2d"][1]/1000 * width)
            norm_y2 = int(bounding_box["box_2d"][2]/1000 * height)
            norm_x2 = int(bounding_box["box_2d"][3]/1000 * width)
            self.y = ((norm_y1 + norm_y2) // 2)
            self.x = ((norm_x1 + norm_x2) // 2)
            print(f"ymin: {norm_y1}, xmin: {norm_x1}, ymax: {norm_y2}, xmax: {norm_x2}")
            print(f"Center of bounding box: ({self.y}, {self.x})")
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
            self.depth = cv_image[self.y, self.x]
            print(f"Distance = {self.depth} meters")

    def get_3d_position(self):
        # Calculate angle per pixel
        angle_per_pixel_x = self.h_fov / self.image_width
        angle_per_pixel_y = self.v_fov / self.image_height

        # Offset from image center
        x_offset = self.x - self.image_width / 2
        y_offset = self.y - self.image_height / 2

        # Calculate angles in radians
        x_ang = x_offset * angle_per_pixel_x
        y_ang = y_offset * angle_per_pixel_y

        # Project to 3D (assuming depth is along camera Z axis)
        z = self.depth
        x = z * math.tan(x_ang)
        y = z * math.tan(y_ang)

        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        self.object_3d_pub.publish(p)

        m = Marker()
        m.header.frame_id = self.camera_frame
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.1
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0

        self.object_marker_pub.publish(m)

    def get_tool_response(self, tool_name, parameters):

        if tool_name == "navigate_to_location":
            location = parameters.get("location", "unknown")
            x = parameters.get("x", 0.0)
            y = parameters.get("y", 0.0)
            
            self.get_logger().info(f'Navigating to {location} at coordinates ({x}, {y})')
            
            # Check if Nav2 action server is available
            if not self.nav_to_pose_client.server_is_ready():
                server_ready = self.wait_for_nav_server()
                if not server_ready:
                    return f"Navigation to {location} failed: Nav2 action server not available."
            
            # Send the navigation goal
            success = self.send_nav_goal(x, y)
            
            if not success:
                return f"Failed to send navigation goal to {location}."
            
        elif tool_name == "image_input":
            prompt = parameters.get("prompt")
            self.gemini_image_understanding(prompt)
            response = self.image_response
            return f"Image analysis result: {response}"

        elif tool_name == "get_distance":
            prompt = parameters.get("prompt")
            if self.latest_depth_image is None:
                return "No depth image available. Please ensure the depth camera is publishing images."
            self.gemini_image_understanding(prompt)
            self.get_distance()
            self.get_3d_position()
            return f"Distance calculation result: {self.depth} meters"

        else:
            return "Unknown tool."
            
    async def ros_spin_task(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)
            await asyncio.sleep(0.01)

    # Main session runner
    async def run_session(self):
        print("\n🚀 Connecting to Gemini API...")
        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                while rclpy.ok():
                    self.session = session
                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue()
                    print("🔄 Starting audio and processing tasks...")
                    # Create tasks
                    tasks = []
                    # Start ROS spin in a background task
                    tasks.append(asyncio.create_task(self.ros_spin_task()))
                    send_text_task = asyncio.create_task(self.send_text())
                    tasks.append(send_text_task)
                    tasks.append(asyncio.create_task(self.send_realtime()))
                    tasks.append(asyncio.create_task(self.listen_audio()))
                    tasks.append(asyncio.create_task(self.receive_audio()))
                    tasks.append(asyncio.create_task(self.play_audio()))
                    print("✅ All tasks started successfully!")
                    # Wait for the send_text_task to complete (user exit)
                    await send_text_task
                    print("\n🛑 Stopping system...")
                    # Cancel all other tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    # Wait for all tasks to be cancelled
                    await asyncio.gather(*tasks, return_exceptions=True)
                    print("✅ All tasks stopped cleanly")
                    break
        except Exception as e:
            print(f"🔴 Failed to connect to Gemini API: {str(e)}")
            print("Check your API key and internet connection.")
    



def main(args=None):
    parser = argparse.ArgumentParser(description='MultiModal Gemini Node')
    parser.add_argument('--mode', choices=['robot','sim'], default='sim')
    parser.add_argument('--video', choices=['none', 'camera', 'screen'], default=DEFAULT_MODE, 
                        help='Video input mode (none, camera, or screen)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parsed_args, remaining = parser.parse_known_args(args)
    
    # Set logging level based on command line argument
    if parsed_args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print("🔍 Debug logging enabled")
    
    print("\n===== 🤖 Gemini Multi-Modal ROS2 Node =====")
    print(f"🔑 API Key: {'✓ Set' if GOOGLE_API_KEY else '❌ Missing'}")
    
    
    rclpy.init(args=remaining)
    node = MultiModalGeminiNode(video_mode=parsed_args.video, mode=parsed_args.mode)
    
    try:
        asyncio.run(node.run_session())
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user (KeyboardInterrupt)")
    finally:
        print("\n🛑 Shutting down Gemini Multi-Modal Node")
        # Clean up any audio resources
        if hasattr(node, 'audio_stream') and node.audio_stream:
            node.audio_stream.close()

        # Destroy the node and shut down ROS
        node.destroy_node()
        rclpy.shutdown()
        print("✅ ROS2 node shut down successfully")

if __name__ == "__main__":
    main()
