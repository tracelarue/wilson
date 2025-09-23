# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-live-preview"

DEFAULT_MODE = "none"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GOOGLE_API_KEY"),
)


pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        
        self.mcp_client = MCPClient()

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "🎤 message > ",
            )
            if text.lower() == "q":
                break

            await self.session.send(input=text or ".", end_of_turn=True)
            
    def handle_server_content(self, server_content):
        model_turn = server_content.model_turn
        if model_turn:
            for part in model_turn.parts:
                executable_code = part.executable_code
                if executable_code is not None:
                    print('-------------------------------')
                    print(f'``` python\n{executable_code.code}\n```')
                    print('-------------------------------')

                code_execution_result = part.code_execution_result
                if code_execution_result is not None:
                    print('-------------------------------')
                    print(f'```\n{code_execution_result.output}\n```')
                    print('-------------------------------')

        grounding_metadata = getattr(server_content, 'grounding_metadata', None)
        if grounding_metadata is not None:
            print(grounding_metadata.search_entry_point.rendered_content)

        return
    
    async def handle_tool_call(self, tool_call):
        for fc in tool_call.function_calls:
            result = await self.mcp_client.session.call_tool(
                name=fc.name,
                arguments=fc.args,
            )
            print(result)
            tool_response = types.LiveClientToolResponse(
                function_responses=[types.FunctionResponse(
                    name=fc.name,
                    id=fc.id,
                    response={'result':result},
                )]
            )

            print('\n>>> ', tool_response)
            await self.session.send(input=tool_response)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        print('Microphone:', mic_info['name'])
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print("🤖 ", end="")
                    print(text, end="")
                    
                server_content = response.server_content
                if server_content is not None:
                    self.handle_server_content(server_content)
                    continue

                tool_call = response.tool_call
                if tool_call is not None:
                    await self.handle_tool_call(tool_call)

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        await self.mcp_client.connect_to_server()
        available_tools = await self.mcp_client.session.list_tools()

        #for tool in available_tools.tools:
        #    print('--------------------------------\n')
        #    print(tool)
        #    print('\n--------------------------------\n')

        functional_tools = []
        for tool in available_tools.tools:
            tool_desc = {
                    "name": tool.name,
                    "description": tool.description
                }
            if tool.inputSchema["properties"]:
                tool_desc["parameters"] = {
                    "type": tool.inputSchema["type"],
                    "properties": {},
                }
                for param in tool.inputSchema["properties"]:
                    param_schema = tool.inputSchema["properties"][param]
                    
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
                        param_type = "string"  # fallback default
                    
                    param_def = {
                        "type": param_type,
                        "description": "",
                    }
                    
                    # Handle array types that need items field
                    if param_type == "array" and "items" in param_schema:
                        items_schema = param_schema["items"]
                        if "type" in items_schema:
                            param_def["items"] = {"type": items_schema["type"]}
                        else:
                            # Default to object for complex array items
                            param_def["items"] = {"type": "object"}
                    
                    tool_desc["parameters"]["properties"][param] = param_def
                
            if "required" in tool.inputSchema:
                tool_desc["parameters"]["required"] = tool.inputSchema["required"]
                
            functional_tools.append(tool_desc)
        #print(functional_tools)
        tools = [
            {
                'function_declarations': functional_tools,
                'code_execution': {},
                'google_search': {},
                },
        ]
        
        CONFIG = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
            )
        ),
        system_instruction = types.Content(
            parts=[
                types.Part(
                    text="""When told to go to a location, use publish_once to publish to goal pose.
                            Locations for /goal_pose are:
                            - kitchen = {'position': {'x': -1.0, 'y': 3.0, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}
                        """
                )
            ]
        ),
        tools=tools,
)
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except asyncio.ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())