"""
Gemini Live MCP Client - Main entry point

This module provides backward compatibility by re-exporting all components
from the refactored module structure.
"""

import argparse
import asyncio
import sys

# Support both relative and absolute imports for flexibility
try:
    # Try relative imports first (when used as a module)
    from .config import (
        AUDIO_FORMAT,
        AUDIO_CHANNELS,
        SEND_SAMPLE_RATE,
        RECEIVE_SAMPLE_RATE,
        CHUNK_SIZE,
        OUTPUT_CHUNK_SIZE,
        MODEL,
        DEFAULT_VIDEO_MODE,
        DEFAULT_RESPONSE_MODALITY,
        load_mcp_config,
        load_system_instructions,
        mcp_config,
        server_params,
        client,
    )
    from .audio_utils import (
        BufferPool,
        resample_audio,
        list_audio_devices,
        find_audio_device,
    )
    from .video_capture import VideoCaptureHandler
    from .audio_streams import AudioStreamHandler
    from .tool_handler import ToolHandler
    from .session_manager import AudioLoop
except ImportError:
    # Fall back to absolute imports (when run directly as a script)
    from config import (
        AUDIO_FORMAT,
        AUDIO_CHANNELS,
        SEND_SAMPLE_RATE,
        RECEIVE_SAMPLE_RATE,
        CHUNK_SIZE,
        OUTPUT_CHUNK_SIZE,
        MODEL,
        DEFAULT_VIDEO_MODE,
        DEFAULT_RESPONSE_MODALITY,
        load_mcp_config,
        load_system_instructions,
        mcp_config,
        server_params,
        client,
    )
    from audio_utils import (
        BufferPool,
        resample_audio,
        list_audio_devices,
        find_audio_device,
    )
    from video_capture import VideoCaptureHandler
    from audio_streams import AudioStreamHandler
    from tool_handler import ToolHandler
    from session_manager import AudioLoop

# Python 3.11 compatibility
if sys.version_info < (3, 11, 0):
    import exceptiongroup
    import taskgroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup


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
