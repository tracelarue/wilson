"""Configuration management for Gemini MCP client."""

import json
import os
import pyaudio
from dotenv import load_dotenv
from google import genai
from mcp import StdioServerParameters

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
