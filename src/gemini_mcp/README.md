# Gemini Live with ROS MCP Server

Control ROS robots with natural language voice commands using Google's Gemini Live API.

**Pre-requisites** See the [installation instructions](../../../docs/installation.md) for detailed setup steps.

**Tested In:** Ubuntu 22.04, Python 3.10, ROS2 Humble
(Only works in Ubuntu)

## Quick Setup

1. **Install ROS MCP Server**: Follow the [installation guide](../../../docs/installation.md)

2. **Install system dependencies** (required for audio):
```bash
sudo apt-get update
sudo apt-get install python3-dev portaudio19-dev
```

3. **Install additional dependencies for Gemini Live**:

```bash
# Navigate to the ros-mcp-server root directory
cd ros-mcp-server

# Install the additional dependencies needed for Gemini Live
uv pip install google-genai pyaudio python-dotenv mss exceptiongroup taskgroup
```

**Note**: The main ros-mcp-server project already includes most dependencies (mcp, opencv-python, pillow). We only need to add the Gemini-specific packages.

4. **Get Google API Key**: Visit [Google AI Studio](https://aistudio.google.com) and create an API key

5. **Create a `.env` file in the `gemini_live` folder**:
```bash
cd examples/2_gemini/gemini_live
```

```env
GOOGLE_API_KEY="your_google_api_key_here"
```
Replace with your API key.

6. **Create `mcp_config.json` in the gemini_live folder**:
Replace `/absolute/path/to/ros-mcp-server` with your actual path.
```json
{
   "mcpServers": {
      "ros-mcp-server": {
         "command": "uv",
         "args": [
         "--directory",
         "/absolute/path/to/ros-mcp-server", 
         "run",
         "server.py"
         ]
      }
   }
}
```

## Usage

**Start Gemini Live:**
```bash
# Navigate to the gemini_live folder
cd ros-mcp-server/examples/2_gemini/gemini_live

# Run the client (with defaults: no video, audio responses, mic muting enabled)
uv run gemini_client.py
```

**Command-line options:**

**Video modes** (`--video`):
- `--video=none` - Audio only (default)
- `--video=camera` - Include camera feed
- `--video=screen` - Include screen capture

**Response modes** (`--responses`):
- `--responses=TEXT` - Text responses only
- `--responses=AUDIO` - Audio responses (default)

**Microphone muting** (`--active-muting`):
- `--active-muting=true` - Mute mic during audio playback (default, prevents echo/feedback). Recommended if not using headphones.
- `--active-muting=false` - Keep mic active during audio playback

**Example usage:**
```bash
uv run gemini_client.py --video=camera --responses=TEXT --active-muting=false
```
Type `q` + Enter to quit.

## Test with Turtlesim

**Start rosbridge and turtlesim**:
```bash
# Terminal 1: Launch rosbridge
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```
```bash
# Terminal 2: Start turtlesim
ros2 run turtlesim turtlesim_node
```

**Try these voice commands:**
- "Connect to the robot on ip _ and port _ "
- "What ROS topics are available?"
- "Move the turtle forward at 1 m/s and 0 rad/s"
- "Rotate the turtle at 3 rad/s"
- "Change the pen color to red"


See [Turtlesim Tutorial](../../1_turtlesim/README.md) for more examples.

## Troubleshooting

**Not responding to voice?**
- Check microphone permissions and volume
- Test: `arecord -d 5 test.wav && aplay test.wav`

**Robot not moving?**
- Verify robot/simulation is running
- Check rosbridge is running: `ros2 launch rosbridge_server rosbridge_websocket_launch.xml`
- Check: `ros2 topic list`
- Ask: "List all available tools" to verify MCP connection

**API key errors?**
- Verify `.env` file exists with correct key
- Check key is active in Google AI Studio

**Dependency issues?**
- If you get import errors, make sure you installed the additional dependencies: `uv pip install google-genai pyaudio python-dotenv mss exceptiongroup taskgroup`

##
Contributed by Trace LaRue
traceglarue@gmail.com
[www.traceglarue.com](https://www.traceglarue.com)

