# Gemini Live with MCP

Gemini Live enables realtime transcription and voice/vision interactions. 
This example shows how to use the ros-mcp-server with the Gemini Live API. 

Tested in ubuntu 22.04, python 3.10.12, ROS2 Humble.

# 1. **Installation**
Follow the instructions in [installation guide](docs/installation.md) to install the ros-mcp-server which includes this example. 

# 2. **Create and store Google API Key**:
Go to [aistudio.google.com](https://aistudio.google.com) and create an API key.
Create a `.env` file in the gemini_live.py directory and add your Google API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```

# 3. **Create MCP configuration file**:
Define the ros-mcp-server by creating a `mcp_config.json` file in the project directory. 
Example structure:
```json
{
  "mcpServers": {
    "ros-mcp-server": {
      "command": "uv",
      "args": [
        "--directory",
        "/<ABSOLUTE_PATH>/ros-mcp-server", 
        "run",
        "server.py"
      ]
    }
  }
}
```

## Usage

Open a terminal in the directory containing gemini_live.py using the following command:

```sh
uv run gemini_live.py --mode=none
# mode can be none, camera, screen
```

### Modes
- `none`: Runs without any visual input. 
- `screen`: Sends images of your screen to Gemini. 
- `camera`: Sends images from the camera to Gemini. 

## Response Modalities
Response modalitiy can be specified in gemini_live.py.
- `AUDIO`: Gemini will respond with audio.
- `TEXT` : Gemini will respond with text.

# 4. Test
With Gemini Live running, ask Gemini to list all available tools to confirm connection to the mcp server. 

You can test out your server with any robot that you have running. Just tell your AI to connect to the robot on its target IP. (Default is localhost, so you don't need to tell it to connect if the MCP server is installed on the same machine as your ROS)

✅ **Tip:** If you don't currently have any robots running, turtlesim is considered the hello-ROS robot to experiment with. It does not have any simulation depenencies such as Gazebo or IsaacSim. 

For a complete step-by-step tutorial on using turtlesim with the MCP server and for more information on ROS and turtlesim, see our [Turtlesim Tutorial](../examples/1_turtlesim/README.md).

If you have ROS already installed, you can launch turtlesim with the below command:
**ROS1:**
```
rosrun turtlesim turtlesim_node
```

**ROS2:**
```
ros2 run turtlesim turtlesim_node
```

# 5. Acknowledgements
Portions of this code are adapted from
[gemini-live-mcp](https://github.com/allenbijo/gemini-live-mcp) (MIT License).
Further modifications and development by Trace LaRue.




