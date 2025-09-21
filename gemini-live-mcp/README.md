# Gemini Live MCP

Gemini Live MCP is an MCP client that utilizes the Gemini Live API to enable voice-based interaction with Gemini. Additionally, it now supports adding MCP servers for extended functionality.

## Features
- Voice-based interaction with Gemini via the Gemini Live API
- Support for multiple MCP servers

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/allenbijo/gemini-live-mcp.git
   cd gemini-live-mcp
   ```

2. **Set up environment variables**:
   Create a `.env` file in the project root and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

3. **Create MCP configuration file**:
   Define your MCP servers by creating a `mcp_config.json` file in the project root.
   Example structure:
   ```json
   {
        "mcpServers": {
            "sysinfo": {
                "command": "uv",
                "args": [
                    "--directory",
                    "D:\\WorksOfGreatness\\mcp-sysinfo",
                    "run",
                    "sysinfo.py"
                ]
            }
        }
    }
   ```

## Usage

Run the application using the following command:
```sh
uv run main.py --mode=none
# mode can be none, camera, screen
```

### Modes
- `none`: Runs without any visual interface.
- `screen`: Displays results on the screen.
- `camera`: Uses the camera for interaction.

## License
This project is licensed under the MIT License.

## Contributing
Feel free to submit issues or pull requests to improve the project.

## Contact
For questions or support, reach out via [GitHub Issues](https://github.com/allenbijo/gemini-live-mcp/issues).

