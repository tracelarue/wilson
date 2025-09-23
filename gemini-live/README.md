# Gemini Live with MCP

Gemini Live enables realtime transcription and voice/vision interactions. 
This example shows how to use the ros-mcp-server with the Gemini Live API. 

## Usage

1. **Install ros-mcp-server**
Follow the instructions in installation.md

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

