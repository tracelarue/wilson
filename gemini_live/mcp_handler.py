"""
MCP Client handler for connecting to ROS MCP server.
"""

import asyncio
import json
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configuration file path for MCP server definitions
MCP_CONFIG_FILE = "mcp_config.json"


class MCPClient:
    """
    Client for connecting to and managing MCP server connections.

    This class handles the lifecycle of MCP server connections, providing
    methods to connect, communicate with, and properly clean up server sessions.
    """

    def __init__(self) -> None:
        """
        Initialize the MCP client.
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        # Transport layer components (set during connection)
        self.stdio = None
        self.write = None

    async def connect_to_server(self, server_name: str = "ros-mcp-server") -> None:
        """
        Connect to the specified MCP server and initialize the client session.

        Args:
            server_name: Name of the server to connect to from mcp_config.json

        Raises:
            ValueError: If the specified server is not found in configuration
            FileNotFoundError: If mcp_config.json is not found
            json.JSONDecodeError: If configuration file is malformed
        """
        # Load MCP server configuration
        try:
            with open(MCP_CONFIG_FILE) as config_file:
                server_configurations = json.load(config_file)["mcpServers"]
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{MCP_CONFIG_FILE}' not found")
        except KeyError:
            raise ValueError(
                f"Invalid configuration format in '{MCP_CONFIG_FILE}' - missing 'mcpServers' key"
            )

        # Validate server exists in configuration
        if server_name not in server_configurations:
            available_servers = list(server_configurations.keys())
            raise ValueError(
                f"Server '{server_name}' not found in {MCP_CONFIG_FILE}. "
                f"Available servers: {available_servers}"
            )

        # Extract server configuration
        server_config = server_configurations[server_name]
        server_parameters = StdioServerParameters(
            command=server_config["command"],
            args=server_config["args"],
            env=None,  # Use system environment
        )

        # Initialize stdio transport connection
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_parameters)
        )
        self.stdio, self.write = stdio_transport

        # Create and initialize client session
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        # Discover and display available tools
        tools_response = await self.session.list_tools()
        available_tools = tools_response.tools
        tool_names = [tool.name for tool in available_tools]

        print(f"\nConnected to '{server_name}' with tools: {tool_names}")

    async def cleanup(self) -> None:
        """
        Clean up the MCP client resources.

        Properly closes all async contexts and cleans up transport connections.
        """
        await self.exit_stack.aclose()


async def main() -> None:
    client = MCPClient()
    try:
        await client.connect_to_server()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
