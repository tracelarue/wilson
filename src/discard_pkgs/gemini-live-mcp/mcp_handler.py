import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import json


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        self.messages = []

    async def connect_to_server(self):
        """
        Connect to all MCP servers
        """
        with open('mcp_config.json') as f:
            mcp_servers = json.load(f)['mcpServers']

        server='ros-mcp-server'
        
        server_params = StdioServerParameters(
            command=mcp_servers[server]['command'],
            args=mcp_servers[server]['args'],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        
        # available_tools = [{ 
        #     "name": tool.name,
        #     "description": tool.description,
        #     "input_schema": tool.inputSchema
        # } for tool in response.tools]
        
        print("\nConnected to server with tools:", [tool.name for tool in tools])


if __name__ == "__main__":
    asyncio.run(main())