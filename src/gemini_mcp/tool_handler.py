"""MCP tool execution and handling."""

import json
from google.genai import types


class ToolHandler:
    """Handles MCP tool calls from Gemini."""

    # Configuration
    NAVIGATION_TIMEOUT = 300.0  # 5 minutes for navigation actions

    def __init__(self, mcp_session, session):
        """
        Initialize tool handler.

        Args:
            mcp_session: MCP client session for calling tools
            session: Gemini Live session for sending responses
        """
        self.mcp_session = mcp_session
        self.session = session

    async def handle_tool_call(self, tool_call):
        """
        Process tool calls from Gemini and execute via MCP.

        Args:
            tool_call: Tool call request from Gemini containing function calls
        """
        for function_call in tool_call.function_calls:
            print(f"\n🔧 {function_call.name}({function_call.args})")

            result = await self._execute_tool(function_call)
            response_data = self._parse_result(result)
            await self._send_response(function_call, response_data)

    async def _execute_tool(self, function_call):
        """
        Execute tool with appropriate timeout.

        Args:
            function_call: Function call details from Gemini

        Returns:
            Result from MCP tool execution
        """
        tool_args = function_call.args.copy()

        # Add custom timeout for navigation actions
        if function_call.name == "navigate_to_location":
            tool_args["timeout"] = self.NAVIGATION_TIMEOUT

        return await self.mcp_session.call_tool(
            name=function_call.name,
            arguments=tool_args,
        )

    def _parse_result(self, result) -> dict:
        """
        Parse MCP result into response data.

        Args:
            result: Raw result from MCP tool execution

        Returns:
            Formatted response dictionary
        """
        result_text = self._extract_result_text(result)
        return self._format_response_data(result_text)

    def _extract_result_text(self, result) -> str:
        """
        Extract text from MCP result.

        Args:
            result: Raw MCP result object

        Returns:
            String representation of result
        """
        # Check for structured content first (preferred format)
        if hasattr(result, "structuredContent") and result.structuredContent:
            return json.dumps(result.structuredContent, indent=2)

        # Parse content items
        if not hasattr(result, "content"):
            return str(result)

        result_content = [
            self._parse_content_item(item)
            for item in result.content
        ]
        return "\n".join(result_content) if result_content else str(result)

    def _parse_content_item(self, item) -> str:
        """
        Parse a single content item.

        Args:
            item: Content item from MCP result

        Returns:
            String representation of item
        """
        if hasattr(item, "text"):
            return item.text
        elif hasattr(item, "model_dump"):
            return str(item.model_dump())
        elif isinstance(item, (str, dict)):
            return str(item)
        else:
            return str(item)

    def _format_response_data(self, result_text: str) -> dict:
        """
        Convert result text to response dictionary.

        Args:
            result_text: String result from tool execution

        Returns:
            Dictionary suitable for Gemini response
        """
        try:
            return json.loads(result_text)
        except (json.JSONDecodeError, ValueError):
            return {"result": result_text}

    def _get_scheduling(self, tool_name: str, response_data: dict) -> str:
        """
        Determine scheduling strategy for tool response.

        Args:
            tool_name: Name of the tool being executed
            response_data: Response data from tool execution

        Returns:
            Scheduling strategy string or None
        """
        # Action goals should interrupt to report completion
        if tool_name == "send_action_goal":
            return "SILENT"
        # Default: no special scheduling
        return None

    async def _send_response(self, function_call, response_data: dict):
        """
        Send final response to Gemini with scheduling.

        Args:
            function_call: Original function call from Gemini
            response_data: Formatted response data
        """
        # Determine scheduling based on tool type and result
        scheduling = self._get_scheduling(function_call.name, response_data)

        # Add scheduling to response
        if scheduling:
            response_data["scheduling"] = scheduling

        function_response = types.FunctionResponse(
            name=function_call.name,
            id=function_call.id,
            response=response_data
        )

        try:
            await self.session.send_tool_response(
                function_responses=[function_response]
            )
            print(f"✅ {function_call.name} completed")
        except Exception as e:
            print(f"🔴 Error sending response: {e}")
