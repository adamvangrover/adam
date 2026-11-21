import asyncio
from typing import List, Dict, Any
from .schemas import MCPResource, MCPTool, RefactorProposal

class MCPServer:
    """
    Model Context Protocol (MCP) Server for Adam v24.0.
    Exposes Resources, Tools, and Prompts to agents.
    """
    def __init__(self):
        self.resources: Dict[str, MCPResource] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.prompts: Dict[str, str] = {}
        self._register_defaults()

    def _register_defaults(self):
        # Register default resources
        self.register_resource(MCPResource(
            uri="financial://market/L2_book/BTC-USD",
            mime_type="application/json",
            name="BTC-USD Level 2 Book"
        ))

        # Register tools
        self.register_tool(MCPTool(
            name="propose_refactor",
            description="Proposes a refactor for a specific code module using AST analysis.",
            input_schema=RefactorProposal.model_json_schema()
        ))

        self.register_tool(MCPTool(
            name="execute_trade",
            description="Executes a trade on the Iron Core. REQUIRES HUMAN APPROVAL.",
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "enum": ["Buy", "Sell"]},
                    "quantity": {"type": "number"}
                },
                "required": ["symbol", "side", "quantity"]
            }
        ))

    def register_resource(self, resource: MCPResource):
        self.resources[resource.uri] = resource
        print(f"MCP: Registered resource {resource.uri}")

    def register_tool(self, tool: MCPTool):
        self.tools[tool.name] = tool
        print(f"MCP: Registered tool {tool.name}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], context: Any = None):
        """
        Executes a tool. Enforces Human-in-the-Loop for sensitive tools.
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found.")

        # Human-in-the-Loop Interceptor
        if tool_name == "execute_trade":
            approval = await self._request_human_approval(tool_name, arguments)
            if not approval:
                return {"status": "rejected", "reason": "User denied execution."}

        # Mock Execution
        print(f"MCP: Executing tool {tool_name} with args {arguments}")
        return {"status": "success", "result": "Tool executed successfully (Mock)"}

    async def _request_human_approval(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """
        Simulates sending a request to the Client UI.
        """
        print(f"SECURITY INTERCEPT: User approval required for {tool_name} {args}")
        # In a real system, this would await a WebSocket response
        return True # Auto-approve for demo
