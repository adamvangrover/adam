from typing import Dict, Any, Callable, List
from pydantic import BaseModel

class MCPTool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable

class ToolRegistry:
    """
    Manages the registration and discovery of MCP tools.
    """

    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}

    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], handler: Callable):
        tool = MCPTool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler
        )
        self._tools[name] = tool

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema
            }
            for t in self._tools.values()
        ]

    def get_tool(self, name: str) -> MCPTool:
        return self._tools.get(name)

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool {name} not found.")
        return tool.handler(**arguments)
