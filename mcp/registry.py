from typing import Dict, Any, Callable
from mcp.capabilities import MCPCapability

class ToolRegistry:
    def __init__(self):
        self.capabilities: Dict[str, MCPCapability] = {}
        self.implementations: Dict[str, Callable] = {}

    def register(self, capability: MCPCapability, implementation: Callable):
        self.capabilities[capability.capability_id] = capability
        self.implementations[capability.capability_id] = implementation

    def get_capability(self, capability_id: str) -> MCPCapability:
        return self.capabilities.get(capability_id)

    def get_implementation(self, capability_id: str) -> Callable:
        return self.implementations.get(capability_id)
