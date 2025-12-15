import json
import logging
import time
from typing import Dict, Any, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCPServer")

class MCPServer:
    """
    Model Context Protocol (MCP) Server for Adam v30.0.
    Acts as the nervous system connecting AI agents to the Core.
    """
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.resources: Dict[str, Any] = {}
        # Default sensitive tools
        self.sensitive_tools = ["execute_trade", "transfer_funds"]

    def register_tool(self, name: str, func: Callable, sensitive: bool = False):
        self.tools[name] = func
        if sensitive and name not in self.sensitive_tools:
            self.sensitive_tools.append(name)
        logger.info(f"Registered tool: {name} (Sensitive: {sensitive})")

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a tool call, enforcing Human-in-the-Loop for sensitive tools.
        """
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}

        # Human-in-the-Loop Interceptor
        if tool_name in self.sensitive_tools:
            logger.warning(f"Intercepting sensitive tool call: {tool_name}")
            approval = self._request_human_approval(tool_name, arguments)
            if not approval:
                return {"error": "User denied execution"}

            # Log HMM (Human-Machine Markdown)
            self._log_hmm(tool_name, arguments, "Approved")

        try:
            # Validate against schema (simplified here)
            result = self.tools[tool_name](**arguments)
            return {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Error executing {tool_name}: {e}")
            return {"error": str(e)}

    def _request_human_approval(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """
        Simulates the Human-in-the-Loop approval process.
        In the real v30 system, this pauses execution and pushes a notification to the UI.
        """
        # In a real system, this sends a WebSocket message to the UI
        # For simulation, we assume auto-approval or interactive input
        logger.info(f"[MCP] Human Approval Request: Execute {tool_name} with {args}?")
        # In automated test environments, we default to True, but log clearly.
        return True

    def _log_hmm(self, tool_name, args, action):
        """Logs the interaction in Human-Machine Markdown format."""
        hmm_entry = f"**User Action**: {action} on `{tool_name}` with params `{json.dumps(args)}`"
        logger.info(f"HMM Log: {hmm_entry}")

# --- Example Core Tools Implemented in Python for the Intelligence Layer ---

def execute_trade(symbol: str, side: str, quantity: int, price: float = None):
    """
    Simulates sending an order to the Rust Core via ZMQ/IPC.
    """
    order_id = f"ord_{int(time.time())}"
    logger.info(f"Sending Order to Rust Core: {side} {quantity} {symbol} @ {price if price else 'MKT'}")
    return {"order_id": order_id, "status": "Working"}

def get_market_data(symbol: str):
    """
    Fetches market data from the Time-Series DB (simulated).
    """
    return {"symbol": symbol, "price": 150.00, "bid": 149.95, "ask": 150.05}

# Server Instance (Singleton pattern)
mcp_server = MCPServer()
mcp_server.register_tool("execute_trade", execute_trade, sensitive=True)
mcp_server.register_tool("get_market_data", get_market_data)

if __name__ == "__main__":
    # Test run
    print(mcp_server.call_tool("get_market_data", {"symbol": "AAPL"}))
    print(mcp_server.call_tool("execute_trade", {"symbol": "AAPL", "side": "BUY", "quantity": 100}))
