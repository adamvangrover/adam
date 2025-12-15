from core.v30_architecture.python_intelligence.mcp.server import mcp_server
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EACI_Orchestrator")

class EACIOrchestrator:
    """
    Enterprise Adaptive Core Interface (EACI) v2.0 Orchestrator.
    Manages the lifecycle of high-level intents to low-level MCP tool calls.
    """
    def __init__(self):
        self.active_agents = {}

    def process_intent(self, intent: str, user_role: str):
        """
        Main entry point for user commands.
        """
        logger.info(f"Processing Intent: '{intent}' for role: {user_role}")

        # 1. Decompose Intent (Mocked Logic for v30 Scaffold)
        # In the full implementation, this uses the Neuro-Symbolic Planner
        plan = self._decompose(intent)

        if not plan:
            return {"status": "error", "message": "Could not decompose intent."}

        # 2. Execute Plan
        results = []
        for step in plan:
            logger.info(f"Executing step: {step['description']}")

            # Route to MCP Server
            if step.get('tool'):
                res = mcp_server.call_tool(step['tool'], step['args'])
                results.append({"step": step['description'], "output": res})

                # Stop on error
                if res.get("error"):
                    logger.error(f"Plan failed at step: {step['description']}")
                    break

        return {"status": "completed", "trace": results}

    def _decompose(self, intent):
        """
        Mocks the decompostion of a natural language query into a structured plan.
        """
        intent_lower = intent.lower()

        if "buy" in intent_lower and "aapl" in intent_lower:
            return [
                {"description": "Check Market Data for AAPL", "tool": "get_market_data", "args": {"symbol": "AAPL"}},
                {"description": "Execute Buy Order", "tool": "execute_trade", "args": {"symbol": "AAPL", "side": "BUY", "quantity": 100}}
            ]
        elif "analyze" in intent_lower:
             return [
                {"description": "Check Market Data", "tool": "get_market_data", "args": {"symbol": "TSLA"}},
                 # Future: Add Credit Analysis tool
            ]
        return []

# Singleton instance
orchestrator = EACIOrchestrator()

if __name__ == "__main__":
    # Test
    res = orchestrator.process_intent("I want to buy some AAPL", "PortfolioManager")
    print(res)
