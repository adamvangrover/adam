import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- Configuration & Imports ---

# Attempt to import the actual MCP server from the core v30 architecture path.
# If not found (e.g., running in a standalone test env), fallback to a mock.
try:
    from core.v30_architecture.python_intelligence.mcp.server import mcp_server
except ImportError:
    class MockMCPServer:
        """Fallback mock for standalone testing."""
        def call_tool(self, tool: str, args: Dict):
            return {"status": "success", "source": "MockMCP", "tool": tool, "args": args}
    mcp_server = MockMCPServer()

# Configure Logging
logging.basicConfig(
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger("EACI_Orchestrator")

# --- Data Structures ---

@dataclass
class PlanStep:
    """Represents a single atomic unit of work in the Orchestrator's plan."""
    description: str
    tool: str
    args: Dict[str, Any]
    step_id: int
    dependencies: List[int] = field(default_factory=list)

@dataclass
class ExecutionResult:
    """Standardized result object for orchestrator executions."""
    status: str
    trace: List[Dict[str, Any]]
    error: Optional[str] = None

# --- Core Orchestrator ---

class EACIOrchestrator:
    """
    Enterprise Adaptive Core Interface (EACI) v2.1 Orchestrator.
    
    Manages the lifecycle of high-level intents to low-level MCP tool calls.
    Features:
    - Asynchronous execution pipeline.
    - Backward compatibility for synchronous callers.
    - Internal Agent registration & routing.
    - Enhanced Regex-based intent decomposition.
    """
    
    def __init__(self):
        self.active_agents: Dict[str, Any] = {}
        self.request_history: List[Dict[str, Any]] = []
        logger.info("EACI Orchestrator v2.1 initialized (Async/Sync Hybrid).")

    def register_agent(self, agent_name: str, agent_instance: Any) -> None:
        """Registers a sub-agent (e.g., CreditAnalyst, RiskEngine) with the orchestrator."""
        if agent_name in self.active_agents:
            logger.warning(f"Overwriting existing agent registry: {agent_name}")
        self.active_agents[agent_name] = agent_instance
        logger.info(f"Agent registered: {agent_name}")

    # ------------------------------------------------------------------
    # Entry Points
    # ------------------------------------------------------------------

    async def process_intent_async(self, intent: str, user_role: str, context: Optional[Dict] = None) -> ExecutionResult:
        """
        Primary Asynchronous Entry Point.
        Decomposes intent -> Plans -> Executes via MCP or Internal Agents.
        """
        logger.info(f"Processing Intent (Async): '{intent}' | Role: {user_role}")
        self.request_history.append({"intent": intent, "role": user_role, "context": context})

        try:
            # 1. Decompose Intent
            plan = self._decompose_enhanced(intent)

            if not plan:
                logger.warning("Decomposition failed: No actionable steps identified.")
                return ExecutionResult(status="error", trace=[], error="Could not decompose intent.")

            # 2. Execute Plan
            results = []
            for step in plan:
                logger.info(f"Executing Step {step.step_id}: {step.description}")
                
                # Execute step (routes to MCP or Internal Agent)
                step_result = await self._execute_step(step)
                
                results.append({
                    "step_id": step.step_id,
                    "description": step.description,
                    "output": step_result
                })

                # Basic error handling: Stop sequence on critical failure
                if isinstance(step_result, dict) and step_result.get("error"):
                    error_msg = step_result.get("error")
                    logger.error(f"Plan halted at step {step.step_id}: {error_msg}")
                    return ExecutionResult(status="failed", trace=results, error=error_msg)

            return ExecutionResult(status="completed", trace=results)

        except Exception as e:
            logger.exception(f"Critical Orchestrator Error: {e}")
            return ExecutionResult(status="crashed", trace=[], error=str(e))

    def process_intent(self, intent: str, user_role: str) -> Dict[str, Any]:
        """
        Synchronous Wrapper for backward compatibility with v2.0 calls.
        """
        try:
            # Attempt to run the async loop
            result = asyncio.run(self.process_intent_async(intent, user_role))
            return {"status": result.status, "trace": result.trace, "message": result.error}
        except RuntimeError:
            # Handle running inside existing event loops (e.g., Jupyter, FastAPI)
            logger.warning("Event loop detected. Executing fallback synchronous logic.")
            return self._process_intent_sync_fallback(intent, user_role)

    # ------------------------------------------------------------------
    # Execution Logic
    # ------------------------------------------------------------------

    async def _execute_step(self, step: PlanStep) -> Any:
        """Routes execution to Internal Agents or the MCP Server."""
        
        # Branch 1: Internal Agent Call
        if step.tool == "internal_agent_call":
            agent_name = step.args.get("agent")
            method_name = step.args.get("method")
            kwargs = step.args.get("kwargs", {})

            if agent_name in self.active_agents:
                agent = self.active_agents[agent_name]
                if hasattr(agent, method_name):
                    func = getattr(agent, method_name)
                    # Handle both async and sync agent methods
                    if asyncio.iscoroutinefunction(func):
                        return await func(**kwargs)
                    else:
                        return func(**kwargs)
            return {"error": f"Agent '{agent_name}' or method '{method_name}' not found."}

        # Branch 2: MCP Server Call (External Tools)
        try:
            # Simulate slight async network delay for realism if needed
            # await asyncio.sleep(0.01) 
            return mcp_server.call_tool(step.tool, step.args)
        except Exception as e:
            return {"error": f"MCP Tool Call Failed: {str(e)}"}

    # ------------------------------------------------------------------
    # Planning & Decomposition
    # ------------------------------------------------------------------

    def _decompose_enhanced(self, intent: str) -> List[PlanStep]:
        """
        Parses natural language into structured PlanSteps using Regex patterns.
        """
        intent_lower = intent.lower()
        plan = []
        step_id = 1

        # Pattern A: Buy [Quantity] [Symbol]
        # e.g., "I want to buy 100 shares of AAPL"
        buy_match = re.search(r"buy (\d+)\s*(?:shares of )?([a-z]+)", intent_lower)
        if buy_match:
            quantity, symbol = buy_match.groups()
            symbol = symbol.upper()
            
            # Step 1: Market Data
            plan.append(PlanStep(
                step_id=step_id,
                description=f"Check Market Data for {symbol}",
                tool="get_market_data",
                args={"symbol": symbol}
            ))
            step_id += 1
            
            # Step 2: Execute Trade (Dependent on Step 1)
            plan.append(PlanStep(
                step_id=step_id,
                description=f"Execute Buy Order for {symbol}",
                tool="execute_trade",
                args={"symbol": symbol, "side": "BUY", "quantity": int(quantity)},
                dependencies=[step_id - 1]
            ))
            return plan

        # Pattern B: Analyze [Symbol]
        # e.g., "Analyze NVDA please"
        if "analyze" in intent_lower:
            # Simple heuristic to find ticker
            words = intent_lower.split()
            symbol = "TSLA" # Default mock if not found
            known_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM"]
            
            for word in words:
                clean_word = word.strip(".,!?").upper()
                if clean_word in known_tickers:
                    symbol = clean_word

            # Step 1: Market Data
            plan.append(PlanStep(
                step_id=step_id,
                description=f"Get Market Data for {symbol}",
                tool="get_market_data",
                args={"symbol": symbol}
            ))
            step_id += 1

            # Step 2: Internal Credit Analysis Agent
            plan.append(PlanStep(
                step_id=step_id,
                description=f"Run Risk Analysis for {symbol}",
                tool="internal_agent_call",
                args={
                    "agent": "credit_analyst", 
                    "method": "analyze_risk", 
                    "kwargs": {"ticker": symbol}
                },
                dependencies=[step_id - 1]
            ))
            return plan

        return []

    def _process_intent_sync_fallback(self, intent: str, user_role: str):
        """Fallback logic for when asyncio.run fails (e.g. nested loops)."""
        plan_steps = self._decompose_enhanced(intent)
        results = []
        
        for step in plan_steps:
            # We cannot await here, so we only support sync MCP calls in fallback
            if step.tool == "internal_agent_call":
                 # Simplified sync support for fallback
                 results.append({"step": step.description, "output": "Sync Fallback: Internal Agents not fully supported in fallback mode."})
            else:
                res = mcp_server.call_tool(step.tool, step.args)
                results.append({"step": step.description, "output": res})
        
        return {"status": "completed_fallback", "trace": results}

# --- Singleton Instance ---

orchestrator = EACIOrchestrator()

# --- Execution Test ---

if __name__ == "__main__":
    print("=== EACI Orchestrator v2.1 Test Suite ===\n")

    # 1. Define and Register a Mock Internal Agent
    class MockCreditAgent:
        def analyze_risk(self, ticker):
            return f"Calculated Risk Exposure for {ticker}: MODERATE (Score: 65/100)"

    orchestrator.register_agent("credit_analyst", MockCreditAgent())

    # 2. Test "Buy" Intent (Routes to MCP Tools)
    print("--- Test Case 1: Buy Intent ---")
    res_buy = orchestrator.process_intent("I want to buy 150 AAPL", "PortfolioManager")
    print(res_buy)
    print("\n")

    # 3. Test "Analyze" Intent (Routes to Internal Agent + MCP)
    print("--- Test Case 2: Analyze Intent ---")
    res_analyze = orchestrator.process_intent("Can you analyze NVDA?", "RiskOfficer")
    # Pretty print trace for readability
    for item in res_analyze.get("trace", []):
        print(f"Step: {item['description']}")
        print(f"Output: {item['output']}")
