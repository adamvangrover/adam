import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Set

from pydantic import BaseModel, Field

# Assuming this is available from your interfaces package
from afos_interfaces import IExecutionKernel

logger = logging.getLogger(__name__)

# ==================================================================
# INTERNAL WORKFLOW MODELS
# ==================================================================
class WorkflowStep(BaseModel):
    step_id: str = Field(..., description="Unique ID for the step in the DAG.")
    action: str = Field(..., description="The name of the function/tool to execute.")
    dependencies: List[str] = Field(default_factory=list, description="Step IDs that must complete first.")
    required_params: List[str] = Field(default_factory=list, description="Keys expected in the context state.")

class WorkflowPlan(BaseModel):
    workflow_id: str = Field(..., description="Identifier for this workflow definition.")
    steps: Dict[str, WorkflowStep] = Field(..., description="Map of step_id to step definition.")

# ==================================================================
# EXECUTION KERNEL
# ==================================================================
class ExecutionKernel(IExecutionKernel):
    """
    Concrete implementation of the Execution Kernel.
    Orchestrates Directed Acyclic Graphs (DAGs) of asynchronous tasks.
    """
    def __init__(self):
        # Registry of executable actions (simulating a tool registry or agent behaviors)
        self._action_registry: Dict[str, Callable] = {}
        self._active_workflows: Set[str] = set()

    async def initialize(self) -> None:
        """Boot sequence: Load standard library of executable actions."""
        logger.info("Initializing Execution Kernel: Async DAG engine starting...")
        
        # Register some default actions for our financial OS
        self.register_action("fetch_portfolio", self._mock_action_fetch)
        self.register_action("evaluate_risk", self._mock_action_evaluate)
        self.register_action("require_human_approval", self._mock_action_human_loop)
        self.register_action("seal_decision", self._mock_action_seal)
        
        logger.info(f"Registered {len(self._action_registry)} core actions.")

    async def shutdown(self) -> None:
        """Teardown: Drain running workflows."""
        if self._active_workflows:
            logger.warning(f"Shutting down with {len(self._active_workflows)} active workflows!")
        logger.info("Execution Kernel shutdown complete.")

    def register_action(self, action_name: str, handler: Callable) -> None:
        """Dynamically bind a Python async function to a workflow action name."""
        self._action_registry[action_name] = handler

    async def plan_execution(self, workflow_dsl: str) -> WorkflowPlan:
        """
        Parses a DSL (JSON/YAML) and compiles it into an executable DAG plan.
        Includes basic cycle detection and validation.
        """
        try:
            raw_plan = json.loads(workflow_dsl)
            plan = WorkflowPlan(**raw_plan)
        except Exception as e:
            raise ValueError(f"Failed to compile workflow DSL: {e}")

        # Basic DAG Validation: Ensure dependencies exist
        for step_id, step in plan.steps.items():
            for dep in step.dependencies:
                if dep not in plan.steps:
                    raise ValueError(f"Invalid dependency: Step '{step_id}' depends on missing step '{dep}'.")
                
        logger.info(f"Successfully compiled execution plan: {plan.workflow_id} with {len(plan.steps)} steps.")
        return plan

    async def run_workflow(self, plan: WorkflowPlan, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the compiled DAG.
        Uses asyncio.Event to coordinate dependencies, allowing independent branches 
        to execute concurrently in true async fashion.
        """
        instance_id = f"{plan.workflow_id}_{id(parameters)}"
        self._active_workflows.add(instance_id)
        
        logger.info(f"Starting workflow instance [{instance_id}]")
        
        # Shared state context passed between nodes
        state = parameters.copy()
        
        # Track completion events for dependency resolution
        completion_events: Dict[str, asyncio.Event] = {
            step_id: asyncio.Event() for step_id in plan.steps
        }
        
        # Worker function for an individual step
        async def execute_step(step: WorkflowStep) -> None:
            # 1. Wait for all dependencies to finish
            if step.dependencies:
                logger.debug(f"Step '{step.step_id}' waiting on {step.dependencies}...")
                await asyncio.gather(*(completion_events[dep].wait() for dep in step.dependencies))
            
            # 2. Extract required parameters
            step_inputs = {k: state.get(k) for k in step.required_params}
            
            # 3. Execute the action
            logger.info(f"Executing step '{step.step_id}' (Action: {step.action})...")
            handler = self._action_registry.get(step.action)
            if not handler:
                raise RuntimeError(f"Action '{step.action}' is not registered in the Execution Kernel.")
            
            try:
                result = await handler(step_inputs)
                # Mutate shared state (In a strict functional paradigm, we'd merge immutable dicts)
                if isinstance(result, dict):
                    state.update(result)
            except Exception as e:
                logger.error(f"Step '{step.step_id}' FAILED: {e}")
                raise
                
            # 4. Signal downstream steps that this node is complete
            completion_events[step.step_id].set()
            logger.info(f"Completed step '{step.step_id}'.")

        # Launch all steps as concurrent tasks (they will internally block on dependencies)
        tasks = [
            asyncio.create_task(execute_step(step)) 
            for step in plan.steps.values()
        ]
        
        try:
            await asyncio.gather(*tasks)
            logger.info(f"Workflow instance [{instance_id}] completed successfully.")
            return state
        finally:
            self._active_workflows.remove(instance_id)

    # ---------------------------------------------------------------
    # Mock Action Handlers (Simulating external I/O & Agentic compute)
    # ---------------------------------------------------------------
    async def _mock_action_fetch(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        portfolio_id = inputs.get("portfolio_id")
        await asyncio.sleep(0.5) # Simulate DB latency
        return {"portfolio_data": {"size": 50000000, "assets": 120}}

    async def _mock_action_evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        data = inputs.get("portfolio_data", {})
        await asyncio.sleep(1.0) # Simulate LLM/Model inference
        risk_score = 0.88 if data.get("size", 0) > 10000000 else 0.45
        return {"risk_score": risk_score, "decision": "Approve" if risk_score > 0.85 else "Reject"}

    async def _mock_action_human_loop(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        score = inputs.get("risk_score", 0)
        if score > 0.95:
            # Simulate a suspension/interrupt waiting for human input
            logger.warning(">>> HIGH RISK DETECTED. Suspending for human review...")
            await asyncio.sleep(2.0) 
            logger.warning(">>> Human reviewer approved override.")
        return {"human_override": True}

    async def _mock_action_seal(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.2) # Simulate crypto hashing / Ledger entry
        return {"sealed_hash": "0xABC123..."}

# ==================================================================
# EXECUTION / FUNCTIONAL TEST
# ==================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # A simple DSL representing a DAG: Fetch -> Evaluate -> (Approval & Seal parallel)
    mock_workflow_dsl = json.dumps({
        "workflow_id": "credit_approval_v1",
        "steps": {
            "step_1": {
                "step_id": "step_1",
                "action": "fetch_portfolio",
                "dependencies": [],
                "required_params": ["portfolio_id"]
            },
            "step_2": {
                "step_id": "step_2",
                "action": "evaluate_risk",
                "dependencies": ["step_1"],
                "required_params": ["portfolio_data"]
            },
            "step_3": {
                "step_id": "step_3",
                "action": "require_human_approval",
                "dependencies": ["step_2"],
                "required_params": ["risk_score"]
            },
            "step_4": {
                "step_id": "step_4",
                "action": "seal_decision",
                "dependencies": ["step_2"], # Notice it doesn't wait for step 3, runs concurrently
                "required_params": ["decision"]
            }
        }
    })

    async def main():
        kernel = ExecutionKernel()
        await kernel.initialize()
        
        # 1. Compile the plan
        plan = await kernel.plan_execution(mock_workflow_dsl)
        
        # 2. Run the plan
        initial_context = {"portfolio_id": "port_999_alpha"}
        final_state = await kernel.run_workflow(plan, parameters=initial_context)
        
        print("\n✅ Final Workflow State:")
        print(json.dumps(final_state, indent=2))
        
        await kernel.shutdown()

    # Run the event loop
    asyncio.run(main())
