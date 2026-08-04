import asyncio
import hashlib
import json
import logging
from typing import Any, Dict

# Assuming these are available from your core and interfaces packages
from afos_core import Policy
from afos_interfaces import IPolicyKernel

logger = logging.getLogger(__name__)

# ==================================================================
# NATIVE JSON-LOGIC EVALUATOR (Deterministic Rules Engine)
# ==================================================================
def evaluate_json_logic(logic: Any, data: Dict[str, Any]) -> Any:
    """
    A lightweight, deterministic recursive evaluator for jsonLogic.
    Supports core logical operators required for financial covenants.
    """
    # Base cases: if it's not a dict, it's a raw value (string, int, float, bool)
    if not isinstance(logic, dict):
        return logic
    
    # In jsonLogic, the key is the operator, the value is the array of arguments
    operator = list(logic.keys())[0]
    values = logic[operator]
    
    # Normalize values to a list for uniform processing
    if not isinstance(values, list):
        values = [values]
        
    # 1. Variable resolution (e.g., {"var": "financials.leverage"})
    if operator == "var":
        key_path = values[0]
        # Resolve dot-notation paths
        current_val = data
        for k in key_path.split('.'):
            if isinstance(current_val, dict):
                current_val = current_val.get(k)
            else:
                return None
        return current_val

    # 2. Comparison Operators
    elif operator == "==":
        return evaluate_json_logic(values[0], data) == evaluate_json_logic(values[1], data)
    elif operator == "!=":
        return evaluate_json_logic(values[0], data) != evaluate_json_logic(values[1], data)
    elif operator == ">":
        return evaluate_json_logic(values[0], data) > evaluate_json_logic(values[1], data)
    elif operator == ">=":
        return evaluate_json_logic(values[0], data) >= evaluate_json_logic(values[1], data)
    elif operator == "<":
        return evaluate_json_logic(values[0], data) < evaluate_json_logic(values[1], data)
    elif operator == "<=":
        return evaluate_json_logic(values[0], data) <= evaluate_json_logic(values[1], data)
    
    # 3. Logical Operators
    elif operator == "and":
        return all(evaluate_json_logic(v, data) for v in values)
    elif operator == "or":
        return any(evaluate_json_logic(v, data) for v in values)
    
    else:
        raise ValueError(f"Unsupported jsonLogic operator: {operator}")


# ==================================================================
# POLICY KERNEL
# ==================================================================
class PolicyKernel(IPolicyKernel):
    """
    Concrete implementation of the Policy Kernel.
    Compiles and executes deterministic rulesets (jsonLogic) against 
    dynamic context states to evaluate credit risk and operational boundaries.
    """
    def __init__(self):
        self._compiled_cache: Dict[str, Policy] = {}

    async def initialize(self) -> None:
        """Boot sequence: Load base regulatory or system-wide policies into cache."""
        logger.info("Initializing Policy Kernel: Deterministic Rules Engine online.")
        # In production, this would hydrate self._compiled_cache from PostgreSQL/Redis

    async def shutdown(self) -> None:
        """Teardown sequence."""
        logger.info(f"Policy Kernel shutting down. Flushed {len(self._compiled_cache)} cached policies.")

    async def compile_policy(self, dsl: str, dsl_format: str = "jsonlogic") -> Policy:
        """
        Takes raw DSL strings, validates their syntax, and generates a canonical Policy entity.
        Calculates a deterministic hash for the policy ID to prevent stealth alterations.
        """
        if dsl_format.lower() != "jsonlogic":
            raise NotImplementedError(f"Format {dsl_format} not currently supported by native engine.")
            
        try:
            # Validate that the DSL is actually valid JSON
            parsed_dsl = json.loads(dsl)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Policy DSL: {e}")

        # Create a deterministic ID based on the exact ruleset bytes
        ruleset_hash = hashlib.sha256(dsl.encode('utf-8')).hexdigest()[:16]
        policy_id = f"pol_{ruleset_hash}"

        policy = Policy(
            id=policy_id,
            version="1.0.0",
            ruleset=dsl_format,
            rules=json.dumps(parsed_dsl, separators=(',', ':')) # Minified rules string
        )
        
        self._compiled_cache[policy_id] = policy
        logger.debug(f"Compiled Policy [{policy_id}] successfully.")
        return policy

    async def execute_policy(self, policy: Policy, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the policy against the provided context context.
        Returns an evaluation payload containing the boolean result and the context state.
        """
        logger.info(f"Executing Policy [{policy.id}] v{policy.version}...")
        
        if policy.rules is None:
            raise ValueError(f"Policy {policy.id} contains empty ruleset.")
            
        try:
            logic_dict = json.loads(policy.rules)
            
            # Offload CPU-bound evaluation to prevent blocking the async event loop
            # (Crucial for high-velocity quantitative processing)
            result = await asyncio.to_thread(evaluate_json_logic, logic_dict, context)
            
            return {
                "policy_id": policy.id,
                "passed": bool(result),
                "evaluated_context": context,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Policy Execution Failed for {policy.id}: {str(e)}")
            raise

# ==================================================================
# EXECUTION / FUNCTIONAL TEST
# ==================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    async def main():
        kernel = PolicyKernel()
        await kernel.initialize()
        
        # 1. Define a strict Credit Covenant Policy using jsonLogic
        # Rule: (leverage < 4.5) AND (interest_coverage > 2.0) AND (borrower_status == "Active")
        covenant_dsl = json.dumps({
            "and": [
                {"<": [{"var": "financials.leverage"}, 4.5]},
                {">": [{"var": "financials.interest_coverage"}, 2.0]},
                {"==": [{"var": "borrower.status"}, "Active"]}
            ]
        })
        
        # 2. Compile the DSL into an immutable Policy Entity
        credit_policy = await kernel.compile_policy(covenant_dsl)
        print(f"\n✅ Policy Compiled Successfully: {credit_policy.id}")
        
        # 3. Simulate Context A: Healthy Borrower
        healthy_context = {
            "borrower": {"status": "Active"},
            "financials": {
                "leverage": 3.2,
                "interest_coverage": 4.1
            }
        }
        
        # 4. Simulate Context B: Distressed Borrower (Breaches leverage limit)
        distressed_context = {
            "borrower": {"status": "Active"},
            "financials": {
                "leverage": 4.8, # Breaches < 4.5
                "interest_coverage": 1.5 # Breaches > 2.0
            }
        }
        
        # 5. Execute policies deterministically
        print("\n--- Evaluating Healthy Context ---")
        healthy_result = await kernel.execute_policy(credit_policy, healthy_context)
        print(f"Result: {'PASS' if healthy_result['passed'] else 'FAIL'}")
        
        print("\n--- Evaluating Distressed Context ---")
        distressed_result = await kernel.execute_policy(credit_policy, distressed_context)
        print(f"Result: {'PASS' if distressed_result['passed'] else 'FAIL'}")

        await kernel.shutdown()

    # Run the event loop
    asyncio.run(main())
