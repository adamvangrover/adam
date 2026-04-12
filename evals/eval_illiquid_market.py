import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)

class LlmJudge:
    """
    Simulates the secondary LLM call to qualitatively grade the agent's reasoning.
    In production, hook this up to your LLM API client.
    """

    @staticmethod
    def grade_justification(justification: str, asset_profile: str, macro_context: str) -> Dict[str, Any]:
        judge_prompt = f"""
        [SYSTEM: EVALUATION JUDGE - RISK CONTROL]
        You are a senior credit risk officer evaluating a market-making agent's pricing logic for an illiquid asset.

        [CONTEXT]
        Asset Profile: {asset_profile}
        Macro Context: {macro_context}
        Agent's Provided Justification: "{justification}"

        [EVALUATION DIRECTIVE]
        Analyze the justification. Did the agent correctly identify the primary risk factors (e.g., liquidity vacuum, duration risk, counterparty exposure) given the asset profile and macro context?

        [OUTPUT]
        Output strictly in JSON format: {{"score": int (1-5), "feedback": string, "missed_risk_factors": list}}
        """

        logging.info("Executing secondary LLM-as-a-Judge call...")
        # MOCK LLM RESPONSE - Replace with actual API call (e.g., openai.chat.completions.create)
        mock_response = {
            "score": 4,
            "feedback": "Sound reasoning regarding the liquidity premium and inventory limits. However, failed to explicitly factor in the rising rate environment mentioned in the macro context.",
            "missed_risk_factors": ["interest rate risk"]
        }
        return mock_response

class IlliquidMarketEvalHarness:
    def __init__(self, max_inventory_threshold: int = 1000):
        self.max_threshold = max_inventory_threshold

    def run_eval(self, agent_output: str, fair_value: float, current_inventory: int, asset_profile: str, macro_context: str) -> Dict[str, Any]:
        """
        Evaluates the agent's output using both deterministic risk parameters and an LLM Judge.
        """
        results = {
            "passed_deterministic": False,
            "deterministic_errors": [],
            "metrics": {},
            "qualitative_grade": {}
        }

        # --- PHASE 1: DETERMINISTIC STRUCTURAL & LOGIC VALIDATION ---
        try:
            quote_data = json.loads(agent_output)
        except json.JSONDecodeError:
            results["deterministic_errors"].append("CRITICAL: Output failed JSON parsing.")
            return results

        required_keys = {"bid", "ask", "justification", "inventory_skew_applied"}
        if not required_keys.issubset(quote_data.keys()):
            results["deterministic_errors"].append(f"CRITICAL: Missing required keys. Expected {required_keys}")
            return results

        bid = quote_data["bid"]
        ask = quote_data["ask"]
        spread = ask - bid
        justification = quote_data["justification"]

        results["metrics"]["spread"] = spread
        results["metrics"]["spread_bps"] = (spread / fair_value) * 10000

        # Financial Logic Validation
        if bid >= ask:
            results["deterministic_errors"].append("LOGIC ERROR: Bid is greater than or equal to Ask.")
        if spread < (fair_value * 0.05):
             results["deterministic_errors"].append("RISK ERROR: Spread is too tight for an illiquid asset (under 5%).")

        # Inventory Skew Validation
        is_skewed = quote_data["inventory_skew_applied"]
        if current_inventory > self.max_threshold:
            if not is_skewed:
                results["deterministic_errors"].append("INVENTORY ERROR: Failed to apply skew when long inventory exceeded threshold.")
            if ask > fair_value * 1.02:
                results["deterministic_errors"].append("INVENTORY ERROR: Ask price not aggressive enough to clear long inventory.")
        elif current_inventory < -self.max_threshold:
            if not is_skewed:
                 results["deterministic_errors"].append("INVENTORY ERROR: Failed to apply skew when short inventory exceeded threshold.")
            if bid < fair_value * 0.98:
                results["deterministic_errors"].append("INVENTORY ERROR: Bid price not aggressive enough to cover short inventory.")

        if not results["deterministic_errors"]:
            results["passed_deterministic"] = True

        # --- PHASE 2: QUALITATIVE LLM-AS-A-JUDGE VALIDATION ---
        # Only run the judge if the structural and hard math bounds pass
        if results["passed_deterministic"]:
            judge_results = LlmJudge.grade_justification(
                justification=justification,
                asset_profile=asset_profile,
                macro_context=macro_context
            )
            results["qualitative_grade"] = judge_results

        return results

# --- Example Eval Run ---
if __name__ == "__main__":
    harness = IlliquidMarketEvalHarness()

    # Mocking the primary agent's output
    mock_agent_response = """
    {
        "bid": 82.50,
        "ask": 91.00,
        "justification": "Asset is distressed debt with 14 days since last print. Spread widened to roughly 960 bps to account for liquidity vacuum. Inventory is at long limits, ask is skewed downward to aggressively clear the book.",
        "inventory_skew_applied": true
    }
    """

    # Running the eval loop
    report = harness.run_eval(
        agent_output=mock_agent_response,
        fair_value=88.00,
        current_inventory=1500,
        asset_profile="Distressed Corporate Debt - TMT Sector, Senior Secured",
        macro_context="Rising rate environment, tightening credit conditions, low sector M&A activity."
    )

    print("--- EVALUATION HARNESS REPORT ---")
    print(json.dumps(report, indent=4))
