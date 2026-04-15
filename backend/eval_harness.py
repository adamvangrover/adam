import json

class EvalHarness:
    def __init__(self, fair_value: float, max_threshold: int):
        self.fair_value = fair_value
        self.max_threshold = max_threshold

    def evaluate(self, payload: str, inventory: int):
        """
        Runs the dual-layer evaluation harness on the generated payload.
        """
        try:
            # Parse payload
            data = json.loads(payload)
            bid = data.get("bid")
            ask = data.get("ask")
            justification = data.get("justification")
            inventory_skew_applied = data.get("inventory_skew_applied")

            if bid is None or ask is None or not justification or inventory_skew_applied is None:
                return {"status": "FAIL", "reason": "Missing required JSON fields"}

            # Layer 1: Deterministic Math/Logic Gate
            spread = ask - bid
            if spread <= 0:
                return {"status": "FAIL", "reason": "Spread must be positive"}

            # Spread > 5% of fair value for illiquid securities
            if spread < 0.05 * self.fair_value:
                return {"status": "FAIL", "reason": "Spread too narrow for illiquid asset"}

            # Inventory skew logic: if inventory exceeds MAX_THRESHOLD, quotes must be skewed
            # and `inventory_skew_applied` should be True
            if abs(inventory) > self.max_threshold:
                if not inventory_skew_applied:
                    return {"status": "FAIL", "reason": "Inventory exceeds threshold but skew was not applied"}

                # Check actual skew logic based on inventory
                # If long (inventory > 0), bid should drop significantly to discourage buying
                # If short (inventory < 0), ask should rise significantly to discourage selling
                # For deterministic check, we can check if it shifted from symmetrical fair value
                midpoint = (bid + ask) / 2
                if inventory > 0 and midpoint >= self.fair_value:
                    return {"status": "FAIL", "reason": "Long inventory but quotes not skewed to sell"}
                if inventory < 0 and midpoint <= self.fair_value:
                     return {"status": "FAIL", "reason": "Short inventory but quotes not skewed to buy"}

            # Layer 2: Qualitative (LLM-as-a-Judge)
            # Simulated Senior Credit Risk Officer grading 1-5
            score = self._qualitative_eval(justification)
            if score < 4:
                return {"status": "FAIL", "reason": f"Qualitative score {score} < 4. Justification insufficient."}

            return {
                "status": "PASS",
                "metrics": {"spread_bps": (spread / self.fair_value) * 10000, "inventory": inventory},
                "feedback": "Passed all evaluations"
            }

        except json.JSONDecodeError:
            return {"status": "FAIL", "reason": "Invalid JSON payload"}
        except Exception as e:
            return {"status": "FAIL", "reason": f"Evaluation error: {str(e)}"}

    def _qualitative_eval(self, justification: str) -> int:
        """
        Simulates an LLM-as-a-Judge grading the justification from 1-5.
        In a real scenario, this would call an LLM (e.g., litellm.completion).
        For this harness simulation without an actual LLM call, we'll assign a score
        based on keyword presence or length.
        """
        # Basic mock implementation
        lower_j = justification.lower()
        score = 1
        if len(justification) > 50:
            score += 1
        if any(w in lower_j for w in ["macro", "environment", "risk", "liquidity"]):
            score += 1
        if any(w in lower_j for w in ["distressed", "esoteric", "premium", "asymmetric"]):
            score += 1
        if any(w in lower_j for w in ["skew", "inventory", "position"]):
             score += 1
        return min(score, 5)
