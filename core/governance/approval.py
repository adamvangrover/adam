from typing import Optional, Dict, Any

class ApprovalGate:
    def __init__(self, required_role: str = "admin"):
        self.required_role = required_role

    def evaluate(self, user_context: Dict[str, Any], action_risk: str) -> bool:
        """
        Evaluate if the provided user context satisfies the approval gate for the action.
        """
        if action_risk in ["financial", "admin"]:
            if user_context.get("role") != self.required_role:
                return False
            # Check for multi-factor or explicit human-in-the-loop sign-off here
            if not user_context.get("explicit_approval"):
                return False
        return True
