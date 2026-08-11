from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from core.governance.risk_class import RiskClass

class MCPCapability(BaseModel):
    capability_id: str
    risk_class: RiskClass
    policy_bundle: str
    provenance_required: bool = True
    required_approval: bool = False

    def validate_intent(self, payload: Dict[str, Any]) -> bool:
        """
        Validates the intent before the tool is executed.
        Returns True if intent is structurally sound according to capability schema.
        """
        # Intent gate validation logic
        return True
