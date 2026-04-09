import json
import re
import hashlib
import os
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, model_validator

class CreditMetrics(BaseModel):
    pd: float = Field(..., description="Probabilistic Probability of Default")
    lgd: float = Field(..., description="Deterministic Loss Given Default")
    ead: float = Field(..., description="Exposure at Default")
    expected_loss: float = Field(0.0, description="Calculated Expected Loss")

    @model_validator(mode='after')
    def calculate_expected_loss(self):
        self.expected_loss = self.pd * self.lgd * self.ead
        return self

class DecisionState(BaseModel):
    conviction_score: float = Field(ge=0, le=1)
    routing_path: str # ["AUTOMATED", "HOTL", "HITL_TIER_3"]
    requires_step_up: bool
    audit_hash: str # SHA-256 of the context + prompt

class GuardrailWrapper:
    def __init__(self, market_state: Optional[Dict[str, Any]] = None):
        self.market_state = market_state or {"VIX": 22.5, "mode": "Caution Mode Active"}
        self.no_go_jurisdictions = set()
        self._load_governance()

    def _load_governance(self):
        gov_path = os.path.join(os.path.dirname(__file__), 'governance.json')
        if os.path.exists(gov_path):
            with open(gov_path, 'r') as f:
                gov_data = json.load(f)
                self.no_go_jurisdictions = set(gov_data.get('no_go_jurisdictions', []))
        if not self.no_go_jurisdictions:
            self.no_go_jurisdictions = {"Cayman Islands", "North Korea", "Iran"}

    def sanitize_pii(self, prompt: str) -> str:
        """Intercepts prompt strings; replaces names/SSNs with tokens (e.g., [ENTITY_A])."""
        # Very basic regex for SSN-like patterns
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', prompt)
        # Mocking named entity recognition for demonstration purposes
        sanitized = sanitized.replace("John Doe", "[ENTITY_A]")
        sanitized = sanitized.replace("Jane Smith", "[ENTITY_B]")
        return sanitized

    def inject_context(self, prompt: str) -> str:
        """Appends current market state to the system prompt."""
        context_str = f"Current Market State: {json.dumps(self.market_state)}"
        return f"{context_str}\n\n{prompt}"

    def enforce_redlines(self, jurisdiction: str) -> bool:
        """
        A set of hardcoded 'No-Go' rules.
        Returns True if a redline is breached (e.g., requires Senior Review).
        """
        if jurisdiction in self.no_go_jurisdictions:
            return True
        return False

    def wrap_prompt(self, prompt: str, jurisdiction: str = "USA") -> str:
        """Complete wrapper execution."""
        # 1. PII
        safe_prompt = self.sanitize_pii(prompt)
        # 2. Context
        enriched_prompt = self.inject_context(safe_prompt)
        # 3. Redline check (for logging or throwing exceptions)
        breached = self.enforce_redlines(jurisdiction)
        if breached:
            enriched_prompt = "[REDLINE BREACH: HIGH RISK JURISDICTION] " + enriched_prompt
        return enriched_prompt

    def generate_audit_hash(self, prompt: str, context: Dict[str, Any]) -> str:
        payload = f"{prompt}|{json.dumps(context, sort_keys=True)}"
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

# Added Orchestration Function for easier testing and use
def run_credit_workflow(
    metrics_data: Dict[str, float],
    conviction: float,
    npv_fees: float,
    sigma: Optional[float] = None,
    jurisdiction: str = "USA",
    prompt: str = "",
    context: Optional[Dict[str, Any]] = None
) -> Tuple[DecisionState, str]:
    from core.engine.risk_synthesis import RiskSynthesisEngine

    engine = RiskSynthesisEngine()
    return engine.process_orchestration(
        metrics_data=metrics_data,
        conviction=conviction,
        npv_fees=npv_fees,
        sigma=sigma,
        jurisdiction=jurisdiction,
        prompt=prompt,
        context=context
    )
