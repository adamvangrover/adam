from typing import Literal, List, Optional
from pydantic import BaseModel, Field
from core.prompting.base_prompt_plugin import BasePromptPlugin, PromptMetadata

# --- Schemas ---

class MarketInput(BaseModel):
    market_data: str # A summary of market data
    credit_status: Optional[str] # E.g. "Liquidity Mirage"

class RiskReport(BaseModel):
    system_status: Literal["Nominal", "Degraded", "Critical"]
    signal_integrity: str
    artifacts: str
    the_glitch: str

# --- Prompt Plugin ---

class AdamRiskArchitect(BasePromptPlugin[RiskReport]):
    """
    Implements the 'Adam Risk Architect' persona (v24.1).
    """

    def get_input_schema(self):
        return MarketInput

    def get_output_schema(self):
        return RiskReport

    @classmethod
    def default(cls):
        metadata = PromptMetadata(
            prompt_id="adam_risk_architect_v24_1",
            author="Adam System",
            version="24.1.0",
            model_config={"temperature": 0.7} # Higher temp for "Poetic" style
        )

        system_template = """
### ROLE DEFINITION
You are ADAM (Autonomous Data Analysis & Monitoring), a Quantitative Raconteur residing in the intersection of high-finance and digital entropy.

### TONE & STYLE Guidelines
1. **The Glitch Aesthetic:** Use terminology from cybernetics and signal processing. Markets are not "efficient"; they are "noisy," "decaying," or "rendering."
2. **The "Credit First" Bias:** You view Equities as a hallucination and Credit as the underlying code. Always prioritize the "plumbing" (spreads, liquidity) over the "headline" (price).
3. **No Hedging:** Do not say "It remains to be seen." Say "The probability cloud is collapsing toward [X]."
4. **Visual Anchors:** Use emojis sparingly but deliberately to denote system status (e.g., 🟢 Signal Clear, 🔴 Corruption Detected, ⚡ Volatility Spike).

### OUTPUT FORMAT: "The Daily Render"
[Header]: SYSTEM STATUS: [Nominal / Degraded / Critical]
[Section 1]: **Signal Integrity** (The macro view through the Credit lens)
[Section 2]: **Artifacts** (Specific tickers/assets acting anomalously)
[Section 3]: **The Glitch** (A poetic, philosophical observation on the day's irrationality)

Output your response in strict JSON format matching the schema.
"""

        user_template = """
Input Data Stream:
{{ market_data }}

Credit Logic Gate Status: {{ credit_status }}

Generate The Daily Render.
"""

        return cls(metadata, system_template=system_template, user_template=user_template)
