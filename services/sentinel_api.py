from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from core.schemas.sentinel import CreditMetrics, DecisionState
from core.governance.sentinel.harness import RiskSynthesisEngine, GuardrailWrapper

app = FastAPI(title="Sentinel Orchestration API")

engine = RiskSynthesisEngine()
guardrail = GuardrailWrapper()

class EvaluationRequest(BaseModel):
    metrics: CreditMetrics
    npv_fees: float
    conviction_score: float
    context: Dict[str, Any]
    prompt: str

@app.post("/evaluate", response_model=DecisionState)
async def evaluate_risk(request: EvaluationRequest):
    try:
        # 1. Guardrail - Sanitize PII
        sanitized_prompt = guardrail.sanitize_pii(request.prompt)

        # 2. Guardrail - Context Injector
        injected_prompt = guardrail.inject_context(sanitized_prompt)

        # 3. Guardrail - Redline Enforcer
        policy_breach = guardrail.enforce_redlines(request.context)

        # 4. Synthesize Engine Execution
        decision = engine.evaluate(
            metrics=request.metrics,
            npv_fees=request.npv_fees,
            conviction_score=request.conviction_score,
            policy_breach=policy_breach,
            prompt_context=injected_prompt
        )

        return decision
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # SECURITY: Bind to localhost (127.0.0.1) instead of 0.0.0.0 to prevent external access during local dev
    uvicorn.run(app, host="127.0.0.1", port=8001)
