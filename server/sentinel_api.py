import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from core.governance.sentinel_harness import DecisionState, run_credit_workflow

app = FastAPI(title="Sentinel API", description="Project Sentinel Governance API", version="1.0.0")

class CreditRequest(BaseModel):
    metrics_data: Dict[str, float]
    conviction: float
    npv_fees: float
    sigma: Optional[float] = None
    jurisdiction: str = "USA"
    prompt: str = ""
    context: Optional[Dict[str, Any]] = None

class CreditResponse(BaseModel):
    decision_state: DecisionState
    safe_prompt: str

@app.post("/api/v1/evaluate_credit", response_model=CreditResponse)
def evaluate_credit(request: CreditRequest):
    try:
        decision_state, safe_prompt = run_credit_workflow(
            metrics_data=request.metrics_data,
            conviction=request.conviction,
            npv_fees=request.npv_fees,
            sigma=request.sigma,
            jurisdiction=request.jurisdiction,
            prompt=request.prompt,
            context=request.context
        )
        return CreditResponse(
            decision_state=decision_state,
            safe_prompt=safe_prompt
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    # SECURITY: Bind to localhost (127.0.0.1) instead of 0.0.0.0 to prevent external access during local dev
    uvicorn.run(app, host="127.0.0.1", port=8000)
