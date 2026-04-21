from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from core.services.valuation_service import run_valuation

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Valuation API Wrapper",
    description="Cleanly exposes core valuation business logic without UI presentation coupling."
)

class ValuationRequest(BaseModel):
    ebitda_input: float = 50000000.0
    growth_input: float = 0.05
    debt_input: float = 200000000.0
    interest_input: float = 18000000.0
    entry_mult: float = 10.0
    equity_pct: float = 0.40
    kd_input: float = 0.09
    mock_mode: bool = False

@app.post("/api/v1/valuation")
async def run_valuation_endpoint(req: ValuationRequest):
    try:
        MOCK_MODE = req.mock_mode or os.environ.get("MOCK_MODE", "false").lower() == "true" or os.environ.get("ENV", "").lower() == "demo"
        proj_df, enterprise_value, wacc, risk_model, base_rating, snc_status = run_valuation(
            req.ebitda_input,
            req.growth_input,
            req.debt_input,
            req.interest_input,
            req.entry_mult,
            req.equity_pct,
            req.kd_input,
            MOCK_MODE
        )

        return {
            "enterprise_value": enterprise_value,
            "wacc": wacc,
            "base_rating": base_rating,
            "snc_status": snc_status,
            "proj_data": proj_df.to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Valuation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
