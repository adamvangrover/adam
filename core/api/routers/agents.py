from fastapi import APIRouter, Depends, HTTPException

from core.api.deps import get_orchestrator
from core.api.schemas import AnalysisRequest, AnalysisResponse
from core.engine.meta_orchestrator import MetaOrchestrator

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_task(
    request: AnalysisRequest,
    orchestrator: MetaOrchestrator = Depends(get_orchestrator)
):
    try:
        # Route the request through the MetaOrchestrator brain
        result = await orchestrator.route_request(request.query, context=request.context)
        return AnalysisResponse(status="success", result=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    """
    Simple health check.
    """
    return {"status": "Adam v23.5 Core Online", "version": "23.5.0"}
