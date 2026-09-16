import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any
from core.schemas.agent_schema import AgentOutput
from src.pdil.models import ProvenanceHeader

def calculate_obligor_pd(financials: Dict[str, Any]) -> float:
    """Calculate Obligor-level Probability of Default (PD) representing fundamental creditworthiness."""
    assets = financials.get("total_assets", 1.0)
    liabilities = financials.get("total_liabilities", 0.0)
    ebitda = financials.get("ebitda", 0.1)
    leverage = liabilities / max(assets, 0.01)
    dscr = ebitda / max(financials.get("interest_expense", 0.01), 0.01)
    pd_base = 0.01 + (leverage * 0.05) - (dscr * 0.01)
    return max(0.001, min(0.999, pd_base))

def calculate_facility_rating(obligor_pd: float, facility_details: Dict[str, Any]) -> float:
    """Facility-level ratings incorporate transaction structures like collateral/LGD. Strictly separate from Obligor-level PD."""
    lgd = facility_details.get("lgd", 0.5)
    collateral_value = facility_details.get("collateral_value", 0.0)
    exposure = facility_details.get("exposure", 1.0)
    net_exposure = max(0.0, exposure - collateral_value)
    return obligor_pd * lgd * net_exposure

def assess_credit_risk(entity_id: str, context: Dict[str, Any]) -> AgentOutput:
    financials = context.get("financials", {})
    facility = context.get("facility", {})
    obligor_pd = calculate_obligor_pd(financials)
    facility_expected_loss = calculate_facility_rating(obligor_pd, facility)
    result_data = {
        "entity_id": entity_id,
        "obligor_pd": obligor_pd,
        "facility_expected_loss": facility_expected_loss,
        "warning": "Obligor PD and Facility Risk are strictly separated."
    }
    content_str = json.dumps(result_data, sort_keys=True)
    content_hash = hashlib.sha256(content_str.encode('utf-8')).hexdigest()
    provenance = ProvenanceHeader(
        git_commit_hash="pending",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash=content_hash,
        jsonLogic_version="1.0.0",
        confidence_score=0.95,
        derivation_path="scripts.realtime_pd_model.assess_credit_risk",
        source_data_object=entity_id
    )
    return AgentOutput(
        answer=content_str,
        sources=[entity_id],
        confidence=0.95,
        metadata=result_data,
        provenance_trace=provenance,
        observed_drift=False
    )
