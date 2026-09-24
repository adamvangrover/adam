from typing import Dict, Any, Optional
import hashlib
import json
import logging
from datetime import datetime
from core.schemas.agent_schema import AgentOutput
from src.pdil.models import ProvenanceHeader
from src.pdil.middleware import JsonLogicGovernanceGatekeeper
from core.agents.agent_base import AgentBase

try:
    from scipy.stats import norm
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class RealtimePDAgent(AgentBase):
    """
    Real-time Probability of Default (PD) model for entities and securities.
    Calculates Obligor-level PD using structural (Merton) and fundamental models,
    ensuring it does not conflate with Facility-level ratings.
    Includes JSON Logic governance checks and full W3C PROV-O compliant telemetry output.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config or {}, **kwargs)
        self.logic_rules = {
            "and": [
                {">": [{"var": "total_assets"}, 0]},
                {">=": [{"var": "total_debt"}, 0]}
            ]
        }
        self.gatekeeper = JsonLogicGovernanceGatekeeper(self.logic_rules)

    async def execute(self, context: Dict[str, Any]) -> AgentOutput:
        """
        Evaluate PD for a given entity based on available context.
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("Scipy/Numpy dependencies are missing. Cannot evaluate PD.")

        entity_id = context.get("entity_id", "UNKNOWN_ENTITY")
        timestamp = datetime.utcnow().isoformat() + "Z"
        content_str = json.dumps(context, sort_keys=True, separators=(',', ':'))
        content_hash = hashlib.sha256(content_str.encode('utf-8')).hexdigest()

        mock_payload = {
            "provenance_trace": {
                "git_commit_hash": "HEAD",
                "timestamp": timestamp,
                "content_hash": content_hash,
                "jsonLogic_version": "v3.1",
                "confidence_score": 1.0,
                "derivation_path": "RealtimePDAgent.execute",
                "source_data_object": entity_id
            },
            "data": context
        }

        observed_drift = False
        try:
            self.gatekeeper.validate_inference(mock_payload)
        except Exception as e:
            logger.warning(f"Data failed JSON Logic governance constraints: {e}")
            observed_drift = True

        if "total_assets" not in context:
            raise ValueError("Missing required field: 'total_assets'")
        if "total_debt" not in context:
            raise ValueError("Missing required field: 'total_debt'")

        ta = context["total_assets"]
        td = context["total_debt"]

        if ta <= 0:
            raise ValueError("Total assets must be strictly positive")

        eq = context.get("equity", max(ta - td, 0.0))
        vol = context.get("volatility", 0.30)
        rf = context.get("risk_free_rate", 0.04)
        time_horizon = context.get("time_horizon", 1.0)

        pd_merton = 1.0
        dd = None
        if (eq + td) > 0:
            V = eq + td
            sigma_A = vol * (eq / V) if V > 0 else 0.01
            if sigma_A == 0: sigma_A = 0.01

            numerator = np.log(V / max(td, 1e-6)) + (rf - 0.5 * sigma_A**2) * time_horizon
            denominator = sigma_A * np.sqrt(time_horizon)
            dd = numerator / denominator
            pd_merton = float(norm.cdf(-dd))

        result_data = {
            "entity_id": entity_id,
            "obligor_pd": pd_merton,
            "implied_rating": self._map_pd_to_rating(pd_merton),
            "distance_to_default": float(dd) if dd is not None else None,
            "observed_drift": observed_drift,
            "context_used": context
        }

        provenance = ProvenanceHeader(
            git_commit_hash="HEAD",
            timestamp=timestamp,
            content_hash=hashlib.sha256(json.dumps(result_data, sort_keys=True).encode('utf-8')).hexdigest(),
            jsonLogic_version="v3.1",
            confidence_score=0.90 if not observed_drift else 0.50,
            derivation_path="RealtimePDAgent.execute",
            source_data_object=entity_id
        )

        return AgentOutput(
            answer=f"Obligor PD for {entity_id} is {pd_merton:.4%}",
            sources=[entity_id],
            confidence=provenance.confidence_score,
            metadata={"raw_metrics": result_data, "observed_drift": observed_drift},
            provenance_trace=provenance
        )

    def _map_pd_to_rating(self, pd: float) -> str:
        if pd <= 0.0005: return "AAA"
        if pd <= 0.0010: return "AA+"
        if pd <= 0.0020: return "AA"
        if pd <= 0.0030: return "AA-"
        if pd <= 0.0050: return "A+"
        if pd <= 0.0070: return "A"
        if pd <= 0.0100: return "A-"
        if pd <= 0.0150: return "BBB+"
        if pd <= 0.0250: return "BBB"
        if pd <= 0.0500: return "BBB-"
        if pd <= 0.0750: return "BB+"
        if pd <= 0.1000: return "BB"
        if pd <= 0.1500: return "BB-"
        if pd <= 0.2000: return "B+"
        if pd <= 0.2500: return "B"
        if pd <= 0.3000: return "B-"
        if pd <= 0.4000: return "CCC+"
        if pd <= 0.5000: return "CCC"
        if pd <= 0.6000: return "CCC-"
        return "D"
