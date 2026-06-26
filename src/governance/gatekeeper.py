from src.pdil.models import ProvenanceHeader
from src.pdil.middleware import (
    GovernanceError,
    SecurityGovernanceGatekeeper,
    DriftIntelligenceLayer,
    ProofOfThoughtLogger,
    MilestoneLogger
)

# Legacy wrapper to not break existing dependencies
class GovernanceGatekeeper(SecurityGovernanceGatekeeper):
    def detect_and_heal_drift(self, inference_output, historical_hash):
        layer = DriftIntelligenceLayer(self)
        return layer.detect_and_heal_drift(inference_output, historical_hash)

    def heal_drift(self, inference_output):
        layer = DriftIntelligenceLayer(self)
        return layer.heal_drift(inference_output)
