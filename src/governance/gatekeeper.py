from src.pdil.models import ProvenanceHeader
from src.pdil.middleware import (
    GovernanceError,
    SecurityGovernanceGatekeeper,
    JsonLogicGovernanceGatekeeper,
    DriftIntelligenceLayer,
    ProofOfThoughtLogger,
    MilestoneLogger
)

# Legacy wrapper to not break existing dependencies
class GovernanceGatekeeper(SecurityGovernanceGatekeeper):
    def __init__(self, schema=None, rules=None):
        self._delegate = JsonLogicGovernanceGatekeeper(rules) if rules is not None else None
        super().__init__(schema or {'type': 'object'})
            
    def validate_inference(self, inference_output):
        if self._delegate:
            return self._delegate.validate_inference(inference_output)
        return super().validate_inference(inference_output)
        
    def entry_gate(self, inference_output):
        if self._delegate:
            return self._delegate.entry_gate(inference_output)
        return super().entry_gate(inference_output)
        
    def exit_gate(self, inference_output):
        if self._delegate:
            return self._delegate.exit_gate(inference_output)
        return super().exit_gate(inference_output)

    def detect_and_heal_drift(self, inference_output, historical_hash):
        layer = DriftIntelligenceLayer(self)
        return layer.detect_and_heal_drift(inference_output, historical_hash)

    def heal_drift(self, inference_output):
        layer = DriftIntelligenceLayer(self)
        return layer.heal_drift(inference_output)
