from enum import Enum
from pydantic import BaseModel, Field

class RiskClass(str, Enum):
    READ = "READ"
    FINANCIAL = "FINANCIAL"
    ADMIN = "ADMIN"

class CapabilityRequest(BaseModel):
    capability_id: str
    risk_class: RiskClass
    policy_bundle: str
    approval: bool
    idempotency: bool
    provenance: bool

class CapabilityEngine:
    def __init__(self, default_deny: bool = True):
        self.default_deny = default_deny
        self.capabilities = {}

    def register_capability(self, request: CapabilityRequest):
        self.capabilities[request.capability_id] = request

    def evaluate_request(self, capability_id: str) -> bool:
        if capability_id not in self.capabilities:
            return not self.default_deny
        req = self.capabilities[capability_id]
        if req.risk_class in [RiskClass.FINANCIAL, RiskClass.ADMIN] and not req.approval:
            return False
        return True
