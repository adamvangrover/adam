from typing import Dict, Any, List, Optional
import logging
import time
from enum import Enum
from dataclasses import dataclass, field
from core.adk_patterns.approval_tool import ApprovalTool, RiskImpact, QuorumPolicy, AuthorityLevel, Stakeholder

logger = logging.getLogger(__name__)

class Environment(Enum):
    DEV = "DEV"
    QA = "QA"
    STAGING = "STAGING"
    PROD = "PROD"
    DR = "DR" # Disaster Recovery

class DeploymentStatus(Enum):
    DRAFT = "DRAFT"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    TIME_LOCKED = "TIME_LOCKED"
    DEPLOYED = "DEPLOYED"
    REJECTED = "REJECTED"

@dataclass
class DeploymentRequest:
    request_id: str
    environment: Environment
    artifact_id: str
    description: str
    created_at: float = field(default_factory=time.time)
    approval_request_id: Optional[str] = None
    status: DeploymentStatus = DeploymentStatus.DRAFT
    override_token: Optional[str] = None

    # Time Lock specific
    approved_at: Optional[float] = None

class EnvironmentGate:
    """
    Enforces environment-specific governance policies.
    """

    # Configuration: Time Lock durations (seconds)
    TIME_LOCKS = {
        Environment.DEV: 0,
        Environment.QA: 0,
        Environment.STAGING: 300, # 5 mins
        Environment.PROD: 3600,   # 1 hour
        Environment.DR: 0         # Immediate for emergencies
    }

    def __init__(self, approval_tool: ApprovalTool):
        self.approval_tool = approval_tool
        self.requests: Dict[str, DeploymentRequest] = {}

    def create_deployment(self, env: str, artifact: str, description: str) -> str:
        try:
            env_enum = Environment[env.upper()]
        except KeyError:
            raise ValueError(f"Invalid environment: {env}")

        req = DeploymentRequest(
            request_id=f"DEP-{int(time.time())}",
            environment=env_enum,
            artifact_id=artifact,
            description=description
        )
        self.requests[req.request_id] = req
        return req.request_id

    def initiate_approval(self, request_id: str, conviction: float = 0.9) -> str:
        """
        Calculates required policy and initiates approval workflow.
        """
        req = self.requests.get(request_id)
        if not req:
            raise ValueError("Request not found")

        policy = self._get_policy_for_env(req.environment)
        impact = self._get_impact_for_env(req.environment)

        # TTL based on environment criticality
        ttl = 86400 if req.environment == Environment.PROD else 3600

        approval_id = self.approval_tool.request_approval(
            action=f"DEPLOY:{req.environment.value}:{req.artifact_id}",
            impact=impact.name,
            conviction=conviction,
            ttl=ttl,
            policy_override=policy
        )

        req.approval_request_id = approval_id
        req.status = DeploymentStatus.PENDING_APPROVAL
        return approval_id

    def check_status(self, request_id: str) -> Dict[str, Any]:
        req = self.requests.get(request_id)
        if not req:
            return {"status": "UNKNOWN"}

        if req.status == DeploymentStatus.PENDING_APPROVAL:
            # Sync with Approval Tool
            status = self.approval_tool.get_session_status(req.approval_request_id)
            if status['status'] == "APPROVED":
                # Move to Time Lock or Approved
                req.status = DeploymentStatus.APPROVED
                req.approved_at = time.time()

                # Check Time Lock
                lock_duration = self.TIME_LOCKS.get(req.environment, 0)
                if lock_duration > 0:
                    req.status = DeploymentStatus.TIME_LOCKED

            elif status['status'] == "REJECTED":
                req.status = DeploymentStatus.REJECTED
            elif status['status'] == "EXPIRED":
                req.status = DeploymentStatus.REJECTED

        # Check Time Lock Expiry
        seconds_remaining = 0
        if req.status == DeploymentStatus.TIME_LOCKED:
            elapsed = time.time() - req.approved_at
            lock_duration = self.TIME_LOCKS.get(req.environment, 0)
            if elapsed >= lock_duration:
                req.status = DeploymentStatus.APPROVED # Unlocked
            else:
                seconds_remaining = int(lock_duration - elapsed)

        return {
            "deployment_id": req.request_id,
            "status": req.status.value,
            "environment": req.environment.value,
            "time_lock_remaining": seconds_remaining,
            "approval_status": self.approval_tool.get_session_status(req.approval_request_id) if req.approval_request_id else None
        }

    def execute_deployment(self, request_id: str) -> Dict[str, Any]:
        """
        Final execution step. Returns the token if allowed.
        """
        status = self.check_status(request_id)
        if status['status'] != "APPROVED":
             return {"error": f"Deployment not ready. Current status: {status['status']}"}

        req = self.requests[request_id]

        # Retrieve the token from the approval tool
        # We need a way to get the token back from the session if it was generated
        # For now, we regenerate it or assume the approval tool provides it via a lookup
        # In `ApprovalTool.grant_approval`, it returns the token.
        # But here we need to fetch it later.
        # Let's assume we can re-generate it if approved, or we should have stored it.
        # Ideally, ApprovalTool stores the token or we generate a new one.

        # For this implementation, we will request the token from ApprovalTool
        # (Assuming we modify ApprovalTool to return stored token or we re-generate)
        # Re-generating is safe if the logic is idempotent and we verify status.

        approval_status = self.approval_tool.get_session_status(req.approval_request_id)
        if approval_status['status'] != 'APPROVED':
             return {"error": "Approval verification failed"}

        # Generate token (using the tool's internal method for consistency)
        token = self.approval_tool._generate_token(f"DEPLOY:{req.environment.value}:{req.artifact_id}")

        req.status = DeploymentStatus.DEPLOYED
        return {"status": "SUCCESS", "token": token}

    def _get_policy_for_env(self, env: Environment) -> QuorumPolicy:
        if env == Environment.DEV:
            return QuorumPolicy(min_approvals=1, min_authority_level=AuthorityLevel.OPERATOR)
        elif env == Environment.STAGING:
            return QuorumPolicy(min_approvals=1, min_authority_level=AuthorityLevel.MANAGER)
        elif env == Environment.PROD:
            # PROD requires 2 Directors or above
            return QuorumPolicy(min_approvals=2, min_authority_level=AuthorityLevel.DIRECTOR)
        elif env == Environment.DR:
            # Emergency: Needs CISO but only 1 (speed)
            return QuorumPolicy(min_approvals=1, min_authority_level=AuthorityLevel.CISO)
        return QuorumPolicy()

    def _get_impact_for_env(self, env: Environment) -> RiskImpact:
        if env == Environment.PROD: return RiskImpact.CRITICAL
        if env == Environment.DR: return RiskImpact.CRITICAL
        if env == Environment.STAGING: return RiskImpact.MEDIUM
        return RiskImpact.LOW
