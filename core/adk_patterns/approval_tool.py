from typing import Dict, Any, List, Optional, Set
import logging
import time
import uuid
import hmac
import hashlib
import os
from enum import Enum, auto
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class AuthorityLevel(Enum):
    OPERATOR = 1
    MANAGER = 2
    DIRECTOR = 3
    CISO = 4
    BOARD = 5

class RiskImpact(Enum):
    NEGLIGIBLE = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

@dataclass
class RiskAssessment:
    """
    Quantifies the risk of an operation.
    """
    impact: RiskImpact
    conviction_score: float  # 0.0 to 1.0 (How sure are we this is safe?)
    decay_ttl_seconds: int = 300  # Time To Live for the approval

    @property
    def risk_score(self) -> float:
        uncertainty_penalty = 1.0 + (1.0 - self.conviction_score)
        return self.impact.value * uncertainty_penalty

@dataclass
class Stakeholder:
    user_id: str
    role: str
    authority_level: AuthorityLevel
    public_key: Optional[str] = None # For future crypto verification

@dataclass
class Signature:
    stakeholder: Stakeholder
    timestamp: float
    signature_hash: str # Mock of a digital signature
    comment: Optional[str] = None

class ApprovalStatus(Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class QuorumPolicy:
    """
    Defines complex approval requirements.
    e.g. 2 signers total, must include 1 CISO.
    """
    min_approvals: int = 1
    required_roles: List[str] = field(default_factory=list) # Specific roles like "CISO", "DevOps"
    min_authority_level: AuthorityLevel = AuthorityLevel.OPERATOR

@dataclass
class ApprovalSession:
    request_id: str
    action: str
    risk_assessment: RiskAssessment
    policy: QuorumPolicy
    created_at: float = field(default_factory=time.time)
    signatures: List[Signature] = field(default_factory=list)
    status: ApprovalStatus = ApprovalStatus.PENDING
    audit_log: List[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.risk_assessment.decay_ttl_seconds

    def check_consensus(self) -> bool:
        """
        Validates if the current signatures meet the QuorumPolicy.
        """
        if len(self.signatures) < self.policy.min_approvals:
            return False

        # Check Authority Levels
        valid_signatures = [
            s for s in self.signatures
            if s.stakeholder.authority_level.value >= self.policy.min_authority_level.value
        ]

        if len(valid_signatures) < self.policy.min_approvals:
            return False

        # Check Required Roles (if any)
        # Assuming one person can satisfy one role requirement
        signed_roles = {s.stakeholder.role.lower() for s in valid_signatures}
        for req_role in self.policy.required_roles:
            if req_role.lower() not in signed_roles:
                return False

        return True

class ApprovalTool:
    """
    Centralized Approval Engine for Human-in-the-Loop Governance.
    Manages risk assessment, consensus building, and token issuance.
    """

    def __init__(self, secret_key: str = None):
        self._sessions: Dict[str, ApprovalSession] = {}
        self.secret_key = secret_key or os.environ.get('GOVERNANCE_OVERRIDE_SECRET', 'dev-secret-do-not-use-in-prod')

    def request_approval(self, action: str, impact: str, conviction: float, ttl: int = 300,
                         policy_override: Optional[QuorumPolicy] = None) -> str:
        """
        Initiates a new approval session.
        Returns the request_id.
        """
        try:
            impact_enum = RiskImpact[impact.upper()]
        except KeyError:
            impact_enum = RiskImpact.MEDIUM

        risk = RiskAssessment(impact=impact_enum, conviction_score=conviction, decay_ttl_seconds=ttl)

        if policy_override:
            policy = policy_override
        else:
            policy = self._derive_policy(risk)

        session = ApprovalSession(
            request_id=str(uuid.uuid4()),
            action=action,
            risk_assessment=risk,
            policy=policy
        )
        session.audit_log.append(f"Session Created. Policy: {policy}")

        self._sessions[session.request_id] = session
        logger.info(f"[ApprovalTool] New Request {session.request_id}: Action='{action}', RiskScore={risk.risk_score:.2f}")
        return session.request_id

    def _derive_policy(self, risk: RiskAssessment) -> QuorumPolicy:
        """
        Determines the default policy based on risk score.
        """
        score = risk.risk_score
        if score < 2.0:
            return QuorumPolicy(min_approvals=1, min_authority_level=AuthorityLevel.OPERATOR)
        elif score < 4.0:
            return QuorumPolicy(min_approvals=1, min_authority_level=AuthorityLevel.MANAGER)
        elif score < 6.0:
            return QuorumPolicy(min_approvals=2, min_authority_level=AuthorityLevel.DIRECTOR)
        else: # Critical
            # Require CISO specifically for Critical risks
            return QuorumPolicy(min_approvals=2, min_authority_level=AuthorityLevel.DIRECTOR, required_roles=["CISO"])

    def grant_approval(self, request_id: str, stakeholder: Stakeholder, comment: str = "") -> Dict[str, Any]:
        """
        Registers an approval (signature) from a stakeholder.
        """
        session = self._sessions.get(request_id)
        if not session:
            return {"error": "Session not found"}

        if session.is_expired:
            session.status = ApprovalStatus.EXPIRED
            return {"error": "Request expired due to time decay"}

        # Prevent double signing
        if any(s.stakeholder.user_id == stakeholder.user_id for s in session.signatures):
             return {"error": "User already signed this request"}

        # Create Signature
        # In a real system, we'd verify a crypto signature here.
        mock_hash = hashlib.sha256(f"{stakeholder.user_id}:{time.time()}".encode()).hexdigest()

        signature = Signature(
            stakeholder=stakeholder,
            timestamp=time.time(),
            signature_hash=mock_hash,
            comment=comment
        )

        session.signatures.append(signature)
        session.audit_log.append(f"Approved by {stakeholder.user_id} ({stakeholder.role})")
        logger.info(f"[ApprovalTool] Approval granted for {request_id} by {stakeholder.user_id}")

        # Check Consensus
        if session.check_consensus():
            session.status = ApprovalStatus.APPROVED
            token = self._generate_token(session.action)
            session.audit_log.append("Consensus Reached. Token Issued.")
            logger.info(f"[ApprovalTool] Consensus Reached for {request_id}.")
            return {
                "status": "APPROVED",
                "token": token,
                "message": "Consensus reached. Override token issued."
            }

        return {
            "status": "PENDING",
            "message": f"Approval recorded. Signatures: {len(session.signatures)}/{session.policy.min_approvals}"
        }

    def _generate_token(self, action_path: str) -> str:
        """
        Generates the HMAC-signed override token compatible with GovernanceMiddleware.
        """
        timestamp = int(time.time())
        payload = f"{timestamp}:{action_path}".encode()
        signature = hmac.new(self.secret_key.encode(), payload, hashlib.sha256).hexdigest()
        return f"{timestamp}:{signature}"

    def get_session_status(self, request_id: str) -> Dict[str, Any]:
        session = self._sessions.get(request_id)
        if not session:
            return {"status": "UNKNOWN"}

        return {
            "request_id": session.request_id,
            "status": session.status.value,
            "signatures_count": len(session.signatures),
            "required_approvals": session.policy.min_approvals,
            "seconds_remaining": max(0, int(session.risk_assessment.decay_ttl_seconds - (time.time() - session.created_at))),
            "audit_log": session.audit_log
        }
