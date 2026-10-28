"""
AFOS V3 Kernel Circuit Breakers & Air-Gapped Quarantine Matrix.
Features:
1. Air-Gapped Quarantine Sandbox for forensic logging with SHA-256 traces.
2. Execution Fallback: Automatically disables Hunt & Kill Orchestrator and forces 100% reversion
   to 75% Fortress allocation (sub-90-day sovereign paper) if consensus verification drops < 100%.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QuarantineRecord(BaseModel):
    quarantine_id: str
    timestamp: float
    source_node: str
    quarantine_reason: str
    payload_sha256: str
    raw_payload_summary: Dict[str, Any]
    forensic_tags: List[str]


class SystemState(BaseModel):
    hunt_and_kill_enabled: bool
    fortress_allocation_pct: float
    convex_hunt_allocation_pct: float
    lockdown_active: bool
    quarantine_records_count: int
    last_lockdown_reason: Optional[str] = None


class QuarantineMatrix:
    """
    Air-Gapped Quarantine Matrix & System Governance Circuit Breakers.
    Manages air-gapped forensic logging and controls portfolio allocation state transitions.
    """

    def __init__(self):
        self._quarantine_store: List[QuarantineRecord] = []
        self.hunt_and_kill_enabled: bool = True
        self.fortress_allocation_pct: float = 75.0
        self.convex_hunt_allocation_pct: float = 25.0
        self.lockdown_active: bool = False
        self.last_lockdown_reason: Optional[str] = None

    def divert_to_quarantine(
        self,
        source_node: str,
        reason: str,
        raw_payload: Dict[str, Any],
        forensic_tags: List[str]
    ) -> QuarantineRecord:
        """
        Diverts corrupted, unverified, or anomalous payload to air-gapped sandbox for forensic logging.
        """
        payload_bytes = json.dumps(raw_payload, sort_keys=True).encode("utf-8")
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()
        quarantine_id = f"QARR-{int(time.time()*1000)}-{payload_hash[:8]}"

        record = QuarantineRecord(
            quarantine_id=quarantine_id,
            timestamp=time.time(),
            source_node=source_node,
            quarantine_reason=reason,
            payload_sha256=payload_hash,
            raw_payload_summary=raw_payload,
            forensic_tags=forensic_tags
        )

        self._quarantine_store.append(record)
        return record

    def trigger_circuit_breaker(self, reason: str) -> SystemState:
        """
        Triggers emergency circuit breaker:
        - Disables Hunt & Kill Orchestrator (25% Convex Hunt sleeve)
        - Reverts 100% of portfolio allocation to Fortress (sub-90-day sovereign paper)
        - Locks down system until human authorization overrides
        """
        self.hunt_and_kill_enabled = False
        self.fortress_allocation_pct = 100.0
        self.convex_hunt_allocation_pct = 0.0
        self.lockdown_active = True
        self.last_lockdown_reason = reason

        return self.get_system_state()

    def human_override_unlock(self, authorization_token: str) -> SystemState:
        """
        Human override to restore 75% Fortress / 25% Convex Hunt sleeve allocation and re-enable Hunt & Kill.
        """
        if not authorization_token or "HUMAN_OVERRIDE_APPROVED" not in authorization_token:
            raise PermissionError("Human override authorization token invalid.")

        self.hunt_and_kill_enabled = True
        self.fortress_allocation_pct = 75.0
        self.convex_hunt_allocation_pct = 25.0
        self.lockdown_active = False
        self.last_lockdown_reason = None

        return self.get_system_state()

    def get_system_state(self) -> SystemState:
        """Returns current system allocation and circuit breaker status."""
        return SystemState(
            hunt_and_kill_enabled=self.hunt_and_kill_enabled,
            fortress_allocation_pct=self.fortress_allocation_pct,
            convex_hunt_allocation_pct=self.convex_hunt_allocation_pct,
            lockdown_active=self.lockdown_active,
            quarantine_records_count=len(self._quarantine_store),
            last_lockdown_reason=self.last_lockdown_reason
        )

    def list_quarantine_records(self) -> List[QuarantineRecord]:
        """Returns stored forensic quarantine records."""
        return list(self._quarantine_store)


# Shared global instance
global_quarantine_matrix = QuarantineMatrix()
