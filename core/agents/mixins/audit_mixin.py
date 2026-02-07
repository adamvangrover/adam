import json
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from core.system.state_manager import StateManager

class AuditMixin:
    """
    Mixin to provide standardized audit logging and state snapshotting capabilities.
    Ensures every critical decision is traceable to a specific agent state.
    """

    def __init__(self, *args, **kwargs):
        # We assume the consuming class calls super().__init__
        # and has a 'config' attribute and 'state_manager' if initialized properly.
        # If not, we initialize a local StateManager.
        if not hasattr(self, 'state_manager'):
            self.state_manager = StateManager()
        if not hasattr(self, 'audit_dir'):
            self.audit_dir = "core/libraries_and_archives/audit_trails"
            os.makedirs(self.audit_dir, exist_ok=True)

    def log_decision(self, activity_type: str, details: Dict[str, Any], outcome: Dict[str, Any], snapshot: bool = True) -> str:
        """
        Logs a decision/action to the persistent audit trail.
        Optionally takes a full state snapshot.
        """
        agent_id = getattr(self, "name", "UnknownAgent")
        timestamp = datetime.utcnow().isoformat() + "Z"
        record_id = f"PROV-{uuid.uuid4()}"

        snapshot_id = None
        if snapshot:
            try:
                # Capture current state
                memory = getattr(self, "memory", {})
                context = getattr(self, "context", {})
                if hasattr(self, "recall"): # Check for MemoryMixin
                    # Try to get more complete state if available
                    pass

                snapshot_id = self.state_manager.save_snapshot(
                    agent_id=agent_id,
                    step_description=f"Decision: {activity_type}",
                    memory=memory,
                    context=context
                )
            except Exception as e:
                logging.warning(f"Failed to capture snapshot for audit: {e}")

        log_entry = {
            "record_id": record_id,
            "timestamp": timestamp,
            "agent_id": agent_id,
            "activity_type": activity_type,
            "snapshot_id": snapshot_id,
            "details": details,
            "outcome": outcome
        }

        # Write to disk
        filename = f"{self.audit_dir}/{record_id}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(log_entry, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to write audit log: {e}")

        return record_id

    def log_compliance_check(self, compliance_result: Any) -> str:
        """
        Specialized logger for SNCCreditPolicyResult objects.
        """
        # Convert Pydantic model to dict
        if hasattr(compliance_result, 'model_dump'):
            result_dict = compliance_result.model_dump()
        else:
            result_dict = compliance_result.__dict__

        return self.log_decision(
            activity_type="ComplianceCheck",
            details={"check_type": "SNC_Validator"},
            outcome=result_dict,
            snapshot=False # Compliance checks might be frequent/lightweight
        )
