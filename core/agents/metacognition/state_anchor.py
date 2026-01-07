# core/agents/metacognition/state_anchor.py
import uuid
import json
import time
import hashlib
from typing import Dict, Any, Optional
import logging

class StateAnchorMixin:
    """
    Implements 'State Anchors' to mitigate 'State Drift' in asynchronous workflows.
    Section 7.3 of the Protocol Paradox.
    """

    def __init__(self):
        self._anchors: Dict[str, Dict[str, Any]] = {}

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Computes a deterministic hash of the state dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def create_anchor(self, label: str = "auto") -> str:
        """
        Snapshots the current agent state (HNASP + Context).
        Returns an Anchor ID.
        """
        # Snapshot HNASP State if available
        hnasp_snapshot = {}
        if hasattr(self, 'state') and self.state:
            if hasattr(self.state, 'model_dump'):
                hnasp_snapshot = self.state.model_dump()
            elif isinstance(self.state, dict):
                hnasp_snapshot = self.state.copy()
            else:
                try:
                    hnasp_snapshot = self.state.__dict__.copy()
                except AttributeError:
                    logging.warning("Could not serialize state object for anchor.")

        # Snapshot Legacy Context
        context_snapshot = getattr(self, 'context', {}).copy()

        # Combined State
        full_state = {
            "hnasp": hnasp_snapshot,
            "context": context_snapshot,
            "timestamp": time.time()
        }

        anchor_id = f"anchor_{uuid.uuid4().hex[:8]}"
        state_hash = self._compute_hash(full_state)

        self._anchors[anchor_id] = {
            "id": anchor_id,
            "label": label,
            "hash": state_hash,
            "snapshot": full_state, # In production, store this in Redis/Disk, not RAM
            "timestamp": time.time()
        }

        logging.info(f"State Anchor created: {anchor_id} (Hash: {state_hash[:8]})")
        return anchor_id

    def verify_anchor(self, anchor_id: str) -> bool:
        """
        Checks if the CURRENT state matches the Anchor.
        Returns True if state is consistent (No Drift).
        Returns False if Drift detected.
        """
        if anchor_id not in self._anchors:
            logging.warning(f"Anchor {anchor_id} not found.")
            return False

        anchor = self._anchors[anchor_id]

        # Capture current state
        current_hnasp = {}
        if hasattr(self, 'state') and self.state:
            if hasattr(self.state, 'model_dump'):
                current_hnasp = self.state.model_dump()
            elif isinstance(self.state, dict):
                current_hnasp = self.state.copy()
            else:
                try:
                    current_hnasp = self.state.__dict__.copy()
                except AttributeError:
                    pass

        current_context = getattr(self, 'context', {}).copy()

        current_full = {
            "hnasp": current_hnasp,
            "context": current_context,
            "timestamp": anchor["snapshot"]["timestamp"] # Compare content, ignore timestamp diff
        }

        current_hash = self._compute_hash(current_full)

        is_valid = current_hash == anchor["hash"]

        if not is_valid:
            logging.warning(f"State Drift Detected! Anchor {anchor_id} hash {anchor['hash'][:8]} != current {current_hash[:8]}")

        return is_valid

    def rehydrate_from_anchor(self, anchor_id: str):
        """
        Forcefully restores the agent's state to the anchor point.
        Use with caution - creates a time-travel paradox for the agent's memory.
        """
        if anchor_id not in self._anchors:
            raise ValueError(f"Anchor {anchor_id} not found.")

        anchor = self._anchors[anchor_id]
        snapshot = anchor["snapshot"]

        # Restore Legacy Context
        if hasattr(self, 'context'):
            self.context = snapshot["context"]

        # Restore HNASP (Partial/Mock as full restoration of Pydantic from dict is complex)
        if hasattr(self, 'state'):
            # In a real implementation, we would re-instantiate the Pydantic model
            # self.state = HNASPState(**snapshot["hnasp"])
            logging.info(f"Rehydrated state from anchor {anchor_id}")
