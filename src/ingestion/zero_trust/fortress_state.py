import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class SystemStatus(str, Enum):
    NORMAL = "NORMAL"
    LOCKDOWN = "LOCKDOWN"

class FortressStateManager:
    """
    Phase 4: Fortress Fallback Override & State Manager
    Appends and hardens existing state logic while preserving core 75% Fortress / 25% Convex Hunt logic.
    If consensus drops below threshold (e.g. < 2 valid feeds or spoofing detected),
    triggers automated state override: SYSTEM_STATUS = "LOCKDOWN".
    Halts all 25% Convex Hunt executions and defaults 100% remaining active capital to 75% Fortress sovereign paper (sub-90-day).
    """
    def __init__(self):
        self.system_status: SystemStatus = SystemStatus.NORMAL
        self.fortress_allocation_pct: float = 75.0
        self.convex_hunt_allocation_pct: float = 25.0
        self.manual_meatspace_authorization: bool = False

    def evaluate_state_override(self, valid_feed_count: int, spoofing_detected: bool, total_feeds: int = 3) -> Dict[str, Any]:
        """
        Evaluates system consensus matrix and triggers LOCKDOWN override if consensus threshold fails.
        """
        # Minimum threshold: at least 2 valid feeds AND no spoofing attack
        if valid_feed_count < 2 or spoofing_detected:
            self.trigger_lockdown(
                reason=f"Consensus threshold breach: valid_feeds={valid_feed_count}/{total_feeds}, spoofing={spoofing_detected}"
            )
        return self.get_current_allocation()

    def trigger_lockdown(self, reason: str) -> None:
        self.system_status = SystemStatus.LOCKDOWN
        # Halts 25% Convex Hunt executions; defaults capital to sub-90-day sovereign paper Fortress
        self.fortress_allocation_pct = 100.0
        self.convex_hunt_allocation_pct = 0.0
        self.manual_meatspace_authorization = False
        logger.critical(f"AUTOMATED KERNEL STATE OVERRIDE: SYSTEM_STATUS = 'LOCKDOWN'! Reason: {reason}")

    def authorize_meatspace_override(self) -> None:
        """Manual meatspace authorization to restore normal operations."""
        self.manual_meatspace_authorization = True
        self.system_status = SystemStatus.NORMAL
        self.fortress_allocation_pct = 75.0
        self.convex_hunt_allocation_pct = 25.0
        logger.info("Meatspace authorization granted. Normal 75% Fortress / 25% Convex Hunt operations restored.")

    def get_current_allocation(self) -> Dict[str, Any]:
        return {
            "system_status": self.system_status.value,
            "fortress_allocation_pct": self.fortress_allocation_pct,
            "convex_hunt_allocation_pct": self.convex_hunt_allocation_pct,
            "convex_hunt_execution_allowed": (self.system_status == SystemStatus.NORMAL),
            "asset_class": "SUB_90_DAY_SOVEREIGN_PAPER" if self.system_status == SystemStatus.LOCKDOWN else "DUAL_FORTRESS_HUNT"
        }
