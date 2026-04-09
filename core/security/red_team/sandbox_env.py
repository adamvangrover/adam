from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SandboxEnvironment:
    """
    Autonomous sandbox environment where threats can be detonated safely.
    Contains blast radius and analyzes impact without affecting the main system.
    """
    def __init__(self):
        self.is_isolated: bool = True
        self.blast_radius: int = 0
        self.active_threats: list[Dict[str, Any]] = []

    def detonate(self, threat: Dict[str, Any]) -> None:
        """
        Detonate a threat safely inside the sandbox.
        """
        logger.info(f"Detonating threat in sandbox: {threat.get('type', 'UNKNOWN')}")
        self.active_threats.append(threat)

    def analyze_impact(self) -> int:
        """
        Analyze the impact of all detonated threats in the sandbox.
        Returns the accumulated blast radius.
        """
        for threat in self.active_threats:
            severity = threat.get("severity_score", 0.0)
            if severity >= 9.0:
                self.blast_radius += 10
            elif severity >= 7.0:
                self.blast_radius += 5
            elif severity >= 4.0:
                self.blast_radius += 2
            else:
                self.blast_radius += 1

        # Clear analyzed threats to avoid double counting on subsequent calls
        self.active_threats = []
        return self.blast_radius

    def contain(self) -> Dict[str, str]:
        """
        Ensure the sandbox is contained and reset the blast radius.
        """
        self.is_isolated = True
        contained_radius = self.blast_radius
        self.blast_radius = 0
        self.active_threats = []
        return {
            "status": "SUCCESS",
            "message": f"Sandbox contained successfully. Previous blast radius: {contained_radius}"
        }
