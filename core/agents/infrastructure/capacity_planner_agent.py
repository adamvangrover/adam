from typing import Dict, Any, List
import logging
from core.agents.agent_base import AgentBase
from core.agents.mixins.audit_mixin import AuditMixin

class CapacityPlannerAgent(AgentBase, AuditMixin):
    """
    Monitors system telemetry (CPU, GPU, Memory) and recommends infrastructure scaling actions.
    Functions as the "Site Reliability Engineer" (SRE) of the Analyst OS.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        AuditMixin.__init__(self)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Analyzes hardware telemetry and recommends scaling.
        """
        telemetry = kwargs.get("hardware_telemetry", {})

        cpu_load = telemetry.get("cpu_usage", 0)
        gpu_load = telemetry.get("gpu_usage", 0)
        mem_load = telemetry.get("memory_usage_pct", 0)

        recommendations = []
        severity = "LOW"

        # Heuristic Logic
        if cpu_load > 85:
            recommendations.append("Scale Out: Add 2x CPU nodes.")
            severity = "HIGH"
        elif cpu_load > 70:
            recommendations.append("Warning: CPU saturation imminent.")
            severity = "MEDIUM"

        if gpu_load > 90:
            recommendations.append("Scale Up: Provision A100 GPU instance.")
            severity = "CRITICAL"

        if mem_load > 80:
            recommendations.append("Optimize: Run garbage collection / Flush Semantic Cache.")
            severity = "MEDIUM"

        result = {
            "status": "STABLE" if not recommendations else "ACTION_REQUIRED",
            "severity": severity,
            "telemetry_snapshot": telemetry,
            "recommendations": recommendations
        }

        self.log_decision(
            activity_type="CapacityPlanning",
            details={"telemetry": telemetry},
            outcome=result
        )

        return result

    def get_skill_schema(self) -> Dict[str, Any]:
        return {
            "name": "CapacityPlannerAgent",
            "description": "Monitors system resources and recommends scaling.",
            "skills": [
                {
                    "name": "analyze_capacity",
                    "description": "Reviews hardware metrics and suggests scaling.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hardware_telemetry": {"type": "object"}
                        }
                    }
                }
            ]
        }
