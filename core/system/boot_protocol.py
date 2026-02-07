import time
from typing import Any
from .system_boot_logger import SystemBootLogger, BootLogEntry

class BootProtocol:
    """
    Protocolmixin to standardize agent boot reporting.
    """

    def report_boot_status(self, agent_id: str, prompt: str, conviction: float, status: str = "BOOT_COMPLETE"):
        """
        Logs the boot status to the central version control log.
        """
        entry = BootLogEntry(
            timestamp=time.time(),
            agent_id=agent_id,
            status=status,
            highest_conviction_prompt=prompt,
            conviction_score=conviction
        )
        SystemBootLogger.log_boot(entry)
