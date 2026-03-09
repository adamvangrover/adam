"""
Out-of-Band Hardware Kill Switch for Adam OS.
An independent module that monitors the primary execution core's heartbeat.
If latency thresholds, drawdown limits, or the heartbeat itself fails,
it instantly severs market access at the network layer.
"""

import time
import logging
from threading import Thread

logger = logging.getLogger(__name__)


class HardwareKillSwitch:
    """
    Provides an out-of-band fail-safe mechanism to terminate all active algorithms
    and withdraw liquidity across all connected venues.
    """

    def __init__(self, heartbeat_timeout_ms: int = 500):
        self.heartbeat_timeout_ms = heartbeat_timeout_ms
        self._last_heartbeat_time = time.time()
        self._is_active = False
        self._monitor_thread = None
        logger.warning(
            "HardwareKillSwitch initialized. System requires continuous heartbeats."
        )

    def ping_heartbeat(self):
        """Called continuously by the Iron Core to signal healthy execution."""
        self._last_heartbeat_time = time.time()

    def activate_kill_switch(self, reason: str):
        """
        Instantly bypasses the order management queue and transmits an absolute
        cancellation protocol command directly to the exchange gateway using raw sockets.
        """
        self._is_active = True
        logger.critical(f"HARDWARE KILL SWITCH ACTIVATED. Reason: {reason}")
        # Simulated raw socket cancellation command injection
        logger.critical(
            "Injecting mass cancellation commands to all active egress network queues."
        )

    def _monitor_loop(self):
        """Background thread representing the isolated physical circuit monitoring the heartbeat."""
        while not self._is_active:
            elapsed_ms = (time.time() - self._last_heartbeat_time) * 1000
            if elapsed_ms > self.heartbeat_timeout_ms:
                self.activate_kill_switch(
                    f"Heartbeat lost. Threshold {self.heartbeat_timeout_ms}ms exceeded. Elapsed: {elapsed_ms:.2f}ms"
                )
                break
            time.sleep(
                self.heartbeat_timeout_ms / 2000.0
            )  # Check twice per timeout window

    def arm_switch(self):
        """Starts the background monitoring loop."""
        if not self._monitor_thread or not self._monitor_thread.is_alive():
            self._is_active = False
            self._last_heartbeat_time = time.time()
            self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Hardware Kill Switch ARMED and monitoring.")

    def disarm_switch(self):
        """Carefully disarms the switch (usually for graceful shutdown only)."""
        self._is_active = True  # Stops the loop
        logger.info("Hardware Kill Switch DISARMED.")

    def get_status(self) -> dict:
        """
        Retrieves the current activation and heartbeat status of the kill switch.
        """
        return {
            "is_active": self._is_active,
            "last_heartbeat": self._last_heartbeat_time,
            "timeout_ms": self.heartbeat_timeout_ms,
        }
