"""
Process Scheduling and Coordination Framework for Adam OS.
Handles thread pinning, NUMA-aware memory allocation, and OS-level optimizations
to minimize context switching and CPU cache misses for the Iron Core.
"""

import logging

logger = logging.getLogger(__name__)


class NumaAwareScheduler:
    """
    Manages process affinity and NUMA memory zones to ensure deterministic execution.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.pinned_threads = {}
        logger.info("Initialized NumaAwareScheduler in strict_mode=%s", strict_mode)

    def pin_thread_to_core(self, thread_name: str, core_id: int) -> bool:
        """
        Pins a critical high-frequency strategy thread to a specific CPU core.
        """
        try:
            # Simulated OS-level syscall for CPU affinity
            # os.sched_setaffinity(0, {core_id})
            self.pinned_threads[thread_name] = core_id
            logger.info(
                f"Successfully pinned thread '{thread_name}' to core {core_id}."
            )
            return True
        except Exception as e:
            logger.error(f"Failed to pin thread '{thread_name}': {e}")
            if self.strict_mode:
                raise
            return False

    def allocate_numa_memory(self, size_bytes: int, node_id: int) -> int:
        """
        Requests contiguous memory allocation strictly on the specified NUMA node.
        Returns a mock memory address.
        """
        logger.info(f"Allocating {size_bytes} bytes on NUMA node {node_id}.")
        # Simulated pointer return
        return 0x7F0000000000 + (node_id * 1024)

    def get_pinned_status(self) -> dict:
        return self.pinned_threads
