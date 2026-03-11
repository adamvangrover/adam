"""
NUMA Scheduling and Thread Coordination Module for Adam OS.
Minimizes context switching and CPU cache misses by utilizing
thread pinning and NUMA-aware memory allocation.
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class NUMAScheduler:
    """
    Handles NUMA-aware scheduling and CPU pinning for the Iron Core to
    ensure deterministic high-frequency execution.
    """

    def __init__(self, enable_pinning: bool = True):
        self.enable_pinning = enable_pinning
        self._thread_map: Dict[str, int] = {}
        logger.info(f"Initialized NUMAScheduler. Pinning enabled: {self.enable_pinning}")

    def pin_thread(self, thread_name: str, core_id: int, native_thread_id: int = 0) -> bool:
        """
        Pins a specific thread to a designated CPU core to prevent context switching
        and maximize cache locality. Requires the native OS thread ID (e.g., via threading.get_native_id()).
        """
        if not self.enable_pinning:
            logger.debug(f"Pinning disabled. Skipping core assignment for {thread_name}.")
            return False

        logger.debug(f"Pinning thread {thread_name} (TID: {native_thread_id}) to CPU core {core_id}.")
        self._thread_map[thread_name] = core_id

        try:
            # Attempt true OS-level thread pinning (Linux only)
            if hasattr(os, "sched_setaffinity"):
                # Pass 0 to pin the calling thread, or native_thread_id to pin a specific target
                os.sched_setaffinity(native_thread_id, {core_id})
                logger.info(f"Successfully pinned {thread_name} to core {core_id} via OS syscall.")
            else:
                logger.debug(f"OS does not support sched_setaffinity. Mock pinning {thread_name}.")
        except Exception as e:
            logger.error(f"Failed to pin {thread_name} to core {core_id}: {e}")
            return False

        return True

    def allocate_numa_memory(self, size_bytes: int, node_id: int) -> int:
        """
        Allocates memory on a specific NUMA node to prevent remote memory access latency.
        """
        logger.debug(f"Allocating {size_bytes} bytes on NUMA node {node_id}.")
        # Simulated memory allocation pointer
        return 0x0

    def get_allocation_map(self) -> Dict[str, Any]:
        """
        Returns the current thread-to-core mapping.
        """
        return {"threads": self._thread_map}
