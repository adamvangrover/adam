"""
core/experimental/adamos_bridge.py

Project OMEGA: Pillar 3 - AdamOS Integration.
Implements the Python <-> Rust Bridge (FFI) using ctypes.
Serves as the gateway for the MetaOrchestrator to offload tasks to the Rust Kernel.

Design:
1. Attempt to load `libadamos_kernel.so` (or .dll/.dylib).
2. If found, expose Rust functions via `ctypes`.
3. If not found, fallback to a Python Mock Kernel (Graceful Degradation).
"""

import ctypes
import os
import sys
import platform
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# --- Paths ---
# Adjust based on where `cargo build --release` outputs the binary
KERNEL_PATH = os.path.join(os.path.dirname(__file__), "adamos_kernel", "target", "release")

def get_lib_name():
    system = platform.system()
    if system == "Linux":
        return "libadamos_kernel.so"
    elif system == "Darwin":
        return "libadamos_kernel.dylib"
    elif system == "Windows":
        return "adamos_kernel.dll"
    return "libadamos_kernel.so"

class AdamOSKernel:
    """
    Python Wrapper for the Rust Kernel.
    """
    def __init__(self):
        self._lib: Optional[ctypes.CDLL] = None
        self._mock_mode = False
        self._load_kernel()

    def _load_kernel(self):
        lib_file = get_lib_name()
        lib_path = os.path.join(KERNEL_PATH, lib_file)

        try:
            if os.path.exists(lib_path):
                self._lib = ctypes.CDLL(lib_path)
                logger.info(f"AdamOS Kernel loaded successfully from {lib_path}")
                self._setup_signatures()
            else:
                raise FileNotFoundError(f"Kernel binary not found at {lib_path}")
        except Exception as e:
            logger.warning(f"Failed to load AdamOS Kernel: {e}. Falling back to Python Mock.")
            self._mock_mode = True

    def _setup_signatures(self):
        """Define C-types for Rust functions."""
        # Example: fn register_agent(name: *const c_char) -> *const c_char
        self._lib.register_agent_ffi.argtypes = [ctypes.c_char_p]
        self._lib.register_agent_ffi.restype = ctypes.c_char_p

        self._lib.process_message_ffi.argtypes = [ctypes.c_char_p]
        self._lib.process_message_ffi.restype = ctypes.c_char_p

    def register_agent(self, name: str) -> str:
        """Register a new agent with the Kernel."""
        if self._mock_mode:
            return f"mock_agent_{name}_{os.getpid()}"

        # FFI Call
        name_bytes = name.encode('utf-8')
        ptr = self._lib.register_agent_ffi(name_bytes)
        result = ctypes.cast(ptr, ctypes.c_char_p).value.decode('utf-8')
        # Rust side should free the string, but for prototype we skip memory management complexity
        return result

    def send_message(self, sender: str, recipient: str, content: str) -> str:
        """Route a message through the Kernel."""
        if self._mock_mode:
            logger.info(f"[MOCK KERNEL] Routing: {sender} -> {recipient}: {content[:20]}...")
            return "delivered_mock"

        # In a real FFI, we'd pass a struct or JSON string
        payload = f"{sender}|{recipient}|{content}".encode('utf-8')
        ptr = self._lib.process_message_ffi(payload)
        return ctypes.cast(ptr, ctypes.c_char_p).value.decode('utf-8')

# --- Testing ---
if __name__ == "__main__":
    kernel = AdamOSKernel()
    agent_id = kernel.register_agent("RiskManager")
    print(f"Registered Agent ID: {agent_id}")

    status = kernel.send_message("MetaOrchestrator", agent_id, "Analyze VIX")
    print(f"Message Status: {status}")
