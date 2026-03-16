import logging
import random
from typing import Any, List

logger = logging.getLogger(__name__)

# ==============================================================================
# PyTorch Mocks
# ==============================================================================

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not found. Using NSSF MockTorch implementation.")

    class MockTensor:
        def __init__(self, data=None, shape=None):
            self.data = data if data is not None else []
            if shape is not None:
                self.shape = shape
            else:
                if isinstance(self.data, list):
                    self.shape = (len(self.data),)
                else:
                    self.shape = (0,)

        def __repr__(self):
            return f"MockTensor(shape={self.shape})"

        def view(self, *shape):
            # Update shape for view operations
            if len(shape) == 1 and isinstance(shape[0], tuple):
                 self.shape = shape[0]
            elif len(shape) > 0:
                 self.shape = shape
            return self

        def __add__(self, other):
            return MockTensor(shape=self.shape)

        def __mul__(self, other):
            return MockTensor(shape=self.shape)

        def __rmul__(self, other):
            return MockTensor(shape=self.shape)

        def __truediv__(self, other):
             return MockTensor(shape=self.shape)

        def __rtruediv__(self, other):
             return MockTensor(shape=self.shape)

        def __sub__(self, other):
             return MockTensor(shape=self.shape)

        def __neg__(self):
             return MockTensor(shape=self.shape)

        def __pow__(self, power):
             return MockTensor(shape=self.shape)

        def __getitem__(self, idx):
            if isinstance(self.data, list) and isinstance(idx, int) and 0 <= idx < len(self.data):
                return self.data[idx]
            # Return a generic mock tensor or 0.0 if likely a scalar access
            # But t[1] is expected to be a number (time).
            # If data is populated, return it.
            return 0.1 # Fallback for time steps in mock

    class MockModule:
        def __init__(self):
            self._parameters = {}
            self.training = True

        def forward(self, *args, **kwargs):
            raise NotImplementedError

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class MockParameter(MockTensor):
        def __init__(self, data=None, requires_grad=True):
            if isinstance(data, MockTensor):
                super().__init__(data=data.data, shape=data.shape)
            else:
                super().__init__(data=data)

    class MockNN:
        Module = MockModule
        Parameter = MockParameter

    class MockTorchLib:
        nn = MockNN

        def randn(self, *size):
            return MockTensor(shape=size)

        def randn_like(self, tensor):
            return MockTensor(shape=tensor.shape)

        def matmul(self, a, b):
            # Naive shape inference
            # If a=(M, K) and b=(K, N), result=(M, N)
            shape = (1, 1)
            if hasattr(a, 'shape') and hasattr(b, 'shape'):
                if len(a.shape) == 2 and len(b.shape) == 2:
                    shape = (a.shape[0], b.shape[1])
            return MockTensor(shape=shape)

        def cos(self, tensor):
            return MockTensor(shape=tensor.shape)

        def abs(self, tensor):
            return MockTensor(shape=tensor.shape)

        def tensor(self, data):
             return MockTensor(data=data)

        def view(self, *shape):
            return self

    torch = MockTorchLib()
    nn = torch.nn

# ==============================================================================
# Qiskit Mocks (Re-export or additional)
# ==============================================================================
# Note: core.risk_engine.quantum_model already mocks Qiskit components.
# We might need additional ones here if NSSF uses specific circuits.

import sys
from types import ModuleType

# Mock 'mcp' if it is missing, as it is required by quantum_model.py
# We check if we can import the specific required module, otherwise we mock the whole chain
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    logger.warning("Mocking MCP for NSSF")
    if "mcp" not in sys.modules:
        mcp = ModuleType("mcp")
        sys.modules["mcp"] = mcp
    else:
        mcp = sys.modules["mcp"]

    if "mcp.server" not in sys.modules:
        mcp_server = ModuleType("mcp.server")
        sys.modules["mcp.server"] = mcp_server
        mcp.server = mcp_server
    else:
        mcp_server = sys.modules["mcp.server"]

    # Force fastmcp existence
    mcp_server_fastmcp = ModuleType("mcp.server.fastmcp")
    sys.modules["mcp.server.fastmcp"] = mcp_server_fastmcp
    mcp_server.fastmcp = mcp_server_fastmcp

    class MockFastMCP:
        def __init__(self, *args, **kwargs):
            pass

    mcp_server_fastmcp.FastMCP = MockFastMCP
