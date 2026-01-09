
import os
import logging

class CapabilityRegistry:
    """
    Central registry for system capabilities.
    Allows for graceful degradation if dependencies (like Torch, GPU, API keys) are missing.
    """

    def __init__(self):
        self.capabilities = {}
        self._logger = logging.getLogger("core.capabilities")
        self._detect_capabilities()

    def _detect_capabilities(self):
        """Auto-detects available features."""

        # Check for PyTorch
        try:
            import torch
            self.capabilities["torch"] = True
            self.capabilities["cuda"] = torch.cuda.is_available()
        except ImportError:
            self.capabilities["torch"] = False
            self.capabilities["cuda"] = False
            self._logger.warning("PyTorch not found. Deep learning features disabled.")

        # Check for OpenAI API
        if os.getenv("OPENAI_API_KEY"):
            self.capabilities["openai"] = True
        else:
            self.capabilities["openai"] = False
            self._logger.warning("OPENAI_API_KEY not found. LLM features will be mocked.")

        # Check for Crypto/Web3
        try:
            # Check if we have a node connection string
            if os.getenv("WEB3_PROVIDER_URI"):
                self.capabilities["web3"] = True
            else:
                self.capabilities["web3"] = False
        except:
             self.capabilities["web3"] = False

    def has_capability(self, capability: str) -> bool:
        return self.capabilities.get(capability, False)

    def require(self, capability: str):
        """Raises error if capability missing."""
        if not self.has_capability(capability):
            raise RuntimeError(f"Required capability '{capability}' is missing.")

# Singleton
caps = CapabilityRegistry()
