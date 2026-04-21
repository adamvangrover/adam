import os
import sys

try:
    from core.llm_plugin import MockLLM
    print("MockLLM exists")
except ImportError:
    print("MockLLM not found in core.llm_plugin")
