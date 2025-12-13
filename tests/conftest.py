import sys
from unittest.mock import MagicMock
import pytest

# Global mocks for heavy/missing dependencies
# These must be mocked BEFORE any test imports the modules that use them.
# Placing them here ensures they are applied during pytest collection.

if "spacy" not in sys.modules:
    sys.modules["spacy"] = MagicMock()

if "transformers" not in sys.modules:
    sys.modules["transformers"] = MagicMock()
    sys.modules["transformers.pipeline"] = MagicMock()

if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = MagicMock()

# Mock facebook_scraper if missing (used in social_media_api)
if "facebook_scraper" not in sys.modules:
    sys.modules["facebook_scraper"] = MagicMock()

# We can also mock core.llm_plugin to avoid API key checks during collection/init
# But maybe we should let tests mock it explicitly if they need to test logic around it.
# However, AgentOrchestrator imports it and initializes it.
# Let's mock the class inside the module if possible, or leave it.
# tests/test_agent_orchestrator.py patches it.

@pytest.fixture(autouse=True)
def mock_settings_env(monkeypatch):
    """Set dummy env vars for all tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")
    monkeypatch.setenv("OPENAI_ORG_ID", "dummy_org")
    monkeypatch.setenv("COHERE_API_KEY", "dummy_key")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "dummy_key")
