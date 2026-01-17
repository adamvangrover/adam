import sys
import types
from unittest.mock import MagicMock
import pytest

# Global mocks for heavy/missing dependencies
# These must be mocked BEFORE any test imports the modules that use them.
# Placing them here ensures they are applied during pytest collection.


def create_mock_module(name):
    m = types.ModuleType(name)
    m.__file__ = f"mock_{name}.py"
    # Attach a MagicMock for attribute access if code tries to use the module
    # But we can't easily proxy all attrs.
    # Instead, we set specific attrs if needed.
    # For now, let's keep it minimal to satisfy import inspection.
    return m

# Helper to attach recursive mock capabilities if needed,
# but ModuleType doesn't support __getattr__ dynamically unless subclassed.
# We'll stick to ModuleType and manually add mocks if tests fail on attr access.
# Or use a MagicMock that behaves better.
# Let's try MagicMock again but ensure __path__ is also a list of strings if accessed.


def create_magic_mock_module(name):
    mock = MagicMock()
    mock.__spec__ = MagicMock()
    mock.__file__ = f"mock_{name}.py"
    mock.__path__ = [f"mock_{name}_path"]  # List of strings, not Mock
    return mock


if "spacy" not in sys.modules:
    sys.modules["spacy"] = create_magic_mock_module("spacy")

if "transformers" not in sys.modules:
    sys.modules["transformers"] = create_magic_mock_module("transformers")
    sys.modules["transformers.pipeline"] = create_magic_mock_module("transformers.pipeline")

if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = create_magic_mock_module("tensorflow")

# Mock facebook_scraper if missing (used in social_media_api)
if "facebook_scraper" not in sys.modules:
    sys.modules["facebook_scraper"] = create_magic_mock_module("facebook_scraper")

# Prevent torch._dynamo from scanning modules and crashing on mocks
if "torch._dynamo" not in sys.modules:
    mock_dynamo = create_magic_mock_module("torch._dynamo")
    # Make disable() a pass-through decorator
    mock_dynamo.disable = MagicMock(side_effect=lambda fn=None, recursive=True, **kwargs: (lambda x: x) if fn is None else fn)
    sys.modules["torch._dynamo"] = mock_dynamo

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
