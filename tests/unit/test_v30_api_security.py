import pytest
import sys
import os
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# Mock dependencies before importing core.api.main
sys.modules['core.engine.meta_orchestrator'] = MagicMock()
sys.modules['core.engine.neuro_symbolic_planner'] = MagicMock()
sys.modules['langgraph'] = MagicMock()
sys.modules['langgraph.graph'] = MagicMock()

# Ensure MetaOrchestrator class exists on the mock
sys.modules['core.engine.meta_orchestrator'].MetaOrchestrator = MagicMock

# Import app after mocks
from core.api.main import app
from core.settings import settings

@pytest.fixture
def client():
    return TestClient(app)

def test_v30_api_unauthorized_access(client):
    """
    Test that requests without a valid API key are rejected with 401.
    """
    payload = {
        "query": "What is the meaning of life?",
        "context": {}
    }

    # Case 1: No API Key
    response = client.post("/api/v1/agents/analyze", json=payload)
    assert response.status_code == 401
    assert response.json() == {"detail": "Missing API Key"}

    # Case 2: Invalid API Key
    headers = {"X-API-Key": "wrong-key"}
    response = client.post("/api/v1/agents/analyze", json=payload, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API Key"}

def test_v30_api_authorized_access(client):
    """
    Test that requests with a valid API key are accepted (not 401).
    Note: We expect 500 because dependencies are mocked.
    """
    payload = {
        "query": "What is the meaning of life?",
        "context": {}
    }

    # Use the configured API key
    api_key = settings.adam_api_key
    headers = {"X-API-Key": api_key}

    response = client.post("/api/v1/agents/analyze", json=payload, headers=headers)

    # We expect 500 (Internal Server Error) due to mocks, but definitely NOT 401
    assert response.status_code != 401
    assert response.status_code == 500
