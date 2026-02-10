
import pytest
import sys
import os
from unittest.mock import patch
from fastapi.testclient import TestClient

# Ensure core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import apps
from core.v30_architecture.python_intelligence.bridge.neural_mesh import app as mesh_app
from core.v30_architecture.python_intelligence.bridge.neural_link import app as link_app

@pytest.fixture
def mesh_client():
    return TestClient(mesh_app)

@pytest.fixture
def link_client():
    return TestClient(link_app)

def test_mesh_websocket_cors_wildcard_handling(mesh_client):
    """
    Test that Neural Mesh handles wildcard '*' in allowed_origins correctly.
    Should allow connection from any origin.
    """
    with patch("core.v30_architecture.python_intelligence.bridge.neural_mesh.allowed_origins", ["*"]):
        try:
            with mesh_client.websocket_connect("/ws/mesh", headers={"Origin": "http://random-origin.com"}) as websocket:
                assert websocket.scope['type'] == 'websocket'
        except Exception as e:
            pytest.fail(f"Connection blocked unexpectedly despite wildcard '*': {e}")

def test_mesh_websocket_no_origin(mesh_client):
    """
    Test that Neural Mesh allows connection without Origin header (e.g. server-to-server).
    """
    # allowed_origins doesn't matter much if origin is None, but let's set it to strict
    with patch("core.v30_architecture.python_intelligence.bridge.neural_mesh.allowed_origins", ["http://localhost:3000"]):
        try:
            # No Origin header
            with mesh_client.websocket_connect("/ws/mesh") as websocket:
                assert websocket.scope['type'] == 'websocket'
        except Exception as e:
            pytest.fail(f"Connection blocked unexpectedly when no Origin header provided: {e}")

def test_link_websocket_cors_wildcard_handling(link_client):
    """
    Test that Neural Link handles wildcard '*' in allowed_origins correctly.
    """
    with patch("core.v30_architecture.python_intelligence.bridge.neural_link.allowed_origins", ["*"]):
        try:
            with link_client.websocket_connect("/ws/stream", headers={"Origin": "http://random-origin.com"}) as websocket:
                assert websocket.scope['type'] == 'websocket'
        except Exception as e:
            pytest.fail(f"Connection blocked unexpectedly despite wildcard '*': {e}")

def test_link_websocket_no_origin(link_client):
    """
    Test that Neural Link allows connection without Origin header.
    """
    with patch("core.v30_architecture.python_intelligence.bridge.neural_link.allowed_origins", ["http://localhost:3000"]):
        try:
            with link_client.websocket_connect("/ws/stream") as websocket:
                assert websocket.scope['type'] == 'websocket'
        except Exception as e:
            pytest.fail(f"Connection blocked unexpectedly when no Origin header provided: {e}")
