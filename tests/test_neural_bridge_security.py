
import pytest
import sys
import os
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

def test_mesh_websocket_cors_allowed(mesh_client):
    """
    Test that Neural Mesh accepts localhost.
    """
    with mesh_client.websocket_connect("/ws/mesh", headers={"Origin": "http://localhost:3000"}) as websocket:
        assert websocket.scope['type'] == 'websocket'

def test_mesh_websocket_cors_blocked(mesh_client):
    """
    Test that Neural Mesh rejects arbitrary origins.
    """
    try:
        with mesh_client.websocket_connect("/ws/mesh", headers={"Origin": "http://evil.com"}) as websocket:
            # If we connect successfully, fail the test
            pytest.fail("Should have rejected connection from evil.com")
    except Exception as e:
        # Expected failure (connection rejected)
        pass

def test_link_websocket_cors_allowed(link_client):
    """
    Test that Neural Link accepts localhost.
    """
    with link_client.websocket_connect("/ws/stream", headers={"Origin": "http://localhost:3000"}) as websocket:
        assert websocket.scope['type'] == 'websocket'

def test_link_websocket_cors_blocked(link_client):
    """
    Test that Neural Link rejects arbitrary origins.
    """
    try:
        with link_client.websocket_connect("/ws/stream", headers={"Origin": "http://evil.com"}) as websocket:
            pytest.fail("Should have rejected connection from evil.com")
    except Exception as e:
        pass
