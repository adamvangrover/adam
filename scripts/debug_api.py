import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from src.adam.api.main import app, state_manager
from src.adam.api.auth import get_current_user

# Override dependency
async def override_get_current_user():
    return {"sub": "test_user"}

app.dependency_overrides[get_current_user] = override_get_current_user

client = TestClient(app)

def reproduction():
    with patch.object(state_manager, 'load_state', return_value=None) as mock_load:
        with patch.object(state_manager, 'save_state') as mock_save:
            payload = {
                "session_id": "sess_A",
                "config": {"algorithm": "adamw", "learning_rate": 0.1},
                "parameters": [1.0, 1.0],
                "gradients": [0.1, 0.1]
            }

            response = client.post("/optimize", json=payload)
            print(f"Status Code: {response.status_code}")
            print(f"Response Body: {response.text}")

if __name__ == "__main__":
    reproduction()
