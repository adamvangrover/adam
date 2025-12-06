import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import torch

from src.adam.api.main import app, state_manager
from src.adam.api.auth import get_current_user

# Override dependency
async def override_get_current_user():
    return {"sub": "test_user"}

app.dependency_overrides[get_current_user] = override_get_current_user

client = TestClient(app)

def test_optimization_flow_adamw():
    """
    Test that optimization works and state is saved/loaded.
    """
    # 1. First Step (No state)
    with patch.object(state_manager, 'load_state', return_value=None) as mock_load:
        with patch.object(state_manager, 'save_state') as mock_save:
            payload = {
                "session_id": "sess_A",
                "config": {"algorithm": "adamw", "learning_rate": 0.1},
                "parameters": [1.0, 1.0],
                "gradients": [0.1, 0.1]
            }

            response = client.post("/optimize", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == "success"

            # Verify Save
            mock_save.assert_called_once()
            call_args = mock_save.call_args
            key = call_args[0][0]
            state = call_args[0][1]

            assert key == "opt_state:sess_A"
            # Verify internal structure of state dict
            # PyTorch optimizer state_dict has 'state' and 'param_groups'
            assert 'state' in state
            assert 'param_groups' in state

            # Save this state for next step
            saved_state_step_1 = state

    # 2. Second Step (With State)
    with patch.object(state_manager, 'load_state', return_value=saved_state_step_1) as mock_load:
        with patch.object(state_manager, 'save_state') as mock_save:
            payload = {
                "session_id": "sess_A",
                "config": {"algorithm": "adamw", "learning_rate": 0.1},
                "parameters": [0.99, 0.99], # Parameters updated externally or by prev step
                "gradients": [0.1, 0.1]
            }

            response = client.post("/optimize", json=payload)
            assert response.status_code == 200

            mock_load.assert_called_once_with("opt_state:sess_A")

            # Verify Step Count Increased
            new_state = mock_save.call_args[0][1]

            # Extract step from first param state
            # state keys are usually param IDs (int).
            param_state_values = list(new_state['state'].values())
            assert len(param_state_values) > 0
            assert param_state_values[0]['step'] == 2

def test_adam_mini_support():
    """
    Test that Adam-mini endpoint accepts request and runs (even if mocked/approx).
    """
    with patch.object(state_manager, 'save_state'):
         payload = {
                "session_id": "sess_B",
                "config": {"algorithm": "adam-mini", "learning_rate": 0.01},
                "parameters": [1.0] * 128, # Block size 128
                "gradients": [0.1] * 128
            }
         response = client.post("/optimize", json=payload)
         assert response.status_code == 200
