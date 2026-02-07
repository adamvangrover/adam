import pytest
import numpy as np
import torch
from core.oswm.trainer import OSWMTrainer
from core.oswm.inference import OSWMInference

def test_oswm_integration():
    """
    Verifies that OSWM Trainer and Inference can be initialized and run one step.
    """
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Init Trainer
    try:
        trainer = OSWMTrainer(device='cpu')
    except Exception as e:
        pytest.fail(f"Failed to init trainer: {e}")

    # 2. Train Step
    try:
        loss, eval_pos = trainer.train_step()
        assert not np.isnan(loss), "Loss is NaN"
        assert isinstance(eval_pos, int)
    except Exception as e:
        pytest.fail(f"Failed train step: {e}")

    # 3. Init Inference
    try:
        inference = OSWMInference(trainer.model, device='cpu')
    except Exception as e:
        pytest.fail(f"Failed to init inference: {e}")

    # 4. Set Context
    try:
        transitions = []
        for i in range(100):
            s = np.random.randn(7)
            a = np.random.randn(2)
            ns = np.random.randn(7)
            r = np.random.randn(1)
            transitions.append((s, a, ns, r))

        inference.set_context(transitions)
    except Exception as e:
        pytest.fail(f"Failed set context: {e}")

    # 5. Inference Step
    try:
        curr_s = np.random.randn(7)
        curr_a = np.random.randn(2)

        next_s, next_r = inference.step(curr_s, curr_a)

        assert next_s.shape == (7,)
        assert next_r.shape == ()
        assert not np.isnan(next_s).any()
        assert not np.isnan(next_r)

    except Exception as e:
        pytest.fail(f"Failed inference step: {e}")
