import pytest
import torch
from src.adam.core.optimizers import AdamW, Lion

@pytest.mark.parametrize("optimizer_class", [AdamW, Lion])
def test_optimizer_basic_step(optimizer_class):
    """
    Test that the optimizer actually updates parameters on a simple convex loss.
    Loss = x^2 + y^2. Minimum at (0,0).
    Starting at (1, 2). Should move towards 0.
    """
    params = torch.tensor([1.0, 2.0], requires_grad=True)
    # Use a relatively high LR to ensure movement
    opt = optimizer_class([params], lr=0.1)

    # Zero grad
    opt.zero_grad()

    # Compute loss and gradients
    loss = (params ** 2).sum()
    loss.backward()

    # Verify gradients exist
    assert params.grad is not None

    # Step
    opt.step()

    # Check if params moved towards 0
    assert params[0] < 1.0
    assert params[1] < 2.0

    # Check if they didn't explode
    assert params[0] > -2.0
    assert params[1] > -2.0

def test_adamw_weight_decay():
    """
    Test decoupled weight decay mechanism.
    Update rule: theta_t = theta_{t-1} - lr * (grad + weight_decay * theta_{t-1})
    If we decouple: theta_t = theta_{t-1} * (1 - lr * weight_decay) - lr * grad
    """
    params = torch.tensor([10.0], requires_grad=True)
    # Set gradient to 0 to isolate weight decay
    params.grad = torch.zeros_like(params)

    lr = 0.1
    wd = 0.1
    opt = AdamW([params], lr=lr, weight_decay=wd)

    opt.step()

    # Expected: 10 * (1 - 0.1 * 0.1) = 10 * 0.99 = 9.9
    expected = 10.0 * (1.0 - lr * wd)
    assert torch.isclose(params, torch.tensor([expected]), atol=1e-5)

def test_lion_sign_update():
    """
    Test that Lion uses the sign of the update.
    """
    params = torch.tensor([1.0, -1.0], requires_grad=True)
    opt = Lion([params], lr=0.1, betas=(0.9, 0.99))

    # Gradients: [positive, negative]
    params.grad = torch.tensor([0.5, -0.5])

    opt.step()

    # Lion update is roughly -lr * sign(update).
    # Since exp_avg starts at 0, update = (1-beta1)*grad.
    # sign(positive) = 1, sign(negative) = -1.
    # params[0] should decrease by lr (approx)
    # params[1] should increase by lr (approx)

    assert params[0] < 1.0
    assert params[1] > -1.0
