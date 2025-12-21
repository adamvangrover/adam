import math

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """
    Implements AdamW algorithm with decoupled weight decay.

    References:
    - Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class Lion(Optimizer):
    """
    Implements Lion algorithm (Evolved Sign Momentum).

    References:
    - Symbolic Discovery of Optimization Algorithms: https://arxiv.org/abs/2302.06675
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight decay
                if group['weight_decay'] > 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Parameter update
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign()
                p.add_(update, alpha=-group['lr'])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

class AdamMini(Optimizer):
    """
    Implements Adam-mini (Memory Efficient).

    Features:
    - Block-wise quantization (simulated via block averaging).
    - Reduces second moment memory footprint by factor of 'block_size'.

    References:
    - Adam-mini: Use 1/block_size fewer stats for v.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, block_size=128):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, block_size=block_size)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            block_size = group['block_size']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamMini does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # For Adam-mini, second moment is reduced.
                    numel = p.numel()
                    num_blocks = (numel + block_size - 1) // block_size
                    # We store one value per block
                    state['exp_avg_sq'] = torch.zeros(num_blocks, device=p.device, dtype=p.dtype)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Weight decay
                if group['weight_decay'] != 0:
                     p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # 1. Update Momentum (m) - Standard
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 2. Update Second Moment (v) - Block-wise
                flat_grad = grad.view(-1)
                numel = flat_grad.numel()
                pad_len = (block_size - (numel % block_size)) % block_size
                if pad_len > 0 and pad_len != block_size:
                     padded_grad = F.pad(flat_grad, (0, pad_len))
                else:
                    padded_grad = flat_grad

                # Fix pad logic: if numel % block_size == 0, pad_len is block_size using formula above
                # Correct logic:
                if numel % block_size != 0:
                     pad_len = block_size - (numel % block_size)
                     padded_grad = F.pad(flat_grad, (0, pad_len))
                else:
                     pad_len = 0
                     padded_grad = flat_grad

                reshaped_grad = padded_grad.view(-1, block_size)

                # Calculate mean of squared gradients per block
                g_sq_mean = reshaped_grad.pow(2).mean(dim=1)

                # Update running average of block means
                exp_avg_sq.mul_(beta2).add_(g_sq_mean, alpha=1 - beta2)

                # 3. Calculate Update
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Block-wise step size
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # Expand denom back to parameter size
                denom_expanded = denom.repeat_interleave(block_size)

                # Remove padding
                if pad_len > 0:
                    denom_expanded = denom_expanded[:numel]

                denom_final = denom_expanded.view_as(p)

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg, denom_final, value=-step_size)

        return loss
