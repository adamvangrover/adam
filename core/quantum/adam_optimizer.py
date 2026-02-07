import numpy as np

class AdamOptimizer:
    """
    A NumPy-based implementation of the Adam optimization algorithm.
    Designed for optimizing quantum control parameters.

    References:
    Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
    """

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def initialize(self, params):
        """Initializes momentum and velocity terms for the parameters."""
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)
        self.t = 0

    def step(self, params, grads):
        """
        Performs a single Adam optimization step.

        Args:
            params (np.ndarray): Current parameters.
            grads (np.ndarray): Gradients with respect to parameters.

        Returns:
            np.ndarray: Updated parameters.
        """
        if self.m is None:
            self.initialize(params)

        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update parameters
        params_updated = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params_updated
