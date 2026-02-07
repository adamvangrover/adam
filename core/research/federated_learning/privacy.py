import torch
import torch.nn.functional as F

class PrivacyEngine:
    """
    Handles Differential Privacy mechanisms like Laplacian Noise.
    """
    def __init__(self, epsilon=1.0, sensitivity=1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        # Scale = sensitivity / epsilon
        self.scale = sensitivity / (epsilon + 1e-10)

    def add_noise(self, state_dict):
        """
        Adds Laplacian noise to the model state dictionary (gradients/weights).
        """
        noisy_state = {}
        for k, v in state_dict.items():
            noise = torch.distributions.laplace.Laplace(0, self.scale).sample(v.shape)
            noisy_state[k] = v + noise
        return noisy_state

class MSGuard:
    """
    Defense against model poisoning attacks (e.g., ScaleSign).
    """
    @staticmethod
    def filter_updates(updates):
        """
        Filters out potentially malicious updates based on Cosine Similarity to the consensus.
        Returns the list of valid updates.
        """
        if not updates:
            return []

        # 1. Flatten all updates
        flat_updates = []
        for update in updates:
            # Flatten only float tensors
            flat = torch.cat([p.flatten().float() for p in update.values()])
            flat_updates.append(flat)

        stack = torch.stack(flat_updates)

        # 2. Compute mean update (Consensus)
        mean_update = torch.mean(stack, dim=0)

        # 3. Compute cosine similarity of each update to mean
        # eps=1e-8 to avoid div by zero
        sims = F.cosine_similarity(stack, mean_update.unsqueeze(0), dim=1)

        # 4. Filter
        valid_updates = []
        for i, sim in enumerate(sims):
            # If similarity is positive, it aligns with consensus.
            # ScaleSign attacks often try to hide, but we'll use a basic positive alignment check.
            if sim > 0.0:
                valid_updates.append(updates[i])

        # If we filtered everything, fallback to all (to avoid halting)
        if not valid_updates:
            return updates

        return valid_updates
