import torch
import torch.nn as nn
import torch.optim as optim

class GNNExplainer:
    """
    A simple implementation of GNNExplainer.
    It learns a mask over the adjacency matrix and features to maximize mutual information
    between the original prediction and the masked prediction.
    """
    def __init__(self, model, epochs=100, lr=0.01):
        self.model = model
        self.epochs = epochs
        self.lr = lr

    def explain_node(self, node_idx, features, adj):
        """
        Explains the prediction for a specific node.

        Args:
            node_idx (int): The index of the node to explain.
            features (torch.Tensor): Full feature matrix.
            adj (torch.Tensor): Full adjacency matrix (sparse or dense).

        Returns:
            Dict: Importance scores for features and edges.
        """
        self.model.eval()

        # Get original prediction
        with torch.no_grad():
            orig_pred = self.model(features, adj)[node_idx]

        # Initialize trainable masks
        # Feature mask: [num_features]
        feature_mask = nn.Parameter(torch.rand(features.shape[1], requires_grad=True))

        # Edge mask: For simplicity in this demo, we won't learn a per-edge mask
        # because dealing with sparse tensor gradients manually is complex without PyG.
        # We will focus on Feature Importance.

        optimizer = optim.Adam([feature_mask], lr=self.lr)
        loss_fn = nn.MSELoss()

        for _ in range(self.epochs):
            optimizer.zero_grad()

            # Apply mask (sigmoid to keep it 0-1)
            masked_features = features * torch.sigmoid(feature_mask)

            # Forward pass
            pred = self.model(masked_features, adj)[node_idx]

            # Loss: Minimize difference from original prediction + Regularization (sparsity)
            # We want the prediction to stay the same using FEWER features.
            loss = loss_fn(pred, orig_pred) + 0.01 * torch.norm(feature_mask, 1)

            loss.backward()
            optimizer.step()

        # Extract importance
        feature_importance = torch.sigmoid(feature_mask).detach().numpy()

        return {
            "feature_importance": feature_importance.tolist(),
            "original_score": orig_pred.item()
        }
