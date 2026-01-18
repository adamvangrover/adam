import torch
import torch.nn as nn
import torch.optim as optim

class GNNExplainer:
    """
    GNNExplainer implementation for explaining GNN predictions by identifying
    critical subgraphs and node features.
    """
    def __init__(self, model, epochs=50, lr=0.01):
        self.model = model
        self.epochs = epochs
        self.lr = lr

    def explain_node(self, node_idx, x, adj):
        """
        Explain a single node's prediction.
        x: (N, F) features
        adj: (N, N) adjacency (can be sparse or dense)
        """
        self.model.eval()

        # Ensure adj is dense for masking operations
        if adj.is_sparse:
            adj = adj.to_dense()

        # Trainable masks
        # We initialize with values that sigmoid to ~0.5
        mask_adj = torch.zeros_like(adj, requires_grad=True)
        mask_feat = torch.zeros_like(x, requires_grad=True)

        optimizer = optim.Adam([mask_adj, mask_feat], lr=self.lr)

        # Get original prediction target
        with torch.no_grad():
            orig_out = self.model(x, adj)
            orig_pred = orig_out[node_idx]

        # Optimization loop
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Apply masks (sigmoid to keep in 0-1)
            # We treat original adj as a "hard" mask, and learn a "soft" mask on top
            sigmoid_adj = torch.sigmoid(mask_adj)
            sigmoid_feat = torch.sigmoid(mask_feat)

            masked_adj = adj * sigmoid_adj
            masked_x = x * sigmoid_feat

            out = self.model(masked_x, masked_adj)
            pred = out[node_idx]

            # Loss: Minimize difference from original prediction
            # Regularization: Minimize size of mask (sparsity)
            # loss = dist(pred, orig_pred) + lambda * (|mask_adj| + |mask_feat|)
            loss = torch.abs(pred - orig_pred) + 1e-4 * (torch.sum(sigmoid_adj) + torch.sum(sigmoid_feat))

            loss.backward()
            optimizer.step()

        # Return masked importance
        return torch.sigmoid(mask_adj).detach(), torch.sigmoid(mask_feat).detach()
