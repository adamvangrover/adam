import torch
import torch.nn as nn
import torch.optim as optim

class GNNExplainer:
    """
    GNNExplainer implementation for explaining GNN predictions by identifying
    critical subgraphs and node features. 

    Learns masks over the adjacency matrix and features to maximize mutual information
    between the original prediction and the masked prediction.
    """
    def __init__(self, model, epochs=100, lr=0.01):
        self.model = model
        self.epochs = epochs  # Adopted 100 from feature branch for better convergence
        self.lr = lr

    def explain_node(self, node_idx, x, adj):
        """
        Explain a single node's prediction.

        Args:
            node_idx (int): Index of node to explain.
            x (torch.Tensor): Feature matrix (N, F).
            adj (torch.Tensor): Adjacency matrix (N, N). Can be sparse.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (edge_mask, feature_mask)
        """
        self.model.eval()

        # Ensure adj is dense for masking operations (Main branch logic for full edge support)
        if adj.is_sparse:
            adj = adj.to_dense()

        # Initialize trainable masks
        # We initialize with zeros (sigmoid(0) = 0.5) to allow gradients to flow either way
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
            # We treat original adj/x as "hard" constraints, and learn a "soft" mask on top
            sigmoid_adj = torch.sigmoid(mask_adj)
            sigmoid_feat = torch.sigmoid(mask_feat)

            masked_adj = adj * sigmoid_adj
            masked_x = x * sigmoid_feat

            out = self.model(masked_x, masked_adj)
            pred = out[node_idx]

            # Loss Function:
            # 1. Prediction Fidelity: Minimize difference from original prediction
            # 2. Sparsity Regularization: Minimize size of mask (L1 norm)
            loss_dist = torch.abs(pred - orig_pred) 
            loss_reg = 1e-4 * (torch.sum(sigmoid_adj) + torch.sum(sigmoid_feat))
            
            loss = loss_dist + loss_reg

            loss.backward()
            optimizer.step()

        # Return masked importance as probabilities (0-1)
        # Compatible with GraphRiskEngine's explain_risk method
        return torch.sigmoid(mask_adj).detach(), torch.sigmoid(mask_feat).detach()