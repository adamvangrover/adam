import torch
import networkx as nx
import numpy as np
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph
from .model import GCN, GAT, GraphSAGE
from .explainer import GNNExplainer

class GraphRiskEngine:
    def __init__(self, model_type="GCN"):
        self.ukg = UnifiedKnowledgeGraph()
        self.graph = self.ukg.graph
        self.node_list = list(self.graph.nodes())
        self.node_map = {node: i for i, node in enumerate(self.node_list)}
        self.num_nodes = len(self.node_list)

        self.adj = self._build_adjacency_matrix()
        self.features = self._build_feature_matrix()

        self.model_type = model_type
        input_dim = self.features.shape[1]

        # Initialize GNN based on type
        if model_type == "GAT":
            self.model = GAT(nfeat=input_dim, nhid=16, nclass=1)
        elif model_type == "GraphSAGE":
            self.model = GraphSAGE(nfeat=input_dim, nhid=16, nclass=1)
        else:
            self.model = GCN(nfeat=input_dim, nhid=16, nclass=1)

        self.explainer = GNNExplainer(self.model)

    def _build_adjacency_matrix(self):
        """Builds normalized sparse adjacency matrix."""
        adj = nx.adjacency_matrix(self.graph, nodelist=self.node_list)
        adj = adj.tocoo()

        # Normalize: D^-0.5 A D^-0.5 approx (skipped for simplicity, just using A + I)
        # Adding self-loops
        indices = torch.LongTensor(np.vstack((adj.row, adj.col)))
        values = torch.FloatTensor(adj.data)

        # Add eye
        eye_indices = torch.arange(self.num_nodes)
        eye_indices = torch.stack((eye_indices, eye_indices))
        eye_values = torch.ones(self.num_nodes)

        indices = torch.cat((indices, eye_indices), dim=1)
        values = torch.cat((values, eye_values))

        shape = torch.Size(adj.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def _build_feature_matrix(self):
        """Builds node features based on 'type' attribute."""
        node_types = set()
        for n in self.node_list:
            node_types.add(self.graph.nodes[n].get('type', 'Unknown'))

        type_list = list(node_types)
        type_map = {t: i for i, t in enumerate(type_list)}

        num_features = len(type_list)
        features = torch.zeros(self.num_nodes, num_features)

        for i, node in enumerate(self.node_list):
            t = self.graph.nodes[node].get('type', 'Unknown')
            features[i, type_map[t]] = 1.0

        return features

    def predict_risk(self):
        """Runs GNN inference to predict risk for all nodes."""
        self.model.eval()

        # GAT needs dense adj usually, handled inside GAT forward or here?
        # Our implemented GAT handles conversion if sparse.

        with torch.no_grad():
            risk_scores = self.model(self.features, self.adj)

        results = {}
        for i, node in enumerate(self.node_list):
            results[node] = risk_scores[i].item()

        return results

    def explain_risk(self, node_id):
        """
        Returns explanation (masked adj, masked features) for a node.
        """
        if node_id not in self.node_map:
            return None
        idx = self.node_map[node_id]
        mask_adj, mask_feat = self.explainer.explain_node(idx, self.features, self.adj)
        return mask_adj, mask_feat
