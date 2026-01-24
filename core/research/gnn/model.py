import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolutionLayer, GraphAttentionLayer, GraphSAGELayer

class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN)
    Two-layer GCN with ReLU activation and Dropout.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(nfeat, nhid)
        self.gc2 = GraphConvolutionLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return torch.sigmoid(x)

class GAT(nn.Module):
    """
    Graph Attention Network (GAT) 
    Multi-head attention mechanism.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=2):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return torch.sigmoid(x)

class GraphSAGE(nn.Module):
    """
    GraphSAGE 
    Inductive learning model using neighbor sampling and aggregation.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.sage1 = GraphSAGELayer(nfeat, nhid)
        self.sage2 = GraphSAGELayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.sage1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sage2(x, adj)
        return torch.sigmoid(x)