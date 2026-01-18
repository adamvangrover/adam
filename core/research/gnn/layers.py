import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphConvolutionLayer(nn.Module):
    """
    Simple GCN Layer: H' = A * H * W
    Supports sparse adjacency matrix A.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        input: (N, in_features)
        adj: (N, N) sparse or dense
        """
        # Support = H * W
        support = torch.mm(input, self.weight)

        # Output = A * Support
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphAttentionLayer(nn.Module):
    """
    GAT Layer: Uses attention mechanism to weigh neighbors.
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # Wh.shape: (N, out_features)
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)

        # Handle sparse adj by converting to dense for demo purposes
        # In production, use scatter/gather operations
        if adj.is_sparse:
             adj = adj.to_dense()

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

class GraphSAGELayer(nn.Module):
    """
    GraphSAGE Layer: Inductive learning with neighborhood aggregation (Mean).
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphSAGELayer, self).__init__()
        self.linear_neigh = nn.Linear(in_features, out_features, bias=bias)
        self.linear_self = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, h, adj):
        # Normalize adjacency to act as Mean Aggregator
        if adj.is_sparse:
             adj_dense = adj.to_dense()
             deg = adj_dense.sum(dim=1, keepdim=True) + 1e-5
             adj_norm = adj_dense / deg
             h_neigh = torch.mm(adj_norm, h)
        else:
             deg = adj.sum(dim=1, keepdim=True) + 1e-5
             adj_norm = adj / deg
             h_neigh = torch.mm(adj_norm, h)

        out = self.linear_neigh(h_neigh) + self.linear_self(h)
        return F.relu(out)
