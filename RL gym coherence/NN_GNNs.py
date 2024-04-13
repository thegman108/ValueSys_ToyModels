import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GCNConv, GATConv

class GraphLevelGCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GraphLevelGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.linear = torch.nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # edge_weights = data.edge_attr
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Aggregate node features to graph-level features
        x = global_mean_pool(x, batch)
        
        # Make a binary classification prediction
        x = self.linear(x)
        return torch.sigmoid(x)

class GATGraphLevelBinary(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GATGraphLevelBinary, self).__init__()
        self.conv1 = GATConv(num_node_features, 8, heads=8, dropout=0.6)
        # Increase the number of output features from the first GAT layer
        self.conv2 = GATConv(8 * 8, 16, heads=1, concat=False, dropout=0.6)
        # Additional GAT layer for richer node representations
        self.linear = torch.nn.Linear(16, 1)
        # Final linear layer to produce a graph-level output

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        x = global_mean_pool(x, batch)  # Aggregate node features to graph-level
        x = self.linear(x)
        return torch.sigmoid(x)  # Sigmoid activation function for binary classification