import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
import misc_methods

# Define a simple GCN model
from torch_geometric.data import Data
class GCN(torch.nn.Module):
    def __init__(self, data):
        super(GCN, self).__init__()
        # Define the GCN layers
        self.conv1 = GCNConv(data.num_node_features, 4)  # Input features to hidden
        self.conv2 = GCNConv(4, 2)  # Hidden to output features
        self.data = data

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Pass data through the first GCN layer, then apply ReLU
        x = torch.relu(self.conv1(x, edge_index))
        # Pass data through the second GCN layer
        x = self.conv2(x, edge_index)
        return x
    
    
def nn_to_data(model: nn.Module) -> Data:
    edges = []

    # Counter for global neuron index
    idx = 0

    # Iterate over each layer in the network
    base = next(model.children())
    if isinstance(base, nn.Sequential):
        layers = list(base.children())
        layers2 = list(base.children())
    else:
        layers = list(model.children()) # iterator over the layers of the model
        layers2 = list(model.children())
    
    num_nodes = layers2[0].weight.shape[1] + sum([layer.weight.shape[0] for layer in layers2 if isinstance(layer, nn.Linear)])
    num_node_features = num_nodes
    node_features = torch.zeros(num_nodes, num_node_features)
    # shape = (num_nodes, num_node_features), where the node features are the bias of each node
    # and the weights of the edges to each node (zero if there is no edge)

    for layer in layers:
        if isinstance(layer, nn.Linear):
            # Update edges based on the weight matrix
            input_dim = layer.weight.shape[1]
            output_dim = layer.weight.shape[0]
            for i in range(input_dim):  # Input neurons (e.g. 4)
                for j in range(output_dim):  # Output neurons (e.g. 64)
                    edges.append((idx + i, idx + input_dim + j))
            
            # Update node features (e.g., biases)
            biases = torch.tensor(layer.bias.detach().numpy())
            edge_weights = torch.tensor(layer.weight.detach().numpy().T)
            node_features[idx + input_dim:idx + input_dim + output_dim, 0] = biases
            node_features[idx:idx + input_dim, 1+idx:1+idx+output_dim] = edge_weights
            node_features[idx + input_dim:idx + input_dim + output_dim, 1+idx:1+idx+input_dim] = edge_weights.T
            
            # Update the global neuron index
            idx += input_dim

    # Convert lists to PyTorch tensors
    num_nonzero = [np.count_nonzero(node_features[i]) for i in range(node_features.shape[0])]
    # print(num_nonzero)
    row_mean, row_median, row_var = torch.mean(node_features[:, 1:], dim=1), torch.median(node_features[:, 1:], dim=1)[0], torch.var(node_features[:, 1:], dim=1)
    x = torch.stack([node_features[:, 0], row_mean, row_median, row_var]).T
    # print(x.shape)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

if __name__ == "__main__":
    agent = misc_methods.train_dqn(env_name = "CartPole-v1", episodes = 1000, verbose = True, return_reward = False)
    data = nn_to_data(agent.model)
    gcn = GCN(data)
    # data.x.shape, data.edge_index.shape
    # print(data.x)
