import torch
import torch.nn as nn

class FCNNBinary(nn.Module):
    def __init__(self, num_node_features):
        super(FCNNBinary, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_node_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x = data.x
        x = self.fc(x)
        return torch.sigmoid(x)