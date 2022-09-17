import torch.nn as nn
import torch
from torch.nn.functional import normalize

class Network(nn.Module):
    def __init__(self, resnet, feature_dim):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        
    def forward(self, x_i, x_j):
        
        x_i = self.resnet(x_i)
        x_j = self.resnet(x_j)
        
        z_i = normalize(self.instance_projector(x_i), dim=1)
        z_j = normalize(self.instance_projector(x_j), dim=1)
        
        return z_i, z_j
