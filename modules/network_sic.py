import torch.nn as nn
import torch
from torch.nn import Parameter
from torch.nn.functional import normalize

class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num, cluster_centers=None, alpha=1.0):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.cluster_centers = cluster_centers
        self.alpha = alpha
        
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        
        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, x, x_i, x_j):
        h = self.get_embeddings(x)
        c = self.get_cluster_prob(h)
    
        z_i = normalize(self.forward_instance_embeddings(x_i), dim=1)
        z_j = normalize(self.forward_instance_embeddings(x_j), dim=1)
        
        return c, z_i, z_j

    def forward_instance_embeddings(self, x):
        h = self.resnet(x)
        z = self.instance_projector(h)
        return z

    def get_embeddings(self, x):
        h = self.resnet(x)
        return h

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
