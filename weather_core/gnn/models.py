import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class SphericalGraphConv(nn.Module):
    """
    Native PyTorch message-passing graph convolution layer operating on spherical edge graphs.
    H^{(l+1)} = GELU( D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)} + b )
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [N, in_features]
        edge_index: [2, E]
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Linear projection
        h = torch.matmul(x, self.weight)  # [N, out_features]

        # Compute node degrees for symmetric normalization
        deg = torch.zeros(N, device=x.device)
        deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1.0), -0.5)

        # Message aggregation
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        msg = h[src] * norm.unsqueeze(1)  # [E, out_features]

        out = torch.zeros_like(h)
        out.index_add_(0, dst, msg)
        out = out + self.bias

        return F.gelu(out)


class WeatherGNNPredictor(nn.Module):
    """
    Lightweight Graph Neural Network for atmospheric spatial anomaly representation learning.
    """

    def __init__(
        self,
        in_features: int = 8,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        self.conv1 = SphericalGraphConv(in_features, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = SphericalGraphConv(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Node Anomaly Intensity Regression Head
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        # Event Representation Embedding Head (Global Pooling)
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [N, in_features]
        edge_index: [2, E]

        Returns:
            node_anomaly_scores: [N, 1]
            event_embedding: [1, 64]
        """
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = self.norm2(h)

        # 1. Node Anomaly Intensity Predictions
        node_scores = self.anomaly_head(h)

        # 2. Graph Event Embedding (Global Mean Pooling)
        graph_mean = torch.mean(h, dim=0, keepdim=True)
        event_embed = self.embedding_head(graph_mean)

        return node_scores, event_embed
