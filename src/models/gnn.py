"""
GNN Models
Standard GNN architectures for experiments.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class GCN(torch.nn.Module):
    """Graph Convolutional Network."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_weight)
        return x


class GAT(torch.nn.Module):
    """Graph Attention Network."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.5
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        
        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_weight=None, return_attention_weights=False):
        attention_weights = []
        
        for i, conv in enumerate(self.convs[:-1]):
            if return_attention_weights:
                x, (edge_idx, attn) = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append((edge_idx, attn))
            else:
                x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention_weights:
            x, (edge_idx, attn) = self.convs[-1](x, edge_index, return_attention_weights=True)
            attention_weights.append((edge_idx, attn))
            return x, attention_weights
        else:
            x = self.convs[-1](x, edge_index)
            return x


class GraphClassifier(torch.nn.Module):
    """Graph-level classification model."""
    
    def __init__(
        self,
        encoder: torch.nn.Module,
        hidden_channels: int,
        num_classes: int
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch=None, edge_weight=None):
        # Node embeddings
        x = self.encoder(x, edge_index, edge_weight)
        
        # Global pooling
        if batch is None:
            x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        else:
            x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x
