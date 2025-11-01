"""
Baseline Explainer Methods
Implementations of standard GNN explanation methods for comparison.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import numpy as np

from .base import BaseExplainer


class GNNExplainer(BaseExplainer):
    """
    GNNExplainer baseline implementation.
    Based on "GNNExplainer: Generating Explanations for Graph Neural Networks"
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.epochs = epochs
        self.lr = lr
    
    def explain(
        self,
        graph: Data,
        target_node: Optional[int] = None,
        granularity: str = 'node',
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Generate explanation using GNNExplainer."""
        graph = graph.to(self.device)
        num_edges = graph.edge_index.size(1)
        
        # Initialize edge mask
        edge_mask = torch.randn(num_edges, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)
        
        # Get target class
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            if target_node is not None:
                pred = out[target_node]
            else:
                pred = out.mean(dim=0)
            target_class = pred.argmax(dim=-1)
        
        # Optimize edge mask
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Apply sigmoid to get mask in [0, 1]
            mask = torch.sigmoid(edge_mask)
            
            # Masked forward pass
            masked_edge_index = graph.edge_index
            out = self.model(graph.x, masked_edge_index, edge_weight=mask)
            
            if target_node is not None:
                pred = out[target_node]
            else:
                pred = out.mean(dim=0)
            
            # Loss: maximize prediction + regularization
            loss = -F.log_softmax(pred, dim=-1)[target_class]
            loss = loss + 0.01 * mask.sum()  # Sparsity regularization
            
            loss.backward()
            optimizer.step()
        
        # Final mask
        with torch.no_grad():
            edge_importance = torch.sigmoid(edge_mask)
            edge_importance = edge_importance / (edge_importance.max() + 1e-8)
        
        # Compute node importance from edges
        node_importance = torch.zeros(graph.x.size(0), device=self.device)
        for i in range(num_edges):
            src, dst = graph.edge_index[0, i], graph.edge_index[1, i]
            node_importance[src] += edge_importance[i]
            node_importance[dst] += edge_importance[i]
        node_importance = node_importance / (node_importance.max() + 1e-8)
        
        return {
            'node_importance': node_importance,
            'edge_importance': edge_importance,
            'granularity': granularity
        }


class GradCAM(BaseExplainer):
    """
    Grad-CAM for GNNs.
    Gradient-based explanation method.
    """
    
    def explain(
        self,
        graph: Data,
        target_node: Optional[int] = None,
        granularity: str = 'node',
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Generate explanation using Grad-CAM."""
        graph = graph.to(self.device)
        graph.x.requires_grad = True
        
        # Forward pass
        self.model.eval()
        out = self.model(graph.x, graph.edge_index)
        
        if target_node is not None:
            pred = out[target_node]
        else:
            pred = out.mean(dim=0)
        
        target_class = pred.argmax(dim=-1)
        
        # Backward pass
        self.model.zero_grad()
        pred[target_class].backward()
        
        # Node importance from gradients
        node_importance = graph.x.grad.abs().sum(dim=-1)
        node_importance = node_importance / (node_importance.max() + 1e-8)
        
        return {
            'node_importance': node_importance.detach(),
            'edge_importance': None,
            'granularity': granularity
        }


class GraphMask(BaseExplainer):
    """
    GraphMask baseline.
    Learns to mask edges for explanation.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.epochs = epochs
        self.lr = lr
    
    def explain(
        self,
        graph: Data,
        target_node: Optional[int] = None,
        granularity: str = 'node',
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Generate explanation using GraphMask."""
        graph = graph.to(self.device)
        num_edges = graph.edge_index.size(1)
        
        # Initialize gate
        gate = torch.nn.Parameter(torch.randn(num_edges, device=self.device))
        optimizer = torch.optim.Adam([gate], lr=self.lr)
        
        # Get target
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            if target_node is not None:
                pred = out[target_node]
            else:
                pred = out.mean(dim=0)
            target_class = pred.argmax(dim=-1)
        
        # Train gate
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Sample mask using Gumbel-Softmax
            mask = torch.sigmoid(gate)
            
            # Forward with mask
            out = self.model(graph.x, graph.edge_index, edge_weight=mask)
            
            if target_node is not None:
                pred = out[target_node]
            else:
                pred = out.mean(dim=0)
            
            # Loss
            pred_loss = -F.log_softmax(pred, dim=-1)[target_class]
            size_loss = 0.01 * mask.sum()
            loss = pred_loss + size_loss
            
            loss.backward()
            optimizer.step()
        
        # Get final importance
        with torch.no_grad():
            edge_importance = torch.sigmoid(gate)
            edge_importance = edge_importance / (edge_importance.max() + 1e-8)
        
        # Node importance
        node_importance = torch.zeros(graph.x.size(0), device=self.device)
        for i in range(num_edges):
            src, dst = graph.edge_index[0, i], graph.edge_index[1, i]
            node_importance[src] += edge_importance[i]
            node_importance[dst] += edge_importance[i]
        node_importance = node_importance / (node_importance.max() + 1e-8)
        
        return {
            'node_importance': node_importance,
            'edge_importance': edge_importance,
            'granularity': granularity
        }


class PGExplainer(BaseExplainer):
    """
    PGExplainer baseline.
    Parameterized graph explainer.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 50,
        lr: float = 0.01,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.epochs = epochs
        self.lr = lr
    
    def explain(
        self,
        graph: Data,
        target_node: Optional[int] = None,
        granularity: str = 'node',
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Generate explanation using PGExplainer."""
        graph = graph.to(self.device)
        
        # Use simple gradient-based approach
        # (Full PGExplainer requires training a separate explainer network)
        graph.x.requires_grad = True
        
        self.model.eval()
        out = self.model(graph.x, graph.edge_index)
        
        if target_node is not None:
            pred = out[target_node]
        else:
            pred = out.mean(dim=0)
        
        target_class = pred.argmax(dim=-1)
        
        # Backward
        self.model.zero_grad()
        pred[target_class].backward()
        
        # Node importance
        node_importance = graph.x.grad.abs().sum(dim=-1)
        node_importance = node_importance / (node_importance.max() + 1e-8)
        
        # Edge importance from node gradients
        num_edges = graph.edge_index.size(1)
        edge_importance = torch.zeros(num_edges, device=self.device)
        for i in range(num_edges):
            src, dst = graph.edge_index[0, i], graph.edge_index[1, i]
            edge_importance[i] = (node_importance[src] + node_importance[dst]) / 2
        
        return {
            'node_importance': node_importance.detach(),
            'edge_importance': edge_importance,
            'granularity': granularity
        }
