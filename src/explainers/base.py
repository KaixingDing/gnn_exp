"""
Base Explainer Abstract Class
Provides a unified API for all explainer methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union
import torch
from torch_geometric.data import Data


class BaseExplainer(ABC):
    """
    Abstract base class for GNN explainers.
    
    All explainer implementations should inherit from this class and implement
    the explain() method.
    """
    
    def __init__(self, model: torch.nn.Module, **kwargs):
        """
        Initialize the explainer.
        
        Args:
            model: The GNN model to explain
            **kwargs: Additional arguments specific to the explainer
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()
        
    @abstractmethod
    def explain(
        self,
        graph: Data,
        target_node: Optional[int] = None,
        granularity: str = 'node',
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate explanation for the given graph.
        
        Args:
            graph: Input graph (torch_geometric Data object)
            target_node: Target node index (for node classification tasks)
            granularity: Explanation granularity ('node', 'edge', 'subgraph', 'global')
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing explanation results:
                - 'node_importance': Node importance scores (torch.Tensor)
                - 'edge_importance': Edge importance scores (torch.Tensor)
                - 'subgraph_nodes': Subgraph node indices (Optional)
                - 'subgraph_edges': Subgraph edge indices (Optional)
        """
        pass
    
    def _forward_pass(
        self,
        graph: Data,
        target_node: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the model.
        
        Args:
            graph: Input graph
            target_node: Target node index
            
        Returns:
            Tuple of (predictions, embeddings)
        """
        with torch.no_grad():
            self.model.eval()
            out = self.model(graph.x, graph.edge_index)
            
            if target_node is not None:
                pred = out[target_node]
            else:
                pred = out
                
        return pred, out
    
    def _get_prediction_class(
        self,
        graph: Data,
        target_node: Optional[int] = None
    ) -> int:
        """
        Get the predicted class for the target.
        
        Args:
            graph: Input graph
            target_node: Target node index
            
        Returns:
            Predicted class index
        """
        pred, _ = self._forward_pass(graph, target_node)
        return pred.argmax(dim=-1).item()
