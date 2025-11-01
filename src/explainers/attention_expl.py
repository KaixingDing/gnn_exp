"""
Multi-Granularity Attention-based Explainer
Core implementation of the proposed method.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Set
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
import numpy as np

from .base import BaseExplainer


class MultiGranularityAttentionExplainer(BaseExplainer):
    """
    Multi-granularity attention-based explainer supporting:
    - Node-level explanations
    - Edge-level explanations
    - Subgraph-level explanations (greedy/beam search)
    - Global-level explanations
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        use_attention: bool = True,
        subgraph_method: str = 'greedy',  # 'greedy' or 'beam'
        beam_width: int = 5,
        max_subgraph_size: int = 10,
        **kwargs
    ):
        """
        Initialize the multi-granularity explainer.
        
        Args:
            model: GNN model to explain
            use_attention: Whether to use model's attention weights (if available)
            subgraph_method: Method for subgraph discovery ('greedy' or 'beam')
            beam_width: Width for beam search
            max_subgraph_size: Maximum subgraph size
        """
        super().__init__(model, **kwargs)
        self.use_attention = use_attention
        self.subgraph_method = subgraph_method
        self.beam_width = beam_width
        self.max_subgraph_size = max_subgraph_size
        
    def explain(
        self,
        graph: Data,
        target_node: Optional[int] = None,
        granularity: str = 'node',
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate multi-granularity explanations.
        
        Args:
            graph: Input graph
            target_node: Target node index
            granularity: 'node', 'edge', 'subgraph', or 'global'
            
        Returns:
            Dictionary with explanation results
        """
        graph = graph.to(self.device)
        
        if granularity == 'node':
            return self._explain_node_level(graph, target_node)
        elif granularity == 'edge':
            return self._explain_edge_level(graph, target_node)
        elif granularity == 'subgraph':
            return self._explain_subgraph_level(graph, target_node)
        elif granularity == 'global':
            return self._explain_global_level(graph, target_node)
        else:
            raise ValueError(f"Unknown granularity: {granularity}")
    
    def _explain_node_level(
        self,
        graph: Data,
        target_node: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate node-level explanations using gradient-based importance.
        """
        # Enable gradient computation
        graph.x.requires_grad = True
        
        # Forward pass
        self.model.eval()
        out = self.model(graph.x, graph.edge_index)
        
        # Get target prediction
        if target_node is not None:
            pred = out[target_node]
            target_class = pred.argmax(dim=-1)
        else:
            # For graph-level tasks
            pred = out.mean(dim=0)
            target_class = pred.argmax(dim=-1)
        
        # Backward pass to get gradients
        self.model.zero_grad()
        pred[target_class].backward()
        
        # Node importance = gradient magnitude
        node_importance = graph.x.grad.abs().sum(dim=-1)
        node_importance = node_importance / (node_importance.max() + 1e-8)
        
        return {
            'node_importance': node_importance.detach(),
            'edge_importance': None,
            'granularity': 'node'
        }
    
    def _explain_edge_level(
        self,
        graph: Data,
        target_node: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate edge-level explanations using edge masking.
        """
        num_edges = graph.edge_index.size(1)
        edge_importance = torch.zeros(num_edges, device=self.device)
        
        # Get original prediction
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            if target_node is not None:
                orig_pred = out[target_node]
            else:
                orig_pred = out.mean(dim=0)
            orig_prob = F.softmax(orig_pred, dim=-1)
            target_class = orig_pred.argmax(dim=-1)
        
        # Compute edge importance by masking
        for edge_idx in range(num_edges):
            # Create masked edge_index
            mask = torch.ones(num_edges, dtype=torch.bool, device=self.device)
            mask[edge_idx] = False
            masked_edge_index = graph.edge_index[:, mask]
            
            # Forward pass with masked edge
            with torch.no_grad():
                out = self.model(graph.x, masked_edge_index)
                if target_node is not None:
                    masked_pred = out[target_node]
                else:
                    masked_pred = out.mean(dim=0)
                masked_prob = F.softmax(masked_pred, dim=-1)
            
            # Importance = drop in prediction probability
            edge_importance[edge_idx] = (
                orig_prob[target_class] - masked_prob[target_class]
            ).item()
        
        # Normalize
        edge_importance = torch.clamp(edge_importance, min=0)
        edge_importance = edge_importance / (edge_importance.max() + 1e-8)
        
        return {
            'node_importance': None,
            'edge_importance': edge_importance,
            'granularity': 'edge'
        }
    
    def _explain_subgraph_level(
        self,
        graph: Data,
        target_node: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate subgraph-level explanations using greedy or beam search.
        """
        if self.subgraph_method == 'greedy':
            return self._greedy_subgraph_search(graph, target_node)
        elif self.subgraph_method == 'beam':
            return self._beam_subgraph_search(graph, target_node)
        else:
            raise ValueError(f"Unknown subgraph method: {self.subgraph_method}")
    
    def _greedy_subgraph_search(
        self,
        graph: Data,
        target_node: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Greedy search for important subgraph.
        
        Strategy: Start from high-importance node, iteratively add neighbors
        that maximize prediction probability.
        """
        num_nodes = graph.x.size(0)
        
        # First, get node importance to find starting node
        node_expl = self._explain_node_level(graph, target_node)
        node_importance = node_expl['node_importance']
        
        # Get original prediction
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            if target_node is not None:
                orig_pred = out[target_node]
                start_node = target_node
            else:
                orig_pred = out.mean(dim=0)
                start_node = node_importance.argmax().item()
            target_class = orig_pred.argmax(dim=-1).item()
        
        # Initialize subgraph with starting node
        subgraph_nodes = {start_node}
        
        # Get edge_index as list for easier manipulation
        edge_index = graph.edge_index.cpu().numpy()
        
        # Build adjacency list
        neighbors = {i: set() for i in range(num_nodes)}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            neighbors[src].add(dst)
            neighbors[dst].add(src)
        
        # Greedy expansion
        for _ in range(self.max_subgraph_size - 1):
            # Find candidate neighbors
            candidates = set()
            for node in subgraph_nodes:
                candidates.update(neighbors[node])
            candidates -= subgraph_nodes
            
            if not candidates:
                break
            
            # Evaluate each candidate
            best_score = -float('inf')
            best_candidate = None
            
            for candidate in candidates:
                # Create subgraph with candidate
                test_nodes = list(subgraph_nodes | {candidate})
                test_nodes_tensor = torch.tensor(test_nodes, dtype=torch.long, device=self.device)
                
                # Get subgraph
                sub_edge_index, _ = subgraph(
                    test_nodes_tensor,
                    graph.edge_index,
                    relabel_nodes=True
                )
                
                # Evaluate prediction
                with torch.no_grad():
                    sub_x = graph.x[test_nodes_tensor]
                    out = self.model(sub_x, sub_edge_index)
                    
                    if target_node is not None and target_node in test_nodes:
                        new_target_idx = test_nodes.index(target_node)
                        pred = out[new_target_idx]
                    else:
                        pred = out.mean(dim=0)
                    
                    score = F.softmax(pred, dim=-1)[target_class].item()
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate is not None:
                subgraph_nodes.add(best_candidate)
        
        # Create final subgraph
        subgraph_nodes_tensor = torch.tensor(
            list(subgraph_nodes),
            dtype=torch.long,
            device=self.device
        )
        
        sub_edge_index, edge_mask = subgraph(
            subgraph_nodes_tensor,
            graph.edge_index,
            relabel_nodes=False
        )
        
        # Create node and edge masks
        node_mask = torch.zeros(num_nodes, device=self.device)
        node_mask[subgraph_nodes_tensor] = 1.0
        
        edge_mask_full = torch.zeros(graph.edge_index.size(1), device=self.device)
        edge_mask_full[edge_mask] = 1.0
        
        return {
            'node_importance': node_mask,
            'edge_importance': edge_mask_full,
            'subgraph_nodes': subgraph_nodes_tensor,
            'subgraph_edges': edge_mask,
            'granularity': 'subgraph'
        }
    
    def _beam_subgraph_search(
        self,
        graph: Data,
        target_node: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Beam search for important subgraph.
        
        Maintains top-k partial subgraphs and expands them.
        """
        num_nodes = graph.x.size(0)
        
        # Get node importance
        node_expl = self._explain_node_level(graph, target_node)
        node_importance = node_expl['node_importance']
        
        # Get target class
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            if target_node is not None:
                orig_pred = out[target_node]
                start_node = target_node
            else:
                orig_pred = out.mean(dim=0)
                start_node = node_importance.argmax().item()
            target_class = orig_pred.argmax(dim=-1).item()
        
        # Build adjacency list
        edge_index = graph.edge_index.cpu().numpy()
        neighbors = {i: set() for i in range(num_nodes)}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            neighbors[src].add(dst)
            neighbors[dst].add(src)
        
        # Initialize beam with starting node
        # Each beam item: (nodes_set, score)
        beam = [({start_node}, 0.0)]
        
        # Beam search
        for depth in range(self.max_subgraph_size - 1):
            candidates = []
            
            for nodes_set, _ in beam:
                # Find neighbors to expand
                next_nodes = set()
                for node in nodes_set:
                    next_nodes.update(neighbors[node])
                next_nodes -= nodes_set
                
                if not next_nodes:
                    candidates.append((nodes_set, self._evaluate_subgraph(
                        graph, list(nodes_set), target_node, target_class
                    )))
                    continue
                
                # Expand with each neighbor
                for next_node in next_nodes:
                    new_nodes = nodes_set | {next_node}
                    score = self._evaluate_subgraph(
                        graph, list(new_nodes), target_node, target_class
                    )
                    candidates.append((new_nodes, score))
            
            # Keep top-k
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:self.beam_width]
            
            if not beam:
                break
        
        # Return best subgraph
        best_nodes_set, _ = beam[0]
        subgraph_nodes_tensor = torch.tensor(
            list(best_nodes_set),
            dtype=torch.long,
            device=self.device
        )
        
        sub_edge_index, edge_mask = subgraph(
            subgraph_nodes_tensor,
            graph.edge_index,
            relabel_nodes=False
        )
        
        # Create masks
        node_mask = torch.zeros(num_nodes, device=self.device)
        node_mask[subgraph_nodes_tensor] = 1.0
        
        edge_mask_full = torch.zeros(graph.edge_index.size(1), device=self.device)
        edge_mask_full[edge_mask] = 1.0
        
        return {
            'node_importance': node_mask,
            'edge_importance': edge_mask_full,
            'subgraph_nodes': subgraph_nodes_tensor,
            'subgraph_edges': edge_mask,
            'granularity': 'subgraph'
        }
    
    def _evaluate_subgraph(
        self,
        graph: Data,
        nodes: List[int],
        target_node: Optional[int],
        target_class: int
    ) -> float:
        """
        Evaluate a subgraph by its prediction score.
        """
        nodes_tensor = torch.tensor(nodes, dtype=torch.long, device=self.device)
        
        sub_edge_index, _ = subgraph(
            nodes_tensor,
            graph.edge_index,
            relabel_nodes=True
        )
        
        with torch.no_grad():
            sub_x = graph.x[nodes_tensor]
            out = self.model(sub_x, sub_edge_index)
            
            if target_node is not None and target_node in nodes:
                new_target_idx = nodes.index(target_node)
                pred = out[new_target_idx]
            else:
                pred = out.mean(dim=0)
            
            score = F.softmax(pred, dim=-1)[target_class].item()
        
        return score
    
    def _explain_global_level(
        self,
        graph: Data,
        target_node: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate global-level explanations by aggregating node/edge importance.
        """
        # Get node-level importance
        node_expl = self._explain_node_level(graph, target_node)
        node_importance = node_expl['node_importance']
        
        # Get edge-level importance (sample-based for efficiency)
        num_edges = graph.edge_index.size(1)
        sample_size = min(100, num_edges)
        edge_indices = torch.randperm(num_edges)[:sample_size]
        
        edge_importance = torch.zeros(num_edges, device=self.device)
        
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            if target_node is not None:
                orig_pred = out[target_node]
            else:
                orig_pred = out.mean(dim=0)
            orig_prob = F.softmax(orig_pred, dim=-1)
            target_class = orig_pred.argmax(dim=-1)
            
            for edge_idx in edge_indices:
                mask = torch.ones(num_edges, dtype=torch.bool, device=self.device)
                mask[edge_idx] = False
                masked_edge_index = graph.edge_index[:, mask]
                
                out = self.model(graph.x, masked_edge_index)
                if target_node is not None:
                    masked_pred = out[target_node]
                else:
                    masked_pred = out.mean(dim=0)
                masked_prob = F.softmax(masked_pred, dim=-1)
                
                edge_importance[edge_idx] = (
                    orig_prob[target_class] - masked_prob[target_class]
                ).item()
        
        # Normalize
        edge_importance = torch.clamp(edge_importance, min=0)
        edge_importance = edge_importance / (edge_importance.max() + 1e-8)
        
        # Global aggregation
        global_node_score = node_importance.mean().item()
        global_edge_score = edge_importance.mean().item()
        
        return {
            'node_importance': node_importance,
            'edge_importance': edge_importance,
            'global_node_score': global_node_score,
            'global_edge_score': global_edge_score,
            'granularity': 'global'
        }
