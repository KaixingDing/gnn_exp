"""
Evaluation Metrics for GNN Explainability
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict
from torch_geometric.data import Data


def fidelity_plus(
    model: torch.nn.Module,
    graph: Data,
    explanation: Dict[str, torch.Tensor],
    target_node: Optional[int] = None,
    top_k: float = 0.1
) -> float:
    """
    Fidelity+ (Fidelity-increase):
    Measures how well the explanation preserves the prediction.
    
    Higher is better - explanation should maintain prediction when keeping
    only the important parts.
    
    Args:
        model: GNN model
        graph: Input graph
        explanation: Explanation dict with 'node_importance' or 'edge_importance'
        target_node: Target node index
        top_k: Fraction of top important features to keep
        
    Returns:
        Fidelity+ score
    """
    device = next(model.parameters()).device
    graph = graph.to(device)
    
    # Get original prediction
    with torch.no_grad():
        model.eval()
        out_orig = model(graph.x, graph.edge_index)
        if target_node is not None:
            pred_orig = out_orig[target_node]
        else:
            pred_orig = out_orig.mean(dim=0)
        prob_orig = F.softmax(pred_orig, dim=-1)
        target_class = pred_orig.argmax(dim=-1).item()
    
    # Create masked graph keeping top-k important edges
    if 'edge_importance' in explanation and explanation['edge_importance'] is not None:
        edge_importance = explanation['edge_importance']
        num_edges = graph.edge_index.size(1)
        k = max(1, int(num_edges * top_k))
        
        # Get top-k edges
        top_edges = torch.topk(edge_importance, k).indices
        mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        mask[top_edges] = True
        
        masked_edge_index = graph.edge_index[:, mask]
        
        # Predict with masked graph
        with torch.no_grad():
            out_masked = model(graph.x, masked_edge_index)
            if target_node is not None:
                pred_masked = out_masked[target_node]
            else:
                pred_masked = out_masked.mean(dim=0)
            prob_masked = F.softmax(pred_masked, dim=-1)
        
        # Fidelity+ = how much probability is retained
        fid_plus = prob_masked[target_class].item() / (prob_orig[target_class].item() + 1e-8)
    else:
        # Use node importance as fallback
        fid_plus = 1.0
    
    return fid_plus


def fidelity_minus(
    model: torch.nn.Module,
    graph: Data,
    explanation: Dict[str, torch.Tensor],
    target_node: Optional[int] = None,
    top_k: float = 0.1
) -> float:
    """
    Fidelity- (Fidelity-decrease):
    Measures how much prediction drops when removing important parts.
    
    Higher is better - removing important parts should decrease prediction.
    
    Args:
        model: GNN model
        graph: Input graph
        explanation: Explanation dict
        target_node: Target node index
        top_k: Fraction of top important features to remove
        
    Returns:
        Fidelity- score
    """
    device = next(model.parameters()).device
    graph = graph.to(device)
    
    # Get original prediction
    with torch.no_grad():
        model.eval()
        out_orig = model(graph.x, graph.edge_index)
        if target_node is not None:
            pred_orig = out_orig[target_node]
        else:
            pred_orig = out_orig.mean(dim=0)
        prob_orig = F.softmax(pred_orig, dim=-1)
        target_class = pred_orig.argmax(dim=-1).item()
    
    # Create masked graph removing top-k important edges
    if 'edge_importance' in explanation and explanation['edge_importance'] is not None:
        edge_importance = explanation['edge_importance']
        num_edges = graph.edge_index.size(1)
        k = max(1, int(num_edges * top_k))
        
        # Get top-k edges to remove
        top_edges = torch.topk(edge_importance, k).indices
        mask = torch.ones(num_edges, dtype=torch.bool, device=device)
        mask[top_edges] = False
        
        masked_edge_index = graph.edge_index[:, mask]
        
        # Predict without important edges
        with torch.no_grad():
            out_masked = model(graph.x, masked_edge_index)
            if target_node is not None:
                pred_masked = out_masked[target_node]
            else:
                pred_masked = out_masked.mean(dim=0)
            prob_masked = F.softmax(pred_masked, dim=-1)
        
        # Fidelity- = how much probability drops
        fid_minus = prob_orig[target_class].item() - prob_masked[target_class].item()
        fid_minus = max(0, fid_minus)  # Ensure non-negative
    else:
        fid_minus = 0.0
    
    return fid_minus


def sparsity(
    explanation: Dict[str, torch.Tensor],
    threshold: float = 0.1
) -> float:
    """
    Sparsity: Measures how sparse the explanation is.
    
    Higher is better - fewer features should be selected.
    
    Args:
        explanation: Explanation dict
        threshold: Threshold for considering a feature as selected
        
    Returns:
        Sparsity score (1 - fraction of selected features)
    """
    if 'edge_importance' in explanation and explanation['edge_importance'] is not None:
        importance = explanation['edge_importance']
    elif 'node_importance' in explanation and explanation['node_importance'] is not None:
        importance = explanation['node_importance']
    else:
        return 0.0
    
    # Count features above threshold
    selected = (importance > threshold).sum().item()
    total = importance.numel()
    
    # Sparsity = 1 - density
    sparsity_score = 1.0 - (selected / total)
    
    return sparsity_score


def stability(
    model: torch.nn.Module,
    graph: Data,
    explainer,
    target_node: Optional[int] = None,
    num_runs: int = 5,
    noise_level: float = 0.1
) -> float:
    """
    Stability: Measures consistency of explanations under small perturbations.
    
    Higher is better - explanations should be stable.
    
    Args:
        model: GNN model
        graph: Input graph
        explainer: Explainer instance
        target_node: Target node index
        num_runs: Number of perturbed runs
        noise_level: Noise level for perturbation
        
    Returns:
        Stability score (average cosine similarity)
    """
    device = next(model.parameters()).device
    graph = graph.to(device)
    
    # Get base explanation
    base_expl = explainer.explain(graph, target_node)
    if 'edge_importance' in base_expl and base_expl['edge_importance'] is not None:
        base_importance = base_expl['edge_importance']
    else:
        base_importance = base_expl['node_importance']
    
    similarities = []
    
    # Run with perturbations
    for _ in range(num_runs):
        # Add noise to node features
        perturbed_graph = graph.clone()
        noise = torch.randn_like(perturbed_graph.x) * noise_level
        perturbed_graph.x = perturbed_graph.x + noise
        
        # Get explanation for perturbed graph
        perturbed_expl = explainer.explain(perturbed_graph, target_node)
        if 'edge_importance' in perturbed_expl and perturbed_expl['edge_importance'] is not None:
            perturbed_importance = perturbed_expl['edge_importance']
        else:
            perturbed_importance = perturbed_expl['node_importance']
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(
            base_importance.unsqueeze(0),
            perturbed_importance.unsqueeze(0)
        ).item()
        similarities.append(similarity)
    
    return sum(similarities) / len(similarities)


def evaluate_explanation(
    model: torch.nn.Module,
    graph: Data,
    explanation: Dict[str, torch.Tensor],
    explainer = None,
    target_node: Optional[int] = None,
    compute_stability: bool = False
) -> Dict[str, float]:
    """
    Comprehensive evaluation of an explanation.
    
    Args:
        model: GNN model
        graph: Input graph
        explanation: Explanation dict
        explainer: Explainer instance (needed for stability)
        target_node: Target node index
        compute_stability: Whether to compute stability (slower)
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'fidelity_plus': fidelity_plus(model, graph, explanation, target_node),
        'fidelity_minus': fidelity_minus(model, graph, explanation, target_node),
        'sparsity': sparsity(explanation)
    }
    
    if compute_stability and explainer is not None:
        metrics['stability'] = stability(model, graph, explainer, target_node)
    
    return metrics
