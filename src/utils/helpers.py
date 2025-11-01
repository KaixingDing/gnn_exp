"""
Utility Functions
Helper functions for visualization and data processing.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from typing import Optional, Dict, List


def visualize_graph(
    graph: Data,
    node_importance: Optional[torch.Tensor] = None,
    edge_importance: Optional[torch.Tensor] = None,
    node_labels: Optional[List[str]] = None,
    title: str = "Graph Visualization",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    Visualize a graph with optional importance highlighting.
    
    Args:
        graph: PyG graph data
        node_importance: Node importance scores
        edge_importance: Edge importance scores
        node_labels: Node labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    # Convert to NetworkX
    G = to_networkx(graph, to_undirected=True)
    
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    
    # Node colors based on importance
    if node_importance is not None:
        node_colors = node_importance.cpu().numpy()
        node_colors = (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min() + 1e-8)
    else:
        node_colors = 'lightblue'
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=500,
        cmap='YlOrRd',
        vmin=0, vmax=1
    )
    
    # Draw edges
    if edge_importance is not None:
        # Get edge weights
        edge_weights = edge_importance.cpu().numpy()
        edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-8)
        
        # Draw edges with varying width
        edges = list(G.edges())
        for i, (u, v) in enumerate(edges):
            if i < len(edge_weights):
                width = 0.5 + edge_weights[i] * 3
                alpha = 0.3 + edge_weights[i] * 0.7
                nx.draw_networkx_edges(
                    G, pos,
                    [(u, v)],
                    width=width,
                    alpha=alpha,
                    edge_color='red'
                )
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Draw labels
    if node_labels is not None:
        labels = {i: label for i, label in enumerate(node_labels)}
        nx.draw_networkx_labels(G, pos, labels)
    else:
        nx.draw_networkx_labels(G, pos)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()


def visualize_molecule(
    graph: Data,
    node_importance: Optional[torch.Tensor] = None,
    edge_importance: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    title: str = "Molecule Visualization"
):
    """
    Visualize a molecular graph.
    
    For now, uses standard graph visualization.
    For production, this should use RDKit for proper molecular rendering.
    """
    visualize_graph(
        graph,
        node_importance,
        edge_importance,
        title=title,
        save_path=save_path
    )


def save_results_dict(results: Dict, filepath: str):
    """
    Save results dictionary to file.
    
    Args:
        results: Results dictionary
        filepath: Output file path
    """
    import json
    import pickle
    
    # Try JSON first
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    except (TypeError, ValueError):
        # Fall back to pickle
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved results as pickle to {filepath}")
    else:
        print(f"Saved results as JSON to {filepath}")


def load_results_dict(filepath: str) -> Dict:
    """
    Load results dictionary from file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Results dictionary
    """
    import json
    import pickle
    
    # Try JSON first
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Fall back to pickle
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    
    return results


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
