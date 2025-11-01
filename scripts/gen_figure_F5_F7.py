"""
Generate Case Visualization Figures F5-F7
Visualize explanation examples on different datasets.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datasets import get_dataset
from models import GCN
from explainers import MultiGranularityAttentionExplainer
from utils import visualize_graph, set_seed


def generate_figure_f5_synthetic(output_dir: Path, device: str = 'cpu'):
    """
    Generate Figure F5: Synthetic graph case.
    
    Highlights important substructure.
    """
    print("Generating Figure F5: Synthetic graph case...")
    
    set_seed(42)
    
    # Load Synthetic dataset
    dataset = get_dataset('SYNTHETIC')
    graph = dataset[5].to(device)  # Use 6th graph
    
    # Create model
    num_features = graph.x.size(1)
    num_classes = 2
    
    model = GCN(
        in_channels=num_features,
        hidden_channels=64,
        out_channels=num_classes,
        num_layers=2
    ).to(device)
    model.eval()
    
    # Get explanation
    explainer = MultiGranularityAttentionExplainer(
        model=model,
        subgraph_method='greedy',
        max_subgraph_size=8
    )
    
    explanation = explainer.explain(
        graph,
        target_node=0,
        granularity='subgraph'
    )
    
    # Visualize
    output_file = output_dir / 'F5_synthetic_case.png'
    visualize_graph(
        graph,
        node_importance=explanation['node_importance'],
        edge_importance=explanation['edge_importance'],
        title='Figure F5: Synthetic Graph - Subgraph Explanation Example',
        save_path=str(output_file),
        figsize=(12, 10)
    )
    
    print(f"Figure F5 saved to {output_file}")


def generate_figure_f6_synthetic(output_dir: Path, device: str = 'cpu'):
    """
    Generate Figure F6: Synthetic graph case 2.
    
    Highlights different structure.
    """
    print("Generating Figure F6: Synthetic graph case 2...")
    
    set_seed(123)
    
    # Load Synthetic dataset
    dataset = get_dataset('SYNTHETIC')
    graph = dataset[10].to(device)  # Use 11th graph
    
    # Create model
    num_features = graph.x.size(1)
    num_classes = 2
    
    model = GCN(
        in_channels=num_features,
        hidden_channels=64,
        out_channels=num_classes,
        num_layers=2
    ).to(device)
    model.eval()
    
    # Get explanation
    explainer = MultiGranularityAttentionExplainer(
        model=model,
        subgraph_method='greedy',
        max_subgraph_size=10
    )
    
    explanation = explainer.explain(
        graph,
        target_node=0,
        granularity='subgraph'
    )
    
    # Visualize
    output_file = output_dir / 'F6_synthetic_case2.png'
    visualize_graph(
        graph,
        node_importance=explanation['node_importance'],
        edge_importance=explanation['edge_importance'],
        title='Figure F6: Synthetic Graph - Alternative Subgraph Discovery',
        save_path=str(output_file),
        figsize=(12, 10)
    )
    
    print(f"Figure F6 saved to {output_file}")


def generate_figure_f7_comparison(output_dir: Path, device: str = 'cpu'):
    """
    Generate Figure F7: Comparison across granularities.
    
    Shows the same graph with different granularity explanations.
    """
    print("Generating Figure F7: Multi-granularity comparison...")
    
    set_seed(42)
    
    # Load simple dataset for clear visualization
    dataset = get_dataset('SIMPLE')
    graph = dataset[0].to(device)
    
    # Create model
    num_features = graph.x.size(1)
    num_classes = 2
    
    model = GCN(
        in_channels=num_features,
        hidden_channels=32,
        out_channels=num_classes,
        num_layers=2
    ).to(device)
    model.eval()
    
    # Create explainer
    explainer = MultiGranularityAttentionExplainer(
        model=model,
        subgraph_method='greedy',
        max_subgraph_size=5
    )
    
    # Get explanations at different granularities
    granularities = ['node', 'edge', 'subgraph']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, granularity in enumerate(granularities):
        explanation = explainer.explain(
            graph,
            target_node=0,
            granularity=granularity
        )
        
        # Convert to NetworkX for plotting
        from torch_geometric.utils import to_networkx
        G = to_networkx(graph, to_undirected=True)
        
        ax = axes[idx]
        pos = nx.spring_layout(G, seed=42)
        
        # Get importance scores
        if granularity == 'node':
            node_colors = explanation['node_importance'].cpu().numpy()
            edge_colors = 'gray'
        elif granularity == 'edge':
            node_colors = 'lightblue'
            edge_colors = explanation['edge_importance'].cpu().numpy()
        else:  # subgraph
            node_colors = explanation['node_importance'].cpu().numpy()
            edge_colors = 'gray'
        
        # Normalize colors
        if isinstance(node_colors, (list, tuple)) or hasattr(node_colors, '__iter__'):
            if len(node_colors) > 0 and not isinstance(node_colors, str):
                node_colors = (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min() + 1e-8)
        
        # Draw
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors if not isinstance(node_colors, str) else [node_colors]*len(G.nodes()),
            node_size=600,
            cmap='YlOrRd',
            vmin=0, vmax=1
        )
        
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, width=2)
        nx.draw_networkx_labels(G, pos, ax=ax)
        
        ax.set_title(f'{granularity.capitalize()}-Level', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Figure F7: Multi-Granularity Explanation Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / 'F7_multi_granularity_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure F7 saved to {output_file}")
    plt.close()


def main():
    """Main function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating case visualization figures...")
    print("="*60)
    
    try:
        generate_figure_f5_synthetic(output_dir, device)
        print("="*60)
    except Exception as e:
        print(f"Error generating F5: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        generate_figure_f6_synthetic(output_dir, device)
        print("="*60)
    except Exception as e:
        print(f"Error generating F6: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        generate_figure_f7_comparison(output_dir, device)
        print("="*60)
    except Exception as e:
        print(f"Error generating F7: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nCase visualization figures generated!")


if __name__ == '__main__':
    main()
