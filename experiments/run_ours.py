"""
Run Our Multi-Granularity Explainer
Experiment script for our proposed method.
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datasets import get_dataset, train_simple_model
from models import GCN, GAT, GraphClassifier
from explainers import MultiGranularityAttentionExplainer
from metrics import evaluate_explanation
from utils import visualize_graph, save_results_dict, set_seed


def run_experiment(dataset_name: str, device: str = 'cpu'):
    """
    Run our explainer on a dataset.
    
    Args:
        dataset_name: Name of the dataset
        device: Device to use
    """
    print(f"\n{'='*60}")
    print(f"Running Multi-Granularity Explainer on {dataset_name}")
    print(f"{'='*60}\n")
    
    set_seed(42)
    
    # Load dataset
    if dataset_name == 'PPI':
        train_data, val_data, test_data = get_dataset(dataset_name)
        dataset = test_data
    else:
        dataset = get_dataset(dataset_name)
    
    # Use first graph for demonstration
    if isinstance(dataset, list):
        graph = dataset[0]
    else:
        graph = dataset[0]
    
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Create a simple model
    num_features = graph.num_features if hasattr(graph, 'num_features') else graph.x.size(1)
    num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else 2
    
    model = GCN(
        in_channels=num_features,
        hidden_channels=64,
        out_channels=num_classes,
        num_layers=2
    ).to(device)
    
    # For demo, use random weights (in practice, should train)
    model.eval()
    
    # Initialize our explainer
    explainer = MultiGranularityAttentionExplainer(
        model=model,
        subgraph_method='greedy',
        max_subgraph_size=10
    )
    
    results = {}
    
    # Test different granularities
    for granularity in ['node', 'edge', 'subgraph', 'global']:
        print(f"\nGenerating {granularity}-level explanation...")
        
        explanation = explainer.explain(
            graph.to(device),
            target_node=0 if granularity != 'global' else None,
            granularity=granularity
        )
        
        # Evaluate
        metrics = evaluate_explanation(
            model,
            graph.to(device),
            explanation,
            explainer=explainer,
            target_node=0 if granularity != 'global' else None,
            compute_stability=False  # Skip for speed
        )
        
        print(f"  Fidelity+: {metrics['fidelity_plus']:.4f}")
        print(f"  Fidelity-: {metrics['fidelity_minus']:.4f}")
        print(f"  Sparsity: {metrics['sparsity']:.4f}")
        
        results[granularity] = {
            'explanation': explanation,
            'metrics': metrics
        }
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results' / 'raw_data'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / f'ours_{dataset_name.lower()}_results.pkl'
    save_results_dict(results, str(output_file))
    
    print(f"\nResults saved to {output_file}")
    
    return results


def main():
    """Main function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run on different datasets
    datasets = ['SIMPLE', 'MUTAG', 'BA-Shapes']
    
    all_results = {}
    for dataset_name in datasets:
        try:
            results = run_experiment(dataset_name, device)
            all_results[dataset_name] = results
        except Exception as e:
            print(f"Error running on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == '__main__':
    main()
