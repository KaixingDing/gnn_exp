"""
Automated Evaluation Pipeline
Comprehensive evaluation of all explainers on all datasets.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datasets import get_dataset
from models import GCN
from explainers import (
    MultiGranularityAttentionExplainer,
    GNNExplainer,
    GradCAM,
    GraphMask,
    PGExplainer
)
from metrics import evaluate_explanation
from utils import set_seed


def evaluate_all_methods(
    dataset_name: str,
    device: str = 'cpu',
    num_samples: int = 10
) -> pd.DataFrame:
    """
    Evaluate all explainer methods on a dataset.
    
    Args:
        dataset_name: Name of the dataset
        device: Device to use
        num_samples: Number of graphs to evaluate
        
    Returns:
        DataFrame with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name}")
    print(f"{'='*60}\n")
    
    set_seed(42)
    
    # Load dataset
    if dataset_name == 'PPI':
        _, _, dataset = get_dataset(dataset_name)
    else:
        dataset = get_dataset(dataset_name)
    
    # Limit samples
    if isinstance(dataset, list):
        graphs = dataset[:num_samples]
    else:
        graphs = [dataset[i] for i in range(min(num_samples, len(dataset)))]
    
    # Get model parameters
    graph_sample = graphs[0]
    num_features = graph_sample.num_features if hasattr(graph_sample, 'num_features') else graph_sample.x.size(1)
    num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else 2
    
    # Create model
    model = GCN(
        in_channels=num_features,
        hidden_channels=64,
        out_channels=num_classes,
        num_layers=2
    ).to(device)
    model.eval()
    
    # Initialize explainers
    explainer_configs = {
        'Ours-Node': ('ours', 'node'),
        'Ours-Edge': ('ours', 'edge'),
        'Ours-Subgraph': ('ours', 'subgraph'),
        'Ours-Global': ('ours', 'global'),
        'GNNExplainer': ('baseline', 'GNNExplainer'),
        'GradCAM': ('baseline', 'GradCAM'),
        'GraphMask': ('baseline', 'GraphMask'),
        'PGExplainer': ('baseline', 'PGExplainer')
    }
    
    # Our explainer
    our_explainer = MultiGranularityAttentionExplainer(
        model=model,
        subgraph_method='greedy',
        max_subgraph_size=10
    )
    
    # Baseline explainers
    baseline_explainers = {
        'GNNExplainer': GNNExplainer(model, epochs=30),
        'GradCAM': GradCAM(model),
        'GraphMask': GraphMask(model, epochs=30),
        'PGExplainer': PGExplainer(model, epochs=20)
    }
    
    # Collect results
    results = []
    
    for graph_idx, graph in enumerate(graphs):
        print(f"\nEvaluating graph {graph_idx + 1}/{len(graphs)}...")
        
        graph = graph.to(device)
        target_node = 0 if graph.num_nodes > 0 else None
        
        for method_name, config in explainer_configs.items():
            try:
                if config[0] == 'ours':
                    granularity = config[1]
                    explanation = our_explainer.explain(
                        graph,
                        target_node=target_node if granularity != 'global' else None,
                        granularity=granularity
                    )
                    explainer_obj = our_explainer
                else:
                    baseline_name = config[1]
                    explainer_obj = baseline_explainers[baseline_name]
                    explanation = explainer_obj.explain(
                        graph,
                        target_node=target_node,
                        granularity='node'
                    )
                
                # Evaluate
                metrics = evaluate_explanation(
                    model,
                    graph,
                    explanation,
                    explainer=explainer_obj,
                    target_node=target_node,
                    compute_stability=False
                )
                
                results.append({
                    'Dataset': dataset_name,
                    'Graph': graph_idx,
                    'Method': method_name,
                    'Fidelity+': metrics['fidelity_plus'],
                    'Fidelity-': metrics['fidelity_minus'],
                    'Sparsity': metrics['sparsity']
                })
                
                print(f"  {method_name}: Fid+={metrics['fidelity_plus']:.3f}, "
                      f"Fid-={metrics['fidelity_minus']:.3f}, "
                      f"Sparsity={metrics['sparsity']:.3f}")
                
            except Exception as e:
                print(f"  Error with {method_name}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    return df


def main():
    """Main function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Datasets to evaluate
    datasets = ['SIMPLE', 'MUTAG', 'BA-Shapes']
    
    all_results = []
    
    for dataset_name in datasets:
        try:
            df = evaluate_all_methods(dataset_name, device, num_samples=5)
            all_results.append(df)
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Combine results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save to CSV
        results_dir = Path(__file__).parent.parent / 'results' / 'raw_data'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = results_dir / 'evaluation_metrics.csv'
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"Results saved to {output_file}")
        print(f"{'='*60}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("="*60)
        summary = combined_df.groupby(['Dataset', 'Method']).mean()
        print(summary)
        
        return combined_df
    
    return None


if __name__ == '__main__':
    main()
