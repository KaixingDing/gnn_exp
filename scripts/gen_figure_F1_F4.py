"""
Generate Performance Figures F1-F4
Visualization of performance metrics.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def generate_figure_f1_fidelity_vs_sparsity(df: pd.DataFrame, output_dir: Path):
    """
    Generate Figure F1: Fidelity vs. Sparsity scatter plot.
    
    Shows the trade-off between explanation quality and conciseness.
    """
    plt.figure(figsize=(10, 8))
    
    # Group by method
    methods = df['Method'].unique()
    
    # Color palette
    colors = sns.color_palette("husl", len(methods))
    method_colors = {method: colors[i] for i, method in enumerate(methods)}
    
    # Plot each method
    for method in methods:
        method_data = df[df['Method'] == method]
        
        # Mark our methods differently
        if 'Ours' in method:
            marker = 'o'
            size = 120
            alpha = 0.8
        else:
            marker = '^'
            size = 100
            alpha = 0.6
        
        plt.scatter(
            method_data['Sparsity'],
            method_data['Fidelity+'],
            label=method,
            marker=marker,
            s=size,
            alpha=alpha,
            color=method_colors[method]
        )
    
    plt.xlabel('Sparsity', fontsize=14)
    plt.ylabel('Fidelity+', fontsize=14)
    plt.title('Figure F1: Fidelity vs. Sparsity Trade-off', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'F1_fidelity_vs_sparsity.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure F1 saved to {output_file}")
    plt.close()


def generate_figure_f2_method_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Generate Figure F2: Bar chart comparing methods across metrics.
    """
    # Compute average metrics per method
    avg_metrics = df.groupby('Method')[['Fidelity+', 'Fidelity-', 'Sparsity']].mean()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Fidelity+', 'Fidelity-', 'Sparsity']
    titles = ['Fidelity+ (Higher is Better)', 'Fidelity- (Higher is Better)', 'Sparsity (Higher is Better)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Sort by metric value
        data = avg_metrics[metric].sort_values(ascending=False)
        
        # Color our methods differently
        colors = ['#FF6B6B' if 'Ours' in method else '#4ECDC4' for method in data.index]
        
        data.plot(kind='bar', ax=ax, color=colors)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Figure F2: Method Comparison Across Metrics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / 'F2_method_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure F2 saved to {output_file}")
    plt.close()


def generate_figure_f3_dataset_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Generate Figure F3: Heatmap of performance across datasets.
    """
    # Pivot for heatmap
    heatmap_data = df.pivot_table(
        values='Fidelity+',
        index='Method',
        columns='Dataset',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Fidelity+'},
        linewidths=0.5
    )
    
    plt.title('Figure F3: Performance Heatmap Across Datasets', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Method', fontsize=14)
    plt.tight_layout()
    
    output_file = output_dir / 'F3_dataset_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure F3 saved to {output_file}")
    plt.close()


def generate_figure_f4_granularity_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Generate Figure F4: Our method's performance at different granularities.
    """
    # Filter only our methods
    our_methods = df[df['Method'].str.contains('Ours')]
    
    if our_methods.empty:
        print("No 'Ours' methods found for F4")
        return
    
    # Average across datasets
    granularity_metrics = our_methods.groupby('Method')[['Fidelity+', 'Fidelity-', 'Sparsity']].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(granularity_metrics.index))
    width = 0.25
    
    ax.bar(x - width, granularity_metrics['Fidelity+'], width, label='Fidelity+', color='#FF6B6B')
    ax.bar(x, granularity_metrics['Fidelity-'], width, label='Fidelity-', color='#4ECDC4')
    ax.bar(x + width, granularity_metrics['Sparsity'], width, label='Sparsity', color='#95E1D3')
    
    ax.set_xlabel('Granularity Level', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Figure F4: Multi-Granularity Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(granularity_metrics.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = output_dir / 'F4_granularity_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure F4 saved to {output_file}")
    plt.close()


def main():
    """Main function."""
    # Load evaluation results
    data_dir = Path(__file__).parent.parent / 'results' / 'raw_data'
    input_file = data_dir / 'evaluation_metrics.csv'
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        print("Please run experiments/run_evaluation.py first.")
        return
    
    df = pd.read_csv(input_file)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating performance figures...")
    print("="*60)
    
    # Generate figures
    generate_figure_f1_fidelity_vs_sparsity(df, output_dir)
    print("="*60)
    
    generate_figure_f2_method_comparison(df, output_dir)
    print("="*60)
    
    generate_figure_f3_dataset_comparison(df, output_dir)
    print("="*60)
    
    generate_figure_f4_granularity_comparison(df, output_dir)
    print("="*60)
    
    print("\nAll performance figures generated successfully!")


if __name__ == '__main__':
    main()
