# User Manual: Multi-Granularity GNN Explainability Framework

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Using Explainers](#4-using-explainers)
5. [Running Experiments](#5-running-experiments)
6. [Generating Visualizations](#6-generating-visualizations)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Advanced Usage](#8-advanced-usage)
9. [Troubleshooting](#9-troubleshooting)
10. [FAQ](#10-faq)

---

## 1. Introduction

This framework provides a comprehensive toolkit for explaining Graph Neural Network (GNN) predictions at multiple granularities: node, edge, subgraph, and global levels.

### Key Features

- **Multi-Granularity Explanations**: Support for 4 different explanation levels
- **Multiple Methods**: Our method plus 4 baseline methods (GNNExplainer, GradCAM, GraphMask, PGExplainer)
- **Comprehensive Evaluation**: Fidelity, Sparsity, and Stability metrics
- **Visualization Tools**: Graph and molecular visualization
- **Pre-configured Datasets**: MUTAG, BA-Shapes, PPI

---

## 2. Installation

### Prerequisites

- Python >= 3.9
- CUDA toolkit (optional, for GPU acceleration)

### Step 1: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate gnn_exp
```

### Step 2: Verify Installation

```bash
python -c "import torch; import torch_geometric; print('Installation successful!')"
```

---

## 3. Quick Start

### Basic Example

```python
import torch
from src.datasets import get_dataset
from src.models import GCN
from src.explainers import MultiGranularityAttentionExplainer

# Load dataset
dataset = get_dataset('MUTAG')
graph = dataset[0]

# Create model
model = GCN(in_channels=7, hidden_channels=64, out_channels=2)

# Create explainer
explainer = MultiGranularityAttentionExplainer(model)

# Generate node-level explanation
explanation = explainer.explain(graph, target_node=0, granularity='node')

print("Node importance:", explanation['node_importance'])
```

### Running Pre-configured Experiments

```bash
# Run automated evaluation
python experiments/run_evaluation.py

# Generate tables and figures
python scripts/gen_table_T1_T3.py
python scripts/gen_figure_F1_F4.py
```

---

## 4. Using Explainers

### 4.1 Our Multi-Granularity Explainer

```python
from src.explainers import MultiGranularityAttentionExplainer

explainer = MultiGranularityAttentionExplainer(
    model=model,
    subgraph_method='greedy',  # or 'beam'
    max_subgraph_size=10,
    beam_width=5  # only for beam search
)

# Node-level explanation
node_expl = explainer.explain(graph, target_node=0, granularity='node')

# Edge-level explanation
edge_expl = explainer.explain(graph, target_node=0, granularity='edge')

# Subgraph-level explanation
subgraph_expl = explainer.explain(graph, target_node=0, granularity='subgraph')

# Global-level explanation
global_expl = explainer.explain(graph, granularity='global')
```

### 4.2 Baseline Explainers

```python
from src.explainers import GNNExplainer, GradCAM, GraphMask, PGExplainer

# GNNExplainer
gnn_explainer = GNNExplainer(model, epochs=50, lr=0.01)
explanation = gnn_explainer.explain(graph, target_node=0)

# GradCAM
gradcam = GradCAM(model)
explanation = gradcam.explain(graph, target_node=0)

# GraphMask
graphmask = GraphMask(model, epochs=50)
explanation = graphmask.explain(graph, target_node=0)

# PGExplainer
pg_explainer = PGExplainer(model, epochs=30)
explanation = pg_explainer.explain(graph, target_node=0)
```

### 4.3 Explanation Format

All explainers return a dictionary with the following structure:

```python
{
    'node_importance': torch.Tensor,  # Shape: [num_nodes]
    'edge_importance': torch.Tensor,  # Shape: [num_edges]
    'subgraph_nodes': torch.Tensor,   # Optional, for subgraph explanations
    'subgraph_edges': torch.Tensor,   # Optional
    'granularity': str                # 'node', 'edge', 'subgraph', or 'global'
}
```

---

## 5. Running Experiments

### 5.1 Automated Evaluation Pipeline

```bash
python experiments/run_evaluation.py
```

This script:
1. Loads datasets (MUTAG, BA-Shapes, Simple)
2. Runs all explainer methods
3. Computes evaluation metrics
4. Saves results to `results/raw_data/evaluation_metrics.csv`

### 5.2 Running Individual Methods

```bash
# Run our method
python experiments/run_ours.py

# Run baseline methods
python experiments/run_baselines.py
```

### 5.3 Custom Evaluation

```python
from src.metrics import evaluate_explanation

metrics = evaluate_explanation(
    model=model,
    graph=graph,
    explanation=explanation,
    target_node=0,
    compute_stability=True  # Optional, slower
)

print(f"Fidelity+: {metrics['fidelity_plus']:.3f}")
print(f"Fidelity-: {metrics['fidelity_minus']:.3f}")
print(f"Sparsity: {metrics['sparsity']:.3f}")
```

---

## 6. Generating Visualizations

### 6.1 Graph Visualization

```python
from src.utils import visualize_graph

visualize_graph(
    graph=graph,
    node_importance=explanation['node_importance'],
    edge_importance=explanation['edge_importance'],
    title='My Explanation',
    save_path='my_explanation.png'
)
```

### 6.2 Generating All Figures

```bash
# Performance figures F1-F4
python scripts/gen_figure_F1_F4.py

# Case study figures F5-F7
python scripts/gen_figure_F5_F7.py
```

### 6.3 Generating Tables

```bash
# Performance tables T1-T3
python scripts/gen_table_T1_T3.py

# User study tables T4-T5
python scripts/gen_table_T4_T5.py
```

---

## 7. Evaluation Metrics

### 7.1 Fidelity Metrics

**Fidelity+** measures how well important features preserve prediction:
```python
from src.metrics import fidelity_plus

fid_plus = fidelity_plus(
    model=model,
    graph=graph,
    explanation=explanation,
    target_node=0,
    top_k=0.1  # Keep top 10% features
)
```

**Fidelity-** measures prediction drop when removing important features:
```python
from src.metrics import fidelity_minus

fid_minus = fidelity_minus(
    model=model,
    graph=graph,
    explanation=explanation,
    target_node=0,
    top_k=0.1  # Remove top 10% features
)
```

### 7.2 Sparsity

Measures explanation conciseness:
```python
from src.metrics import sparsity

sparse_score = sparsity(
    explanation=explanation,
    threshold=0.1  # Features above this threshold are selected
)
```

### 7.3 Stability

Measures consistency under perturbations:
```python
from src.metrics import stability

stable_score = stability(
    model=model,
    graph=graph,
    explainer=explainer,
    target_node=0,
    num_runs=5,
    noise_level=0.1
)
```

---

## 8. Advanced Usage

### 8.1 Custom Datasets

```python
from torch_geometric.data import Data
import torch

# Create custom graph
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 10)  # 3 nodes, 10 features each
y = torch.tensor([0], dtype=torch.long)

graph = Data(x=x, edge_index=edge_index, y=y)

# Use with explainer
explanation = explainer.explain(graph, target_node=0, granularity='node')
```

### 8.2 Custom GNN Models

Your model must be compatible with PyTorch Geometric:

```python
import torch.nn as nn
from torch_geometric.nn import GCNConv

class CustomGNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)
    
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

model = CustomGNN(in_channels=7, out_channels=2)
explainer = MultiGranularityAttentionExplainer(model)
```

### 8.3 Batch Processing

```python
results = []
for graph in dataset[:10]:  # Process first 10 graphs
    explanation = explainer.explain(graph, target_node=0, granularity='subgraph')
    metrics = evaluate_explanation(model, graph, explanation)
    results.append(metrics)

# Aggregate results
import pandas as pd
df = pd.DataFrame(results)
print(df.mean())
```

### 8.4 GPU Acceleration

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Move model and graph to GPU
model = model.to(device)
graph = graph.to(device)

# Explainer will automatically use the same device
explainer = MultiGranularityAttentionExplainer(model)
explanation = explainer.explain(graph, target_node=0)
```

---

## 9. Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or use smaller max_subgraph_size
```python
explainer = MultiGranularityAttentionExplainer(
    model=model,
    max_subgraph_size=5  # Reduce from default 10
)
```

### Issue: Slow Edge-Level Explanation

**Solution**: The full edge evaluation can be slow for large graphs. Use sampling:
```python
# Modify src/explainers/attention_expl.py
# In _explain_edge_level method, sample edges instead of evaluating all
```

### Issue: Import Errors

**Solution**: Ensure you're in the project root and Python path is set:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Issue: Dataset Download Fails

**Solution**: Manually download datasets:
```bash
mkdir -p data
# Download MUTAG, BA-Shapes, or PPI from PyG repository
```

---

## 10. FAQ

### Q1: Which granularity should I use?

**A**: Depends on your task:
- **Node-level**: Quick overview, node classification tasks
- **Edge-level**: Understanding information flow, link prediction
- **Subgraph-level**: Identifying important structures, molecular analysis
- **Global-level**: Overall decision understanding, graph classification

### Q2: How do I interpret the importance scores?

**A**: Scores are normalized to [0, 1]:
- **>0.7**: High importance, critical for prediction
- **0.3-0.7**: Moderate importance
- **<0.3**: Low importance, may be ignored

### Q3: Can I use this with pre-trained models?

**A**: Yes! Load your model and pass it to the explainer:
```python
model = torch.load('my_pretrained_model.pth')
model.eval()
explainer = MultiGranularityAttentionExplainer(model)
```

### Q4: How do I cite this work?

**A**: See README.md for citation information.

### Q5: Can I contribute to the project?

**A**: Yes! The project is open-source. Submit pull requests or open issues on GitHub.

### Q6: What if my graph has node labels?

**A**: Pass them to the visualization function:
```python
visualize_graph(
    graph=graph,
    node_importance=explanation['node_importance'],
    node_labels=['A', 'B', 'C', ...]
)
```

### Q7: How do I save explanations for later?

**A**: Use pickle or JSON:
```python
from src.utils import save_results_dict, load_results_dict

save_results_dict(explanation, 'my_explanation.pkl')
loaded_explanation = load_results_dict('my_explanation.pkl')
```

### Q8: Can I use this for directed graphs?

**A**: The current implementation works best with undirected graphs. For directed graphs, the edge-level explanation will respect edge direction.

---

## Additional Resources

- **Technical Report**: See `docs/tech_report.md` for implementation details
- **Paper**: See `paper/paper_EN.md` for methodology
- **Artifact Manifest**: See `artifact_manifest.md` for complete code-figure mapping
- **Examples**: Check `experiments/` for complete examples

---

## Support

For issues and questions:
1. Check this manual and the FAQ
2. Review the technical report
3. Open an issue on GitHub
4. Contact the maintainers

---

**Version**: 1.0  
**Last Updated**: 2025-11-01
