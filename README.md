# Multi-Granularity Graph Neural Network Explainability Framework

## Overview

This repository contains a comprehensive implementation of a multi-granularity GNN explainability framework that provides explanations at four different levels:
- **Node-level**: Individual node importance
- **Edge-level**: Edge importance scores
- **Subgraph-level**: Connected subgraph discovery
- **Global-level**: Graph-level importance aggregation

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate gnn_exp
```

## Project Structure

```
/project_root
|-- /src                    # Core algorithms and evaluation code
|   |-- /explainers         # Explainer modules
|   |-- /datasets           # Data loading and processing
|   |-- /metrics            # Evaluation metrics
|   |-- /models             # GNN models (GCN, GAT, etc.)
|   |-- /utils              # Utility functions
|-- /experiments            # Executable experiment scripts
|-- /scripts                # Chart generation scripts
|-- /results                # Experimental data and charts
|-- /paper                  # Paper drafts (CN/EN)
|-- /docs                   # Technical documentation
```

## Usage

### Running Experiments

```bash
# Run baseline methods
python experiments/run_baselines.py

# Run our method
python experiments/run_ours.py

# Run automated evaluation
python experiments/run_evaluation.py
```

### Generating Charts

```bash
# Generate performance tables T1-T3
python scripts/gen_table_T1_T3.py

# Generate case study tables T4-T5
python scripts/gen_table_T4_T5.py

# Generate performance figures F1-F4
python scripts/gen_figure_F1_F4.py

# Generate case visualization figures F5-F7
python scripts/gen_figure_F5_F7.py
```

## Datasets

The framework supports:
- **MUTAG**: Molecular dataset for mutagenicity prediction
- **BA-Shapes**: Synthetic dataset with ground-truth structure
- **PPI**: Protein-protein interaction network

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gnn_exp_2025,
  title={Multi-Granularity Graph Neural Network Explainability Framework},
  author={},
  year={2025}
}
```

## License

MIT License
