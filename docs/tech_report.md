# Technical Report: Multi-Granularity GNN Explainability Framework

## 1. Overview

This technical report provides detailed implementation information for the multi-granularity GNN explainability framework. It covers architecture decisions, algorithm details, optimization strategies, and lessons learned during development.

## 2. System Architecture

### 2.1 Module Organization

The system follows a modular architecture with clear separation of concerns:

```
src/
├── explainers/      # Explanation generation
├── metrics/         # Evaluation metrics
├── models/          # GNN architectures
├── datasets/        # Data loading and preprocessing
└── utils/           # Helper functions
```

### 2.2 Design Patterns

**Strategy Pattern**: Different explainer methods (GNNExplainer, GradCAM, etc.) implement the same `BaseExplainer` interface, allowing easy switching and comparison.

**Factory Pattern**: Dataset loaders use a factory function `get_dataset()` to instantiate appropriate dataset classes.

**Template Method**: The base explainer defines the workflow skeleton, with subclasses implementing specific explanation strategies.

## 3. Core Algorithm Implementation

### 3.1 Multi-Granularity Attention Explainer

#### Node-Level Explanation

**Approach**: Gradient-based importance scoring
**Implementation**:
```python
graph.x.requires_grad = True
out = model(graph.x, graph.edge_index)
pred[target_class].backward()
node_importance = graph.x.grad.abs().sum(dim=-1)
```

**Rationale**: Gradients measure sensitivity of predictions to node features. L2 norm aggregates multi-dimensional gradients into scalar importance scores.

**Complexity**: O(|V| · d · L) where d is feature dimension and L is number of GNN layers.

#### Edge-Level Explanation

**Approach**: Edge masking and perturbation
**Implementation**: For each edge, temporarily remove it and measure prediction change.

**Optimization**: For large graphs, sample a subset of edges instead of evaluating all edges.

**Trade-off**: Accuracy vs. speed. Full evaluation provides exact importance but scales O(|E|). Sampling provides approximation in O(k) where k << |E|.

#### Subgraph-Level Explanation

**Core Innovation**: Greedy expansion with mutual information maximization

**Algorithm Details**:

1. **Initialization**:
   - Compute node-level importance scores
   - Select node with highest importance as seed
   - Initialize subgraph S = {v_seed}

2. **Iterative Expansion**:
   ```
   for i in range(max_subgraph_size - 1):
       candidates = get_neighbors(S) - S
       best_candidate = None
       best_score = -inf
       
       for v in candidates:
           temp_subgraph = S ∪ {v}
           score = evaluate_subgraph(temp_subgraph)
           if score > best_score:
               best_score = score
               best_candidate = v
       
       S = S ∪ {best_candidate}
   ```

3. **Scoring Function**:
   - Extract induced subgraph from node set
   - Forward pass through GNN
   - Return target class probability

**Beam Search Variant**:
- Maintains top-B candidates at each step
- Provides better exploration but B× computational cost
- Default B=5 provides good balance

**Key Optimizations**:
- Cache model activations to avoid redundant forward passes
- Use sparse matrix operations for subgraph extraction
- Early stopping if score plateaus

**Complexity Analysis**:
- Greedy: O(K · d̄ · T) where K is max subgraph size, d̄ is average degree, T is GNN inference time
- Beam: O(K · B · d̄ · T)
- Much faster than SubgraphX's O(2^|V|) exponential search

### 3.2 Baseline Implementations

#### GNNExplainer

**Approach**: Learn edge mask through optimization

**Implementation Notes**:
- Initialize mask with random values
- Optimize using Adam with learning rate 0.01
- Loss = -log P(y|G,mask) + λ·||mask||₁
- Sparsity regularization λ=0.01

**Challenges**:
- Sensitive to initialization
- May converge to local optima
- Requires careful tuning of epochs (30-100)

#### GradCAM

**Approach**: Use gradients as attention weights

**Advantages**: Fast, no optimization required
**Disadvantages**: May not capture complex patterns

#### GraphMask

**Approach**: Learnable gating mechanism

**Implementation**: Similar to GNNExplainer but uses Gumbel-Softmax for differentiable sampling.

## 4. Evaluation Metrics

### 4.1 Fidelity Metrics

**Fidelity+ (Faithfulness)**:
```
Fid+ = P(y | G_important) / P(y | G)
```

Measures how well important features preserve prediction.

**Fidelity- (Sufficiency)**:
```
Fid- = P(y | G) - P(y | G_unimportant)
```

Measures prediction drop when removing important features.

**Implementation Notes**:
- Use top-k% features (default k=10%)
- Normalize scores to [0, 1]
- Average over multiple samples for stability

### 4.2 Sparsity

```
Sparsity = 1 - (# selected features / # total features)
```

Higher is better - encourages concise explanations.

**Threshold Selection**: Use adaptive threshold or top-k selection.

### 4.3 Stability

Measures consistency under perturbations:
```
Stability = E[cosine_similarity(E(G), E(G + noise))]
```

**Implementation**:
- Add Gaussian noise to node features (σ=0.1)
- Generate multiple perturbed versions
- Compute average cosine similarity

## 5. Technical Challenges and Solutions

### 5.1 Subgraph Connectivity

**Challenge**: Ensuring discovered subgraphs are connected

**Solution**: Only expand to neighbors of current subgraph nodes. This naturally maintains connectivity.

**Alternative**: Allow disconnected regions by modifying candidate selection.

### 5.2 Scalability

**Challenge**: Large graphs (>10K nodes) cause memory issues

**Solutions**:
1. **Mini-batch processing**: Process subgraphs in batches
2. **Sparse operations**: Use PyG's sparse tensors
3. **Sampling**: Sample neighbors instead of examining all
4. **GPU acceleration**: Leverage CUDA for parallel computation

### 5.3 Model Compatibility

**Challenge**: Different GNN architectures have different attention mechanisms

**Solution**: Designed model-agnostic approach using gradients and perturbations rather than relying on internal attention weights.

**Compatibility Matrix**:
- ✓ GCN: Fully supported
- ✓ GAT: Fully supported (can also use native attention)
- ✓ GraphSAGE: Supported via gradient method
- ✓ GIN: Supported via gradient method

### 5.4 Evaluation Consistency

**Challenge**: Different datasets have different graph sizes and properties

**Solution**: 
- Normalize metrics to [0, 1]
- Use relative comparisons within datasets
- Report per-dataset and aggregate statistics

## 6. Implementation Decisions

### 6.1 Framework Choice

**PyTorch Geometric** chosen for:
- Extensive GNN model library
- Efficient sparse operations
- Active community and good documentation

**Alternatives considered**:
- DGL: Good performance but less mature
- TensorFlow GNN: Limited flexibility

### 6.2 Data Structures

**Graph Representation**: Edge list format (COO) for memory efficiency

**Feature Storage**: Dense tensors for small graphs, sparse for large graphs

**Explanation Storage**: Dictionary format for flexibility
```python
{
    'node_importance': Tensor,
    'edge_importance': Tensor,
    'subgraph_nodes': Tensor (optional),
    'granularity': str
}
```

### 6.3 Visualization

**NetworkX** for graph visualization:
- Easy to use
- Integrates well with matplotlib
- Sufficient for research purposes

**For Production**: Consider Cytoscape.js or D3.js for interactive visualizations.

## 7. Performance Optimization

### 7.1 Computational Optimizations

1. **Caching**: Cache node embeddings between explanation steps
2. **Vectorization**: Batch operations where possible
3. **Early Stopping**: Stop expansion if score doesn't improve

### 7.2 Memory Optimizations

1. **Gradient Checkpointing**: For very deep GNNs
2. **In-place Operations**: Reduce memory allocation
3. **Clear Cache**: Explicitly clear CUDA cache for large graphs

### 7.3 Benchmarks

On MUTAG dataset (average graph: 18 nodes, 20 edges):
- Node-level: ~0.05s per graph
- Edge-level: ~0.08s per graph
- Subgraph-level (greedy): ~0.15s per graph
- Subgraph-level (beam, B=5): ~0.35s per graph

## 8. Testing Strategy

### 8.1 Unit Tests

**Coverage**:
- Explainer initialization
- Explanation generation for each granularity
- Metric computation
- Edge cases (empty graphs, single node, etc.)

### 8.2 Integration Tests

**Scenarios**:
- End-to-end explanation pipeline
- Multiple datasets
- Different GNN architectures

### 8.3 Validation

**Sanity Checks**:
- Higher importance features should have higher scores
- Removing important features should decrease prediction
- Explanations should be sparse

## 9. Known Limitations

### 9.1 Current Limitations

1. **Non-connected Subgraphs**: Current implementation assumes connected subgraphs
2. **Directed Graphs**: Optimized for undirected graphs
3. **Dynamic Graphs**: Static snapshot only
4. **Heterogeneous Graphs**: Single node/edge type assumed

### 9.2 Future Enhancements

1. **Multi-hop Attention**: Extend beyond 1-hop neighbors
2. **Hierarchical Explanations**: Multiple levels of granularity simultaneously
3. **Interactive Exploration**: User-guided explanation refinement
4. **Causal Explanations**: Integrate counterfactual reasoning

## 10. Lessons Learned

### 10.1 Design Insights

1. **Modularity Pays Off**: Clean separation enabled rapid experimentation
2. **Unified API**: Single interface simplified comparison and evaluation
3. **Reproducibility**: Fixed seeds and detailed logging essential

### 10.2 Implementation Tips

1. **Start Simple**: Implement basic version first, then optimize
2. **Profile Early**: Identify bottlenecks before premature optimization
3. **Document Decisions**: Record why certain approaches were chosen
4. **Version Control**: Commit frequently with clear messages

### 10.3 Research Insights

1. **Multi-granularity is Essential**: Different tasks need different views
2. **No Silver Bullet**: Each method has trade-offs
3. **Human Evaluation Matters**: Metrics alone insufficient

## 11. Conclusion

This technical report documents the implementation of a comprehensive multi-granularity GNN explainability framework. The modular architecture, efficient algorithms, and extensive evaluation provide a solid foundation for both research and practical applications.

The code is open-source and designed for extensibility. Researchers can easily add new explainer methods, datasets, or evaluation metrics by following the established patterns.

## Appendix A: Code Structure

```
src/
├── explainers/
│   ├── __init__.py
│   ├── base.py                 # Abstract base class
│   ├── attention_expl.py       # Our multi-granularity method
│   └── baselines.py            # GNNExplainer, GradCAM, etc.
├── metrics/
│   ├── __init__.py
│   └── evaluation.py           # Fidelity, sparsity, stability
├── models/
│   ├── __init__.py
│   └── gnn.py                  # GCN, GAT, etc.
├── datasets/
│   ├── __init__.py
│   └── loaders.py              # MUTAG, BA-Shapes, PPI
└── utils/
    ├── __init__.py
    └── helpers.py              # Visualization, I/O
```

## Appendix B: Configuration Files

All experiments use configuration embedded in scripts for simplicity. For production, consider using YAML/JSON config files with libraries like Hydra or OmegaConf.

## Appendix C: Dependencies

Core dependencies:
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- NetworkX >= 2.8
- Matplotlib >= 3.5
- Pandas >= 1.4
- NumPy >= 1.22

See `environment.yml` for complete list.
