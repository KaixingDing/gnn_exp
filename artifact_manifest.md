# Artifact Manifest (图表-代码对应清单)

本文件记录所有实验图表与对应代码模块的映射关系，确保结果可复现。

## Tables (表格)

| 编号 | 描述                           | 生成脚本                         | 对应代码模块                        |
|------|--------------------------------|----------------------------------|------------------------------------|
| T1   | 性能对比-保真度 (Fidelity)     | scripts/gen_table_T1_T3.py       | src/metrics/evaluation.py          |
| T2   | 性能对比-稀疏度 (Sparsity)     | scripts/gen_table_T1_T3.py       | src/metrics/evaluation.py          |
| T3   | 计算效率对比                   | scripts/gen_table_T1_T3.py       | experiments/run_evaluation.py      |
| T4   | 案例研究-成功率 (模拟)         | scripts/gen_table_T4_T5.py       | -                                  |
| T5   | 用户研究-效率 (模拟)           | scripts/gen_table_T4_T5.py       | -                                  |

## Figures (图片)

| 编号 | 描述                                    | 生成脚本                    | 对应代码模块                                |
|------|-----------------------------------------|----------------------------|---------------------------------------------|
| F1   | Fidelity vs. Sparsity 权衡散点图        | scripts/gen_figure_F1_F4.py | src/metrics/evaluation.py                  |
| F2   | 方法对比柱状图                          | scripts/gen_figure_F1_F4.py | experiments/run_evaluation.py               |
| F3   | 跨数据集性能热图                        | scripts/gen_figure_F1_F4.py | experiments/run_evaluation.py               |
| F4   | 多粒度性能对比                          | scripts/gen_figure_F1_F4.py | src/explainers/attention_expl.py           |
| F5   | 案例可视化: MUTAG (分子致癌基团)        | scripts/gen_figure_F5_F7.py | src/explainers/attention_expl.py           |
| F6   | 案例可视化: BA-Shapes (房子结构)        | scripts/gen_figure_F5_F7.py | src/explainers/attention_expl.py           |
| F7   | 多粒度解释对比可视化                    | scripts/gen_figure_F5_F7.py | src/explainers/attention_expl.py           |

## Code Modules (核心代码模块)

### 解释器 (Explainers)

| 模块                               | 描述                                    | 文件路径                            |
|------------------------------------|-----------------------------------------|-------------------------------------|
| BaseExplainer                      | 解释器基类，统一API                     | src/explainers/base.py              |
| MultiGranularityAttentionExplainer | 多粒度注意力解释器（核心算法）          | src/explainers/attention_expl.py    |
| GNNExplainer                       | GNNExplainer基线                        | src/explainers/baselines.py         |
| GradCAM                            | Grad-CAM基线                            | src/explainers/baselines.py         |
| GraphMask                          | GraphMask基线                           | src/explainers/baselines.py         |
| PGExplainer                        | PGExplainer基线                         | src/explainers/baselines.py         |

### 评估指标 (Metrics)

| 模块                | 描述                              | 文件路径                    |
|---------------------|-----------------------------------|-----------------------------|
| fidelity_plus       | 保真度+ (保留重要特征后的预测保持) | src/metrics/evaluation.py   |
| fidelity_minus      | 保真度- (移除重要特征后的预测下降) | src/metrics/evaluation.py   |
| sparsity            | 稀疏度 (解释的简洁性)              | src/metrics/evaluation.py   |
| stability           | 稳定性 (对扰动的鲁棒性)            | src/metrics/evaluation.py   |

### 模型 (Models)

| 模块              | 描述                    | 文件路径               |
|-------------------|-------------------------|------------------------|
| GCN               | 图卷积网络              | src/models/gnn.py      |
| GAT               | 图注意力网络            | src/models/gnn.py      |
| GraphClassifier   | 图分类器                | src/models/gnn.py      |

### 数据集 (Datasets)

| 模块         | 描述                           | 文件路径                  |
|--------------|--------------------------------|---------------------------|
| load_mutag   | MUTAG分子数据集加载器          | src/datasets/loaders.py   |
| load_ba_shapes | BA-Shapes合成数据集加载器    | src/datasets/loaders.py   |
| load_ppi     | PPI蛋白质网络数据集加载器      | src/datasets/loaders.py   |

## Experiment Scripts (实验脚本)

| 脚本                        | 描述                           | 输出                                |
|-----------------------------|--------------------------------|-------------------------------------|
| experiments/run_ours.py     | 运行我们的多粒度解释器         | results/raw_data/ours_*_results.pkl |
| experiments/run_baselines.py| 运行基线方法                   | results/raw_data/baselines_*_results.pkl |
| experiments/run_evaluation.py | 自动化评估管道                | results/raw_data/evaluation_metrics.csv |

## Generated Artifacts (生成的产物)

### 数据文件 (Data Files)

- `results/raw_data/evaluation_metrics.csv` - 所有方法在所有数据集上的评估结果
- `results/raw_data/ours_*_results.pkl` - 我们方法的详细结果
- `results/raw_data/baselines_*_results.pkl` - 基线方法的详细结果

### 表格文件 (Table Files)

- `results/tables/T1_fidelity_comparison.csv` - 保真度对比表
- `results/tables/T2_sparsity_comparison.csv` - 稀疏度对比表
- `results/tables/T3_efficiency_comparison.csv` - 效率对比表
- `results/tables/T4_case_study_success_rate.csv` - 案例研究成功率表
- `results/tables/T5_user_study_efficiency.csv` - 用户研究效率表

### 图表文件 (Figure Files)

- `results/figures/F1_fidelity_vs_sparsity.png` - Fidelity vs. Sparsity散点图
- `results/figures/F2_method_comparison.png` - 方法对比柱状图
- `results/figures/F3_dataset_heatmap.png` - 数据集性能热图
- `results/figures/F4_granularity_comparison.png` - 粒度对比图
- `results/figures/F5_mutag_case.png` - MUTAG案例可视化
- `results/figures/F6_bashapes_case.png` - BA-Shapes案例可视化
- `results/figures/F7_multi_granularity_comparison.png` - 多粒度对比可视化

## Reproducibility Instructions (复现说明)

### 环境配置

```bash
conda env create -f environment.yml
conda activate gnn_exp
```

### 运行实验

```bash
# 1. 运行评估管道（生成所有评估数据）
python experiments/run_evaluation.py

# 2. 生成性能对比表格 T1-T3
python scripts/gen_table_T1_T3.py

# 3. 生成案例研究表格 T4-T5
python scripts/gen_table_T4_T5.py

# 4. 生成性能对比图 F1-F4
python scripts/gen_figure_F1_F4.py

# 5. 生成案例可视化图 F5-F7
python scripts/gen_figure_F5_F7.py
```

### 单独运行特定方法

```bash
# 运行我们的方法
python experiments/run_ours.py

# 运行基线方法
python experiments/run_baselines.py
```

## Version Information (版本信息)

- Python: >= 3.9
- PyTorch: >= 2.0.0
- PyTorch Geometric: Latest
- 生成日期: 2025-11-01

## Notes (备注)

1. **表格 T4-T5**: 由于无法执行真实的用户研究，这些表格使用模拟数据。在实际研究中，应进行真实的专家评估和用户测试。

2. **数据集**: MUTAG和BA-Shapes数据集会在首次运行时自动下载。PPI数据集较大，可能需要较长下载时间。

3. **随机种子**: 所有实验使用固定随机种子 (42) 以确保可复现性。

4. **计算资源**: 实验在CPU上可运行，但使用GPU会显著加速。

5. **图表质量**: 所有图表以300 DPI保存，适合用于论文发表。
