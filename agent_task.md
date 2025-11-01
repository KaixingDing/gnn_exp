### 📇 项目执行指令：多粒度图神经网络可解释性框架

**交付目标：** 提交一个完整的GitHub仓库，包含所有可复现的代码、实验数据、图表生成脚本，以及中英文论文初稿。

**核心指令：** Agent必须严格按照以下任务序列执行。每个任务的产出（尤其是代码和图表）必须在`artifact_manifest.md`中被精确记录。

#### 1\. 项目基础架构 (Project Structure)

在开始任何编码之前，必须按以下结构初始化项目目录：

```
/project_root
|
|-- /src                 # 核心算法与评估代码
|   |-- /explainers        # 解释器模块
|   |   |-- base.py          # 解释器基类
|   |   |-- attention_expl.py # 核心：多粒度注意力解释器 (T1.1)
|   |   |-- baselines.py       # 基线方法 (T2.1)
|   |-- /datasets          # 数据加载与处理
|   |-- /metrics           # 评估指标 (T2.2)
|   |-- /models            # GNN模型 (e.g., GCN, GAT)
|   |-- /utils             # 辅助函数
|
|-- /experiments         # 可执行的实验脚本
|   |-- run_baselines.py     # 运行基线 (T2.3)
|   |-- run_ours.py          # 运行我们的方法 (T1)
|   |-- run_evaluation.py    # 自动化评估管道 (T2.3)
|
|-- /scripts             # 图表生成脚本
|   |-- gen_table_T1_T3.py   # 生成性能对比表 (T2.4)
|   |-- gen_table_T4_T5.py   # 生成应用案例/用户研究表 (T3.4)
|   |-- gen_figure_F1_F4.py  # 生成性能图 (e.g., 保真度vs稀疏度)
|   |-- gen_figure_F5_F7.py  # 生成案例可视化图 (T3.3)
|
|-- /results             # 实验原始数据和生成的图表
|   |-- /raw_data          # .csv, .json, .pkl (T2.3)
|   |-- /tables            # T1-T5
|   |-- /figures           # F1-F7
|
|-- /paper               # 论文产出
|   |-- paper_CN.md        # 中文初稿
|   |-- paper_EN.md        # 英文初稿
|   |-- references.bib     # 参考文献
|
|-- /docs                # 技术文档与用户手册
|   |-- user_manual.md
|   |-- tech_report.md
|
|-- artifact_manifest.md   # [核心要求] 图表-代码对应清单 (T4.1)
|-- environment.yml        # Conda/Pip环境
|-- README.md              # 项目说明
```

-----

#### 2\. 详细任务分解 (Task Breakdown)

##### 任务 T0：环境与文献准备 (Setup)

  * **T0.1: 环境配置：** 创建 `environment.yml`。
      * **依赖：** `python>=3.9`, `pytorch`, `pyg (torch_geometric)`, `rdkit` (用于分子), `networkx`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
  * **T0.2: 文献综述（用于论文 T4.3）：**
      * **指令：** 搜索并下载以下主题的最新5篇高引论文（2022-2025年）：
        1.  "GNN Explainability OR XAI"
        2.  "Multi-granularity GNN Explanation"
        3.  "Subgraph-level GNN Explanation"
        4.  "Limitations of Attention as Explanation" (针对 T4 挑战)
        5.  "Counterfactual Explanations for Graphs"
      * **产出：** 汇总文献摘要并存入 `paper/references.bib`。

##### 任务 T1：核心算法实现 (Core Algorithm)

  * **T1.1: 实现多粒度注意力解释器（`src/explainers/attention_expl.py`）：**
      * **(a) 节点/边粒度：** 提取GNN模型（如GAT）的原始注意力权重，或实现一个基于梯度/扰动的注意力生成器。
      * **(b) 子图粒度（挑战点）：** 实现一个**动态子图发现算法**。
          * **策略：** 从一个高重要性节点开始，使用**贪心搜索**（Greedy Search）或**束搜索**（Beam Search），基于“互信息最大化”或“预测概率最大化”标准，逐步扩展形成一个连通子图。
      * **(c) 全局粒度：** 实现一个基于全局池化（Global Pooling）层注意力的解释器，或对所有节点/边重要性进行聚合。
  * **T1.2: 建立统一API（`src/explainers/base.py`）：**
      * 定义一个 `BaseExplainer` 抽象类，包含 `explain(graph, target_node, ...)` 方法。
      * `attention_expl.py` 中的解释器必须继承此基类，并允许通过参数（`granularity='node'` or `'subgraph'`）切换解释粒度。
  * **T1.3: 更新清单：** 在 `artifact_manifest.md` 中记录此模块。

##### 任务 T2：基线与评估框架 (Baseline & Evaluation)

  * **T2.1: 复现基线方法（`src/explainers/baselines.py`）：**
      * **指令：** 优先使用 `torch_geometric.explain` 库中的标准实现。
      * **目标：** `GNNExplainer`, `GraphMask`, `PGExplainer`, `Grad-CAM` (或其图版本)。
  * **T2.2: 实现评估指标（`src/metrics/`）：**
      * **指令：** 编写可重用的评估函数。
      * **指标：**
        1.  **Fidelity+** (保真度-增)
        2.  **Fidelity-** (保真度-减)
        3.  **Sparsity** (稀疏度)
        4.  **Stability** (稳定性，可选但加分)
  * **T2.3: 建立自动化评估管道（`experiments/run_evaluation.py`）：**
      * **指令：** 编写一个脚本，该脚本可以
        1.  加载一个数据集（e.g., MUTAG, BA-Shapes）。
        2.  加载一个训练好的GNN模型。
        3.  循环调用**所有**解释器（我们自己的 T1.1 和基线 T2.1）。
        4.  使用 T2.2 的指标评估每个解释器。
      * **产出：** 将所有评估结果保存为 `results/raw_data/evaluation_metrics.csv`。
  * **T2.4: [要求1] 生成图表 T1-T3：**
      * **指令：** 编写 `scripts/gen_table_T1_T3.py`。
      * **动作：** 读取 `evaluation_metrics.csv`，使用 `pandas` 和 `matplotlib` (或 `seaborn`) 生成：
          * **T1:** Fidelity+/Fidelity- 对比表（行：方法，列：数据集）。
          * **T2:** Sparsity 对比表。
          * **F1:** Fidelity vs. Sparsity 散点图。
      * **产出：** 将图表保存到 `results/tables/` 和 `results/figures/`。

##### 任务 T3：应用验证与可视化 (Application & Visualization)

  * **T3.1: 运行应用场景实验：**
      * **指令：** 使用 `experiments/run_ours.py` 和 `run_baselines.py` 在三个领域的数据集上生成解释案例。
      * **数据集：**
        1.  **分子：** MUTAG (或 BBBP)
        2.  **社交：** BA-Shapes (用于验证“ground-truth”)
        3.  **生物：** PPI (蛋白质互作)
  * **T3.2: 开发可视化脚本（`scripts/gen_figure_F5_F7.py`）：**
      * **指令：** 编写一个函数，能接受一个图和一组解释权重（节点/边/子图），并高亮显示它们。
      * **技术：** 使用 `networkx` 和 `matplotlib` 绘图，或 `rdkit` 绘制高亮分子。
  * **T3.3: [要求1] 生成图表 F5-F7：**
      * **指令：** 使用 T3.1 的结果和 T3.2 的脚本，生成定性分析图。
          * **F5:** MUTAG 分子案例（高亮致癌基团 - 亚硝基）。
          * **F6:** BA-Shapes 案例（高亮“房子”结构）。
          * **F7:** PPI 案例（高亮核心蛋白）。
  * **T3.4: [要求1 & 3] 用户研究与表格 T4-T5：**
      * **指令：** Agent 无法执行物理的用户研究。**替代方案：**
        1.  **生成 T4：** 基于 F5-F7 的可视化结果，模拟一个“领域专家评估表”，对比基线方法和我们的方法在识别关键结构上的“成功率”（模拟数据）。
        2.  **生成 T5：** 模拟一个A/B测试结果（例如，使用我们的多粒度解释，识别“房子”结构的时间 vs. GNNExplainer）。
        3.  **产出：** 编写 `docs/user_study_protocol.md`（描述如何进行真实研究）和 `scripts/gen_table_T4_T5.py`（生成模拟数据表）。

##### 任务 T4：文档与论文产出 (Documentation & Paper)

  * **T4.1: [要求1] 汇总图表-代码清单（`artifact_manifest.md`）：**
      * **指令：** 这是整个项目的**核心检查点**。此文件必须包含如下格式的清单：
    <!-- end list -->
    ```markdown
    # Artifact Manifest (图表-代码对应清单)

    ## Tables (表格)
    | 编号 | 描述                  | 生成脚本                         | 对应代码模块               |
    |------|-----------------------|----------------------------------|--------------------------|
    | T1   | 性能对比-保真度       | scripts/gen_table_T1_T3.py       | src/metrics/fidelity.py  |
    | T2   | 性能对比-稀疏度       | scripts/gen_table_T1_T3.py       | src/metrics/sparsity.py  |
    | T3   | 计算效率对比          | experiments/run_evaluation.py    | ...                      |
    | T4   | 案例研究-成功率 (模拟)| scripts/gen_table_T4_T5.py       | ...                      |
    | T5   | 用户研究-效率 (模拟)  | scripts/gen_table_T4_T5.py       | ...                      |

    ## Figures (图片)
    | 编号 | 描述                  | 生成脚本                         | 对应代码模块               |
    |------|-----------------------|----------------------------------|--------------------------|
    | F1   | Fidelity vs. Sparsity | scripts/gen_figure_F1_F4.py      | ...                      |
    | ...  | ...                     | ...                              | ...                      |
    | F5   | 案例: MUTAG           | scripts/gen_figure_F5_F7.py      | src/explainers/attention_expl.py |
    | F6   | 案例: BA-Shapes       | scripts/gen_figure_F5_F7.py      | ...                      |
    | F7   | 案例: PPI             | scripts/gen_figure_F5_F7.py      | ...                      |
    ```
  * **T4.2: [要求2] 撰写论文 - Introduction (CN/EN)：**
      * **指令：** 结合 T0.2 的文献综述，撰写 `paper/paper_CN.md` 和 `paper/paper_EN.md` 的引言部分。
      * **结构要求：**
        1.  **(段落1-2) GNN的广泛应用与“黑盒”问题。**
        2.  **(段落3-4) 现有GNN可解释性的局限性。** (引用 T0.2 文献) 重点抨击“单一粒度”问题（如GNNExplainer只看节点/边）。引用 Yuan et al. (2024) 指出多粒度是核心挑战。
        3.  **(段落5) 我们的工作优势。** (重点) 引入我们的“多粒度图神经网络可解释性框架”。**强调创新点：** (a) 业界首个统一的四粒度（节点、边、子图、全局）解释框架；(b) 基于注意力与贪心搜索的高效子图解释器；(c) 包含自动化评估与用户研究（模拟）的完整验证。
        4.  **(段落6) 贡献总结。**
  * **T4.3: [要求2] 撰写论文 - Methods & Experiments (CN/EN)：**
      * **指令：**
        1.  **Methods：** 详细描述 T1.1 中实现的算法（尤其是 T1.1b 的子图搜索策略）。
        2.  **Experiments：** 描述 T2 的评估设置和 T3 的应用场景。
        3.  **[核心要求] 引用图表：** 在文本中必须**显式引用** T4.1 中定义的所有编号。
              * *(示例)* “...我们在保真度和稀疏度指标上超越了所有基线方法（**见 T1, T2**）。具体而言，我们的方法在Fidelity-指标上比GNNExplainer提升了22%。Fidelity与Sparsity的权衡曲线如**图F1**所示...”
              * *(示例)* “...为了验证解释的有效性，我们在MUTAG数据集上进行了案例分析（**见图F5**）。我们的‘子图’解释器成功高亮了[XXX]官能团，这与化学知识一致...”
  * **T4D.4: 撰写技术文档（`docs/tech_report.md`）：**
      * **指令：** 详细记录 T4 “技术挑战与局限”中的内容，以及 T1.1b 的算法细节和调参过程。

-----

### 3\. 成功标准 (Success Criteria Check)

Agent 在执行完毕后，必须自检是否满足以下所有条件：

1.  **[Check]** `artifact_manifest.md` 文件是否完整且所有链接有效？
2.  **[Check]** `results/` 目录下的所有图表（T1-T5, F1-F7）是否全部生成？
3.  **[Check]** `paper_CN.md` 和 `paper_EN.md` 是否包含 Introduction, Methods, Experiments, Conclusion 四个章节？
4.  **[Check]** 论文初稿中是否**至少引用了10次** T1-T5 或 F1-F7？
5.  **[Check]** Introduction 部分是否明确突出了“多粒度统一框架”的优势？
6.  **[Check]** 代码是否（模拟）达到了“比基线方法提升20%以上”的指标（如 T1 所示）？（如果模拟不达标，应在 T4.4 中报告挑战）。
