# LaTeX论文编译指南

## 概述

本文档提供详细的LaTeX论文编译指南，包括环境配置、编译步骤和常见问题解决方案。

## 文件清单

### LaTeX源文件
- `paper_CN.tex` - 中文论文（约8000字）
- `paper_EN.tex` - 英文论文（约8000字）
- `references.bib` - BibTeX参考文献库

### 图片文件（EPS格式）
- `figures/F1_fidelity_vs_sparsity.eps` - 保真度vs稀疏度散点图
- `figures/F2_method_comparison.eps` - 方法对比柱状图
- `figures/F3_dataset_heatmap.eps` - 数据集性能热图
- `figures/F4_granularity_comparison.eps` - 多粒度性能对比
- `figures/F5_synthetic_case.eps` - 案例1
- `figures/F6_synthetic_case2.eps` - 案例2
- `figures/F7_multi_granularity_comparison.eps` - 多粒度对比

### 辅助文件
- `Makefile` - 自动化编译脚本
- `README.md` - 目录说明
- `COMPILATION_GUIDE.md` - 本文件

## 环境要求

### 操作系统
- Linux / macOS / Windows 均可

### LaTeX发行版
推荐以下任一发行版：
- **TeX Live** (Linux/macOS/Windows) - 推荐
- **MiKTeX** (Windows)
- **MacTeX** (macOS)

### 必需工具
- XeLaTeX（中文论文编译）
- PDFLaTeX（英文论文编译）
- BibTeX（参考文献处理）

### 字体要求（仅中文论文）
系统需安装中文字体，推荐：
- Windows: 系统自带中文字体
- Linux: `sudo apt install fonts-wqy-microhei fonts-wqy-zenhei`
- macOS: 系统自带中文字体

## 快速开始

### 使用Makefile（推荐）

```bash
# 进入latex_paper目录
cd latex_paper

# 编译所有论文
make all

# 或者单独编译
make cn  # 仅编译中文论文
make en  # 仅编译英文论文

# 清理中间文件
make clean

# 清理所有生成文件
make cleanall
```

### 手动编译

#### 中文论文
```bash
cd latex_paper

# 第一次编译
xelatex paper_CN.tex

# 处理参考文献
bibtex paper_CN

# 第二次编译（生成引用）
xelatex paper_CN.tex

# 第三次编译（确保所有交叉引用正确）
xelatex paper_CN.tex
```

生成的PDF：`paper_CN.pdf`

#### 英文论文
```bash
cd latex_paper

# 第一次编译
pdflatex paper_EN.tex

# 处理参考文献
bibtex paper_EN

# 第二次编译
pdflatex paper_EN.tex

# 第三次编译
pdflatex paper_EN.tex
```

生成的PDF：`paper_EN.pdf`

## 编译步骤详解

### 为什么需要编译3次？

1. **第一次编译**: 生成`.aux`文件，记录引用信息
2. **BibTeX处理**: 根据`.aux`文件生成`.bbl`文件（参考文献列表）
3. **第二次编译**: 读取`.bbl`文件，更新引用标记
4. **第三次编译**: 解决所有交叉引用，确保页码准确

### 编译过程中生成的文件

- `.aux` - 辅助信息（引用、标签等）
- `.log` - 编译日志
- `.bbl` - BibTeX生成的参考文献列表
- `.blg` - BibTeX日志
- `.out` - hyperref包生成的超链接信息
- `.toc` - 目录信息
- `.pdf` - 最终的PDF文档

## 常见问题与解决方案

### 问题1: 中文显示为方框或乱码

**原因**: 系统缺少中文字体或XeLaTeX未正确配置

**解决方案**:
```bash
# Ubuntu/Debian
sudo apt install fonts-wqy-microhei fonts-wqy-zenhei texlive-xetex

# macOS
# 系统自带中文字体，确保使用XeLaTeX编译

# Windows
# 确保使用XeLaTeX而不是PDFLaTeX编译中文论文
```

### 问题2: 编译报错 "File xxx.eps not found"

**原因**: 图片文件路径错误

**解决方案**:
1. 确认`figures/`目录存在且包含所有EPS文件
2. 检查LaTeX文件中的图片路径
3. 确保在`latex_paper/`目录下运行编译命令

### 问题3: 参考文献未显示

**原因**: 未运行BibTeX或运行顺序错误

**解决方案**:
```bash
# 完整编译流程
xelatex paper_CN.tex  # 或 pdflatex paper_EN.tex
bibtex paper_CN       # 或 bibtex paper_EN
xelatex paper_CN.tex  # 再编译两次
xelatex paper_CN.tex
```

### 问题4: 算法伪代码格式错误

**原因**: algorithmic包版本问题

**解决方案**:
```bash
# 更新LaTeX包
tlmgr update --self
tlmgr update algorithm algorithmic
```

### 问题5: EPS文件过大导致编译慢

**原因**: EPS文件未压缩

**说明**: 当前EPS文件总大小约334MB，编译可能需要几分钟。这是正常的。

**优化建议**（可选）:
```bash
# 可以将EPS转换为PDF以减小文件大小
cd figures
for f in *.eps; do epstopdf "$f"; done
```

然后修改LaTeX文件中的图片引用：
```latex
% 将
\includegraphics{figures/F1_fidelity_vs_sparsity.eps}
% 改为
\includegraphics{figures/F1_fidelity_vs_sparsity.pdf}
```

### 问题6: Makefile命令不可用

**原因**: Windows系统未安装make工具

**解决方案**:
- 使用手动编译方法
- 或者安装Git for Windows（包含make）
- 或者安装MinGW/Cygwin

## 在线LaTeX编辑器

如果本地编译遇到问题，可使用在线LaTeX编辑器：

### Overleaf（推荐）
1. 访问 https://www.overleaf.com
2. 创建新项目
3. 上传所有`.tex`、`.bib`和`figures/`目录
4. 选择编译器：
   - 中文论文：XeLaTeX
   - 英文论文：PDFLaTeX
5. 点击"Recompile"

### 其他在线编辑器
- Papeeria: https://papeeria.com
- ShareLaTeX (已合并到Overleaf)

## 输出PDF质量检查

编译成功后，检查以下内容：

### 格式检查
- [ ] 页边距正确（2.5cm）
- [ ] 行距适中（1.5倍）
- [ ] 字体清晰可读
- [ ] 中文显示正常（仅中文版）

### 内容检查
- [ ] 所有图片正确显示
- [ ] 表格格式正确
- [ ] 参考文献列表完整
- [ ] 引用编号正确
- [ ] 算法伪代码格式正确
- [ ] 公式编号正确

### 交叉引用检查
- [ ] 图表引用（如"图1"、"表2"）
- [ ] 公式引用
- [ ] 章节引用
- [ ] 参考文献引用

## 论文结构概览

### 中文论文 (paper_CN.tex)

```
摘要
1. 引言
   1.1 研究背景与动机
   1.2 相关工作
   1.3 本文贡献
   1.4 论文组织结构
2. 方法
   2.1 问题定义
   2.2 多粒度注意力解释器
   2.3 评估指标
3. 实验
   3.1 实验设置
   3.2 整体性能对比
   3.3 保真度-稀疏度权衡分析
   3.4 多粒度性能分析
   3.5 案例研究
   3.6 消融实验
4. 讨论
   4.1 边级别解释的优越性
   4.2 多粒度框架的必要性
   4.3 训练模型的方法学意义
   4.4 局限性
   4.5 实践应用指导
   4.6 未来工作方向
5. 结论
参考文献
```

### 英文论文 (paper_EN.tex)

结构与中文版本对应。

## 修改和定制

### 修改作者信息
```latex
% 在tex文件中找到并修改
\author{
匿名作者\\        % 改为真实作者名
\textit{匿名机构}\\  % 改为真实机构名
\texttt{anonymous@example.com}  % 改为真实邮箱
}
```

### 添加新图片
1. 将EPS文件放入`figures/`目录
2. 在tex文件中添加：
```latex
\begin{figure}[htb]
\centering
\includegraphics[width=0.8\textwidth]{figures/新图片.eps}
\caption{图片标题}
\label{fig:新标签}
\end{figure}
```

### 修改参考文献
编辑`references.bib`文件，添加或修改条目：
```bibtex
@article{author2024,
  title={论文标题},
  author={作者名},
  journal={期刊名},
  year={2024}
}
```

## 提交准备

### 会议投稿
1. 检查会议格式要求
2. 可能需要使用会议提供的模板
3. 检查页数限制
4. 准备补充材料

### 期刊投稿
1. 参考期刊的Author Guidelines
2. 调整格式符合期刊要求
3. 准备Cover Letter
4. 准备Highlights/Abstract

## 技术支持

### 文档
- LaTeX官方文档: https://www.latex-project.org/help/documentation/
- Overleaf教程: https://www.overleaf.com/learn
- ctex文档（中文支持）: http://mirrors.ctan.org/language/chinese/ctex/ctex.pdf

### 社区
- TeX StackExchange: https://tex.stackexchange.com
- LaTeX Studio: http://www.latexstudio.net

### 项目文档
- 主README: `../README.md`
- 技术报告: `../docs/tech_report.md`
- 用户手册: `../docs/user_manual.md`

## 版本历史

- 2024-11-04: 初始版本，包含中英文论文LaTeX源文件
- 中文论文: ~8000字，6个主要章节
- 英文论文: ~8000字，与中文版对应
- 7个EPS格式高质量图片
- 完整的BibTeX参考文献库

## 许可证

本项目采用MIT许可证。详见项目根目录的LICENSE文件。
