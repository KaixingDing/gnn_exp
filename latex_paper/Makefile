# Makefile for compiling LaTeX papers
# 用于编译LaTeX论文的Makefile

# 论文文件名（不含扩展名）
CN_PAPER = paper_CN
EN_PAPER = paper_EN

# 编译器
XELATEX = xelatex
PDFLATEX = pdflatex
BIBTEX = bibtex

# 编译选项
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# 默认目标：编译所有论文
.PHONY: all
all: cn en

# 编译中文论文（使用XeLaTeX支持中文）
.PHONY: cn
cn: $(CN_PAPER).pdf

$(CN_PAPER).pdf: $(CN_PAPER).tex references.bib
	@echo "=== Compiling Chinese paper (1st pass) ==="
	$(XELATEX) $(LATEX_FLAGS) $(CN_PAPER).tex
	@echo "=== Running BibTeX ==="
	-$(BIBTEX) $(CN_PAPER)
	@echo "=== Compiling Chinese paper (2nd pass) ==="
	$(XELATEX) $(LATEX_FLAGS) $(CN_PAPER).tex
	@echo "=== Compiling Chinese paper (3rd pass) ==="
	$(XELATEX) $(LATEX_FLAGS) $(CN_PAPER).tex
	@echo "=== Chinese paper compiled successfully: $(CN_PAPER).pdf ==="

# 编译英文论文（使用PDFLaTeX）
.PHONY: en
en: $(EN_PAPER).pdf

$(EN_PAPER).pdf: $(EN_PAPER).tex references.bib
	@echo "=== Compiling English paper (1st pass) ==="
	$(PDFLATEX) $(LATEX_FLAGS) $(EN_PAPER).tex
	@echo "=== Running BibTeX ==="
	-$(BIBTEX) $(EN_PAPER)
	@echo "=== Compiling English paper (2nd pass) ==="
	$(PDFLATEX) $(LATEX_FLAGS) $(EN_PAPER).tex
	@echo "=== Compiling English paper (3rd pass) ==="
	$(PDFLATEX) $(LATEX_FLAGS) $(EN_PAPER).tex
	@echo "=== English paper compiled successfully: $(EN_PAPER).pdf ==="

# 清理中间文件
.PHONY: clean
clean:
	@echo "=== Cleaning intermediate files ==="
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lot *.lof
	rm -f *.fdb_latexmk *.fls *.synctex.gz
	@echo "=== Cleanup complete ==="

# 清理所有生成文件（包括PDF）
.PHONY: cleanall
cleanall: clean
	@echo "=== Removing PDF files ==="
	rm -f $(CN_PAPER).pdf $(EN_PAPER).pdf
	@echo "=== All generated files removed ==="

# 查看帮助信息
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make all      - Compile both Chinese and English papers"
	@echo "  make cn       - Compile Chinese paper only"
	@echo "  make en       - Compile English paper only"
	@echo "  make clean    - Remove intermediate files"
	@echo "  make cleanall - Remove all generated files including PDFs"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - Chinese paper: XeLaTeX, BibTeX, Chinese fonts"
	@echo "  - English paper: PDFLaTeX or XeLaTeX, BibTeX"
	@echo ""
	@echo "Output files:"
	@echo "  - $(CN_PAPER).pdf (Chinese paper)"
	@echo "  - $(EN_PAPER).pdf (English paper)"
