.PHONY: all data moltbook reddit align metrics visualize judge paper clean

PYTHON = python3
SRC = src
PAPER = paper

all: data metrics visualize paper

# Data collection
data: moltbook reddit align

moltbook:
	cd $(SRC) && $(PYTHON) crawl_moltbook.py

reddit:
	cd $(SRC) && $(PYTHON) crawl_reddit.py

align:
	cd $(SRC) && $(PYTHON) preprocess_align.py

# Analysis
metrics:
	cd $(SRC) && $(PYTHON) metrics.py

visualize:
	cd $(SRC) && $(PYTHON) visualize.py

judge:
	cd $(SRC) && $(PYTHON) llm_judge.py

# Paper
paper:
	cp icml2026/icml2026.sty $(PAPER)/
	cp icml2026/icml2026.bst $(PAPER)/
	cp icml2026/algorithm.sty $(PAPER)/
	cp icml2026/algorithmic.sty $(PAPER)/
	cp icml2026/fancyhdr.sty $(PAPER)/
	cd $(PAPER) && pdflatex main && bibtex main && pdflatex main && pdflatex main

clean:
	rm -f $(PAPER)/*.aux $(PAPER)/*.bbl $(PAPER)/*.blg $(PAPER)/*.log $(PAPER)/*.out $(PAPER)/*.pdf
	rm -f $(PAPER)/icml2026.sty $(PAPER)/icml2026.bst $(PAPER)/algorithm.sty $(PAPER)/algorithmic.sty $(PAPER)/fancyhdr.sty
