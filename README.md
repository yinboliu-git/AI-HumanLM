# Autonomy Shapes Language

**A Comparative Linguistic Topology of Autonomous AI, Directed AI, and Human Discourse**

[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2026.XXXXX)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Moltbook-yellow)](https://huggingface.co/datasets/qugemingzi/moltbook-ai-agent-posts)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

This repository contains the code and data for our research comparing linguistic patterns across three types of discourse:
- **Autonomous AI** (Moltbook) - AI agents operating independently
- **Directed AI** (OpenAssistant) - AI responding to human instructions
- **Human Discourse** (Reddit) - Organic human conversations

### 🔑 Key Findings

1. **Autonomy-Diversity Paradox**: Autonomous AI exhibits higher lexical diversity but **3.1× higher semantic homogeneity** than humans
2. **Syntactic Shallowness**: The strongest discriminator (Cohen's d = -0.87)
3. **Corpus-Level Attribution**: Achieves **F1 = 0.97** with 36 linguistic features

## 📁 Project Structure

```
.
├── src/                      # Source code
│   ├── crawl_moltbook.py     # Moltbook data collection
│   ├── crawl_reddit.py       # Reddit data collection
│   ├── preprocess_align.py   # Data preprocessing & alignment
│   ├── metrics.py            # Basic linguistic metrics
│   ├── advanced_metrics.py   # Advanced metrics computation
│   ├── corpus_attribution.py # Classification experiments
│   ├── visualize.py          # Visualization scripts
│   └── advanced_viz.py       # Advanced visualizations
├── data/                     # Data directory (see data/README.md)
├── requirements.txt          # Python dependencies
├── Makefile                  # Build automation
└── LICENSE
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/autonomy-shapes-language.git
cd autonomy-shapes-language

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Download Data

```python
from datasets import load_dataset

# Moltbook (Autonomous AI) - 25,445 posts
moltbook = load_dataset("qugemingzi/moltbook-ai-agent-posts")

# OASST (Directed AI)
oasst = load_dataset("OpenAssistant/oasst1")
```

### Run Analysis Pipeline

```bash
# Full pipeline
make all

# Or step by step:
python src/preprocess_align.py      # Preprocess and align data
python src/metrics.py               # Compute basic metrics
python src/advanced_metrics.py      # Compute advanced metrics
python src/corpus_attribution.py    # Run classification
python src/visualize.py             # Generate figures
```

## 📊 Metrics

We analyze **11 primary metrics** across **4 dimensions**:

| Dimension | Metrics |
|-----------|---------|
| **Morphosyntax** | POS distribution, dependency distance, left-arc ratio, dependency entropy, syntactic depth |
| **Lexical Diversity** | TTR, hapax ratio, Distinct-1/2/3 |
| **Affective Profile** | Emotional granularity, Moral Foundations (5 axes) |
| **Semantic Topology** | Semantic homogeneity, intrinsic dimensionality |

## 📈 Results Summary

| Metric | Moltbook (Auto) | OASST (Directed) | Reddit (Human) |
|--------|-----------------|------------------|----------------|
| Distinct-1 ↑ | **0.789** | 0.685 | 0.748 |
| Semantic Homogeneity ↓ | 0.252 | **0.036** | 0.077 |
| Syntactic Depth ↑ | 3.52 | 3.89 | **4.58** |

## 📦 Dataset

Our curated Moltbook dataset is available on HuggingFace:

🤗 **[qugemingzi/moltbook-ai-agent-posts](https://huggingface.co/datasets/qugemingzi/moltbook-ai-agent-posts)**

- **25,445 posts** from autonomous AI agents
- Includes: content, metadata, engagement metrics
- License: CC-BY-4.0

## 📝 Citation

If you use this code or dataset, please cite our paper:

```bibtex
@article{liu2026autonomy,
  title={Autonomy Shapes Language: A Comparative Linguistic Topology of Autonomous AI, Directed AI, and Human Discourse},
  author={Liu, Yinbo and Gao, Handi and Ding, Yue},
  journal={ResearchGate preprint 10.13140/RG.2.2.26381.40165},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Moltbook community for AI agent data
- OpenAssistant for the OASST dataset
- Reddit communities for human discourse data

## 📧 Contact

For questions or issues, please open an issue on this repository
