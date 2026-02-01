"""
Advanced Visualization Module.
Generates figures for expanded three-way analysis.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from config import PROCESSED_DIR, FIGURES_DIR

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.linewidth'] = 0.8

COLORS = {"moltbook": "#E74C3C", "reddit": "#3498DB", "oasst": "#2ECC71"}
LABELS = {"moltbook": "Moltbook (Auto. AI)", "reddit": "Reddit (Human)", "oasst": "OASST (Dir. AI)"}
LABELS_LONG = {"moltbook": "Moltbook\n(Autonomous AI)", "reddit": "Reddit\n(Human)", "oasst": "OASST\n(Directed AI)"}


def plot_three_way_bar(df, metrics, metric_labels, title, filename):
    """Grouped bar chart for three-way comparison."""
    sources = ["moltbook", "reddit", "oasst"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, src in enumerate(sources):
        vals = [df[df["source"] == src][m].mean() for m in metrics]
        ax.bar(x + i * width, vals, width, label=LABELS_LONG[src], color=COLORS[src], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_pos_comparison(df, filename="pos_distribution.pdf"):
    """POS tag distribution comparison."""
    pos_tags = ["pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV", "pos_PRON",
                "pos_ADP", "pos_DET", "pos_AUX", "pos_PUNCT"]
    pos_labels = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "DET", "AUX", "PUNCT"]
    sources = ["moltbook", "reddit", "oasst"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pos_tags))
    width = 0.25

    for i, src in enumerate(sources):
        vals = [df[df["source"] == src][p].mean() for p in pos_tags]
        ax.bar(x + i * width, vals, width, label=LABELS_LONG[src], color=COLORS[src], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(pos_labels, fontsize=9)
    ax.set_ylabel("Proportion")
    ax.set_title("Part-of-Speech Distribution")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_distinct_n(df, filename="distinct_n.pdf"):
    """Distinct-n comparison."""
    metrics = ["distinct_1", "distinct_2", "distinct_3"]
    labels = ["Distinct-1", "Distinct-2", "Distinct-3"]
    plot_three_way_bar(df, metrics, labels, "N-gram Diversity (Distinct-N)", filename)


def plot_dependency_metrics(df, filename="dependency_metrics.pdf"):
    """Dependency distance and direction."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sources = ["moltbook", "reddit", "oasst"]

    # Mean dependency distance - KDE
    ax = axes[0]
    for src in sources:
        vals = df[df["source"] == src]["mean_dep_distance"].dropna()
        sns.kdeplot(vals, ax=ax, color=COLORS[src], label=LABELS[src],
                    fill=True, alpha=0.2, linewidth=2)
    ax.set_title("Mean Dependency Distance")
    ax.set_xlabel("Distance (tokens)")
    ax.legend(frameon=False, fontsize=7)

    # Left arc ratio
    ax = axes[1]
    for src in sources:
        vals = df[df["source"] == src]["left_arc_ratio"].dropna()
        sns.kdeplot(vals, ax=ax, color=COLORS[src], fill=True, alpha=0.2, linewidth=2)
    ax.set_title("Left Arc Ratio")
    ax.set_xlabel("Proportion left-directed")

    # Dep types entropy
    ax = axes[2]
    for src in sources:
        vals = df[df["source"] == src]["dep_types_entropy"].dropna()
        sns.kdeplot(vals, ax=ax, color=COLORS[src], fill=True, alpha=0.2, linewidth=2)
    ax.set_title("Dependency Type Entropy")
    ax.set_xlabel("Entropy (bits)")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_semantic_homogeneity(filename="semantic_homogeneity.pdf"):
    """Bar chart of within-group pairwise similarity."""
    homo_path = os.path.join(PROCESSED_DIR, "semantic_homogeneity.json")
    if not os.path.exists(homo_path):
        return
    with open(homo_path) as f:
        homo = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Pairwise similarity
    ax = axes[0]
    sources = list(homo.keys())
    means = [homo[s]["mean_pairwise_sim"] for s in sources]
    stds = [homo[s]["std_pairwise_sim"] for s in sources]
    colors = [COLORS.get(s, "#999999") for s in sources]
    xlabels = [LABELS.get(s, s) for s in sources]
    ax.bar(xlabels, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax.set_ylabel("Mean Pairwise Cosine Similarity")
    ax.set_title("Within-Group Semantic Homogeneity")

    # Explained variance at 10D
    ax = axes[1]
    ev = [homo[s]["explained_var_10d"] for s in sources]
    ax.bar(xlabels, ev, color=colors, alpha=0.8)
    ax.set_ylabel("Explained Variance (first 10 PCs)")
    ax.set_title("Intrinsic Dimensionality")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_hapax_comparison(df, filename="hapax_ratio.pdf"):
    """KDE of hapax legomena ratio."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for src in ["moltbook", "reddit", "oasst"]:
        vals = df[df["source"] == src]["hapax_ratio"].dropna()
        sns.kdeplot(vals, ax=ax, color=COLORS[src],
                    label=LABELS[src],
                    fill=True, alpha=0.2, linewidth=2)
    ax.set_xlabel("Hapax Legomena Ratio")
    ax.set_ylabel("Density")
    ax.set_title("Hapax Legomena Ratio Distribution")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    csv_path = os.path.join(PROCESSED_DIR, "advanced_metrics.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from advanced metrics")

    print("\nGenerating advanced figures...")
    plot_pos_comparison(df)
    plot_distinct_n(df)
    plot_dependency_metrics(df)
    plot_semantic_homogeneity()
    plot_hapax_comparison(df)

    print("\nAll advanced figures generated!")


if __name__ == "__main__":
    main()
