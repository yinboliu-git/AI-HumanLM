"""
Visualization Module.
Generates all figures from the unpaired distributional comparison.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.manifold import TSNE
from config import PROCESSED_DIR, FIGURES_DIR

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.linewidth'] = 0.8

COLORS = {"moltbook": "#E74C3C", "reddit": "#3498DB"}


def plot_kde(df, col, title, xlabel, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    molt = df[df["source"] == "moltbook"][col].dropna()
    red = df[df["source"] == "reddit"][col].dropna()
    sns.kdeplot(molt, ax=ax, color=COLORS["moltbook"],
                label="Moltbook (AI)", fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(red, ax=ax, color=COLORS["reddit"],
                label="Reddit (Human)", fill=True, alpha=0.3, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_radar(df, filename="radar_moral_foundations.pdf"):
    foundations = ["care", "fairness", "loyalty", "authority", "sanctity"]
    labels = [f.title() for f in foundations]

    molt = df[df["source"] == "moltbook"]
    red = df[df["source"] == "reddit"]
    molt_means = [molt[f"mf_{f}"].mean() for f in foundations]
    red_means = [red[f"mf_{f}"].mean() for f in foundations]

    angles = np.linspace(0, 2 * np.pi, len(foundations), endpoint=False).tolist()
    angles += angles[:1]
    molt_means += molt_means[:1]
    red_means += red_means[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, molt_means, 'o-', color=COLORS["moltbook"], linewidth=2, label="Moltbook (AI)")
    ax.fill(angles, molt_means, alpha=0.15, color=COLORS["moltbook"])
    ax.plot(angles, red_means, 's-', color=COLORS["reddit"], linewidth=2, label="Reddit (Human)")
    ax.fill(angles, red_means, alpha=0.15, color=COLORS["reddit"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=False)
    ax.set_title("Moral Foundations Distribution", y=1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_tsne(filename="tsne_projection.pdf"):
    molt_emb = np.load(os.path.join(PROCESSED_DIR, "moltbook_embeddings.npy"))
    red_emb = np.load(os.path.join(PROCESSED_DIR, "reddit_embeddings.npy"))

    max_n = 2000
    if len(molt_emb) > max_n:
        idx = np.random.choice(len(molt_emb), max_n, replace=False)
        molt_emb = molt_emb[idx]
    if len(red_emb) > max_n:
        idx = np.random.choice(len(red_emb), max_n, replace=False)
        red_emb = red_emb[idx]

    all_emb = np.vstack([molt_emb, red_emb])
    labels = np.array([0] * len(molt_emb) + [1] * len(red_emb))

    print("  Computing t-SNE...")
    coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(all_emb)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[labels == 0, 0], coords[labels == 0, 1],
               c=COLORS["moltbook"], alpha=0.4, s=10, label="Moltbook (AI)")
    ax.scatter(coords[labels == 1, 0], coords[labels == 1, 1],
               c=COLORS["reddit"], alpha=0.4, s=10, label="Reddit (Human)")
    ax.legend(frameon=False, markerscale=3)
    ax.set_title("t-SNE Projection of Semantic Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_boxplots(df, filename="metric_boxplots.pdf"):
    metrics = [
        ("ttr", "TTR"),
        ("emotional_granularity", "Emotional\nGranularity"),
        ("syntactic_depth", "Syntactic\nDepth"),
        ("flesch_kincaid", "Flesch-Kincaid\nGrade"),
        ("burstiness", "Burstiness"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    for ax, (col, label) in zip(axes, metrics):
        molt_vals = df[df["source"] == "moltbook"][col].dropna().values
        red_vals = df[df["source"] == "reddit"][col].dropna().values
        bp = ax.boxplot([molt_vals, red_vals],
                        tick_labels=["AI", "Human"],
                        patch_artist=True, widths=0.6)
        bp["boxes"][0].set_facecolor(COLORS["moltbook"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(COLORS["reddit"])
        bp["boxes"][1].set_alpha(0.6)
        ax.set_title(label, fontsize=10)

    plt.suptitle("Metric Comparison: AI vs. Human", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_stats_table(filename="stats_summary.pdf"):
    """Render statistical tests as a publication-quality table figure."""
    csv_path = os.path.join(PROCESSED_DIR, "statistical_tests.csv")
    if not os.path.exists(csv_path):
        return
    sdf = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    cols = ["metric", "molt_mean", "red_mean", "ks_statistic", "ks_p_value", "cohens_d", "wasserstein_distance"]
    headers = ["Metric", "AI Mean", "Human Mean", "KS Stat", "p-value", "Cohen's d", "W₁ Dist"]
    cell_text = []
    for _, row in sdf.iterrows():
        cell_text.append([
            row["metric"],
            f"{row['molt_mean']:.4f}",
            f"{row['red_mean']:.4f}",
            f"{row['ks_statistic']:.4f}",
            f"{row['ks_p_value']:.2e}",
            f"{row['cohens_d']:.4f}",
            f"{row['wasserstein_distance']:.4f}",
        ])
    table = ax.table(cellText=cell_text, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    csv_path = os.path.join(PROCESSED_DIR, "metrics_raw.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} metric rows")

    print("\nGenerating figures...")
    plot_kde(df, "emotional_granularity", "Emotional Granularity Distribution",
             "Shannon Entropy (bits)", "kde_emotional_granularity.pdf")
    plot_kde(df, "ttr", "Lexical Diversity (Type-Token Ratio)", "TTR", "kde_ttr.pdf")
    plot_kde(df, "flesch_kincaid", "Reading Complexity Distribution",
             "Flesch-Kincaid Grade Level", "kde_flesch_kincaid.pdf")
    plot_kde(df, "burstiness", "Sentence Length Burstiness",
             "Burstiness (Var/Mean)", "kde_burstiness.pdf")
    plot_radar(df)
    plot_tsne()
    plot_boxplots(df)
    plot_stats_table()

    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
