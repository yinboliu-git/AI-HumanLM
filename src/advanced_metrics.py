"""
Advanced Metrics Module.
Extends analysis with:
1. POS tag distribution (morphological analysis)
2. Dependency arc length & direction (syntactic optimization)
3. Vocabulary richness (Zipf, hapax, distinct-n)
4. Within-group semantic homogeneity (pairwise sim, intrinsic dimensionality)
5. Three-way comparison: Moltbook / OASST / Reddit
"""

import json
import os
import re
import math
import numpy as np
import pandas as pd
import spacy
from collections import Counter
from scipy import stats
from tqdm import tqdm
from config import RAW_DIR, PROCESSED_DIR

nlp = spacy.load("en_core_web_sm")


def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[*_~`#>]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ===================== POS ANALYSIS =====================

def compute_pos_distribution(doc):
    """Return normalized POS tag counts."""
    counts = Counter(token.pos_ for token in doc)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {pos: count / total for pos, count in counts.items()}


# ===================== DEPENDENCY ANALYSIS =====================

def compute_dependency_metrics(doc):
    """Compute dependency arc lengths, direction ratio, and DLM."""
    arc_lengths = []
    left_arcs = 0
    right_arcs = 0
    dep_types = Counter()

    for token in doc:
        if token.dep_ == "ROOT":
            continue
        dist = abs(token.i - token.head.i)
        arc_lengths.append(dist)
        dep_types[token.dep_] += 1
        if token.i < token.head.i:
            left_arcs += 1
        else:
            right_arcs += 1

    if not arc_lengths:
        return {
            "mean_dep_distance": 0, "std_dep_distance": 0,
            "left_arc_ratio": 0, "dep_types_entropy": 0,
        }

    total_arcs = left_arcs + right_arcs
    dep_total = sum(dep_types.values())
    probs = [c / dep_total for c in dep_types.values()]
    dep_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    return {
        "mean_dep_distance": float(np.mean(arc_lengths)),
        "std_dep_distance": float(np.std(arc_lengths)),
        "left_arc_ratio": left_arcs / total_arcs if total_arcs > 0 else 0,
        "dep_types_entropy": dep_entropy,
    }


# ===================== VOCABULARY RICHNESS =====================

def compute_vocab_richness(text):
    """Compute Zipf coefficient, hapax ratio, distinct-n."""
    tokens = text.lower().split()
    if len(tokens) < 10:
        return {"hapax_ratio": 0, "distinct_1": 0, "distinct_2": 0, "distinct_3": 0}

    freq = Counter(tokens)
    hapax = sum(1 for w, c in freq.items() if c == 1)
    hapax_ratio = hapax / len(freq) if freq else 0

    # Distinct-n: unique n-grams / total n-grams
    def distinct_n(toks, n):
        ngrams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
        if not ngrams:
            return 0
        return len(set(ngrams)) / len(ngrams)

    return {
        "hapax_ratio": hapax_ratio,
        "distinct_1": distinct_n(tokens, 1),
        "distinct_2": distinct_n(tokens, 2),
        "distinct_3": distinct_n(tokens, 3),
    }


# ===================== MAIN ANALYSIS =====================

def analyze_three_way(sources):
    """Run advanced metrics on all three sources."""
    all_results = []

    for source_name, entries in sources.items():
        print(f"\n--- Analyzing {source_name} ({len(entries)} entries) ---")

        pos_agg = Counter()
        n_docs = 0

        for entry in tqdm(entries, desc=f"Advanced ({source_name})"):
            text = clean_text(entry["text"])
            if len(text) < 50:
                continue

            doc = nlp(text[:3000])

            # POS
            pos_dist = compute_pos_distribution(doc)
            for pos, ratio in pos_dist.items():
                pos_agg[pos] += ratio
            n_docs += 1

            # Dependency
            dep_metrics = compute_dependency_metrics(doc)

            # Vocabulary
            vocab_metrics = compute_vocab_richness(text)

            all_results.append({
                "source": source_name,
                "id": entry["id"],
                **dep_metrics,
                **vocab_metrics,
                **{f"pos_{pos}": ratio for pos, ratio in pos_dist.items()},
            })

        # Normalize POS aggregates
        if n_docs > 0:
            for pos in pos_agg:
                pos_agg[pos] /= n_docs
            print(f"  Top POS: {pos_agg.most_common(8)}")

    return pd.DataFrame(all_results)


def compute_semantic_homogeneity():
    """Compute within-group pairwise similarity and intrinsic dimensionality."""
    results = {}
    for name in ["moltbook", "oasst", "reddit"]:
        emb_path = os.path.join(PROCESSED_DIR, f"{name}_embeddings.npy")
        if not os.path.exists(emb_path):
            continue
        emb = np.load(emb_path)
        # Subsample for speed
        if len(emb) > 1000:
            idx = np.random.choice(len(emb), 1000, replace=False)
            emb = emb[idx]

        # Pairwise cosine similarity (sample 5000 pairs)
        n = len(emb)
        sims = []
        for _ in range(5000):
            i, j = np.random.randint(0, n, 2)
            if i != j:
                cos = np.dot(emb[i], emb[j]) / (np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]))
                sims.append(cos)

        # Intrinsic dimensionality via PCA explained variance
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, len(emb)))
        pca.fit(emb)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        dim_90 = int(np.searchsorted(cumvar, 0.9)) + 1
        dim_95 = int(np.searchsorted(cumvar, 0.95)) + 1

        results[name] = {
            "mean_pairwise_sim": float(np.mean(sims)),
            "std_pairwise_sim": float(np.std(sims)),
            "intrinsic_dim_90": dim_90,
            "intrinsic_dim_95": dim_95,
            "explained_var_10d": float(cumvar[9]) if len(cumvar) > 9 else float(cumvar[-1]),
        }
        print(f"\n{name}:")
        print(f"  Mean pairwise similarity: {results[name]['mean_pairwise_sim']:.4f}")
        print(f"  Intrinsic dim (90% var): {dim_90}, (95% var): {dim_95}")

    return results


def run_three_way_tests(df):
    """Run KW tests for three-way comparisons."""
    metrics = [
        "mean_dep_distance", "std_dep_distance", "left_arc_ratio",
        "dep_types_entropy", "hapax_ratio", "distinct_1", "distinct_2", "distinct_3",
    ]

    test_results = []
    sources = df["source"].unique()
    for m in metrics:
        groups = [df[df["source"] == s][m].dropna().values for s in sources]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        h_stat, p_val = stats.kruskal(*groups)

        means = {s: df[df["source"] == s][m].mean() for s in sources}
        test_results.append({
            "metric": m,
            **{f"mean_{s}": means.get(s, 0) for s in sources},
            "kruskal_H": float(h_stat),
            "kruskal_p": float(p_val),
            "significant": p_val < 0.05,
        })

    return pd.DataFrame(test_results)


def run_pairwise_posthoc(df):
    """Run pairwise Mann-Whitney U tests with Bonferroni correction."""
    metrics = [
        "mean_dep_distance", "left_arc_ratio",
        "dep_types_entropy", "hapax_ratio", "distinct_1", "distinct_2", "distinct_3",
    ]
    pairs = [("moltbook", "reddit"), ("moltbook", "oasst"), ("oasst", "reddit")]
    results = []
    for m in metrics:
        for s1, s2 in pairs:
            g1 = df[df["source"] == s1][m].dropna().values
            g2 = df[df["source"] == s2][m].dropna().values
            if len(g1) < 2 or len(g2) < 2:
                continue
            u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            p_bonf = min(p_val * 3, 1.0)  # Bonferroni for 3 comparisons
            # Rank-biserial r as effect size
            n1, n2 = len(g1), len(g2)
            r_rb = 1 - (2 * u_stat) / (n1 * n2)
            results.append({
                "metric": m, "pair": f"{s1} vs {s2}",
                "U": float(u_stat), "p_raw": float(p_val),
                "p_bonferroni": float(p_bonf),
                "rank_biserial_r": float(r_rb),
                "significant": p_bonf < 0.05,
            })
    return pd.DataFrame(results)


def main():
    # Load all three sources
    sources = {}
    for fname, key in [("moltbook_data.json", "moltbook"),
                       ("reddit_data.json", "reddit"),
                       ("oasst_data.json", "oasst")]:
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            # Sample for efficiency
            import random
            if len(data) > 3000:
                data = random.sample(data, 3000)
            sources[key] = data

    print(f"Sources: {', '.join(f'{k}({len(v)})' for k, v in sources.items())}")

    # Advanced metrics
    df = analyze_three_way(sources)
    csv_path = os.path.join(PROCESSED_DIR, "advanced_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved advanced metrics to {csv_path}")

    # Three-way statistical tests
    print("\n=== Three-Way Statistical Tests (Kruskal-Wallis) ===")
    test_df = run_three_way_tests(df)
    test_path = os.path.join(PROCESSED_DIR, "advanced_tests.csv")
    test_df.to_csv(test_path, index=False)
    print(test_df.to_string(index=False))

    # Pairwise post-hoc tests
    print("\n=== Pairwise Post-Hoc Tests (Mann-Whitney U, Bonferroni) ===")
    posthoc_df = run_pairwise_posthoc(df)
    posthoc_path = os.path.join(PROCESSED_DIR, "posthoc_tests.csv")
    posthoc_df.to_csv(posthoc_path, index=False)
    print(posthoc_df.to_string(index=False))

    # Semantic homogeneity
    print("\n=== Semantic Homogeneity ===")
    homo = compute_semantic_homogeneity()
    homo_path = os.path.join(PROCESSED_DIR, "semantic_homogeneity.json")
    with open(homo_path, "w") as f:
        json.dump(homo, f, indent=2)

    print(f"\nAll advanced analyses complete!")


if __name__ == "__main__":
    main()
