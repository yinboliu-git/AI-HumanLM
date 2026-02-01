"""
Metric Calculation Module.
Uses the full (unpaired) corpus for population-level distributional comparison.
Computes:
- Lexical Diversity (TTR)
- Moral Foundations Distribution (empath)
- Emotional Granularity (Shannon Entropy)
- Cognitive Complexity (Flesch-Kincaid, Syntactic Depth)
- Burstiness
- Statistical Tests (KS test, Cohen's d, Wasserstein distance)
"""

import json
import os
import math
import numpy as np
import pandas as pd
import spacy
import textstat
from scipy import stats
from empath import Empath
from tqdm import tqdm
from config import PROCESSED_DIR

nlp = spacy.load("en_core_web_sm")
lexicon = Empath()

MFD_MAPPING = {
    "care": ["sympathy", "suffering", "help", "children", "warmth"],
    "fairness": ["law", "government", "politics", "stealing", "giving"],
    "loyalty": ["military", "family", "friends", "heroic", "pride"],
    "authority": ["leader", "dominance", "power", "worship", "office"],
    "sanctity": ["religion", "divine", "cleaning", "body", "disgust"],
}

EMOTION_CATEGORIES = [
    "joy", "trust", "fear", "surprise", "sadness",
    "disgust", "anger", "anticipation", "love", "optimism",
    "pessimism", "shame", "pride", "envy", "gratitude",
]


def compute_ttr(text):
    tokens = text.lower().split()
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_moral_foundations(text):
    scores = {}
    for foundation, categories in MFD_MAPPING.items():
        total = 0
        for cat in categories:
            result = lexicon.analyze(text, categories=[cat], normalize=True)
            if result and cat in result:
                total += result[cat]
        scores[foundation] = total / len(categories)
    return scores


def compute_emotional_granularity(text):
    emotion_scores = []
    for emo in EMOTION_CATEGORIES:
        result = lexicon.analyze(text, categories=[emo], normalize=True)
        score = result.get(emo, 0) if result else 0
        emotion_scores.append(score)

    total = sum(emotion_scores)
    if total == 0:
        return 0.0

    probs = [s / total for s in emotion_scores if s > 0]
    entropy = -sum(p * math.log2(p) for p in probs)
    return entropy


def compute_syntactic_depth(text):
    doc = nlp(text[:3000])
    depths = []
    for sent in doc.sents:
        def tree_depth(token):
            children = list(token.children)
            if not children:
                return 0
            return 1 + max(tree_depth(c) for c in children)
        depths.append(tree_depth(sent.root))
    if not depths:
        return 0.0
    return float(np.mean(depths))


def compute_flesch_kincaid(text):
    return textstat.flesch_kincaid_grade(text)


def compute_burstiness(text):
    doc = nlp(text[:3000])
    sent_lengths = [len(sent) for sent in doc.sents]
    if len(sent_lengths) < 2:
        return 0.0
    mean_l = np.mean(sent_lengths)
    if mean_l == 0:
        return 0.0
    return float(np.var(sent_lengths) / mean_l)


def analyze_entries(entries, label):
    """Compute all metrics for a list of text entries."""
    results = []
    for entry in tqdm(entries, desc=f"Metrics ({label})"):
        text = entry["text"]
        results.append({
            "source": label,
            "id": entry["id"],
            "subreddit": entry.get("subreddit", ""),
            "ttr": compute_ttr(text),
            **{f"mf_{k}": v for k, v in compute_moral_foundations(text).items()},
            "emotional_granularity": compute_emotional_granularity(text),
            "syntactic_depth": compute_syntactic_depth(text),
            "flesch_kincaid": compute_flesch_kincaid(text),
            "burstiness": compute_burstiness(text),
            "text_length": len(text.split()),
        })
    return results


def run_statistical_tests(df):
    """KS tests, Cohen's d, Wasserstein distance for all metrics (two-way: moltbook vs reddit)."""
    molt_df = df[df["source"] == "moltbook"]
    red_df = df[df["source"] == "reddit"]

    metrics = [
        ("ttr", "Type-Token Ratio"),
        ("emotional_granularity", "Emotional Granularity"),
        ("syntactic_depth", "Syntactic Depth"),
        ("flesch_kincaid", "Flesch-Kincaid Grade"),
        ("burstiness", "Burstiness"),
        ("text_length", "Text Length (words)"),
    ]
    for f in ["care", "fairness", "loyalty", "authority", "sanctity"]:
        metrics.append((f"mf_{f}", f"Moral Foundation: {f.title()}"))

    test_results = []
    for col, name in metrics:
        m_vals = molt_df[col].dropna().values
        r_vals = red_df[col].dropna().values

        ks_stat, ks_p = stats.ks_2samp(m_vals, r_vals)
        pooled_std = np.sqrt((np.var(m_vals) + np.var(r_vals)) / 2)
        cohens_d = (np.mean(m_vals) - np.mean(r_vals)) / pooled_std if pooled_std > 0 else 0
        w_dist = stats.wasserstein_distance(m_vals, r_vals)

        test_results.append({
            "metric": name,
            "molt_mean": float(np.mean(m_vals)),
            "molt_std": float(np.std(m_vals)),
            "red_mean": float(np.mean(r_vals)),
            "red_std": float(np.std(r_vals)),
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_p),
            "cohens_d": float(cohens_d),
            "wasserstein_distance": float(w_dist),
            "significant": ks_p < 0.05,
        })

    return pd.DataFrame(test_results)


def main():
    # Load all three sources
    from config import RAW_DIR
    import random

    sources = {}
    for fname, key in [("moltbook_data.json", "moltbook"),
                       ("reddit_data.json", "reddit"),
                       ("oasst_data.json", "oasst")]:
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            # Sample 5000 for balance
            if len(data) > 5000:
                data = random.sample(data, 5000)
            sources[key] = data
            print(f"{key} entries: {len(data)}")

    # Compute metrics for all groups
    all_results = []
    for source_name, entries in sources.items():
        results = analyze_entries(entries, source_name)
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    csv_path = os.path.join(PROCESSED_DIR, "metrics_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw metrics to {csv_path}")

    # Statistical tests
    print("\n=== Statistical Test Results ===")
    test_df = run_statistical_tests(df)
    test_csv_path = os.path.join(PROCESSED_DIR, "statistical_tests.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(test_df.to_string(index=False))
    print(f"\nSaved to {test_csv_path}")


if __name__ == "__main__":
    main()
