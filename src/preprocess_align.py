"""
Preprocessing & Corpus Construction Pipeline.
1. Clean text (remove HTML, URLs, excessive whitespace)
2. Encode with SBERT (all-mpnet-base-v2)
3. Two-track approach:
   a) Best-match paired corpus (top-1 nearest neighbor per Moltbook entry)
   b) Full unpaired corpus for population-level distributional comparison
4. Output aligned_corpus.json (paired) and full_corpus.json (unpaired)
"""

import json
import os
import re
import random
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import RAW_DIR, PROCESSED_DIR, SBERT_MODEL


def clean_text(text):
    """Remove HTML tags, URLs, and normalize whitespace."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # markdown links
    text = re.sub(r'[*_~`#>]', '', text)  # markdown formatting
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data(filename):
    path = os.path.join(RAW_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading data...")
    moltbook = load_data("moltbook_data.json")
    reddit = load_data("reddit_data.json")

    print(f"  Moltbook entries: {len(moltbook)}")
    print(f"  Reddit entries: {len(reddit)}")

    # Clean text
    print("Cleaning text...")
    for entry in moltbook + reddit:
        entry["text_clean"] = clean_text(entry["text"])

    # Filter out too-short cleaned texts
    moltbook = [e for e in moltbook if len(e["text_clean"]) >= 50]
    reddit = [e for e in reddit if len(e["text_clean"]) >= 50]

    print(f"  After cleaning - Moltbook: {len(moltbook)}, Reddit: {len(reddit)}")

    # Encode with SBERT
    print(f"Loading SBERT model ({SBERT_MODEL})...")
    model = SentenceTransformer(SBERT_MODEL)

    print("Encoding Moltbook texts...")
    molt_texts = [e["text_clean"] for e in moltbook]
    molt_embeddings = model.encode(molt_texts, show_progress_bar=True, batch_size=64)

    print("Encoding Reddit texts...")
    red_texts = [e["text_clean"] for e in reddit]
    red_embeddings = model.encode(red_texts, show_progress_bar=True, batch_size=64)

    # Save embeddings for t-SNE etc.
    np.save(os.path.join(PROCESSED_DIR, "moltbook_embeddings.npy"), molt_embeddings)
    np.save(os.path.join(PROCESSED_DIR, "reddit_embeddings.npy"), red_embeddings)

    # ===== Track A: Best-match paired corpus =====
    print("\n--- Track A: Best-match pairing (top-1 nearest neighbor) ---")
    aligned_pairs = []
    batch_size = 100
    for i in tqdm(range(0, len(molt_embeddings), batch_size), desc="Pairing"):
        batch_emb = molt_embeddings[i:i + batch_size]
        sims = cosine_similarity(batch_emb, red_embeddings)
        for j, sim_row in enumerate(sims):
            molt_idx = i + j
            top_idx = int(np.argmax(sim_row))
            top_sim = float(sim_row[top_idx])
            aligned_pairs.append({
                "moltbook": {
                    "id": moltbook[molt_idx]["id"],
                    "text": moltbook[molt_idx]["text_clean"],
                    "author": moltbook[molt_idx]["author"],
                    "subreddit": moltbook[molt_idx]["subreddit"],
                    "score": moltbook[molt_idx]["score"],
                },
                "reddit": {
                    "id": reddit[top_idx]["id"],
                    "text": reddit[top_idx]["text_clean"],
                    "author": reddit[top_idx]["author"],
                    "subreddit": reddit[top_idx]["subreddit"],
                    "category": reddit[top_idx]["category"],
                    "score": reddit[top_idx]["score"],
                },
                "similarity": top_sim,
            })

    # Sort by similarity and report distribution
    aligned_pairs.sort(key=lambda x: x["similarity"], reverse=True)
    sims_arr = [p["similarity"] for p in aligned_pairs]
    print(f"  Total pairs: {len(aligned_pairs)}")
    print(f"  Similarity: mean={np.mean(sims_arr):.3f}, "
          f"median={np.median(sims_arr):.3f}, "
          f"max={np.max(sims_arr):.3f}, min={np.min(sims_arr):.3f}")
    for thresh in [0.8, 0.7, 0.6, 0.5]:
        n = sum(1 for s in sims_arr if s >= thresh)
        print(f"  Pairs with sim >= {thresh}: {n}")

    output_path = os.path.join(PROCESSED_DIR, "aligned_corpus.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aligned_pairs, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {output_path}")

    # ===== Track B: Full unpaired corpus for distributional comparison =====
    print("\n--- Track B: Full unpaired corpus ---")

    # Balance sample sizes for fair comparison
    n_samples = min(len(moltbook), len(reddit), 5000)
    molt_sample = random.sample(range(len(moltbook)), n_samples)
    red_sample = random.sample(range(len(reddit)), n_samples)

    full_corpus = {
        "moltbook": [{
            "id": moltbook[i]["id"],
            "text": moltbook[i]["text_clean"],
            "author": moltbook[i]["author"],
            "subreddit": moltbook[i]["subreddit"],
            "score": moltbook[i]["score"],
        } for i in molt_sample],
        "reddit": [{
            "id": reddit[i]["id"],
            "text": reddit[i]["text_clean"],
            "author": reddit[i]["author"],
            "subreddit": reddit[i]["subreddit"],
            "category": reddit[i]["category"],
            "score": reddit[i]["score"],
        } for i in red_sample],
    }

    full_path = os.path.join(PROCESSED_DIR, "full_corpus.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_corpus, f, ensure_ascii=False, indent=2)
    print(f"  Moltbook sample: {len(full_corpus['moltbook'])}")
    print(f"  Reddit sample: {len(full_corpus['reddit'])}")
    print(f"  Saved to {full_path}")


if __name__ == "__main__":
    main()
