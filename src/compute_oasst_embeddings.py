"""
Generate OASST embeddings for semantic homogeneity calculation.
"""

import json
import os
import re
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import RAW_DIR, PROCESSED_DIR, SBERT_MODEL


def clean_text(text):
    """Remove HTML tags, URLs, and normalize whitespace."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[*_~`#>]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    print("Loading OASST data...")
    oasst_path = os.path.join(RAW_DIR, "oasst_data.json")
    with open(oasst_path, "r", encoding="utf-8") as f:
        oasst = json.load(f)

    print(f"  OASST entries: {len(oasst)}")

    # Clean text
    print("Cleaning text...")
    for entry in oasst:
        entry["text_clean"] = clean_text(entry["text"])

    # Filter out too-short texts
    oasst = [e for e in oasst if len(e["text_clean"]) >= 50]
    print(f"  After cleaning: {len(oasst)}")

    # Load SBERT model
    print(f"Loading SBERT model ({SBERT_MODEL})...")
    model = SentenceTransformer(SBERT_MODEL)

    # Encode
    print("Encoding OASST texts...")
    oasst_texts = [e["text_clean"] for e in oasst]
    oasst_embeddings = model.encode(oasst_texts, show_progress_bar=True, batch_size=64)

    # Save embeddings
    output_path = os.path.join(PROCESSED_DIR, "oasst_embeddings.npy")
    np.save(output_path, oasst_embeddings)
    print(f"Saved embeddings to {output_path}")
    print(f"Shape: {oasst_embeddings.shape}")


if __name__ == "__main__":
    main()
