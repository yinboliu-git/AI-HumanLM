"""
Reddit data collector using HuggingFace datasets.
Sources:
  - HuggingFaceGECLM/REDDIT_comments: philosophy, psychology, relationship_advice
  - derek-thomas/dataset-creator-reddit-amitheasshole: moral judgment
"""

import json
import os
import random
from datasets import load_dataset
from tqdm import tqdm
from config import RAW_DIR, MIN_TEXT_LENGTH

# Map HF subreddit splits to our semantic categories
HF_REDDIT_SUBS = {
    "philosophy": "philosophical",
    "psychology": "psychological",  # not available, use fallback below
    "relationship_advice": "emotional",
    "socialskills": "casual",
    "changemyview": "moral_judgment",
}

# Max entries per subreddit to keep dataset balanced
MAX_PER_SUB = 3000


def load_reddit_comments(sub_name, category):
    """Load comments from HuggingFaceGECLM/REDDIT_comments for a subreddit."""
    print(f"  Loading r/{sub_name} from HuggingFace...")
    try:
        ds = load_dataset(
            "HuggingFaceGECLM/REDDIT_comments",
            name="default",
            split=sub_name,
            streaming=True,
        )
    except Exception as e:
        print(f"  WARNING: Could not load r/{sub_name}: {e}")
        return []

    entries = []
    for i, row in enumerate(tqdm(ds, desc=f"r/{sub_name}", total=MAX_PER_SUB)):
        if i >= MAX_PER_SUB * 3:  # scan up to 3x to find enough good entries
            break
        body = row.get("body", "")
        if not body or len(body) < MIN_TEXT_LENGTH:
            continue
        if body in ("[deleted]", "[removed]"):
            continue

        entries.append({
            "source": "reddit",
            "subreddit": sub_name,
            "category": category,
            "type": "comment",
            "id": row.get("id", f"{sub_name}_{len(entries)}"),
            "author": "[anonymized]",
            "title": "",
            "text": body,
            "score": int(row.get("score", 0)),
            "num_comments": 0,
        })
        if len(entries) >= MAX_PER_SUB:
            break

    return entries


def load_aita():
    """Load AmItheAsshole posts from derek-thomas dataset."""
    print("  Loading r/AmItheAsshole from HuggingFace...")
    try:
        ds = load_dataset(
            "derek-thomas/dataset-creator-reddit-amitheasshole",
            split="train",
        )
    except Exception as e:
        print(f"  WARNING: Could not load AITA: {e}")
        return []

    entries = []
    for row in tqdm(ds, desc="r/AmItheAsshole"):
        content = row.get("content", "")
        if not content or len(content) < MIN_TEXT_LENGTH:
            continue

        entries.append({
            "source": "reddit",
            "subreddit": "AmItheAsshole",
            "category": "moral_judgment",
            "type": "post",
            "id": row.get("id", f"aita_{len(entries)}"),
            "author": "[anonymized]",
            "title": row.get("title", ""),
            "text": content,
            "score": int(row.get("score", 0)),
            "num_comments": 0,
        })
        if len(entries) >= MAX_PER_SUB:
            break

    return entries


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    all_data = []

    # Load from HuggingFaceGECLM/REDDIT_comments
    for sub_name, category in HF_REDDIT_SUBS.items():
        entries = load_reddit_comments(sub_name, category)
        all_data.extend(entries)
        print(f"  r/{sub_name}: {len(entries)} entries")

    # Load AITA dataset
    aita_entries = load_aita()
    all_data.extend(aita_entries)
    print(f"  r/AmItheAsshole: {len(aita_entries)} entries")

    # Shuffle
    random.shuffle(all_data)

    output_path = os.path.join(RAW_DIR, "reddit_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\nTotal Reddit entries: {len(all_data)}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
