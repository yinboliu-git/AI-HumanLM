"""
OpenAssistant (OASST1) data collector.
Downloads assistant responses from OpenAssistant/oasst1 (public, HuggingFace).
Provides a third comparison group: AI in human-directed dialogue context.
"""

import json
import os
import random
from datasets import load_dataset
from tqdm import tqdm
from config import RAW_DIR, MIN_TEXT_LENGTH

MAX_ENTRIES = 5000


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Loading OpenAssistant oasst1...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")

    entries = []
    for row in tqdm(ds, desc="OASST1"):
        if len(entries) >= MAX_ENTRIES:
            break
        # Only English assistant messages
        if row.get("lang", "") != "en":
            continue
        if row.get("role", "") != "assistant":
            continue

        text = row.get("text", "")
        if len(text) < MIN_TEXT_LENGTH or len(text) > 5000:
            continue

        entries.append({
            "source": "oasst",
            "subreddit": "openassistant",
            "category": "human_directed_ai",
            "type": "response",
            "id": row.get("message_id", f"oasst_{len(entries)}"),
            "author": "assistant",
            "title": "",
            "text": text,
            "score": int(row.get("rank") or 0),
            "num_comments": 0,
        })

    random.shuffle(entries)
    entries = entries[:MAX_ENTRIES]

    output_path = os.path.join(RAW_DIR, "oasst_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"\nTotal OASST entries: {len(entries)}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
