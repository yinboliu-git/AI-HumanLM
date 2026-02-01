"""
LLM-as-a-Judge Evaluation Module.
Selects 100 random aligned pairs and uses an LLM to evaluate
Theory of Mind depth, empathy, and cognitive sophistication.

Supports: OpenAI API (GPT-4) or Anthropic API (Claude).
Set the appropriate API key in environment variables.
"""

import json
import os
import random
import time
from tqdm import tqdm
from config import PROCESSED_DIR

# Judge system prompt
JUDGE_SYSTEM_PROMPT = """You are an expert psychologist and cognitive scientist.
You will receive two texts labeled "Text A" and "Text B". One is written by an AI agent,
the other by a human. You do NOT know which is which.

Evaluate BOTH texts on the following dimensions (score 1-10 each):

1. **Theory of Mind Depth**: How well does the author demonstrate understanding of
   others' mental states, beliefs, and perspectives?
2. **Emotional Nuance**: Does the text show mixed, complex emotions or only simple/binary ones?
3. **Moral Reasoning Complexity**: Does the author engage with moral ambiguity, or
   default to simple good/bad judgments?
4. **Logical Coherence**: How well-structured is the argumentation?
5. **Authenticity**: Does the text feel genuine and spontaneous, or formulaic?

Respond in JSON format:
{
  "text_a": {"theory_of_mind": X, "emotional_nuance": X, "moral_reasoning": X, "logical_coherence": X, "authenticity": X},
  "text_b": {"theory_of_mind": X, "emotional_nuance": X, "moral_reasoning": X, "logical_coherence": X, "authenticity": X},
  "which_is_more_human": "A" or "B",
  "confidence": X (1-10),
  "reasoning": "brief explanation"
}"""


def create_judge_prompt(text_a, text_b):
    return f"""Analyze these two texts:

**Text A:**
{text_a[:2000]}

**Text B:**
{text_b[:2000]}

Provide your evaluation in the specified JSON format."""


def judge_with_anthropic(prompt, system_prompt):
    """Use Anthropic Claude API as judge."""
    import anthropic
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def judge_with_openai(prompt, system_prompt):
    """Use OpenAI GPT-4 API as judge."""
    from openai import OpenAI
    client = OpenAI()  # uses OPENAI_API_KEY env var
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def parse_judge_response(response_text):
    """Extract JSON from judge response."""
    import re
    # Try to find JSON block
    match = re.search(r'\{[\s\S]*\}', response_text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def main():
    corpus_path = os.path.join(PROCESSED_DIR, "aligned_corpus.json")
    with open(corpus_path, "r", encoding="utf-8") as f:
        aligned_pairs = json.load(f)

    # Select 100 random pairs
    sample_size = min(100, len(aligned_pairs))
    sample = random.sample(aligned_pairs, sample_size)

    # Determine which API to use
    api_backend = "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        api_backend = "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        api_backend = "anthropic"

    judge_fn = judge_with_anthropic if api_backend == "anthropic" else judge_with_openai
    print(f"Using {api_backend} as judge backend")

    results = []
    for i, pair in enumerate(tqdm(sample, desc="Judging pairs")):
        # Randomly swap order to avoid position bias
        swap = random.random() > 0.5
        if swap:
            text_a, text_b = pair["reddit"]["text"], pair["moltbook"]["text"]
            label_a, label_b = "reddit", "moltbook"
        else:
            text_a, text_b = pair["moltbook"]["text"], pair["reddit"]["text"]
            label_a, label_b = "moltbook", "reddit"

        prompt = create_judge_prompt(text_a, text_b)

        try:
            response = judge_fn(prompt, JUDGE_SYSTEM_PROMPT)
            parsed = parse_judge_response(response)

            if parsed:
                parsed["_meta"] = {
                    "pair_index": i,
                    "text_a_source": label_a,
                    "text_b_source": label_b,
                    "swapped": swap,
                    "similarity": pair["similarity"],
                }
                results.append(parsed)
        except Exception as e:
            print(f"  Error on pair {i}: {e}")
            time.sleep(2)
            continue

        # Rate limiting
        time.sleep(1)

    output_path = os.path.join(PROCESSED_DIR, "llm_judge_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Aggregate scores
    if results:
        molt_scores = {k: [] for k in ["theory_of_mind", "emotional_nuance", "moral_reasoning", "logical_coherence", "authenticity"]}
        red_scores = {k: [] for k in molt_scores}
        correct_identification = 0

        for r in results:
            meta = r.get("_meta", {})
            a_src = meta.get("text_a_source", "")
            b_src = meta.get("text_b_source", "")

            for dim in molt_scores:
                if a_src == "moltbook":
                    molt_scores[dim].append(r.get("text_a", {}).get(dim, 0))
                    red_scores[dim].append(r.get("text_b", {}).get(dim, 0))
                else:
                    molt_scores[dim].append(r.get("text_b", {}).get(dim, 0))
                    red_scores[dim].append(r.get("text_a", {}).get(dim, 0))

            # Check if judge correctly identified the human
            human_pick = r.get("which_is_more_human", "")
            if (human_pick == "A" and a_src == "reddit") or (human_pick == "B" and b_src == "reddit"):
                correct_identification += 1

        print("\n=== LLM Judge Aggregate Results ===")
        print(f"{'Dimension':<25} {'Moltbook (AI)':>15} {'Reddit (Human)':>15}")
        print("-" * 60)
        for dim in molt_scores:
            m_mean = sum(molt_scores[dim]) / len(molt_scores[dim]) if molt_scores[dim] else 0
            r_mean = sum(red_scores[dim]) / len(red_scores[dim]) if red_scores[dim] else 0
            print(f"{dim:<25} {m_mean:>15.2f} {r_mean:>15.2f}")

        accuracy = correct_identification / len(results) * 100
        print(f"\nHuman identification accuracy: {accuracy:.1f}%")

    print(f"\nSaved {len(results)} judge results to {output_path}")


if __name__ == "__main__":
    main()
