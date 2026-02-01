"""
Main pipeline orchestrator.
Run all steps sequentially: crawl -> preprocess -> metrics -> visualize.
"""

import subprocess
import sys
import os

STEPS = [
    ("Step 1: Collect Moltbook data", "crawl_moltbook.py"),
    ("Step 2: Collect Reddit data", "crawl_reddit.py"),
    ("Step 3: Preprocess & Align", "preprocess_align.py"),
    ("Step 4: Compute Metrics", "metrics.py"),
    ("Step 5: Generate Visualizations", "visualize.py"),
    # Step 6 (LLM Judge) is optional and requires API key
]

def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))

    for desc, script in STEPS:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}\n")

        result = subprocess.run(
            [sys.executable, os.path.join(src_dir, script)],
            cwd=src_dir,
        )
        if result.returncode != 0:
            print(f"\nERROR: {script} failed with return code {result.returncode}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}")
    print("\nOptional: Run LLM-as-a-Judge (requires ANTHROPIC_API_KEY or OPENAI_API_KEY):")
    print(f"  python3 {os.path.join(src_dir, 'llm_judge.py')}")
    print("\nTo compile the paper:")
    print("  cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main")


if __name__ == "__main__":
    main()
