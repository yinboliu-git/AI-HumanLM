"""
Configuration for the MoltbookPaper project.
All data is sourced from HuggingFace datasets (no API keys needed for data collection).
"""

# Minimum text length (characters) to keep
MIN_TEXT_LENGTH = 50

# Embedding model
SBERT_MODEL = "all-mpnet-base-v2"
# Semantic alignment threshold
ALIGNMENT_THRESHOLD = 0.8

# Paths
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
