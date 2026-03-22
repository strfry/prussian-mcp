"""Configuration management for Prussian Dictionary."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

DICTIONARY_PATH = DATA_DIR / "prussian_dictionary.json"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings_e5_prefix"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"

# LLM Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v3")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-oss-20b-int4-ov")

# Embedding configuration
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
