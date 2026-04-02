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
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings_with_prussian"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"

# LLM Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v3")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "eurollm-22b-instruct-int4")

# Embedding configuration (now served by local LLM server)
EMBEDDING_MODEL = "Qwen3-Embedding-0.6B-int8-ov"
EMBEDDING_DIM = 1024
QUERY_PREFIX = "Instruct: Given a search query, retrieve the corresponding multilingual dictionary entry containing translations in German, English, Lithuanian, Latvian, Polish, and Russian.\nQuery: "
PASSAGE_PREFIX = ""

# Reranker configuration
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen3-Reranker-0.6B-int8-ov")
