"""Configuration management for Prussian Dictionary."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

DICTIONARY_PATH = DATA_DIR / "prussian_dictionary.json"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings_with_prussian"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"

# Embedding & Reranking API Configuration (defaults: Jina AI)
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "") or os.getenv("JINA_API_KEY", "")
RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", "") or os.getenv(
    "JINA_BASE_URL", "https://api.jina.ai"
)
RERANK_EMBEDDING_MODEL = os.getenv(
    "RERANK_EMBEDDING_MODEL", "jina-embeddings-v5-text-small"
)
RERANK_EMBEDDING_DIM = 1024
RERANK_RERANKER_MODEL = os.getenv(
    "RERANK_RERANKER_MODEL", "jina-reranker-v2-base-multilingual"
)

# LLM Configuration (for chat/llm features)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v3")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "eurollm-22b-instruct-int4")

# Query prefix for embedding model (E5-style)
QUERY_PREFIX = ""
