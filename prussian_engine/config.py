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

# Embedding & Reranking API Configuration
# Defaults: Jina AI. Override via environment variables for local models (e.g. OVMS + Qwen).
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "") or os.getenv("JINA_API_KEY", "")
RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", "") or os.getenv(
    "JINA_BASE_URL", "https://api.jina.ai"
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v5-text-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "jina-reranker-v2-base-multilingual")

# Asymmetric search prefixes (Qwen3-Embedding style).
# Set via env vars when using models that need instruct prefixes.
# For Jina, leave empty. For Qwen3-Embedding, set:
#   QUERY_PREFIX="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
#   PASSAGE_PREFIX="Instruct: Represent this word and its translations for retrieval\n"
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "")
PASSAGE_PREFIX = os.getenv("PASSAGE_PREFIX", "")

# LLM Configuration (for chat/llm features)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v3")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "eurollm-22b-instruct-int4")
