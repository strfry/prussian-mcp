"""Configuration management for Prussian Dictionary."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

DICTIONARY_PATH = DATA_DIR / "twanksta_entries.json"
AGENT_PROMPT_PATH = PROMPTS_DIR / "agent_system_en.md"

# ── Embedding backend ────────────────────────────────────────────────────────
# "model2vec" -> local, CPU-only static embeddings (default, no API needed)
# "api"       -> remote OpenAI-/Jina-compatible embedding endpoint (legacy)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "model2vec").lower()

# model2vec model: a HuggingFace id or a local directory produced with
# StaticModel.save_pretrained(). Point MODEL2VEC_MODEL at a local path to run
# without any HuggingFace access.
MODEL2VEC_MODEL = os.getenv("MODEL2VEC_MODEL", "minishlab/potion-multilingual-128M")

# Precomputed corpus embeddings live at "<EMBEDDINGS_DIR>/<EMBEDDINGS_NAME>.*".
# They must be generated with the same backend/model used for query encoding.
_DEFAULT_EMB_NAME = (
    "embeddings_model2vec" if EMBEDDING_BACKEND == "model2vec" else "embeddings_voyage"
)
EMBEDDINGS_NAME = os.getenv("EMBEDDINGS_NAME", _DEFAULT_EMB_NAME)
EMBEDDINGS_PATH = EMBEDDINGS_DIR / EMBEDDINGS_NAME

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

# Asymmetric search prefixes.
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "Query: ")
PASSAGE_PREFIX = os.getenv("PASSAGE_PREFIX", "Document: ")

# LLM Configuration (for chat/llm features)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v3")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "eurollm-22b-instruct-int4")

# FSG/CG check: prussian-fst wird als editierbare uv-Path-Dependency
# importiert (pyproject [tool.uv.sources]) — Checkout-Pfad dort anpassen,
# nicht mehr per PRUSSIAN_FST_DIR-Umgebungsvariable.
