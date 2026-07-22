"""Configuration management for Prussian Dictionary."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT.parent / "embeddings" / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

DICTIONARY_PATH = DATA_DIR / "twanksta_entries.json"
AGENT_PROMPT_PATH = PROMPTS_DIR / "agent_system_en.md"

# ── Embedding backend (deprecated pass-throughs) ──────────────────────────────
# These env vars are now consumed by prussian_embeddings.env_config() at package init.
# They are kept for backwards compat with existing env files (.sh, .env).
# Backend selection, model loading, API config all happen in prussian_embeddings package.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "fastembed").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "jina-reranker-v2-base-multilingual")
API_KEY = (
    os.getenv("API_KEY", "")
    or os.getenv("JINA_API_KEY", "")
    or os.getenv("RERANK_API_KEY", "")
)
API_BASE_URL = (
    os.getenv("API_BASE_URL", "")
    or os.getenv("RERANK_BASE_URL", "")
    or os.getenv("JINA_BASE_URL", "https://api.jina.ai")
)

# ── Embedding storage and search ──────────────────────────────────────────────
# Precomputed corpus embeddings live at "<EMBEDDINGS_DIR>/<EMBEDDINGS_NAME>.*".
EMBEDDINGS_NAME = os.getenv("EMBEDDINGS_NAME", "embeddings_fastembed")
EMBEDDINGS_PATH = EMBEDDINGS_DIR / EMBEDDINGS_NAME

# Asymmetric query encoding prefix
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "Query: ")

# ── Chunk mode ────────────────────────────────────────────────────────────────
# Non-empty CHUNK_EMBEDDINGS_NAME enables chunk mode (one vector per
# lemma-cluster instead of per headword).  QUERY_PREFIX must match the
# chunk model's expected prefix (e.g. "query: " for e5-large).
CHUNK_EMBEDDINGS_NAME = os.getenv("CHUNK_EMBEDDINGS_NAME", "")
CHUNK_EMBEDDINGS_PATH = EMBEDDINGS_DIR / CHUNK_EMBEDDINGS_NAME if CHUNK_EMBEDDINGS_NAME else None

# BM25 + dense RRF hybrid recall (set "0" to force dense-only for debugging).
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "1") not in ("0", "false", "no")

# Max chunks sent to the cross-encoder for context reranking (per-line cost).
CHUNK_RERANK_TOPN = int(os.getenv("CHUNK_RERANK_TOPN", "10"))

# LLM Configuration (for chat/llm features)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v3")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "eurollm-22b-instruct-int4")

# FSG/CG check: prussian-fst wird als editierbare uv-Path-Dependency
# importiert (pyproject [tool.uv.sources]) — Checkout-Pfad dort anpassen,
# nicht mehr per PRUSSIAN_FST_DIR-Umgebungsvariable.
