"""Configuration management for Prussian Dictionary."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT.parent / "embeddings" / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

DICTIONARY_PATH = DATA_DIR / "twanksta_entries.json"
AGENT_PROMPT_PATH = PROMPTS_DIR / "agent_system_en.md"

# ── Embedding backend ────────────────────────────────────────────────────────
# "model2vec" -> local, CPU-only static embeddings (default, no API needed)
# "api"       -> remote OpenAI-/Jina-compatible embedding endpoint
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "fastembed").lower()

# Model name — works for both backends:
#   model2vec: HuggingFace id or local dir from StaticModel.save_pretrained()
#   api:       remote model id (e.g. "jina-embeddings-v5-text-small")
_DEFAULT_MODEL = {
    "model2vec": "minishlab/potion-multilingual-128M",
    "api": "jina-embeddings-v5-text-small",
}.get(EMBEDDING_BACKEND, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL)

# Embedding dimension — only used by the API backend (model2vec reads it from
# the loaded model at runtime).
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# Reranker model (only relevant when API backend is active)
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "jina-reranker-v2-base-multilingual"
)

# Precomputed corpus embeddings live at "<EMBEDDINGS_DIR>/<EMBEDDINGS_NAME>.*".
# They must be generated with the same backend/model used for query encoding.
_DEFAULT_EMB_NAME = {
    "model2vec": "embeddings_model2vec",
    "api": "embeddings_voyage",
}.get(EMBEDDING_BACKEND, "embeddings_fastembed")
EMBEDDINGS_NAME = os.getenv("EMBEDDINGS_NAME", _DEFAULT_EMB_NAME)
EMBEDDINGS_PATH = EMBEDDINGS_DIR / EMBEDDINGS_NAME

# API server — embedding + reranking endpoints (same server)
API_KEY = (
    os.getenv("API_KEY", "")
    or os.getenv("JINA_API_KEY", "")
    or os.getenv("RERANK_API_KEY", "")  # backwards-compat
)
API_BASE_URL = (
    os.getenv("API_BASE_URL", "")
    or os.getenv("RERANK_BASE_URL", "")  # backwards-compat
    or os.getenv("JINA_BASE_URL", "https://api.jina.ai")
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
