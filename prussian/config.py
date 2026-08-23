"""Configuration management for Prussian Dictionary."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT.parent / "embeddings" / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
#PROMPTS_DIR = Path("../../prompts")

# Kanonisches Dictionary: wird im corpus-Repo gebaut (scripts/twanksta_parse.py)
# und von allen Konsumenten direkt dort referenziert — keine Kopien in den Repos.
CORPUS_PARSED = PROJECT_ROOT.parent / "corpus" / "parsed"
DICTIONARY_PATH = Path(os.getenv(
    "PRUSSIAN_DICTIONARY", str(CORPUS_PARSED / "twanksta_entries.json")))
AGENT_PROMPT_PATH = PROMPTS_DIR / "agent_system_en.md"

# ── Embedding storage and search ──────────────────────────────────────────────
# Backend/Modell/API-Keys liest ausschließlich prussian_embeddings.env_config()
# aus den Env-Vars (Quelle: mcp/env.<provider>.sh).  Precomputed corpus
# embeddings live at "<EMBEDDINGS_DIR>/<EMBEDDINGS_NAME>.*"; derselbe Name
# steuert auch den Schreibpfad von `make store` im embeddings-Repo.
EMBEDDINGS_NAME = os.getenv("EMBEDDINGS_NAME", "embeddings_fastembed")
EMBEDDINGS_PATH = EMBEDDINGS_DIR / EMBEDDINGS_NAME

# Query-Prefix-Fallback für Stores OHNE meta["query_prefix"] (Legacy-e5-Stores).
# Aktuelle Stores tragen ihren Prefix in der Meta — die hat Vorrang.
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "Query: ")

# ── Chunk retrieval ───────────────────────────────────────────────────────────
# Chunk store: one vector per lemma-cluster instead of per headword.

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
