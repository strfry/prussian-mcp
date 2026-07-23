"""Shared lazy runtime for the tool adapters.

The ``SearchEngine`` (loads the dictionary + embeddings) and the reranker
(loads a cross-encoder) are heavyweight singletons.  Every adapter — the
FastMCP server, the smolagents CLI, and the inspect-ai eval — used to keep
its own module-level ``_engine`` / ``_reranker`` pair with identical lazy
init.  They share this one pair instead.

Construction is deferred so that importing an adapter stays cheap and
side-effect free, and so ``prussian.config`` reads the embedding / LLM env
vars only *after* an env file has been sourced (see the CLI startup order).
"""

from __future__ import annotations

_engine = None
_reranker = None


def get_engine():
    """Return the shared ``SearchEngine``, constructing it on first use."""
    global _engine
    if _engine is None:
        from prussian.engine.search import SearchEngine
        _engine = SearchEngine()
    return _engine


def get_reranker():
    """Return the shared cross-encoder reranker (may be ``None`` if unavailable)."""
    global _reranker
    if _reranker is None:
        from prussian.engine.embeddings.rerank import build_reranker
        _reranker = build_reranker()
    return _reranker
