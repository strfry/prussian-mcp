"""Dictionary + FST engine internals.

Sub-packages:

- :mod:`prussian.engine.search` — the ``SearchEngine`` (semantic search,
  reverse lookup, form generation).
- :mod:`prussian.engine.morphology` — Prussian Glossing Rules (PGR)
  parsing and feature utilities.
- :mod:`prussian.engine.embeddings` — embedding backends, HTTP client,
  and the reranked search wrapper.
- :mod:`prussian.engine.fst` — FST tag analysis and the CG3 grammar
  validator.
"""

from .search import SearchEngine

__all__ = ["SearchEngine"]
