"""prussian — dictionary + FST engine, shared tools, and framework adapters.

Public surface:

- :class:`prussian.engine.search.SearchEngine` (re-exported here as
  ``prussian.SearchEngine``) — the dictionary/embedding engine.
- :mod:`prussian.tools` — the four canonical tool functions
  (``search_tool``, ``lookup_tool``, ``wordforms_tool``,
  ``validate_tool``) shared 1:1 by every adapter.
- :mod:`prussian.adapters` — thin framework wrappers around those
  tools (``mcp``, ``inspect``, ``agent``).
"""

from .engine.search import SearchEngine
from .config import (
    DICTIONARY_PATH,
    EMBEDDINGS_PATH,
    AGENT_PROMPT_PATH,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
)

__version__ = "3.0.0"


def load() -> "SearchEngine":
    """Construct and return a :class:`SearchEngine` instance."""
    return SearchEngine()


__all__ = [
    "SearchEngine",
    "load",
    "DICTIONARY_PATH",
    "EMBEDDINGS_PATH",
    "AGENT_PROMPT_PATH",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "__version__",
]
