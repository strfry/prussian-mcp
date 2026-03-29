"""Prussian Dictionary Engine - RAG system for Old Prussian language."""

from .search import SearchEngine
from .config import (
    DICTIONARY_PATH,
    EMBEDDINGS_PATH,
    SYSTEM_PROMPT_PATH,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
)

__version__ = "2.0.0"

__all__ = [
    "SearchEngine",
    "load",
]


def load():
    """
    Load and initialize the Prussian Dictionary engine.

    Returns:
        SearchEngine instance
    """
    return SearchEngine()
