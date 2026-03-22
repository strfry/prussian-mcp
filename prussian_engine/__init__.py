"""Prussian Dictionary Engine - RAG system for Old Prussian language."""

from .search import SearchEngine
from .chat import ChatEngine
from .config import (
    DICTIONARY_PATH,
    EMBEDDINGS_PATH,
    SYSTEM_PROMPT_PATH,
    OPENAI_MODEL,
    OPENAI_BASE_URL
)

__version__ = "2.0.0"

__all__ = [
    "SearchEngine",
    "ChatEngine",
    "load",
]


def load():
    """
    Load and initialize the Prussian Dictionary engine.

    Returns:
        Tuple of (search_engine, chat_engine)
    """
    search_engine = SearchEngine()
    chat_engine = ChatEngine(search_engine)
    return search_engine, chat_engine
