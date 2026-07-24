"""The four prussian tools as inspect-ai ``@tool``s.

Signatures mirror the other adapters.  Tool descriptions come from
``prussian.tools.spec`` (the single source of truth): each ``execute``
callable's ``__doc__`` is set from the canonical spec, so the eval, the MCP
server and the CLI all present identical tool text.  Optional parameters use
defaults (not nullable) so the schemas stay in OpenAI-compatible format
without ``anyOf``.  Each tool delegates to the same raw callables
(``prussian.tools``) the other adapters use and shares one lazily-built
``SearchEngine`` via ``prussian.tools.runtime``.
"""

from __future__ import annotations

import json
from typing import Any

import anyio

from inspect_ai.tool import tool

from prussian.tools import lookup_tool, search_tool, spec, validate_tool, wordforms_tool
from prussian.tools.runtime import get_engine as _get_engine
from prussian.tools.runtime import get_reranker as _get_reranker
from prussian.tools.search import format_search_results


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


@tool
def search_dictionary():
    async def execute(
        query: str,
        top_k: int = 10,
        filter_tags: str = "",
        context: str = "",
    ) -> str:
        engine = _get_engine()
        result = await anyio.to_thread.run_sync(
            lambda: search_tool(
                engine,
                query,
                top_k=top_k,
                filter_tags=filter_tags or None,
                reranker=_get_reranker() if context else None,
                context=context or None,
            )
        )
        return format_search_results(result)

    execute.__doc__ = spec.docstring(spec.SEARCH)
    return execute


@tool
def lookup_prussian_word():
    async def execute(text: str, fuzzy: bool = False) -> str:
        engine = _get_engine()
        result = await anyio.to_thread.run_sync(
            lambda: lookup_tool(engine, text, fuzzy=bool(fuzzy))
        )
        return _dumps(result)

    execute.__doc__ = spec.docstring(spec.LOOKUP)
    return execute


@tool
def get_word_forms():
    async def execute(lemma: str, features: str = "") -> str:
        engine = _get_engine()
        result = await anyio.to_thread.run_sync(
            lambda: wordforms_tool(engine, lemma, features=features or None)
        )
        return _dumps(result)

    execute.__doc__ = spec.docstring(spec.WORDFORMS)
    return execute


@tool
def validate_prussian():
    async def execute(text: str, include_conllu: bool = False) -> str:
        result = await anyio.to_thread.run_sync(
            lambda: validate_tool(text, include_conllu=bool(include_conllu))
        )
        # validate_tool already returns a JSON string.
        return result

    execute.__doc__ = spec.docstring(spec.VALIDATE)
    return execute
