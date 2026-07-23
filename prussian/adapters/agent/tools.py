"""In-process smolagents tools for the ``prussian-agent`` CLI.

The four tools — ``search_dictionary``, ``lookup_prussian_word``,
``get_word_forms``, ``validate_prussian`` — have names and signatures
identical to the FastMCP server in ``prussian.adapters.mcp``, and their
descriptions come from the shared ``prussian.tools.spec`` (the single source
of truth), so the in-process agent path and the remote-MCP path present
identical tool text.  smolagents is deprecated; this path is kept working but
is no longer a design driver.

``SearchEngine`` and ``run_validate`` are imported **lazily** inside
``build_local_toolset``.  This keeps ``prussian-mcp`` importable from
``prussian-llm``'s editable install without forcing ``prussian-fst`` to
be present, and it lets users ``source env.hf-voyage.sh`` *before*
launching the CLI so that ``prussian.config`` picks up the right
embedding / LLM env vars at import time.
"""

from __future__ import annotations

from typing import Any

from smolagents import tool


def build_local_toolset(engine=None) -> list:
    """Build the four in-process smolagents tools.

    Imports of ``SearchEngine`` happen here (not at module top) so the
    package stays importable without ``prussian-fst`` and so env vars
    are read at the right time.

    Args:
        engine: optional ``SearchEngine`` instance.  If ``None`` a fresh
            one is constructed (loads dictionary + embeddings, which is
            the heavyweight bit — ~1–2 s on a warm disk).

    Returns:
        List of four smolagents ``Tool`` objects.
    """
    # Lazy import — see module docstring.  When no engine is injected, reuse
    # the shared lazy singleton so all adapters share one SearchEngine.
    if engine is None:
        from prussian.tools.runtime import get_engine
        engine = get_engine()

    from prussian.tools import search_tool, lookup_tool, wordforms_tool, validate_tool
    from prussian.tools import spec

    # The docstrings below are terse placeholders: smolagents builds a tool by
    # parsing this source (name, signature, per-arg docs), so every parameter
    # must be documented here.  The user-visible description and argument docs
    # are then replaced from ``prussian.tools.spec`` (the single source of
    # truth) via ``apply_to_smolagents_tool`` before the tools are returned, so
    # this adapter presents exactly the same text as the MCP and inspect-ai
    # adapters.

    @tool
    def search_dictionary(
        query: str,
        top_k: int = 10,
        filter_tags: str | None = None,
        context: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search in the Prussian dictionary (description from spec).

        Args:
            query: source-language query.
            top_k: number of results to return.
            filter_tags: optional FST tag filter.
            context: optional usage context for reranking.
        """
        from prussian.tools.runtime import get_reranker
        reranker = get_reranker() if context else None
        return search_tool(engine, query, top_k=top_k,
                           filter_tags=filter_tags,
                           reranker=reranker, context=context)

    @tool
    def lookup_prussian_word(
        text: str,
        fuzzy: bool = False,
    ) -> list[dict[str, Any]]:
        """Analyze a Prussian sentence via the FST cascade (description from spec).

        Args:
            text: Prussian text (one or more sentences).
            fuzzy: set True for Levenshtein fallback on OOV tokens.
        """
        return lookup_tool(engine, text, fuzzy=fuzzy)

    @tool
    def get_word_forms(lemma: str, features: str | None = None) -> list[dict[str, Any]]:
        """Inflect a Prussian lemma into its forms (description from spec).

        Args:
            lemma: Prussian base form.
            features: optional comma-separated feature filter.
        """
        return wordforms_tool(engine, lemma, features=features)

    @tool
    def validate_prussian(text: str, include_conllu: bool = False) -> str:
        """Grammar-check Prussian text via FST + CG3 (description from spec).

        Args:
            text: Prussian sentence(s) to check.
            include_conllu: also include each sentence's CoNLL-U analysis.
        """
        return validate_tool(text, include_conllu=include_conllu)

    tools = [search_dictionary, lookup_prussian_word, get_word_forms, validate_prussian]
    for _tool, _spec in zip(tools, (spec.SEARCH, spec.LOOKUP, spec.WORDFORMS, spec.VALIDATE)):
        spec.apply_to_smolagents_tool(_tool, _spec)
    return tools
