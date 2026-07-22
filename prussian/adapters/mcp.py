"""FastMCP adapter — the four prussian tools over MCP (stdio / SSE).

A thin wrapper: every tool delegates to :mod:`prussian.tools`, the exact
same functions the inspect eval (:mod:`prussian.adapters.inspect`) and
the CLI agent (:mod:`prussian.adapters.agent`) use.  No LLM proxy, no
prompts, no resources — just the tools plus a startup FST health check.

Run it::

    python -m prussian.adapters.mcp            # stdio (Claude Code / Desktop)
    python -m prussian.adapters.mcp --web      # streamable-http (Claude Web)
    prussian-mcp                               # console-script entry point
"""

import argparse
import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from prussian.engine.fst.validate import check_fsg_pipeline
from prussian.engine.search import SearchEngine
from prussian.tools import lookup_tool, search_tool, validate_tool, wordforms_tool

# ── Server ────────────────────────────────────────────────────────────────────

# Allow strfry.org for remote access via SSH tunnel.
_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=["127.0.0.1:*", "localhost:*", "strfry.org:*", "strfry.org"],
    allowed_origins=[
        "http://127.0.0.1:*",
        "http://localhost:*",
        "https://strfry.org",
        "https://strfry.org:*",
    ],
)
mcp = FastMCP(
    "Prussian Dictionary",
    transport_security=_security,
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8001")),
)

# The engine and reranker are loaded lazily so that importing this module
# (e.g. for the console-script entry point) stays cheap and side-effect free.
_search_engine: SearchEngine | None = None
_reranker = None


def _engine() -> SearchEngine:
    global _search_engine
    if _search_engine is None:
        _search_engine = SearchEngine()
    return _search_engine


def _get_reranker():
    global _reranker
    if _reranker is None:
        from prussian.engine.embeddings.rerank import build_reranker
        _reranker = build_reranker()
    return _reranker


# ── Tools (1:1 with prussian.tools) ──────────────────────────────────────────


@mcp.tool()
def search_dictionary(
    query: str,
    top_k: int = 10,
    filter_tags: str | None = None,
    context: str = "",
) -> list[dict[str, Any]]:
    """Semantic search in the Prussian dictionary.

    Use this when you have a concept or modern-language word and want
    to find the Prussian equivalent.  Do NOT use for looking up
    known Prussian forms -- use ``lookup_prussian_word`` instead.

    Args:
        query: search query (German, English, Lithuanian, Latvian,
            Polish, Russian).  Never add "prussian"/"preußisch" --
            it's implicit.
        top_k: number of results to return.
        filter_tags: optional FST tag filter, e.g. ``"Akk+Sg"``,
            ``"Part+Pass"``, ``"Opt"``.  When set, each entry's
            forms are filtered to those matching the tags.
        context: usage context for reranking (enables cross-encoder
            when set).  In chunk mode each top chunk is annotated
            with ``best_line`` / ``lines``; in entry mode results
            are reranked by relevance.

    Returns:
        List of entries ``{word, translations, forms?, gender?}``
        (entry mode) or ``{lemma, members, pos, score, text, entries,
        best_line?, lines?}`` (chunk mode).
    """
    return search_tool(
        _engine(),
        query,
        top_k=top_k,
        filter_tags=filter_tags,
        reranker=_get_reranker() if context else None,
        context=context or None,
    )


@mcp.tool()
def lookup_prussian_word(text: str, fuzzy: bool = False) -> list[dict[str, Any]]:
    """Look up a Prussian sentence: tokenize, FST-analyze, enrich from dictionary.

    Each token is analyzed via the FST cascade and enriched with
    translations.  Tokens without FST analyses fall back to
    dictionary lookup.

    Args:
        text: Prussian text (one or more sentences).  Whole
            sentences are the normal case.
        fuzzy: set True for Levenshtein fallback on OOV tokens.
    """
    return lookup_tool(_engine(), text, fuzzy=fuzzy)


@mcp.tool()
def get_word_forms(lemma: str, features: str | None = None) -> list[dict[str, Any]]:
    """Get all declension or conjugation forms for a Prussian lemma.

    Returns a flat list of forms with their FST tags.  For verbs,
    the default shows only indicative present forms plus a list of
    available features.  Use ``features`` to request specific
    categories.

    Args:
        lemma: Prussian base form (from ``lookup_prussian_word``).
        features: optional comma-separated feature filter.  Accepts
            human-readable names (``participle``, ``conjunctive``,
            ``optative``, ``present``, ``preterite``,
            ``infinitive``, ``adverb``) or raw FST tags (``Part+Pass``,
            ``Ind``, ``Gen+Pl``).
    """
    return wordforms_tool(_engine(), lemma, features=features)


@mcp.tool()
def validate_prussian(text: str, include_conllu: bool = False) -> str:
    """Grammar check of Prussian text (FST + CG3 pipeline, three-valued).

    The single grammar tool: validates sentences and can also return
    the full dependency analysis.

    Args:
        text: Prussian sentence(s) to check (one or more sentences).
        include_conllu: also include each sentence's CoNLL-U
            dependency analysis (field "conllu", with rule provenance
            ``Rule=/`` / ``AgrParent=`` in MISC) — same pipeline
            run, no extra cost.  Set True when you need the parse,
            not just the verdict.

    Returns:
        JSON: ``{"overall": {"status", "n_sentences", "n_violations"},
        "sentences": [{sent_id, text, status, violations, coverage,
        conllu?}, ...]}``.

    Interpretation guide (IMPORTANT):

    - ``verified_in_coverage`` is the ONLY positive evidence of
      well-formedness: all words analyzed, low ambiguity, and at
      least one check family applied to the sentence.
    - ``out_of_coverage`` does NOT mean correct — the checker simply
      cannot verify (unknown words, collapsed analyses, residual
      ambiguity, or no applicable check; see coverage.reasons).
      Never treat it as approval.
    - ``violations_found``: each ``violations[]`` entry names the
      rule, the offending form (with index and surviving reading),
      a message, and a severity — ``"error"`` (case government /
      valency / person clash) is reliable; ``"warning"`` (adjective
      agreement, nominative in PP) is often a loanword paradigm
      gap rather than a real error.
    """
    return validate_tool(text, include_conllu=include_conllu)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prussian Dictionary MCP server (four tools + FST health).",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        default=os.getenv("MCP_TRANSPORT") == "sse",
        help="Serve over streamable-http (Claude Web) instead of stdio.",
    )
    parser.add_argument("--host", default=None, help="Host for --web mode.")
    parser.add_argument("--port", type=int, default=None, help="Port for --web mode.")
    args = parser.parse_args()

    if args.host:
        mcp.settings.host = args.host
    if args.port:
        mcp.settings.port = args.port

    print("Loading Prussian Dictionary search engine...")
    _engine()
    print("Search engine loaded successfully!")

    # Quick health check of the FST/CG3 grammar pipeline.
    _ok, fsg_msg = check_fsg_pipeline()
    print(fsg_msg)

    if args.web:
        print(
            f"Starting MCP server (streamable-http) on "
            f"http://{mcp.settings.host}:{mcp.settings.port}"
        )
        mcp.run(transport="streamable-http")
    else:
        print("Starting MCP server (stdio)")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
