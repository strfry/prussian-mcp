"""In-process smolagents tools for the ``prussian-agent`` CLI.

The four tools — ``search_dictionary``, ``lookup_prussian_word``,
``get_word_forms``, ``validate_prussian`` — have names and signatures
identical to the FastMCP server in ``mcp_server.py`` so that the system
prompt's tool descriptions stay interchangeable between the in-process
agent path and the remote-MCP path.

``SearchEngine`` and ``run_validate`` are imported **lazily** inside
``build_local_toolset``.  This keeps ``prussian-mcp`` importable from
``prussian-llm``'s editable install without forcing ``prussian-fst`` to
be present, and it lets users ``source env.hf-voyage.sh`` *before*
launching the CLI so that ``prussian_engine.config`` picks up the right
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
    # Lazy import — see module docstring.
    from prussian_engine.search import SearchEngine

    if engine is None:
        engine = SearchEngine()

    from tools import search_tool, lookup_tool, wordforms_tool, validate_tool

    @tool
    def search_dictionary(
        query: str,
        top_k: int = 10,
        use_reranker: bool = True,
        filter_tags: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search in the Prussian dictionary.

        Use this when you have a concept or modern-language word and want
        to find the Prussian equivalent.  Do NOT use for looking up
        known Prussian forms — use ``lookup_prussian_word`` instead.

        Args:
            query: search query (German, English, Lithuanian, Latvian,
                Polish, Russian).  Never add "prussian"/"preußisch" —
                it's implicit.
            top_k: number of results to return.
            use_reranker: accepted for signature parity with the MCP
                server; currently ignored (no reranker in-process).
            filter_tags: optional FST tag filter, e.g. ``"Akk+Sg"``,
                ``"Part+Pass"``, ``"Opt"``.  When set, each entry's
                forms are filtered to those matching the tags.

        Returns:
            List of entries ``{word, translations, forms?, gender?}``.
            ``forms`` / ``gender`` are added only when ``filter_tags``
            matches forms for that entry.
        """
        return search_tool(engine, query, top_k=top_k,
                           use_reranker=False, filter_tags=filter_tags)

    @tool
    def lookup_prussian_word(
        text: str,
        fuzzy: bool = False,
    ) -> list[dict[str, Any]]:
        """Look up a Prussian sentence: tokenize, FST-analyze, enrich from dictionary.

        Each token is analyzed via the FST cascade and enriched with
        translations.  Tokens without FST analyses fall back to
        dictionary lookup.

        Args:
            text: Prussian text (one or more sentences).  Whole
                sentences are the normal case.
            fuzzy: set True for Levenshtein fallback on OOV tokens.
        """
        return lookup_tool(engine, text, fuzzy=fuzzy)

    @tool
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
        return wordforms_tool(engine, lemma, features=features)

    @tool
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

    return [search_dictionary, lookup_prussian_word, get_word_forms, validate_prussian]
