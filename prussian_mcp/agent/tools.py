"""In-process Haystack tools for the ``prussian-agent`` CLI.

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

import json
from typing import Any

from haystack.tools import create_tool_from_function


def _prune_bool_defaults(schema: dict) -> list[str]:
    """Remove boolean properties carrying a ``default`` from a JSON-Schema
    in place; return the names of removed properties.

    1:1 move from ``haystack_runner.py`` (Z.249–269).  Some models
    (Llama-3.3-style) emit string values like ``"true"`` for boolean args
    that carry a default, which strict inference providers reject with
    HTTP 400.  Removing them from the schema means the tool simply uses
    its own default.
    """
    if not isinstance(schema, dict):
        return []
    props = schema.get("properties")
    if not isinstance(props, dict):
        return []
    removed: list[str] = []
    for name, prop in list(props.items()):
        if (
            isinstance(prop, dict)
            and prop.get("type") == "boolean"
            and "default" in prop
        ):
            props.pop(name, None)
            removed.append(name)
    required = schema.get("required")
    if isinstance(required, list) and removed:
        schema["required"] = [r for r in required if r not in removed]
    return removed


def _simplify_optional_strings(schema: dict) -> None:
    """Flatten ``anyOf: [{type: string}, {type: null}]`` (Pydantic's
    rendering of ``str | None = None``) into a plain ``{type: string}``
    and drop ``default: null``.

    Some tool-calling backends (Llama-3.3 on TGI) reject ``null`` values
    for parameters whose schema uses ``anyOf`` — they look at the first
    type branch (``string``) and error on ``null``.  Removing the null
    branch and the ``default: null`` makes the parameter appear as an
    optional string, so the model either passes a string or omits it.
    The tool function still accepts ``None`` at the Python level.
    """
    if not isinstance(schema, dict):
        return
    props = schema.get("properties")
    if not isinstance(props, dict):
        return
    for prop in props.values():
        if not isinstance(prop, dict):
            continue
        any_of = prop.get("anyOf")
        if isinstance(any_of, list) and len(any_of) == 2:
            types = {b.get("type") for b in any_of}
            if types == {"string", "null"}:
                prop.pop("anyOf", None)
                prop["type"] = "string"
                if prop.get("default") is None:
                    prop.pop("default", None)


def build_local_toolset(engine=None) -> list:
    """Build the four in-process Haystack tools.

    Imports of ``SearchEngine`` happen here (not at module top) so the
    package stays importable without ``prussian-fst`` and so env vars
    are read at the right time.

    Args:
        engine: optional ``SearchEngine`` instance.  If ``None`` a fresh
            one is constructed (loads dictionary + embeddings, which is
            the heavyweight bit — ~1–2 s on a warm disk).

    Returns:
        List of four Haystack ``Tool`` objects, with bool-default params
        pruned from their JSON schemas (models that emit ``"true"``
        strings would otherwise HTTP-400 some providers).
    """
    # Lazy import — see module docstring.
    from prussian_engine.search import SearchEngine

    if engine is None:
        engine = SearchEngine()

    from tools import search_tool, lookup_tool, wordforms_tool, validate_tool

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
                ``infinitive``) or raw FST tags (``Part+Pass``,
                ``Ind``, ``Gen+Pl``).
        """
        return wordforms_tool(engine, lemma, features=features)

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

    tools = [
        create_tool_from_function(
            function=search_dictionary,
            description=(
                "Semantic search in the Prussian dictionary. Use this when "
                "you have a concept or modern-language word and want to find "
                "the Prussian equivalent. Do NOT use for looking up known "
                "Prussian forms — use lookup_prussian_word instead. "
                "Optional filter_tags (e.g. 'Akk+Sg', 'Part+Pass') "
                "restricts results to matching forms."
            ),
        ),
        create_tool_from_function(
            function=lookup_prussian_word,
            description=(
                "Look up a Prussian sentence: tokenize, FST-analyze, "
                "enrich from dictionary. Each token gets FST analyses "
                "(lemma + tags) and dictionary translations. Whole "
                "sentences are the normal case."
            ),
        ),
        create_tool_from_function(
            function=get_word_forms,
            description=(
                "Get declension/conjugation forms for a Prussian lemma. "
                "Returns flat list of forms with FST tags. For verbs, "
                "default shows indicative present only; use features "
                "(e.g. 'participle', 'Gen+Pl') for specific categories."
            ),
        ),
        create_tool_from_function(
            function=validate_prussian,
            description=(
                "Grammar check of Prussian text (FST + CG3 pipeline, "
                "three-valued). Returns JSON with per-sentence status: "
                "'verified_in_coverage' (only positive evidence), "
                "'out_of_coverage' (NOT correct — checker cannot verify), "
                "or 'violations_found' (rule violations with severity "
                "error/warning). Use to check grammar/agreement of a "
                "Prussian sentence, NOT for dictionary lookups."
            ),
        ),
    ]

    # Prune bool-default params from each tool's schema so models that
    # emit "true"/"false" strings for boolean args don't HTTP-400 strict
    # providers.  Mirrors haystack_runner.py Z.505–510.
    for tool in tools:
        _prune_bool_defaults(tool.parameters)
        _simplify_optional_strings(tool.parameters)

    return tools