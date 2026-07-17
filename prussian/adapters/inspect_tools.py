"""The four prussian tools as inspect-ai ``@tool``s.

Signatures mirror the smolagents draft (``prussian.adapters.agent.tools``),
but the descriptions are deliberately **instructive** ("Use this tool to …",
workflow anchoring): with the eval's minimal prompt surface the tool
descriptions are the only place the model learns the intended workflow.
Optional parameters use defaults (not nullable) so the schemas stay in
OpenAI-compatible format without ``anyOf``.  Each tool delegates to the
same raw callables (``prussian.tools``) the MCP server and CLI use,
sharing one lazily-built ``SearchEngine``.  No changes are made to the
live server / proxy.
"""

from __future__ import annotations

import json
from typing import Any

import anyio

from inspect_ai.tool import tool

from prussian.tools import lookup_tool, search_tool, validate_tool, wordforms_tool

# Lazily built once — SearchEngine reads embedding/LLM env at construction
# time, so it must be created *after* the env file is sourced.
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from prussian.engine.search import SearchEngine

        _engine = SearchEngine()
    return _engine


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


@tool
def search_dictionary():
    async def execute(
        query: str,
        top_k: int = 10,
        use_reranker: bool = False,
        filter_tags: str | None = None,
    ) -> str:
        """Use this tool to find the Prussian word for a concept — the
        FIRST step for every content word of the sentence you translate.

        Semantic search in the Prussian dictionary.  Never invent or
        guess Prussian words: every word you use must come from this
        dictionary.  Do NOT pass Prussian words as the query — use
        `lookup_prussian_word` for those.

        Args:
            query: the word or concept in a source language (German,
                English, Lithuanian, Latvian, Polish, Russian).  Mixing
                languages in one query (e.g. "Birke birch") sharpens the
                match.  Never add "prussian"/"preußisch" — it's implicit.
            top_k: number of results to return (default 10).
            use_reranker: accepted for signature parity with the MCP
                server; currently ignored (no reranker in-process).
            filter_tags: FST tag filter, e.g. "Akk+Sg", "Part+Pass",
                "Opt", or None for no filter.  When set, each entry's
                forms are filtered to those matching the tags.

        Returns:
            List of entries {word, translations, forms?, gender?}.
        """
        engine = _get_engine()
        result = await anyio.to_thread.run_sync(
            lambda: search_tool(
                engine,
                query,
                top_k=top_k,
                use_reranker=use_reranker,
                filter_tags=filter_tags,
            )
        )
        return _dumps(result)

    return execute


@tool
def lookup_prussian_word():
    async def execute(text: str, fuzzy: bool = False) -> str:
        """Use this tool to analyze Prussian words you already have —
        to confirm a word exists, find its lemma, and see its grammar tags.

        Tokenizes the text, FST-analyzes each token (lemma + tags like
        "V+Ind+Pres+P3+Sg") and enriches it with dictionary translations.
        Tokens without FST analyses fall back to dictionary lookup.  You
        can pass a whole draft sentence to check all words at once.

        Args:
            text: Prussian text (one or more sentences).  Whole
                sentences are the normal case.
            fuzzy: set True for Levenshtein fallback on OOV tokens;
                False for exact matching.
        """
        engine = _get_engine()
        result = await anyio.to_thread.run_sync(
            lambda: lookup_tool(engine, text, fuzzy=bool(fuzzy))
        )
        return _dumps(result)

    return execute


@tool
def get_word_forms():
    async def execute(lemma: str, features: str | None = None) -> str:
        """Use this tool to inflect a Prussian lemma into the exact form
        you need (case, number, gender, person, tense).  Never guess an
        inflected form — always fetch it here.

        Returns a flat list of forms with their FST tags, plus
        `available_features`: the set of FST tags occurring in the
        lemma's paradigm — combine them as a `features` filter.  For
        verbs, the default shows only indicative present forms.

        NOTE: 3rd-person verb forms carry no Sg/Pl distinction — request
        them with P3 alone (e.g. "Ind+Pres+P3", never "...+P3+Sg").

        Args:
            lemma: Prussian base form (from `lookup_prussian_word`).
            features: comma-separated feature filter, or None for the
                default view.  Accepts human-readable names (participle,
                conjunctive, optative, present, preterite, infinitive,
                adverb) or raw FST tags (Part+Pass, Gen+Pl, Ind+Pres+P1,
                Ind+Pres+P3).
        """
        engine = _get_engine()
        result = await anyio.to_thread.run_sync(
            lambda: wordforms_tool(engine, lemma, features=features)
        )
        return _dumps(result)

    return execute


@tool
def validate_prussian():
    async def execute(text: str, include_conllu: bool = False) -> str:
        """Use this tool to grammar-check your draft Prussian sentence
        BEFORE you submit it.  Fix every "error" violation it reports,
        then validate again — only submit a sentence you have validated.

        Grammar check of Prussian text (FST + CG3 pipeline, three-valued):
        validates sentences and can also return the full dependency
        analysis.

        Args:
            text: Prussian sentence(s) to check (one or more sentences).
            include_conllu: also include each sentence's CoNLL-U
                dependency analysis (field "conllu"); set True when you
                need the parse, not just the verdict (default False).

        Returns:
            JSON: {"overall": {...}, "sentences": [{sent_id, text, status,
            violations, coverage, conllu?}, ...]}.

        Interpretation guide (IMPORTANT):

        - `verified_in_coverage` is the ONLY positive evidence of
          well-formedness.
        - `out_of_coverage` does NOT mean correct — the checker simply
          cannot verify (unknown words, ambiguity, no applicable check).
          Never treat it as approval.
        - `violations_found`: each violation names the rule, offending
          form, message, and severity (error vs. warning).
        """
        result = await anyio.to_thread.run_sync(
            lambda: validate_tool(text, include_conllu=bool(include_conllu))
        )
        # validate_tool already returns a JSON string.
        return result

    return execute
