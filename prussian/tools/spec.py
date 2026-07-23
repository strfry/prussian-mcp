"""Canonical tool descriptions — the single source of truth for the prose the
model sees for each of the four tools.

Every adapter derives its tool description from here, so the FastMCP server,
the inspect-ai eval and the (legacy) smolagents CLI present identical text and
can never drift again:

* FastMCP and inspect-ai read a callable's ``__doc__``; :func:`docstring`
  renders the full docstring (description + ``Args:``) which those adapters
  assign to ``__doc__`` before registering the tool.
* smolagents parses tool *source* rather than ``__doc__``, so its adapter
  overrides the built tool's ``.description`` (the prose) and each
  ``.inputs[param]["description"]`` (from :attr:`ToolSpec.args`) after
  construction — see :func:`apply_to_smolagents_tool`.

To change what the model reads for a tool, edit it here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolSpec:
    """One tool's canonical description and per-argument docs."""

    name: str
    description: str
    args: dict[str, str] = field(default_factory=dict)


def docstring(spec: ToolSpec) -> str:
    """Render the full docstring (``description`` + ``Args:``) for a tool.

    Used by the ``__doc__``-based adapters (FastMCP, inspect-ai).  The
    ``Args:`` block is what those frameworks parse into per-parameter schema
    descriptions.
    """
    parts = [spec.description.rstrip()]
    if spec.args:
        lines = ["Args:"]
        for name, doc in spec.args.items():
            lines.append(f"    {name}: {doc}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n"


def apply_to_smolagents_tool(tool, spec: ToolSpec) -> None:
    """Point a built smolagents tool at the canonical description.

    smolagents derives its description/inputs by parsing the tool source, so
    we overwrite the two attributes the agent actually renders into its
    prompt: the tool ``description`` and each input's ``description``.
    """
    tool.description = spec.description
    for name, doc in spec.args.items():
        if name in tool.inputs:
            tool.inputs[name]["description"] = doc


# ── The four canonical specs ─────────────────────────────────────────────────

SEARCH = ToolSpec(
    name="search_dictionary",
    description=(
        "Semantic search in the Prussian dictionary.\n\n"
        "Use this when you have a concept or modern-language word and want to "
        "find the Prussian equivalent — the first step for every content word "
        "you translate.  Do NOT use for looking up known Prussian forms — use "
        "lookup_prussian_word instead.\n\n"
        "Returns a list of entries {word, translations, forms?, gender?} "
        "(entry mode) or {lemma, members, pos, score, text, entries, "
        "best_line?, lines?} (chunk mode)."
    ),
    args={
        "query": (
            "search query (German, English, Lithuanian, Latvian, Polish, "
            "Russian).  Never add \"prussian\"/\"preußisch\" — it's implicit."
        ),
        "top_k": "number of results to return.",
        "filter_tags": (
            "optional FST tag filter, e.g. \"Akk+Sg\", \"Part+Pass\", \"Opt\". "
            "When set, each entry's forms are filtered to those matching the "
            "tags."
        ),
        "context": (
            "usage context for reranking (enables the cross-encoder when set). "
            "In chunk mode each top chunk is annotated with best_line / lines; "
            "in entry mode results are reranked by relevance."
        ),
    },
)

LOOKUP = ToolSpec(
    name="lookup_prussian_word",
    description=(
        "Look up a Prussian sentence: tokenize, FST-analyze, enrich from "
        "dictionary.\n\n"
        "Use this to analyze Prussian words you already have — to confirm a "
        "word exists, find its lemma, and see its grammar tags.  Each token is "
        "analyzed via the FST cascade (lemma + tags like \"V+Ind+Pres+P3+Sg\") "
        "and enriched with dictionary translations.  Tokens without FST "
        "analyses fall back to dictionary lookup.  Whole sentences are the "
        "normal case."
    ),
    args={
        "text": "Prussian text (one or more sentences).",
        "fuzzy": "set True for Levenshtein fallback on OOV tokens.",
    },
)

WORDFORMS = ToolSpec(
    name="get_word_forms",
    description=(
        "Get all declension or conjugation forms for a Prussian lemma.\n\n"
        "Use this to inflect a lemma into the exact form you need — never "
        "guess an inflected form.  Returns a flat list of forms with their FST "
        "tags plus the paradigm's available features.  For verbs, the default "
        "shows only indicative present forms; use `features` to request "
        "specific categories.\n\n"
        "Note: 3rd-person verb forms carry no Sg/Pl distinction — request them "
        "with P3 alone (e.g. \"Ind+Pres+P3\", never \"...+P3+Sg\")."
    ),
    args={
        "lemma": "Prussian base form (from lookup_prussian_word).",
        "features": (
            "optional comma-separated feature filter.  Accepts human-readable "
            "names (participle, conjunctive, optative, present, preterite, "
            "infinitive, adverb) or raw FST tags (Part+Pass, Gen+Pl, "
            "Ind+Pres+P1, Ind+Pres+P3)."
        ),
    },
)

VALIDATE = ToolSpec(
    name="validate_prussian",
    description=(
        "Grammar check of Prussian text (FST + CG3 pipeline, three-valued).\n\n"
        "The single grammar tool: validates sentences and can also return the "
        "full dependency analysis.  Grammar-check your draft BEFORE you submit "
        "it; fix every \"error\" violation, then validate again.\n\n"
        "Returns JSON: {\"overall\": {\"status\", \"n_sentences\", "
        "\"n_violations\"}, \"sentences\": [{sent_id, text, status, "
        "violations, coverage, conllu?}, ...]}.\n\n"
        "Interpretation guide (IMPORTANT):\n\n"
        "- verified_in_coverage is the ONLY positive evidence of "
        "well-formedness: all words analyzed, low ambiguity, and at least one "
        "check family applied to the sentence.\n"
        "- out_of_coverage does NOT mean correct — the checker simply cannot "
        "verify (unknown words, collapsed analyses, residual ambiguity, or no "
        "applicable check; see coverage.reasons).  Never treat it as "
        "approval.\n"
        "- violations_found: each violations[] entry names the rule, the "
        "offending form (with index and surviving reading), a message, and a "
        "severity — \"error\" (case government / valency / person clash) is "
        "reliable; \"warning\" (adjective agreement, nominative in PP) is often "
        "a loanword paradigm gap rather than a real error."
    ),
    args={
        "text": "Prussian sentence(s) to check (one or more sentences).",
        "include_conllu": (
            "also include each sentence's CoNLL-U dependency analysis (field "
            "\"conllu\", with rule provenance Rule=/ / AgrParent= in MISC) — "
            "same pipeline run, no extra cost.  Set True when you need the "
            "parse, not just the verdict."
        ),
    },
)

ALL = (SEARCH, LOOKUP, WORDFORMS, VALIDATE)
BY_NAME = {s.name: s for s in ALL}
