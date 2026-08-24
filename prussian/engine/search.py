"""Search engine for Prussian Dictionary using embeddings."""

import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from prussian.config import EMBEDDINGS_PATH, QUERY_PREFIX, HYBRID_SEARCH, DICTIONARY_PATH

from prussian_embeddings import (
    get_embedder,
    EmbeddingStore,
    BM25Index,
    hybrid_query,
    prefix_tokens,
    ngram_tokens,
)
from prussian.engine.morphology import extract_pgr_from_entry, match_pgr, parse_pgr, build_pgr, _parse_pronoun


class SearchEngine:
    """Semantic search engine using precomputed embeddings."""

    # Prefix stripping rules: (prefix_string, rule_name) ordered longest-first
    PREFIX_RULES: list = [
        ("prei", "prei"),
        ("pra", "pra"),
        ("per", "per"),
        ("sen", "sen"), ("san", "san"), ("su", "su"),
        ("nī", "nī"), ("ni", "ni"), ("ne", "ne"),
        ("pa", "pa"),
        ("iz", "iz"),
        ("en", "en"),
        ("et", "et"), ("eb", "eb"),
        ("au", "au"),
    ]

    # Orthographic transformation rules:
    #   (rule_name, [(pattern, replacement, suffix_only), ...])
    # suffix_only=True means only apply at word end
    ORTHO_RULES: list = [
        ("macron_normalize", [("ā", "a", False), ("ē", "e", False), ("ī", "i", False), ("ō", "o", False), ("ū", "u", False)]),
        ("vowel_macron_shift", [("ī", "ē", False), ("ū", "ā", False), ("ū", "ō", False)]),
        ("prothetic_w", [("und", "wund", False), ("āb", "wāb", False)]),
        ("sibilant_onset", [("šl", "skl", False), ("šp", "sp", False)]),
        ("consonant_cluster", [("šš", "ssj", False), ("č", "dž", False)]),
        ("nertiks_vowel", [("lan", "lin", True), ("le", "la", False), ("je", "ja", False)]),
        ("vowel_shift", [("bilin", "bilan", True)]),
        ("diphthong_glide", [("ei", "jai", True), ("eis", "jais", True)]),
    ]

    def __init__(self):
        """Initialize search engine by loading dictionary and embeddings."""
        self.word_to_entry: Dict[str, List[Dict[str, Any]]] = {}
        self.form_to_lemma: Dict[str, List[str]] = {}
        self.form_to_pgr: Dict[str, List[str]] = {}

        print("Loading embedding model...", file=sys.stderr)
        self.embedder = get_embedder()

        print(f"Loading embeddings ({EMBEDDINGS_PATH.name})...", file=sys.stderr)
        self.store = EmbeddingStore.load(str(EMBEDDINGS_PATH))

        # Der Query-Prefix reist mit dem Store (Meta); QUERY_PREFIX aus der
        # Env greift nur für Stores ohne Meta-Eintrag.  Ein LEERER Meta-Wert
        # ist gültig (z. B. Voyage: Asymmetrie via input_type-Parameter).
        self.query_prefix = self.store.meta.get("query_prefix", QUERY_PREFIX)

        # The retrieval store must be a chunk store: records carry text + members.
        recs = self.store.records
        if recs and not ("text" in recs[0] and "members" in recs[0]):
            raise ValueError(
                f"EMBEDDINGS_NAME ({EMBEDDINGS_PATH.name}) is not a chunk store "
                "(records lack 'text'/'members'). Build one with "
                "`prussian-embeddings-build-chunks`."
            )
        # Three lexical channels over the same chunk texts, fused with the
        # dense one in `query()`.  Exact tokens are the precise channel;
        # prefix-truncated tokens and character n-grams cover the inflected
        # queries an agent actually types ("ženklai" for "ženklas"), which
        # exact BM25 misses entirely and the dense channel only half-catches.
        chunk_texts = [r["text"] for r in recs]
        if HYBRID_SEARCH:
            self.bm25 = BM25Index(chunk_texts)
            self.bm25_prefix = BM25Index(chunk_texts, tokenizer=prefix_tokens)
            self.bm25_ngram = BM25Index(chunk_texts, tokenizer=ngram_tokens)
        else:
            self.bm25 = self.bm25_prefix = self.bm25_ngram = None

        # Word/form indices + translation enrichment come from the dictionary
        # (the canonical source), not a second embedding store.
        if not DICTIONARY_PATH.exists():
            raise FileNotFoundError(
                f"Dictionary not found at {DICTIONARY_PATH}. Run `make download` "
                "or set PRUSSIAN_DICTIONARY."
            )
        import json as _json
        with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
            self.entries = _json.load(f)

        print("Building indices...", file=sys.stderr)
        self._build_indices()


    def _build_indices(self):
        """Build lookup indices for words and forms."""
        for entry in self.entries:
            word = entry.get("word", "").lower()
            if word:
                self.word_to_entry.setdefault(word, []).append(entry)

        for entry in self.entries:
            lemma = entry.get("word", "").lower()
            forms_pgr = extract_pgr_from_entry(entry)
            for form, pgr in forms_pgr:
                if form:
                    form_lower = form.lower()
                    lemmas = self.form_to_lemma.setdefault(form_lower, [])
                    if lemma not in lemmas:
                        lemmas.append(lemma)
                    if form_lower not in self.form_to_pgr:
                        self.form_to_pgr[form_lower] = []
                    if pgr not in self.form_to_pgr[form_lower]:
                        self.form_to_pgr[form_lower].append(pgr)

        print(
            f"Indexed {len(self.word_to_entry)} lemmas and {len(self.form_to_lemma)} forms"
        )

    def _exact_lookup(self, word: str) -> list:
        """Look up a word exactly in lemma and form indices.

        Returns:
            List of formatted lookup results (empty if not found).
        """
        results = []

        if word in self.word_to_entry:
            for entry in self.word_to_entry[word]:
                results.append(self._format_lookup_result(entry, matched_form=word))
            return results

        if word in self.form_to_lemma:
            for lemma in self.form_to_lemma[word]:
                for entry in self.word_to_entry.get(lemma, []):
                    result = self._format_lookup_result(entry, matched_form=word)
                    if result not in results:
                        results.append(result)
            return results

        return results

    def _try_prefix_rules(self, word: str, _depth: int = 0) -> list:
        """Try stripping known prefixes and doing exact lookup on the root.

        First match wins; results are annotated with method/rule_applied.
        Composability: if exact on root fails, also tries ortho rules on root.
        """
        if _depth > 1:
            return []
        for prefix, rule_name in self.PREFIX_RULES:
            if word.startswith(prefix) and len(word) > len(prefix):
                root = word[len(prefix):]
                results = self._exact_lookup(root)
                if results:
                    for r in results:
                        r["method"] = "prefix_stripped"
                        r["rule_applied"] = rule_name
                    return results
                # Composability: try ortho transforms on the root
                results = self._try_ortho_rules(root, _depth + 1)
                if results:
                    for r in results:
                        r["method"] = "prefix_stripped"
                        r["rule_applied"] = rule_name
                    return results
        return []

    def _macron_lookup(self, word: str) -> list:
        """Full-index scan with macrons stripped (ā→a, ē→e, etc.).

        Scans all lemmas and inflected forms for a normalized match.
        When an inflected form is found, the dictionary form (with
        original macrons) is passed as ``matched_form`` so that the
        result includes the correct ``form`` and ``pgr`` fields.
        """
        results = []
        word_normalized = self._normalize_macrons(word)

        for lemma, entries in self.word_to_entry.items():
            if self._normalize_macrons(lemma) == word_normalized:
                for entry in entries:
                    results.append(self._format_lookup_result(entry, matched_form=word))

        if not results:
            for form, lemmas in self.form_to_lemma.items():
                if self._normalize_macrons(form) == word_normalized:
                    for lemma in lemmas:
                        for entry in self.word_to_entry.get(lemma, []):
                            result = self._format_lookup_result(
                                entry, matched_form=form
                            )
                            if result not in results:
                                results.append(result)

        return results

    def _try_ortho_rules(self, word: str, _depth: int = 0) -> list:
        """Apply orthographic transformations and try exact lookup.

        First rule in ORTHO_RULES is macron_normalize (full-index scan).
        Remaining rules generate candidate forms via pattern replacement.
        First match wins; results are annotated with method/rule_applied.
        Composability: if exact on candidate fails, also tries prefix on candidate.
        """
        if _depth > 1:
            return []
        for rule_name, patterns in self.ORTHO_RULES:
            if rule_name == "macron_normalize":
                results = self._macron_lookup(word)
                if results:
                    for r in results:
                        r["method"] = "ortho_transform"
                        r["rule_applied"] = rule_name
                    return results
            else:
                for pattern, replacement, suffix_only in patterns:
                    if suffix_only:
                        if word.endswith(pattern):
                            candidate = word[:-len(pattern)] + replacement
                        else:
                            continue
                    else:
                        if pattern not in word:
                            continue
                        candidate = word.replace(pattern, replacement)
                    if candidate == word:
                        continue
                    results = self._exact_lookup(candidate)
                    if results:
                        for r in results:
                            r["method"] = "ortho_transform"
                            r["rule_applied"] = rule_name
                        return results
                    # Composability: try prefix stripping on the candidate
                    results = self._try_prefix_rules(candidate, _depth + 1)
                    if results:
                        for r in results:
                            r["method"] = "ortho_transform"
                            r["rule_applied"] = rule_name
                        return results
        return []


    def query(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Semantic search over the chunk store (BM25+dense RRF or dense-only).

        Args:
            query: Search query (German/English text)
            top_k: Number of results to return

        Returns:
            List of chunks ``{lemma, members, pos, score, text, entries}``.
        """
        if self.store is None:
            return []

        if self.bm25 is not None:
            hits = hybrid_query(self.store, self.embedder, self.bm25, query,
                                k=top_k, query_prefix=self.query_prefix,
                                bm25_prefix=getattr(self, "bm25_prefix", None),
                                bm25_ngram=getattr(self, "bm25_ngram", None))
        else:
            hits = self.store.query(self.embedder, query,
                                    k=top_k, query_prefix=self.query_prefix)

        results = []
        for record, score in hits:
            members = record.get("members", [])
            entries = [
                {"word": e["word"], "translations": e.get("translations", {})}
                for m in members
                for e in self.word_to_entry.get(m.lower(), [])
            ]
            results.append({
                "lemma": record.get("lemma", ""),
                "members": members,
                "pos": record.get("pos", ""),
                "score": float(score),
                "text": record["text"],
                "entries": entries,
            })
        return results

    def get_word_forms(self, lemma: str, filter_pgr: str = None) -> Dict[str, Any]:
        """
        Get all declension or conjugation forms for a Prussian lemma.

        Args:
            lemma: Prussian lemma (base form)
            filter_pgr: Optional PGR filter, e.g. "GEN.PL" or "PRS.1.SG"

        Returns:
            Dictionary with lemma, translations, and structured forms by category
        """
        word_lower = lemma.lower().strip()
        if word_lower not in self.word_to_entry:
            return {"error": f"Word not found: {lemma}"}

        entries = self.word_to_entry[word_lower]
        results = []
        for entry in entries:
            translations = entry.get("translations", {})
            raw_forms = entry.get("forms", {})

            all_forms_pgr = extract_pgr_from_entry(entry)

            filtered_forms = []
            for form, pgr in all_forms_pgr:
                if filter_pgr and not match_pgr(pgr, filter_pgr):
                    continue
                filtered_forms.append({"form": form, "pgr": pgr})

            categorized_forms = self._structure_forms(raw_forms, entry)

            # Determine which categories actually have data
            available_categories = [
                cat for cat, items in categorized_forms.items() if items
            ]

            result = {
                "lemma": entry.get("word", ""),
                "translations": translations,
                "gender": entry.get("gender", ""),
                "forms": categorized_forms,
                "available_categories": available_categories,
                "desc": entry.get("desc", ""),
            }
            if filter_pgr:
                result["filtered_forms"] = filtered_forms

            results.append(result)

        return results

    def _structure_forms(self, raw_forms: Dict, entry: Dict) -> Dict[str, Any]:
        """Convert raw forms into a structured dict with named categories."""
        structured = {}

        if raw_forms.get("indicative"):
            indicative = []
            for mood_data in raw_forms["indicative"]:
                tense = mood_data.get("tense", "")
                for fi in mood_data.get("forms", []):
                    pronoun = fi.get("pronoun", "")
                    form_text = fi.get("form", "")
                    person, number = _parse_pronoun(pronoun)
                    indicative.append({
                        "tense": tense,
                        "person": person,
                        "number": number,
                        "pronoun": pronoun,
                        "form": form_text,
                    })
            structured["indicative"] = indicative

        if raw_forms.get("optative"):
            structured["optative"] = raw_forms["optative"]

        if raw_forms.get("subjunctive"):
            subj = []
            for fi in raw_forms["subjunctive"]:
                pronoun = fi.get("pronoun", "")
                form_text = fi.get("form", "")
                person, number = _parse_pronoun(pronoun)
                subj.append({
                    "person": person,
                    "number": number,
                    "pronoun": pronoun,
                    "form": form_text,
                })
            structured["subjunctive"] = subj

        if raw_forms.get("imperative"):
            imp = []
            for fi in raw_forms["imperative"]:
                pronoun = fi.get("pronoun", "")
                form_text = fi.get("form", "")
                person, number = _parse_pronoun(pronoun)
                imp.append({
                    "person": person,
                    "number": number,
                    "pronoun": pronoun,
                    "form": form_text,
                })
            structured["imperative"] = imp

        if raw_forms.get("participles"):
            structured["participles"] = raw_forms["participles"]

        if raw_forms.get("declension"):
            structured["declension"] = raw_forms["declension"]

        if raw_forms.get("adverb"):
            structured["adverb"] = raw_forms["adverb"]

        if raw_forms.get("comparison"):
            structured["comparison"] = raw_forms["comparison"]

        return structured

    def lookup(self, prussian_word: str, fuzzy: bool = False, apply_rules: bool = True) -> List[Dict[str, Any]]:
        """
        Reverse lookup: Find Prussian word (lemma or inflected form).

        Args:
            prussian_word: Prussian word to look up
            fuzzy: If True, try Levenshtein distance fallback when nothing else matches
            apply_rules: If True, try prefix stripping and orthographic rules
                when exact lookup fails. Macron normalization is included as the
                first orthographic rule.

        Returns:
            List of matching entries with translations, forms, and when
            apply_rules=True: method and rule_applied fields.
        """
        word_lower = prussian_word.lower().strip()
        results = []

        # 1. Exact match (lemma or inflected form)
        results = self._exact_lookup(word_lower)
        if results and apply_rules:
            for r in results:
                r["method"] = "exact"

        # 2. Prefix stripping rules
        if not results and apply_rules:
            results = self._try_prefix_rules(word_lower)
            if results:
                for r in results:
                    r["matched_form"] = word_lower

        # 3. Orthographic transformation rules (macron normalize is first)
        if not results and apply_rules:
            results = self._try_ortho_rules(word_lower)
            if results:
                for r in results:
                    r["matched_form"] = word_lower

        # 4. Levenshtein distance fallback (fuzzy only)
        if not results and fuzzy:
            word_normalized = self._normalize_macrons(word_lower)
            candidates = []
            for lemma, entries in self.word_to_entry.items():
                lemma_norm = self._normalize_macrons(lemma)
                dist = self._levenshtein_distance(word_normalized, lemma_norm)
                if dist <= 2:
                    score = self._fuzzy_score(word_normalized, lemma_norm, dist)
                    for entry in entries:
                        candidates.append((score, dist, lemma, entry))

            candidates.sort(key=lambda x: (-x[0], x[1]))
            top_candidates = candidates[:10]

            for score, dist, lemma, entry in top_candidates[:5]:
                result = self._format_lookup_result(
                    entry, matched_form=word_lower
                )
                if result not in results:
                    results.append(result)

        results = self._follow_references(results)
        return results

    def _follow_references(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add lemma results for reference entries.

        For each reference entry found, resolve to its main lemma
        and add the lemma result with matching form's pgr.
        """
        all_results = list(results)

        for result in results:
            if "desc" in result and result["desc"].startswith("↑"):
                target = self._extract_reference_target(result["desc"])
                if target:
                    lemma_results = self._resolve_reference(target, result["word"])
                    for lr in lemma_results:
                        if lr not in all_results:
                            all_results.append(lr)

        return all_results

    def _normalize_macrons(self, word: str) -> str:
        """Remove macrons for fuzzy matching."""
        return (
            word.replace("ā", "a")
            .replace("ē", "e")
            .replace("ī", "i")
            .replace("ō", "o")
            .replace("ū", "u")
        )

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    def _fuzzy_score(self, query: str, candidate: str, lev_dist: int) -> float:
        """Calculate a fuzzy match score based on multiple factors.

        Higher score = better match.
        """
        score = 0.0

        # Base score from Levenshtein (lower distance = higher score)
        score += 10 - lev_dist * 3

        # Prefix match bonus (first 3-4 chars same)
        for prefix_len in [3, 4]:
            if len(query) >= prefix_len and len(candidate) >= prefix_len:
                if query[:prefix_len] == candidate[:prefix_len]:
                    score += prefix_len * 2
                    break

        # Length similarity bonus (similar length is better)
        len_diff = abs(len(query) - len(candidate))
        score += max(0, 5 - len_diff)

        # Substring bonus (one contains the other as substring)
        if query in candidate:
            score += len(query) * 2
        elif candidate in query:
            score += len(candidate) * 2

        # Common prefix length (longer common prefix = better)
        common_prefix_len = 0
        for i in range(min(len(query), len(candidate))):
            if query[i] == candidate[i]:
                common_prefix_len += 1
            else:
                break
        score += common_prefix_len * 0.5

        # Ending match bonus
        for suffix_len in [2, 3]:
            if (
                query.endswith(candidate[-suffix_len:])
                if len(candidate) >= suffix_len
                else False
            ):
                score += suffix_len
            if (
                candidate.endswith(query[-suffix_len:])
                if len(query) >= suffix_len
                else False
            ):
                score += suffix_len

        return score

    def _extract_reference_target(self, desc: str) -> Optional[str]:
        """Extract target lemma from reference description.

        Args:
            desc: Description like "↑ Abbai dat" or "↑ Dēiws acc"

        Returns:
            Target lemma or None if not a reference
        """
        if not desc or not desc.startswith("↑"):
            return None

        parts = desc[1:].strip().split()
        if parts:
            return parts[0]
        return None

    def _simplify_pgr(self, pgr_string: str) -> str:
        """Collapse a pipe-separated alternation when exactly one feature varies.

        GEN.PL.MASC|GEN.PL.FEM|GEN.PL.NEUT → GEN.PL  (only GENDER varies)
        PST.SG.1.IND|PST.SG.2.IND|PST.SG.3.IND → PST.SG.IND  (only PERSON varies)
        ACC.SG.FEM|NOM.PL.FEM → ACC.SG.FEM|NOM.PL.FEM  (CASE and NUMBER vary)
        """
        if not pgr_string or "|" not in pgr_string:
            return pgr_string

        pgrs = pgr_string.split("|")
        if len(pgrs) == 1:
            return pgr_string

        features_list = [parse_pgr(p) for p in pgrs]

        all_keys = set()
        for features in features_list:
            all_keys |= set(features.keys())

        common_keys = set(features_list[0].keys())
        for features in features_list[1:]:
            common_keys &= set(features.keys())

        differing_keys = {
            k for k in common_keys
            if len({f[k] for f in features_list if k in f}) > 1
        }

        keys_not_in_all = all_keys - common_keys
        differing_keys |= keys_not_in_all

        if len(differing_keys) != 1:
            return pgr_string

        kept_keys = common_keys - differing_keys
        if not kept_keys:
            return pgr_string

        kept_features = {k: features_list[0][k] for k in kept_keys}
        return build_pgr(kept_features)

    def _resolve_reference(self, target: str, ref_word: str) -> List[Dict[str, Any]]:
        """Resolve a reference to a lemma and return formatted results.

        Args:
            target: Target lemma name (e.g., "abbai")
            ref_word: The reference word (e.g., "abbejan")

        Returns:
            List with single formatted result containing pgr from matching forms
        """
        target_lower = target.lower()
        if target_lower not in self.word_to_entry:
            return []

        results = []
        for entry in self.word_to_entry[target_lower]:
            forms_pgr = extract_pgr_from_entry(entry)
            matching = [(f, p) for f, p in forms_pgr if f.lower() == ref_word.lower()]

            if not matching:
                continue

            pgrs = [p for _, p in matching]
            pgr_string = "|".join(pgrs)
            simplified_pgr = self._simplify_pgr(pgr_string)

            results.append({
                "word": entry.get("word", ""),
                "translations": entry.get("translations", {}),
                "matched_form": ref_word,
                "pgr": simplified_pgr,
            })

        return results

    def _format_lookup_result(
        self, entry: Dict[str, Any], matched_form: str = None,
        method: str = None, rule_applied: str = None,
    ) -> Dict[str, Any]:
        """Format an entry for lookup results.

        Always includes ``pgr`` when a form match is found (inflected or
        lemma).  Includes ``form`` for the inflected standard form
        (dictionary form) when it differs from the lemma.  Includes
        ``method`` / ``rule_applied`` metadata when apply_rules is active.
        """
        translations = entry.get("translations", {})

        result: Dict[str, Any] = {
            "word": entry.get("word", ""),
            "translations": translations,
        }

        if entry.get("gender"):
            result["gender"] = entry["gender"]

        if matched_form:
            matched_lower = matched_form.lower()
            lemma_lower = entry.get("word", "").lower()

            # Find all matching forms with their PGR tags
            entry_pgrs = [
                pgr
                for form, pgr in extract_pgr_from_entry(entry)
                if form.lower() == matched_lower
            ]
            if entry_pgrs:
                seen = []
                for pgr in entry_pgrs:
                    if pgr not in seen:
                        seen.append(pgr)
                result["pgr"] = self._simplify_pgr("|".join(seen))

            # Show inflected form when it differs from lemma
            if matched_lower != lemma_lower:
                result["form"] = matched_form

        if method:
            result["method"] = method
        if rule_applied:
            result["rule_applied"] = rule_applied

        desc = entry.get("desc", "")
        if desc:
            result["desc"] = desc

        return result
