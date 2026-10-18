"""Search engine for Prussian Dictionary using embeddings."""

import re
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
from prussian_fst.nvh import read_entries


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
        self.id_to_entry: Dict[str, Dict[str, Any]] = {}

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

        # Word index + translation enrichment come from the dictionary (the
        # lean NVH), not a second embedding store.  Inflected forms are not
        # indexed: the FST analyzer resolves them (see ``_exact_lookup``).
        if not DICTIONARY_PATH.exists():
            raise FileNotFoundError(
                f"Dictionary not found at {DICTIONARY_PATH}. Build the lean NVH "
                "(../corpus, ../fst) or set PRUSSIAN_DICTIONARY."
            )
        self.entries = read_entries(DICTIONARY_PATH)

        print("Building indices...", file=sys.stderr)
        self._build_indices()


    def _build_indices(self):
        """Build the word, entry-id, exact-translation and chunk indices."""
        for entry in self.entries:
            word = entry.get("word", "").lower()
            if word:
                self.word_to_entry.setdefault(word, []).append(entry)
            self.id_to_entry[entry.get("id") or word] = entry

        # Exact-translation index (all six languages) → entry ids, plus
        # entry id → chunk indices, so a literal match can be pinned on top
        # of the fuzzy BM25/dense ranking (see ``query``).
        self.translation_to_lemmas: Dict[str, Dict[str, int]] = {}
        for entry_id, entry in self.id_to_entry.items():
            tr = entry.get("translations")
            if not isinstance(tr, dict):
                continue
            for vals in tr.values():
                for val in ([vals] if isinstance(vals, str) else vals or []):
                    for key, tier in self._translation_keys(val).items():
                        tiers = self.translation_to_lemmas.setdefault(key, {})
                        tiers[entry_id] = min(tier, tiers.get(entry_id, tier))
        self.lemma_to_chunks: Dict[str, List[int]] = {}
        for idx, rec in enumerate(self.store.records):
            for entry_id in self._record_ids(rec):
                self.lemma_to_chunks.setdefault(entry_id, []).append(idx)

        print(f"Indexed {len(self.word_to_entry)} lemmas "
              f"({len(self.id_to_entry)} entries)")

    def _record_ids(self, rec: Dict[str, Any]) -> List[str]:
        """Entry ids of a chunk: ``member_ids``, else all entries of its words."""
        if rec.get("member_ids"):
            return list(rec["member_ids"])
        ids: List[str] = []
        for m in dict.fromkeys(w.lower() for w in rec.get("members", [])):
            for e in self.word_to_entry.get(m, []):
                ids.append(e.get("id") or m)
        return ids

    @staticmethod
    def _translation_keys(text: str) -> Dict[str, int]:
        """Normalized match keys for a translation or query string → tier.

        Lowercased and split on ``,``/``;``.  Tier 0 is the part as written;
        tier 1 drops parentheticals (``to sink (go down)`` → ``to sink``).
        """
        keys: Dict[str, int] = {}

        def add(part: str, tier: int) -> None:
            part = " ".join(part.split())
            if part and tier < keys.get(part, 2):
                keys[part] = tier

        for part in re.split(r"[,;]", text.lower()):
            add(part, 0)
            add(re.sub(r"\([^)]*\)", " ", part), 1)
        return keys

    def _exact_lemmas(self, query: str) -> List[str]:
        """Entry ids with a translation literally equal to ``query``, best first.

        Tier sums rank a match on the full translation text above one that
        only matches after dropping parentheticals.
        """
        best: Dict[str, int] = {}
        for key, q_tier in self._translation_keys(query).items():
            for entry_id, t_tier in self.translation_to_lemmas.get(key, {}).items():
                best[entry_id] = min(q_tier + t_tier, best.get(entry_id, 9))
        return sorted(best, key=lambda entry_id: best[entry_id])

    def _exact_lookup(self, word: str, macron: bool = False) -> list:
        """Look up a word exactly: as a lemma, else as a form via the FST.

        Inflected forms are resolved by the FST analyzer (strict stages, see
        :func:`prussian.engine.fst.tags.lemma_tags`); ``macron=True`` uses the
        macron-loss analyzer instead.

        Returns:
            List of formatted lookup results (empty if not found).
        """
        from prussian.engine.fst.tags import lemma_tags

        analyses = lemma_tags(word, macron=macron)
        results = []
        if not macron and word in self.word_to_entry:
            for entry in self.word_to_entry[word]:
                results.append(self._format_lookup_result(
                    entry, matched_form=word, tags=analyses.get(entry["word"])))
            return results

        for lemma, tags in analyses.items():
            for entry in self.word_to_entry.get(lemma.lower(), []):
                result = self._format_lookup_result(entry, matched_form=word, tags=tags)
                if result not in results:
                    results.append(result)
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
        """Lemma scan with macrons stripped (ā→a, ē→e, etc.), else the FST.

        Lemmas are compared macron-normalized; an inflected form goes
        through the FST's macron-loss analyzer.
        """
        results = []
        word_normalized = self._normalize_macrons(word)

        for lemma, entries in self.word_to_entry.items():
            if self._normalize_macrons(lemma) == word_normalized:
                for entry in entries:
                    results.append(self._format_lookup_result(entry, matched_form=word))

        if not results:
            results = self._exact_lookup(word, macron=True)

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

        Chunks containing a lemma whose translation literally equals the
        query (any language) come first; their matching entries carry
        ``exact: True``.  The fuzzy ranking fills the remaining slots.

        Args:
            query: Search query (German/English text)
            top_k: Number of results to return

        Returns:
            List of chunks ``{lemma, members, pos, score, text, entries}``.
        """
        if self.store is None:
            return []

        exact_ids = self._exact_lemmas(query)
        exact = set(exact_ids)
        pinned: List[int] = []
        for entry_id in exact_ids:
            for idx in self.lemma_to_chunks.get(entry_id, []):
                if idx not in pinned:
                    pinned.append(idx)
        pinned_recs = [self.store.records[i] for i in pinned]
        pinned_ids = {id(r) for r in pinned_recs}

        if self.bm25 is not None:
            hits = hybrid_query(self.store, self.embedder, self.bm25, query,
                                k=top_k, query_prefix=self.query_prefix,
                                bm25_prefix=getattr(self, "bm25_prefix", None),
                                bm25_ngram=getattr(self, "bm25_ngram", None))
        else:
            hits = self.store.query(self.embedder, query,
                                    k=top_k, query_prefix=self.query_prefix)
        hits = ([(r, 1.0) for r in pinned_recs]
                + [(r, s) for r, s in hits if id(r) not in pinned_ids])[:top_k]

        results = []
        for record, score in hits:
            members = record.get("members", [])
            entries = [
                {"id": entry_id, "word": e["word"],
                 "translations": e.get("translations", {}),
                 "gender": e.get("gender", ""),
                 **({"exact": True} if entry_id in exact else {})}
                for entry_id in self._record_ids(record)
                if (e := self.id_to_entry.get(entry_id))
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

    def _resolve_reference(self, target: str, ref_word: str) -> List[Dict[str, Any]]:
        """Resolve a reference to a lemma and return formatted results.

        Args:
            target: Target lemma name (e.g., "abbai")
            ref_word: The reference word (e.g., "abbejan")

        Returns:
            One result per target entry.  The reference itself comes from
            the dictionary (``↑ Target`` in desc); ``pgr`` is added when the
            FST analyzes ``ref_word`` as a form of the target.
        """
        from prussian.engine.fst.tags import lemma_tags

        target_lower = target.lower()
        if target_lower not in self.word_to_entry:
            return []

        analyses = {lemma.lower(): tags for lemma, tags in lemma_tags(ref_word).items()}
        tags = analyses.get(target_lower)
        return [{
            "word": entry.get("word", ""),
            "translations": entry.get("translations", {}),
            "matched_form": ref_word,
            **({"pgr": "|".join(tags)} if tags else {}),
        } for entry in self.word_to_entry[target_lower]]

    def _format_lookup_result(
        self, entry: Dict[str, Any], matched_form: str = None,
        method: str = None, rule_applied: str = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Format an entry for lookup results.

        Includes ``pgr`` (the FST analyses of the matched form, ``|``-joined)
        when ``tags`` are given, and ``form`` when the matched form differs
        from the lemma.  Includes ``method`` / ``rule_applied`` metadata when
        apply_rules is active.
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

            if tags:
                result["pgr"] = "|".join(tags)

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
