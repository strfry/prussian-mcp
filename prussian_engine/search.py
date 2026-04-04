"""Search engine for Prussian Dictionary using embeddings."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import (
    DICTIONARY_PATH,
    EMBEDDINGS_PATH,
    QUERY_PREFIX,
    RERANK_API_KEY,
    RERANK_EMBEDDING_DIM,
)

from .embedding_client import EmbeddingClient
from .pgr import extract_pgr_from_entry, match_pgr, parse_pgr, build_pgr


class SearchEngine:
    """Semantic search engine using precomputed embeddings."""

    def __init__(self):
        """Initialize search engine by loading dictionary and embeddings."""
        self.entries: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.word_to_entry: Dict[str, Dict[str, Any]] = {}
        self.form_to_lemma: Dict[str, str] = {}
        self.form_to_pgr: Dict[str, List[str]] = {}

        if RERANK_API_KEY:
            self.embedding_client = EmbeddingClient()
        else:
            raise ValueError("RERANK_API_KEY environment variable is required")

        self.reranker_available = False  # Reranking handled separately

        self._load_dictionary()
        self._load_embeddings()
        self._build_indices()

    def _load_dictionary(self):
        """Load dictionary entries that match the embeddings."""
        # Load from embeddings directory - these entries match the embedding order!
        entries_file = Path(f"{EMBEDDINGS_PATH}.entries.json")
        if not entries_file.exists():
            raise FileNotFoundError(f"Entries file not found: {entries_file}")

        with open(entries_file, "r", encoding="utf-8") as f:
            self.entries = json.load(f)
        print(f"Loaded {len(self.entries)} dictionary entries")

    def _load_embeddings(self):
        """Load precomputed embeddings from .npy file."""
        emb_file = Path(f"{EMBEDDINGS_PATH}.embeddings.npy")
        if not emb_file.exists():
            raise FileNotFoundError(f"Embeddings not found: {emb_file}")

        self.embeddings = np.load(emb_file)
        print(f"Loaded embeddings: {self.embeddings.shape}")

    def _build_indices(self):
        """Build lookup indices for words and forms."""
        for entry in self.entries:
            word = entry.get("word", "").lower()
            if word:
                self.word_to_entry[word] = entry

        for entry in self.entries:
            lemma = entry.get("word", "").lower()
            forms_pgr = extract_pgr_from_entry(entry)
            for form, pgr in forms_pgr:
                if form:
                    form_lower = form.lower()
                    if form_lower not in self.form_to_lemma:
                        self.form_to_lemma[form_lower] = lemma
                    if form_lower not in self.form_to_pgr:
                        self.form_to_pgr[form_lower] = []
                    if pgr not in self.form_to_pgr[form_lower]:
                        self.form_to_pgr[form_lower].append(pgr)

        print(
            f"Indexed {len(self.word_to_entry)} lemmas and {len(self.form_to_lemma)} forms"
        )

    def _get_query_embedding(self, query_text: str) -> np.ndarray:
        """Encode query using embedding API."""
        try:
            embedding = self.embedding_client.get_embedding(query_text)
            return embedding
        except Exception as e:
            print(f"Error encoding query: {e}")
            return np.zeros(RERANK_EMBEDDING_DIM, dtype=np.float32)

    def _rerank_candidates(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder model."""
        if not candidates or not self.reranker_available:
            return candidates

        # Prepare query-document pairs
        documents = []
        for entry in candidates:
            word = entry.get("word", "")
            de = entry.get("de", "")
            en = entry.get("en", "")
            doc_text = f"{word} {de} {en}".strip()
            documents.append(doc_text)

        try:
            import requests

            base_url = OPENAI_BASE_URL.rstrip("/")
            response = requests.post(
                f"{base_url}/v3/embeddings/rerank",
                json={"model": RERANKER_MODEL, "query": query, "documents": documents},
            )

            if response.status_code == 200:
                data = response.json()
                # Reorder based on reranker scores
                results = [None] * len(candidates)
                for item in data.get("results", data.get("data", [])):
                    index = item.get("index", 0)
                    if index < len(candidates):
                        results[index] = candidates[index]
                return [r for r in results if r is not None][:top_k]
        except Exception as e:
            print(f"Reranker error: {e}")

        return candidates[:top_k]

    def _cosine_similarity(self, vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between vector and matrix."""
        # Normalize query vector
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 1e-10:
            vec = vec / vec_norm

        # Normalize matrix rows
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norms = np.clip(matrix_norms, a_min=1e-10, a_max=None)
        matrix = matrix / matrix_norms

        # Dot product = cosine for normalized vectors
        return np.dot(matrix, vec)

    def query(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Semantic search for dictionary entries.

        Args:
            query: Search query (German/English text)
            top_k: Number of results to return

        Returns:
            List of entries with translations (lemmas only)
        """
        if self.embeddings is None:
            return []

        # Add query prefix
        query_text = f"{QUERY_PREFIX}{query}"

        # Encode query via local embedding server
        query_embedding = self._get_query_embedding(query_text)

        # Compute cosine similarities
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][: top_k * 2]  # Get more to filter

        # Filter for entries with translations and format results
        results = []
        for idx in top_indices:
            if idx >= len(self.entries):
                continue

            entry = self.entries[idx]
            translations = entry.get("translations", {})

            # Only return lemmas with translations (not inflected forms)
            if translations:
                de_trans = translations.get("miks", [])
                en_trans = translations.get("engl", [])
                results.append(
                    {
                        "word": entry.get("word", ""),
                        "de": de_trans[0] if de_trans else "",
                        "en": en_trans[0] if en_trans else "",
                        "score": float(similarities[idx]),
                    }
                )

            if len(results) >= top_k:
                break

        return results

    def get_word_forms(self, lemma: str, filter_pgr: str = None) -> Dict[str, Any]:
        """
        Get all declension or conjugation forms for a Prussian lemma.

        Args:
            lemma: Prussian lemma (base form)
            filter_pgr: Optional PGR filter, e.g. "GEN.PL" or "PRES.1.SG"

        Returns:
            Dictionary with lemma, translations, and forms (flat list with PGR)
        """
        word_lower = lemma.lower().strip()
        if word_lower not in self.word_to_entry:
            return {"error": f"Word not found: {lemma}"}

        entry = self.word_to_entry[word_lower]
        translations = entry.get("translations", {})
        forms_pgr = extract_pgr_from_entry(entry)

        filtered_forms = []
        for form, pgr in forms_pgr:
            if filter_pgr and not match_pgr(pgr, filter_pgr):
                continue
            filtered_forms.append({"form": form, "pgr": pgr})

        return {
            "lemma": entry.get("word", ""),
            "translations": {
                "de": translations.get("miks", [""])[0],
                "en": translations.get("engl", [""])[0],
            },
            "gender": entry.get("gender", ""),
            "forms": filtered_forms,
        }

    def lookup(self, prussian_word: str, fuzzy: bool = True) -> List[Dict[str, Any]]:
        """
        Reverse lookup: Find Prussian word (lemma or inflected form).

        Args:
            prussian_word: Prussian word to look up
            fuzzy: If True, also try macron-normalized lookup if exact match fails

        Returns:
            List of matching entries with translations and forms
        """
        word_lower = prussian_word.lower().strip()
        results = []

        # Try exact match first
        # Check if it's a lemma
        if word_lower in self.word_to_entry:
            entry = self.word_to_entry[word_lower]
            results.append(self._format_lookup_result(entry, matched_form=word_lower))

        # Check if it's an inflected form
        elif word_lower in self.form_to_lemma:
            lemma = self.form_to_lemma[word_lower]
            entry = self.word_to_entry.get(lemma)
            if entry:
                results.append(
                    self._format_lookup_result(entry, matched_form=word_lower)
                )

        # Macron-normalized lookup (always, not just fuzzy)
        if not results:
            word_normalized = self._normalize_macrons(word_lower)

            # Find all lemmata that match when normalized
            for lemma, entry in self.word_to_entry.items():
                if self._normalize_macrons(lemma) == word_normalized:
                    results.append(
                        self._format_lookup_result(entry, matched_form=word_lower)
                    )

            # Find all forms that match when normalized
            if not results:
                for form, lemma in self.form_to_lemma.items():
                    if self._normalize_macrons(form) == word_normalized:
                        entry = self.word_to_entry.get(lemma)
                        if entry:
                            result = self._format_lookup_result(
                                entry, matched_form=word_lower
                            )
                            if result not in results:
                                results.append(result)

            # Levenshtein distance fallback on normalized words (fuzzy only)
            if not results and fuzzy:
                word_norm = word_normalized
                candidates = []
                for lemma, entry in self.word_to_entry.items():
                    lemma_norm = self._normalize_macrons(lemma)
                    dist = self._levenshtein_distance(word_norm, lemma_norm)
                    if dist <= 2:
                        score = self._fuzzy_score(word_norm, lemma_norm, dist)
                        candidates.append((score, dist, lemma, entry))

                # Sort by fuzzy score first, then use reranker on top candidates
                candidates.sort(key=lambda x: (-x[0], x[1]))
                top_candidates = candidates[:10]

                if self.reranker_available:
                    # Use reranker to get better ordering
                    formatted = [
                        self._format_lookup_result(e, matched_form=word_lower)
                        for _, _, _, e in top_candidates
                    ]
                    reranked = self._rerank_candidates(word_lower, formatted, top_k=5)
                    results.extend(reranked)
                else:
                    # Fallback to fuzzy score
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
        """Simplify PGR by outputting only common features.

        GEN.PL.MASC|GEN.PL.FEM|GEN.PL.NEUT → GEN.PL
        NOM.SG.MASC|NOM.SG.FEM → NOM.SG
        NOM.SG.MASC|GEN.PL.FEM → NOM.SG.MASC|GEN.PL.FEM (no simplification)

        Args:
            pgr_string: Pipe-separated PGR strings

        Returns:
            Simplified PGR string
        """
        if not pgr_string or "|" not in pgr_string:
            return pgr_string

        pgrs = pgr_string.split("|")
        if len(pgrs) == 1:
            return pgr_string

        features_list = [parse_pgr(p) for p in pgrs]

        common_keys = set(features_list[0].keys())
        for features in features_list[1:]:
            common_keys &= set(features.keys())

        if not common_keys:
            return pgr_string

        keys_to_remove = []
        for key in common_keys:
            values = set(f[key] for f in features_list if key in f)
            if len(values) > 1:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            common_keys.discard(key)

        if not common_keys:
            return pgr_string

        common_features = {k: features_list[0][k] for k in common_keys}
        return build_pgr(common_features)

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

        entry = self.word_to_entry[target_lower]
        forms_pgr = extract_pgr_from_entry(entry)

        matching = [(f, p) for f, p in forms_pgr if f.lower() == ref_word.lower()]

        if not matching:
            return []

        translations = entry.get("translations", {})
        de_trans = translations.get("miks", [])
        en_trans = translations.get("engl", [])

        pgrs = [p for _, p in matching]
        pgr_string = "|".join(pgrs)
        simplified_pgr = self._simplify_pgr(pgr_string)

        return [
            {
                "word": entry.get("word", ""),
                "de": de_trans[0] if de_trans else "",
                "en": en_trans[0] if en_trans else "",
                "matched_form": ref_word,
                "pgr": simplified_pgr,
            }
        ]

    def _format_lookup_result(
        self, entry: Dict[str, Any], matched_form: str = None
    ) -> Dict[str, Any]:
        """Format an entry for lookup results.

        Reference entries are formatted with desc only (no pgr).
        Lemma entries with matched forms get pgr from form index.
        """
        translations = entry.get("translations", {})
        de_trans = translations.get("miks", [])
        en_trans = translations.get("engl", [])

        result = {
            "word": entry.get("word", ""),
            "de": de_trans[0] if de_trans else "",
            "en": en_trans[0] if en_trans else "",
        }

        if entry.get("gender"):
            result["gender"] = entry["gender"]

        desc = entry.get("desc", "")

        if desc.startswith("↑"):
            result["desc"] = desc
            return result

        if matched_form and matched_form != entry.get("word", "").lower():
            result["matched_form"] = matched_form
            pgrs = self.form_to_pgr.get(matched_form.lower(), [])
            if pgrs:
                pgr_string = "|".join(pgrs)
                result["pgr"] = self._simplify_pgr(pgr_string)

        return result
