"""Search engine for Prussian Dictionary using embeddings."""

import re
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

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
        # indexed: the FST analyzer resolves them (``lookup_prussian_word``).
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
                 "pos": e.get("pos", ""),
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
