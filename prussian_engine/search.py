"""Search engine for Prussian Dictionary using E5 embeddings."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import DICTIONARY_PATH, EMBEDDINGS_PATH, QUERY_PREFIX

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class SearchEngine:
    """Semantic search engine using precomputed E5 embeddings."""

    def __init__(self):
        """Initialize search engine by loading dictionary and embeddings."""
        self.entries: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.word_to_entry: Dict[str, Dict[str, Any]] = {}
        self.form_to_lemma: Dict[str, str] = {}
        self.model = None

        self._load_dictionary()
        self._load_embeddings()
        self._build_indices()
        self._load_model()

    def _load_dictionary(self):
        """Load dictionary entries that match the embeddings."""
        # Load from embeddings directory - these entries match the embedding order!
        entries_file = Path(f"{EMBEDDINGS_PATH}.entries.json")
        if not entries_file.exists():
            raise FileNotFoundError(f"Entries file not found: {entries_file}")

        with open(entries_file, 'r', encoding='utf-8') as f:
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
        # Map lemmas to entries
        for entry in self.entries:
            word = entry.get('word', '').lower()
            if word:
                self.word_to_entry[word] = entry

        # Map inflected forms to lemmas
        for entry in self.entries:
            lemma = entry.get('word', '').lower()
            forms = self._extract_all_forms(entry)
            for form in forms:
                if form and form not in self.form_to_lemma:
                    self.form_to_lemma[form] = lemma

        print(f"Indexed {len(self.word_to_entry)} lemmas and {len(self.form_to_lemma)} forms")

    def _load_model(self):
        """Load E5 model for query encoding."""
        if not HAS_SENTENCE_TRANSFORMERS:
            print("Warning: sentence-transformers not installed, search will not work")
            return

        print("Loading E5 model for queries...")
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        print("Model loaded")

    def _extract_all_forms(self, entry: Dict[str, Any]) -> set:
        """Extract all inflected forms from an entry."""
        forms = set()
        forms_data = entry.get('forms', {})

        # Declension (nouns, adjectives)
        if 'declension' in forms_data:
            for decl in forms_data['declension']:
                for case in decl.get('cases', []):
                    if case.get('singular'):
                        forms.add(case['singular'].lower())
                    if case.get('plural'):
                        forms.add(case['plural'].lower())

        # Conjugation (verbs)
        for mood in ['indicative', 'subjunctive', 'optative', 'imperative']:
            if mood in forms_data:
                for item in forms_data[mood]:
                    if isinstance(item, dict):
                        if 'forms' in item:  # indicative has nested structure
                            for form_obj in item['forms']:
                                if form_obj.get('form'):
                                    forms.add(form_obj['form'].lower())
                        elif 'form' in item:  # other moods
                            forms.add(item['form'].lower())

        # Participles and infinitives
        for category in ['participles', 'infinitives']:
            if category in forms_data:
                for item in forms_data[category]:
                    if item.get('form'):
                        forms.add(item['form'].lower())

        return forms

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
        if self.embeddings is None or self.model is None:
            return []

        # Add E5 query prefix for asymmetric search
        query_text = f"{QUERY_PREFIX}{query}"

        # Encode query
        query_embedding = self.model.encode(query_text, convert_to_numpy=True)

        # Compute cosine similarities
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more to filter

        # Filter for entries with translations and format results
        results = []
        for idx in top_indices:
            if idx >= len(self.entries):
                continue

            entry = self.entries[idx]
            translations = entry.get('translations', {})

            # Only return lemmas with translations (not inflected forms)
            if translations:
                de_trans = translations.get('miks', [])
                en_trans = translations.get('engl', [])
                results.append({
                    'word': entry.get('word', ''),
                    'de': de_trans[0] if de_trans else '',
                    'en': en_trans[0] if en_trans else '',
                    'score': float(similarities[idx])
                })

            if len(results) >= top_k:
                break

        return results

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
                results.append(self._format_lookup_result(entry, matched_form=word_lower))

        # If no exact match and fuzzy is enabled, try macron-normalized lookup
        if not results and fuzzy:
            word_normalized = self._normalize_macrons(word_lower)

            # Find all lemmata that match when normalized
            for lemma, entry in self.word_to_entry.items():
                if self._normalize_macrons(lemma) == word_normalized:
                    results.append(self._format_lookup_result(entry, matched_form=word_lower))

            # Find all forms that match when normalized
            if not results:
                for form, lemma in self.form_to_lemma.items():
                    if self._normalize_macrons(form) == word_normalized:
                        entry = self.word_to_entry.get(lemma)
                        if entry:
                            result = self._format_lookup_result(entry, matched_form=word_lower)
                            if result not in results:
                                results.append(result)

        return results

    def _normalize_macrons(self, word: str) -> str:
        """Remove macrons for fuzzy matching."""
        return word.replace('ā', 'a').replace('ē', 'e').replace('ī', 'i').replace('ō', 'o').replace('ū', 'u')

    def _find_json_paths(self, obj, target: str, path: List[str] = []) -> List[List[str]]:
        """Recursively find all paths in a JSON structure where a value matches target."""
        results = []
        target_lower = target.lower()
        target_normalized = self._normalize_macrons(target_lower)

        if isinstance(obj, str):
            val = obj.lower()
            if val == target_lower or self._normalize_macrons(val) == target_normalized:
                results.append(list(path))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                results.extend(self._find_json_paths(v, target, path + [k]))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                results.extend(self._find_json_paths(v, target, path + [str(i)]))

        return results

    def _format_lookup_result(self, entry: Dict[str, Any], matched_form: str = None) -> Dict[str, Any]:
        """Format an entry for lookup results.

        If matched_form differs from the lemma, only return the paths
        where the form was found, not the entire paradigm.
        """
        translations = entry.get('translations', {})
        de_trans = translations.get('miks', [])
        en_trans = translations.get('engl', [])
        result = {
            'word': entry.get('word', ''),
            'de': de_trans[0] if de_trans else '',
            'en': en_trans[0] if en_trans else '',
        }

        if entry.get('gender'):
            result['gender'] = entry['gender']

        if matched_form and matched_form != entry.get('word', '').lower():
            paths = self._find_json_paths(entry.get('forms', {}), matched_form)
            if paths:
                result['matched_form'] = matched_form
                result['matched_paths'] = ['/'.join(p) for p in paths]

        return result
