"""Search engine for Prussian Dictionary using E5 embeddings."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import DICTIONARY_PATH, EMBEDDINGS_PATH, QUERY_PREFIX


class SearchEngine:
    """Semantic search engine using precomputed E5 embeddings."""

    def __init__(self):
        """Initialize search engine by loading dictionary and embeddings."""
        self.entries: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.word_to_entry: Dict[str, Dict[str, Any]] = {}
        self.form_to_lemma: Dict[str, str] = {}

        self._load_dictionary()
        self._load_embeddings()
        self._build_indices()

    def _load_dictionary(self):
        """Load dictionary from JSON file."""
        with open(DICTIONARY_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.entries = data if isinstance(data, list) else data.get('words', [])
        print(f"Loaded {len(self.entries)} dictionary entries")

    def _load_embeddings(self):
        """Load precomputed embeddings from .npy file."""
        emb_file = Path(f"{EMBEDDINGS_PATH}.npy")
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

        # Add query prefix and create embedding placeholder
        # Note: In production, this would use the actual E5 model to encode the query
        # For now, we'll use a simple cosine similarity with existing embeddings
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('intfloat/multilingual-e5-large')
        query_embedding = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)

        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding)

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
                results.append({
                    'word': entry.get('word', ''),
                    'de': translations.get('miks', [''])[0],
                    'en': translations.get('engl', [''])[0],
                    'score': float(similarities[idx])
                })

            if len(results) >= top_k:
                break

        return results

    def lookup(self, prussian_word: str) -> List[Dict[str, Any]]:
        """
        Reverse lookup: Find Prussian word (lemma or inflected form).

        Args:
            prussian_word: Prussian word to look up

        Returns:
            List of matching entries with translations and forms
        """
        word_lower = prussian_word.lower().strip()
        results = []

        # Check if it's a lemma
        if word_lower in self.word_to_entry:
            entry = self.word_to_entry[word_lower]
            results.append(self._format_lookup_result(entry, include_forms=True))

        # Check if it's an inflected form
        elif word_lower in self.form_to_lemma:
            lemma = self.form_to_lemma[word_lower]
            entry = self.word_to_entry.get(lemma)
            if entry:
                results.append(self._format_lookup_result(entry, include_forms=True))

        return results

    def _format_lookup_result(self, entry: Dict[str, Any], include_forms: bool = False) -> Dict[str, Any]:
        """Format an entry for lookup results."""
        translations = entry.get('translations', {})
        result = {
            'word': entry.get('word', ''),
            'de': translations.get('miks', [''])[0],
            'en': translations.get('engl', [''])[0],
        }

        if include_forms and entry.get('forms'):
            result['forms'] = entry['forms']
            result['paradigm'] = entry.get('paradigm', '')
            result['gender'] = entry.get('gender', '')

        return result
