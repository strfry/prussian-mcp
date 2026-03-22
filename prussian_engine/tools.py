"""Tool definitions for LLM function calling."""

from typing import List, Dict, Any, Callable


# Tool definitions in OpenAI format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_dictionary",
            "description": "Semantic search in the Prussian dictionary. Use descriptive, multi-word queries for best results. IMPORTANT: Use multiple related terms, not just single words! Do NOT include 'prussian' or 'preußisch' in queries - that's implicit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query with multiple descriptive terms in any language (German, Lithuanian, English, etc). Examples: 'Gruß Begrüßung Hallo' (not 'preußischer Gruß'), 'Haus Gebäude Wohnung' (not 'Haus'), 'sveikinimas linkėjimas labas' (Lithuanian). Never add 'prussian'/'preußisch' - it's redundant."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_prussian_word",
            "description": "Look up a Prussian word (lemma or inflected form) to verify its meaning. Use when you already have a Prussian word and need its translation. NOT for finding Prussian words from other languages - use search_dictionary for that!",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "Prussian word to look up (e.g., 'semmē', 'bēiti')"
                    }
                },
                "required": ["word"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_word_forms",
            "description": "Get all declension or conjugation forms for a Prussian lemma.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lemma": {
                        "type": "string",
                        "description": "Prussian lemma (base form) to get forms for"
                    }
                },
                "required": ["lemma"]
            }
        }
    }
]


class ToolExecutor:
    """Executes tool calls using the SearchEngine."""

    def __init__(self, search_engine):
        """
        Initialize tool executor.

        Args:
            search_engine: SearchEngine instance
        """
        self.search_engine = search_engine
        self.tools_map: Dict[str, Callable] = {
            "search_dictionary": self._search_dictionary,
            "lookup_prussian_word": self._lookup_prussian_word,
            "get_word_forms": self._get_word_forms,
        }

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool call.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if tool_name not in self.tools_map:
            return {"error": f"Unknown tool: {tool_name}"}

        return self.tools_map[tool_name](**arguments)

    def _search_dictionary(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Execute search_dictionary tool."""
        results = self.search_engine.query(query, top_k)
        # Return compact format: only word + translations
        return [
            {
                "word": r["word"],
                "de": r["de"],
                "en": r["en"]
            }
            for r in results
        ]

    def _lookup_prussian_word(self, word: str) -> List[Dict[str, Any]]:
        """Execute lookup_prussian_word tool."""
        results = self.search_engine.lookup(word)
        # Return with forms if available
        return [
            {
                "word": r["word"],
                "de": r["de"],
                "en": r["en"],
                **({
                    "paradigm": r.get("paradigm", ""),
                    "gender": r.get("gender", "")
                } if r.get("forms") else {})
            }
            for r in results
        ]

    def _get_word_forms(self, lemma: str) -> Dict[str, Any]:
        """Execute get_word_forms tool."""
        results = self.search_engine.lookup(lemma)
        if not results:
            return {"error": f"Word not found: {lemma}"}

        result = results[0]
        return {
            "lemma": result["word"],
            "translations": {
                "de": result["de"],
                "en": result["en"]
            },
            "paradigm": result.get("paradigm", ""),
            "gender": result.get("gender", ""),
            "forms": result.get("forms", {})
        }
