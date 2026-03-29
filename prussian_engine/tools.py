"""Tool definitions for LLM function calling."""

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
                        "description": "Search query with multiple descriptive terms in any language (German, Lithuanian, English, etc). Examples: 'Gruß Begrüßung Hallo' (not 'preußischer Gruß'), 'Haus Gebäude Wohnung' (not 'Haus'), 'sveikinimas linkėjimas labas' (Lithuanian). Never add 'prussian'/'preußisch' - it's redundant.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
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
                        "description": "Prussian word to look up (e.g., 'semmē', 'bēiti')",
                    }
                },
                "required": ["word"],
            },
        },
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
                        "description": "Prussian lemma (base form) to get forms for",
                    }
                },
                "required": ["lemma"],
            },
        },
    },
]
