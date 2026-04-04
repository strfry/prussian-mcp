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
            "description": "Look up a Prussian word (lemma or inflected form) to verify its meaning. Returns pgr annotation (e.g. 'ACC.SG.MASC' or 'GEN.PL.MASC|ACC.SG.MASC' for ambiguous forms). Use when you already have a Prussian word and need its translation. NOT for finding Prussian words from other languages - use search_dictionary for that!",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "Prussian word to look up (e.g., 'semmē', 'bēiti', 'Dēiwan')",
                    },
                    "fuzzy": {
                        "type": "boolean",
                        "description": "Enable fuzzy matching for approximate/misspelled words",
                        "default": True,
                    },
                },
                "required": ["word"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_word_forms",
            "description": "Get declension or conjugation forms for a Prussian lemma. Returns flat list with form and pgr fields. Use filter to get specific forms (e.g. 'GEN.PL' for genitive plural, 'PRES.1.SG' for present 1st singular).",
            "parameters": {
                "type": "object",
                "properties": {
                    "lemma": {
                        "type": "string",
                        "description": "Prussian lemma (base form) to get forms for",
                    },
                    "filter": {
                        "type": "string",
                        "description": "PGR filter, e.g. 'GEN.PL', 'PRES.1.SG', 'NOM.SG'. Returns only forms matching this pattern.",
                    },
                },
                "required": ["lemma"],
            },
        },
    },
]
