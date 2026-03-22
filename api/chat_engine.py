"""
Chat Engine - Handles LLM conversation with tool calling support.
"""

import json
import re
import requests


# Tool definitions for the LLM (OpenAI format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_dictionary",
            "description": "Searches the Old Prussian dictionary using semantic search. Use descriptive terms in German or English for best results. For example, for 'Hi' search 'greeting salutation hello', not just 'Hi'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Descriptive search terms (German or English). Use semantically rich phrases, not just single short words.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)",
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
            "name": "get_word_forms",
            "description": "Get all inflected forms (declension, conjugation) for a specific Old Prussian lemma.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lemma": {
                        "type": "string",
                        "description": "The Old Prussian word (lemma) to look up",
                    }
                },
                "required": ["lemma"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_prussian_word",
            "description": "Look up a Prussian word to find its German/English translation. Searches both lemmas and all inflected forms (declensions, conjugations). Use this when the user asks about a Prussian word or when you encounter an unfamiliar Prussian word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prussian_word": {
                        "type": "string",
                        "description": "The Prussian word to look up (can be a lemma or any inflected form)",
                    },
                    "fuzzy": {
                        "type": "boolean",
                        "description": "Allow fuzzy matching for diacritics/macrons (default: true)",
                        "default": True,
                    },
                },
                "required": ["prussian_word"],
            },
        },
    },
]


class ChatEngine:
    """Handles LLM conversation with tool calling support."""

    def __init__(self, search, dictionary, hf_token, hf_model):
        """
        Initialize the chat engine.

        Args:
            search: PrussianSearch instance for semantic search
            dictionary: Full dictionary list for forms lookup
            hf_token: HuggingFace API token
            hf_model: HuggingFace model name
        """
        self.search = search
        self.dictionary = dictionary
        self.hf_token = hf_token
        self.hf_model = hf_model

    def send_message(self, user_message, system_prompt, history=None):
        """
        Main conversation loop.

        Args:
            user_message: The user's message
            system_prompt: System prompt for the LLM
            history: Previous conversation history (optional, for multi-turn)

        Returns:
            dict with keys: prussian, german, usedWords, debugInfo
        """
        # Start with provided history or empty list
        conversation_history = list(history) if history else []

        # Add user message to history
        conversation_history.append({"role": "user", "content": user_message})

        all_tool_calls = []
        all_results = []
        all_reasoning = []  # Track reasoning across turns

        # Tool calling loop - keep calling until we get content or tool_calls
        for turn in range(10):
            # Unpack both response and reasoning
            response, reasoning_content = self._call_llm(
                system_prompt, conversation_history
            )

            # Track reasoning if present
            if reasoning_content:
                all_reasoning.append({"turn": turn + 1, "reasoning": reasoning_content})

            # Check for tool use
            tool_use = self._extract_tool_use(response)

            # Check if model only returned reasoning (no tool_call and no content)
            # This is a quirk of our local model - it sometimes needs another turn
            msg = (
                response.get("choices", [{}])[0].get("message", {})
                if response.get("choices")
                else {}
            )
            has_content = bool(msg.get("content"))
            has_tool_call = bool(tool_use)

            if not has_tool_call and not has_content:
                # Model only returned reasoning, continue loop to get next response
                conversation_history.append({"role": "assistant", "content": None})
                continue

            if tool_use:
                # Execute tool locally
                tool_result = self._execute_tool(tool_use["name"], tool_use["input"])

                all_tool_calls.append(
                    {
                        "name": tool_use["name"],
                        "input": tool_use["input"],
                        "result": tool_result,
                    }
                )

                # Track search results
                if tool_use["name"] == "search_dictionary" and tool_result:
                    if isinstance(tool_result, dict) and "results" in tool_result:
                        all_results.extend(tool_result["results"])
                    elif isinstance(tool_result, list):
                        all_results.extend(tool_result)
                elif tool_use["name"] == "lookup_prussian_word" and tool_result:
                    if isinstance(tool_result, dict) and "results" in tool_result:
                        all_results.extend(tool_result["results"])
                    elif isinstance(tool_result, list):
                        all_results.extend(tool_result)

                # Add assistant response with tool call to history
                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_use["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_use["name"],
                                    "arguments": json.dumps(tool_use["input"]),
                                },
                            }
                        ],
                    }
                )

                # Add tool result to history
                conversation_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_use["id"],
                        "content": json.dumps(tool_result),
                    }
                )
            else:
                # Final response without tool use
                text = self._extract_text(response)
                conversation_history.append({"role": "assistant", "content": text})

                # Parse response
                prussian, german = self._parse_response(text)

                # Extract used words
                all_words = [r.get("word") for r in all_results if r.get("word")]
                used_words = [
                    w for w in all_words if w and self._word_in_text(w, prussian)
                ]

                return {
                    "prussian": prussian,
                    "german": german,
                    "usedWords": used_words,
                    "history": conversation_history,
                    "debugInfo": {
                        "query": user_message,
                        "toolCalls": all_tool_calls,
                        "results": all_results,
                        "usedWords": used_words,
                        "systemPrompt": system_prompt,
                        "reasoning": all_reasoning,
                    },
                }

        raise Exception("No final response from LLM")

    def _execute_tool(self, tool_name, tool_input):
        """Execute tool locally (no HTTP calls)."""
        if tool_name == "search_dictionary":
            results = self.search.query(
                tool_input["query"], top_k=tool_input.get("top_k", 10)
            )

            # Format results to match frontend expectations
            formatted = []
            for entry in results:
                cas = entry.get("forms", {}).get("declension", [{}])[0].get("cases", [])
                f = lambda n: next((c for c in cas if c.get("case") == n), {})

                formatted.append(
                    {
                        "word": entry.get("word"),
                        "paradigm": entry.get("paradigm"),
                        "gender": entry.get("gender"),
                        "desc": entry.get("desc"),
                        "de": entry.get("translations", {}).get("miks"),
                        "en": entry.get("translations", {}).get("engl"),
                        "lt": entry.get("translations", {}).get("leit"),
                        "nom_sg": f("Nominative").get("singular"),
                        "gen_sg": f("Genitive").get("singular"),
                        "dat_sg": f("Dative").get("singular"),
                        "acc_sg": f("Accusative").get("singular"),
                        "nom_pl": f("Nominative").get("plural"),
                        "gen_pl": f("Genitive").get("plural"),
                        "dat_pl": f("Dative").get("plural"),
                        "acc_pl": f("Accusative").get("plural"),
                        "present": entry.get("forms", {})
                        .get("indicative", [{}])[0]
                        .get("forms"),
                        "past": entry.get("forms", {})
                        .get("indicative", [{}])[1]
                        .get("forms")
                        if len(entry.get("forms", {}).get("indicative", [])) > 1
                        else None,
                        "imperative": entry.get("forms", {}).get("imperative"),
                        "score": entry.get("score"),
                        "translations": entry.get("translations"),
                    }
                )

            return {
                "results": formatted,
                "words": [r["word"] for r in formatted if r.get("word")],
            }

        elif tool_name == "get_word_forms":
            lemma = tool_input["lemma"]
            entry = next(
                (
                    e
                    for e in self.dictionary
                    if e.get("word", "").lower() == lemma.lower()
                ),
                None,
            )
            if entry:
                return {
                    "lemma": lemma,
                    "forms": entry.get("forms", {}),
                    "translations": entry.get("translations", {}),
                    "paradigm": entry.get("paradigm"),
                    "gender": entry.get("gender"),
                }
            return None

        elif tool_name == "lookup_prussian_word":
            prussian_word = tool_input["prussian_word"]
            fuzzy = tool_input.get("fuzzy", True)

            matches = self._find_prussian_word(prussian_word, fuzzy)

            if matches:
                # Format results similar to search_dictionary
                formatted = []
                for entry in matches:
                    cas = (
                        entry.get("forms", {})
                        .get("declension", [{}])[0]
                        .get("cases", [])
                    )
                    f = lambda n: next((c for c in cas if c.get("case") == n), {})

                    formatted.append(
                        {
                            "word": entry.get("word"),
                            "paradigm": entry.get("paradigm"),
                            "gender": entry.get("gender"),
                            "desc": entry.get("desc"),
                            "de": entry.get("translations", {}).get("miks"),
                            "en": entry.get("translations", {}).get("engl"),
                            "lt": entry.get("translations", {}).get("leit"),
                            "nom_sg": f("Nominative").get("singular"),
                            "gen_sg": f("Genitive").get("singular"),
                            "dat_sg": f("Dative").get("singular"),
                            "acc_sg": f("Accusative").get("singular"),
                            "nom_pl": f("Nominative").get("plural"),
                            "gen_pl": f("Genitive").get("plural"),
                            "dat_pl": f("Dative").get("plural"),
                            "acc_pl": f("Accusative").get("plural"),
                            "present": entry.get("forms", {})
                            .get("indicative", [{}])[0]
                            .get("forms"),
                            "past": entry.get("forms", {})
                            .get("indicative", [{}])[1]
                            .get("forms")
                            if len(entry.get("forms", {}).get("indicative", [])) > 1
                            else None,
                            "imperative": entry.get("forms", {}).get("imperative"),
                            "matched_form": entry.get(
                                "_matched_form"
                            ),  # Track which form matched
                            "translations": entry.get("translations"),
                        }
                    )

                return {
                    "results": formatted,
                    "words": [r["word"] for r in formatted if r.get("word")],
                }

            return None

        return None

    def _call_llm(self, system_prompt, conversation_history):
        """Call HuggingFace LLM with provided history."""
        # Build messages (OpenAI format)
        messages = [{"role": "system", "content": system_prompt}]

        for msg in conversation_history:
            messages.append(msg)

        # Call LLM API (supports both HuggingFace and local models)
        # Determine API URL based on model name
        if self.hf_model.startswith("http://") or self.hf_model.startswith("https://"):
            # Direct URL provided
            api_url = self.hf_model
            model_name = "gpt-oss-20b-int4-ov"  # Default for local
        elif "localhost" in self.hf_model or ":" in self.hf_model:
            # Local model endpoint
            api_url = f"http://{self.hf_model}/chat/completions"
            model_name = "gpt-oss-20b-int4-ov"
        else:
            # HuggingFace model
            api_url = "https://router.huggingface.co/v1/chat/completions"
            model_name = self.hf_model

        headers = {"Content-Type": "application/json"}
        if api_url.startswith("https://router.huggingface.co") and self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        response = requests.post(
            api_url,
            headers=headers,
            json={
                "model": model_name,
                "messages": messages,
                "tools": TOOLS,
                "max_tokens": 1000,
                "temperature": 0.7,
            },
            timeout=120,
        )

        if response.status_code != 200:
            raise Exception(f"LLM API error: {response.status_code} - {response.text}")

        if response.status_code != 200:
            raise Exception(f"LLM API error: {response.status_code} - {response.text}")

        result = response.json()

        # Extract reasoning if present (DeepSeek R1 uses 'reasoning' or 'reasoning_content')
        reasoning_content = None
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
            reasoning_content = message.get("reasoning_content") or message.get(
                "reasoning"
            )

        return result, reasoning_content

    def _extract_tool_use(self, response):
        """Extract tool use from LLM response."""
        if "choices" not in response or len(response["choices"]) == 0:
            return None

        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls")

        if tool_calls and len(tool_calls) > 0:
            tool_call = tool_calls[0]
            return {
                "id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "input": json.loads(tool_call["function"]["arguments"]),
            }

        return None

    def _extract_text(self, response):
        """Extract text content from LLM response."""
        if "choices" not in response or len(response["choices"]) == 0:
            return ""

        message = response["choices"][0]["message"]
        return message.get("content", "")

    def _parse_response(self, text):
        """Extract Prussian text and German translation."""
        de_match = re.search(r"\[DE:\s*(.*?)\]$", text, re.MULTILINE)
        lt_match = re.search(r"\[LT:\s*(.*?)\]$", text, re.MULTILINE)

        translation = (
            de_match.group(1) if de_match else (lt_match.group(1) if lt_match else None)
        )

        prussian = re.sub(r"\[(DE|LT):.*?\]$", "", text, flags=re.MULTILINE).strip()

        return prussian, translation

    def _word_in_text(self, word, text):
        """Check if word appears in text (case-insensitive, word boundary)."""
        if not word or not text:
            return False
        # Escape special regex characters
        escaped = re.escape(word)
        regex = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
        return bool(regex.search(text))

    def _find_prussian_word(self, prussian_word, fuzzy=True):
        """
        Find Prussian word in dictionary, searching lemmas and all inflected forms.

        Args:
            prussian_word: Prussian word to search for
            fuzzy: Allow fuzzy matching (normalize diacritics)

        Returns:
            List of matching dictionary entries with _matched_form annotation
        """
        import unicodedata

        def normalize_prussian(text):
            """Normalize Prussian text for fuzzy matching."""
            if not text:
                return ""
            # Remove diacritics while preserving base characters
            nfd = unicodedata.normalize("NFD", text.lower())
            # Keep only base letters (remove combining marks)
            return "".join(c for c in nfd if not unicodedata.combining(c))

        def extract_all_forms(entry):
            """Extract all inflected forms from an entry."""
            forms = set()

            # Add lemma
            if entry.get("word"):
                forms.add(entry["word"])

            # Declension forms
            if entry.get("forms", {}).get("declension"):
                for decl in entry["forms"]["declension"]:
                    for case in decl.get("cases", []):
                        if case.get("singular"):
                            forms.add(case["singular"])
                        if case.get("plural"):
                            forms.add(case["plural"])

            # Verb forms (indicative, subjunctive, optative, imperative)
            for mood in ["indicative", "subjunctive", "optative", "imperative"]:
                if entry.get("forms", {}).get(mood):
                    for tense_group in entry["forms"][mood]:
                        if isinstance(tense_group, dict) and tense_group.get("forms"):
                            for form_item in tense_group["forms"]:
                                if form_item.get("form"):
                                    forms.add(form_item["form"])

            # Participles and infinitives
            for form_type in ["participles", "infinitives"]:
                if entry.get("forms", {}).get(form_type):
                    for item in entry["forms"][form_type]:
                        if item.get("form"):
                            forms.add(item["form"])

            return forms

        # Normalize search term
        search_normalized = (
            normalize_prussian(prussian_word) if fuzzy else prussian_word.lower()
        )

        matches = []

        for entry in self.dictionary:
            # Get all forms for this entry
            all_forms = extract_all_forms(entry)

            # Check for matches
            for form in all_forms:
                form_check = normalize_prussian(form) if fuzzy else form.lower()

                if form_check == search_normalized:
                    # Add matched form info
                    entry_copy = entry.copy()
                    entry_copy["_matched_form"] = form
                    matches.append(entry_copy)
                    break  # Only add entry once even if multiple forms match

        return matches
