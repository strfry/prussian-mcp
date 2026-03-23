"""Chat engine with LLM integration and tool calling."""

import json
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, SYSTEM_PROMPT_PATH
from .tools import TOOLS, ToolExecutor


class ChatEngine:
    """Chat engine with tool calling support."""

    def __init__(self, search_engine):
        """
        Initialize chat engine.

        Args:
            search_engine: SearchEngine instance for tool execution
        """
        self.search_engine = search_engine
        self.tool_executor = ToolExecutor(search_engine)
        self.client = OpenAI(
            api_key=OPENAI_API_KEY or "dummy",
            base_url=OPENAI_BASE_URL
        )
        self.model = OPENAI_MODEL
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def send_message(
        self,
        message: str,
        language: str = "de",
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message and generate a response.

        Args:
            message: User message
            language: Output language ('de' or 'lt')
            history: Conversation history

        Returns:
            Response dict with prussian, translation, usedWords, debugInfo, history
        """
        if history is None:
            history = []

        # Prepare system prompt with language parameter
        lang_code = "LT" if language == "lt" else "DE"
        system_prompt = self.system_prompt.replace("{lang_code}", lang_code)

        # Build messages array
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        # Track debug info
        debug_info = {
            "query": message,
            "systemPrompt": system_prompt,
            "toolCalls": [],
            "results": [],
            "usedWords": [],
            "reasoning": []
        }

        # Tool calling loop
        max_iterations = 10
        iteration = 0
        all_used_words = set()

        print(f"\n🤖 LLM Tool-Calling Loop:")

        while iteration < max_iterations:
            iteration += 1
            print(f"\n  ⟳ Turn {iteration}:")

            # Debug: Log API call
            api_call = {
                "model": self.model,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto"
            }
            print(f"\n    📤 API CALL:")
            print(json.dumps(api_call, ensure_ascii=False, indent=2))

            # LLM call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )

            # Debug: Log API response
            print(f"\n    📥 API RESPONSE:")
            print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))

            message_obj = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Debug: show what we got
            print(f"    📊 finish_reason: {finish_reason}")
            print(f"    📊 has content: {bool(message_obj.content)}")
            print(f"    📊 has reasoning_content: {hasattr(message_obj, 'reasoning_content') and bool(message_obj.reasoning_content)}")
            if message_obj.content:
                print(f"    📊 content preview: {message_obj.content[:200]}...")

            # Check for reasoning (DeepSeek R1 style)
            has_reasoning = hasattr(message_obj, 'reasoning_content') and message_obj.reasoning_content
            if has_reasoning:
                reasoning = message_obj.reasoning_content
                print(f"    🧠 Reasoning ({len(reasoning)} chars):")
                print(f"       {reasoning[:200]}...")
                debug_info["reasoning"].append({
                    "turn": iteration,
                    "reasoning": reasoning
                })

            # Add assistant message to history (including reasoning if present!)
            assistant_msg = {
                "role": "assistant",
                "content": message_obj.content or "",
            }

            # Include reasoning in message so model can build on previous thinking
            if has_reasoning:
                assistant_msg["reasoning_content"] = message_obj.reasoning_content

            # Include tool calls if present
            if message_obj.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message_obj.tool_calls
                ]

            messages.append(assistant_msg)

            # If no tool calls, we're done
            if not message_obj.tool_calls:
                print(f"    ✓ Final response generated")
                break

            # Execute tool calls
            print(f"    🔧 {len(message_obj.tool_calls)} tool call(s):")
            for tool_call in message_obj.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                # Print tool call
                args_str = json.dumps(arguments, ensure_ascii=False)
                print(f"       • {function_name}({args_str})")

                # Execute tool
                result = self.tool_executor.execute(function_name, arguments)

                # Print result summary
                if isinstance(result, list):
                    result_preview = f"{len(result)} results"
                    if result:
                        words = [r.get('word', '?') for r in result[:3]]
                        result_preview += f": {', '.join(words)}"
                        if len(result) > 3:
                            result_preview += "..."
                    print(f"         → {result_preview}")
                else:
                    print(f"         → {type(result).__name__}")

                # Track for debug
                debug_info["toolCalls"].append({
                    "name": function_name,
                    "input": arguments,
                    "result": result
                })

                # Collect results for debug
                if isinstance(result, list):
                    debug_info["results"].extend(result)

                # Track used words only from get_word_forms (actual usage, not search results)
                if function_name == "get_word_forms" and isinstance(result, dict):
                    if "lemma" in result:
                        all_used_words.add(result["lemma"])
                elif function_name == "lookup_prussian_word" and isinstance(result, list):
                    for item in result:
                        if "word" in item:
                            all_used_words.add(item["word"])

                # Add tool response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })

        # Extract final response
        final_message = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""

        # Parse response for Prussian and translation
        prussian, translation = self._parse_response(final_message, language)

        # Update history (remove system prompt, keep user/assistant/tool messages)
        updated_history = [m for m in messages[1:] if m["role"] in ["user", "assistant", "tool"]]

        # Prepare debug info (exclude verbose reasoning from API response)
        debug_info["usedWords"] = sorted(list(all_used_words))
        # Remove verbose reasoning array from debug info
        debug_info.pop("reasoning", None)

        return {
            "prussian": prussian,
            "translation": translation,
            "usedWords": sorted(list(all_used_words)),
            "debugInfo": debug_info,
            "history": updated_history
        }

    def _parse_response(self, text: str, language: str) -> tuple[str, str]:
        """
        Parse LLM response to extract Prussian text and translation.

        Args:
            text: LLM response text
            language: Expected language ('de' or 'lt')

        Returns:
            Tuple of (prussian_text, translation)
        """
        # Look for translation marker: [DE: ...] or [LT: ...]
        lang_code = "LT" if language == "lt" else "DE"
        pattern = rf'\[{lang_code}:\s*(.+?)\]'
        match = re.search(pattern, text, re.DOTALL)

        if match:
            translation = match.group(1).strip()
            # Everything before the translation is Prussian
            prussian = text[:match.start()].strip()
        else:
            # Fallback: assume entire text is Prussian
            prussian = text.strip()
            translation = ""

        return prussian, translation
