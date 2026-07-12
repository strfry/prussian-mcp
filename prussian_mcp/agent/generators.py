"""ReasoningChatGenerator — OpenAIChatGenerator subclass that round-trips
``reasoning_content`` through Haystack's ``ChatMessage.reasoning`` slot.

Works with any OpenAI-compatible model (DeepSeek, Qwen3, etc.) that
includes ``reasoning_content`` in streaming or non-streaming responses.

Background
----------
Haystack 2.29 ships a ``ReasoningContent`` dataclass on ``ChatMessage``
(PR #9696, merged Aug 2025), but the stock ``OpenAIChatGenerator`` does not
populate it on inbound, nor re-emit ``reasoning_content`` on outbound. The
DeepSeek-reasoner draft (PR #8776) was closed by maintainers in favour of
"each integration handles reasoning itself." This subclass is that
per-integration handler for reasoning-capable models reached via the
OpenAI Chat Completions API.

Some providers (DeepSeek, Qwen3) reject the next turn of a tool-using
conversation unless the previous assistant message's ``reasoning_content``
is echoed back. We therefore:

* override ``run()`` to convert each choice through our own helper, which
  reads ``reasoning_content`` from the OpenAI SDK response and stores it
  on ``ChatMessage.reasoning``;
* override ``_prepare_api_call()`` to inject ``reasoning_content`` back
  into outbound assistant message dicts whenever the source ChatMessage
  has it.

No monkey-patching of core Haystack classes.
"""

from dataclasses import replace as dataclass_replace
from typing import Any

from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.chat.openai import (
    _check_finish_reason,
    _convert_chat_completion_chunk_to_streaming_chunk,
    _convert_chat_completion_to_chat_message,
)
from haystack.components.generators.utils import (
    _convert_streaming_chunks_to_chat_message,
)
from haystack.dataclasses import (
    ChatMessage,
    ComponentInfo,
    StreamingCallbackT,
    SyncStreamingCallbackT,
    select_streaming_callback,
)
from haystack.dataclasses.chat_message import ReasoningContent
from haystack.tools import ToolsType


def _extract_reasoning_content(openai_message) -> str | None:
    """Pull ``reasoning_content`` from an OpenAI SDK ChatCompletionMessage.

    The SDK exposes it either as a direct attribute (when its Pydantic
    model knows about the field) or in ``model_extra`` (otherwise).
    """
    rc = getattr(openai_message, "reasoning_content", None)
    if rc:
        return rc
    extras = getattr(openai_message, "model_extra", None) or {}
    return extras.get("reasoning_content") or None


def _convert_choice_with_reasoning(completion, choice) -> ChatMessage:
    """Like Haystack's ``_convert_chat_completion_to_chat_message`` but
    preserves ``reasoning_content`` on ``ChatMessage.reasoning``."""
    base = _convert_chat_completion_to_chat_message(completion, choice)
    rc = _extract_reasoning_content(choice.message)
    if not rc:
        return base
    return ChatMessage.from_assistant(
        text=base.text,
        tool_calls=base.tool_calls,
        meta=base.meta,
        reasoning=ReasoningContent(reasoning_text=rc),
    )


class ReasoningChatGenerator(OpenAIChatGenerator):
    """OpenAIChatGenerator that round-trips ``reasoning_content``."""

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage] | str,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        *,
        tools: ToolsType | None = None,
        tools_strict: bool | None = None,
    ) -> dict[str, list[ChatMessage]]:
        if not self._is_warmed_up:
            self.warm_up()
        if not messages:
            return {"replies": []}

        streaming_cb = select_streaming_callback(
            init_callback=self.streaming_callback,
            runtime_callback=streaming_callback,
            requires_async=False,
        )

        api_args = self._prepare_api_call(
            messages=messages,
            streaming_callback=streaming_cb,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )
        endpoint = api_args.pop("openai_endpoint")
        method = getattr(self.client.chat.completions, endpoint)
        chat_completion = method(**api_args)

        if streaming_cb is not None:
            completions = self._handle_stream_response(chat_completion, streaming_cb)
        else:
            completions = [
                _convert_choice_with_reasoning(chat_completion, c)
                for c in chat_completion.choices
            ]

        for m in completions:
            _check_finish_reason(m.meta)
        return {"replies": completions}

    def _handle_stream_response(
        self,
        chat_completion,
        callback: SyncStreamingCallbackT,
    ) -> list[ChatMessage]:
        """Streaming variant of ``run()``.

        Wraps the stock OpenAI chunk converter and attaches ``reasoning``
        (a ``ReasoningContent``) to each ``StreamingChunk`` whose delta
        carries ``reasoning_content``. The downstream
        ``_convert_streaming_chunks_to_chat_message`` accumulates these
        into ``ChatMessage.reasoning`` automatically.
        """
        component_info = ComponentInfo.from_component(self)
        chunks = []
        for chunk in chat_completion:
            chunk_delta = _convert_chat_completion_chunk_to_streaming_chunk(
                chunk=chunk,
                previous_chunks=chunks,
                component_info=component_info,
            )
            if chunk.choices:
                delta = chunk.choices[0].delta
                rc = getattr(delta, "reasoning_content", None)
                if rc is None:
                    extras = getattr(delta, "model_extra", None) or {}
                    rc = extras.get("reasoning_content")
                if rc:
                    chunk_delta = dataclass_replace(
                        chunk_delta,
                        reasoning=ReasoningContent(reasoning_text=rc),
                        index=chunk_delta.index or 0,
                    )
            chunks.append(chunk_delta)
            callback(chunk_delta)
        return [_convert_streaming_chunks_to_chat_message(chunks=chunks)]

    def _prepare_api_call(self, *, messages, **kwargs) -> dict[str, Any]:
        result = super()._prepare_api_call(messages=messages, **kwargs)
        out_msgs = result.get("messages")
        if not isinstance(out_msgs, list):
            return result
        for hsk, oai in zip(messages, out_msgs):
            if not isinstance(oai, dict):
                continue
            if oai.get("role") != "assistant":
                continue
            rc = hsk.reasoning
            if rc and getattr(rc, "reasoning_text", None):
                oai["reasoning_content"] = rc.reasoning_text
        return result