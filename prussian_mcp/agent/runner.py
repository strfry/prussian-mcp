"""Runner glue for the ``prussian-agent`` CLI: stream/trace helpers moved
1:1 from ``prussian-llm/scaffolding/haystack_runner.py`` plus the new
``run_agent``/``RunResult`` core.

The agent runs a single ``Agent.run()`` with a system prompt that requires
the model to call ``validate_prussian`` on its draft before emitting the
final ``PRUSSIAN:`` line.  There is no external orchestration loop — the
correction cycle happens inside the model's own tool-calling rounds, gated
by ``max_agent_steps`` on the ``Agent``.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret

from .generators import DeepSeekChatGenerator


# ── ANSI helpers for the live stream (stderr only) ─────────────────────────
_DIM = "\x1b[2m"
_RST = "\x1b[0m"
_BLUE = "\x1b[34m"


def log(msg: str) -> None:
    """Single-line stderr logger, flushed immediately."""
    print(msg, file=sys.stderr, flush=True)


def make_stream_callback(idx):
    """A streaming callback that paints visible content normal, reasoning
    dim, and tool-call deltas blue — all on stderr so the trace stays
    separable from stdout (JSONL output, result block).

    1:1 move from ``haystack_runner.py`` (Z.296–333)."""
    state = {"mode": None}

    def switch(mode: str, header: str) -> None:
        if state["mode"] != mode:
            print(f"\n[{idx}] {header}", file=sys.stderr, flush=True, end=" ")
            state["mode"] = mode

    def cb(chunk):
        # Reasoning delta — print dim
        r = getattr(chunk, "reasoning", None)
        if r and getattr(r, "reasoning_text", None):
            switch("reasoning", "·")
            print(
                f"{_DIM}{r.reasoning_text}{_RST}", end="", flush=True, file=sys.stderr
            )
        # Tool-call deltas — print blue
        if chunk.tool_calls:
            for tcd in chunk.tool_calls:
                name = tcd.tool_name or ""
                args = tcd.arguments or ""
                if name:
                    switch("tool", f"→ {_BLUE}tool {name}({_RST}")
                if args:
                    print(f"{_BLUE}{args}{_RST}", end="", flush=True, file=sys.stderr)
        # Plain text content
        if chunk.content:
            switch("content", "▶")
            print(chunk.content, end="", flush=True, file=sys.stderr)
        # End of an LLM turn — newline so following lines aren't glued on
        if chunk.finish_reason is not None:
            print("", file=sys.stderr, flush=True)
            state["mode"] = None

    return cb


def count_tool_calls(messages: list[ChatMessage]) -> int:
    n = 0
    for m in messages:
        calls = m.tool_calls or []
        n += len(calls)
    return n


def trace_messages(idx, messages: list[ChatMessage], baseline: int = 0) -> None:
    """Print a compact event line per message added since baseline.

    A ``baseline > 0`` skips the pre-built history (e.g. inject mode's
    pre-fetched lookups) so we only show what the model actually produced
    on top.

    1:1 move from ``haystack_runner.py`` (Z.367–399)."""
    for i, m in enumerate(messages[baseline:], start=baseline):
        role = m.role.value if hasattr(m.role, "value") else str(m.role)
        if m.tool_calls:
            for tc in m.tool_calls:
                a = json.dumps(tc.arguments, ensure_ascii=False)
                if len(a) > 140:
                    a = a[:137] + "…"
                log(f"[{idx}] #{i} {role} → tool {tc.tool_name}({a}) [ADDITIONAL]")
        if m.tool_call_results:
            for tr in m.tool_call_results:
                txt = (tr.result or "").replace("\n", " ")
                if len(txt) > 160:
                    txt = txt[:157] + "…"
                err = " [ERROR]" if tr.error else ""
                log(f"[{idx}] #{i} tool_result{err}: {txt}")
        r = getattr(m, "reasoning", None)
        if r and getattr(r, "reasoning_text", None):
            rtxt = r.reasoning_text.replace("\n", " ")
            if len(rtxt) > 240:
                rtxt = rtxt[:237] + "…"
            log(f"[{idx}] #{i} {role} reasoning: {rtxt}")
        if m.text and role in ("assistant",):
            txt = m.text.replace("\n", " ")
            if len(txt) > 200:
                txt = txt[:197] + "…"
            log(f"[{idx}] #{i} {role}: {txt}")


# ── Generator / Agent construction ────────────────────────────────────────


def build_generator(
    *,
    model: str,
    api_key: str,
    api_base_url: str,
    tools,
    temperature: float = 0.2,
    timeout: float = 300.0,
    max_retries: int = 2,
) -> OpenAIChatGenerator:
    """Pick the generator class by model name and build it.

    Selection mirrors ``haystack_runner.py`` main() (Z.512–528): the
    Apertus branch is intentionally NOT migrated — Apertus strips the
    ``tools`` kwarg on the wire, so the agent's self-correction via
    ``validate_prussian`` would not work there. Use gpt-oss / DeepSeek /
    any OpenAI-compatible model that accepts tools.
    """
    gen_cls = (
        DeepSeekChatGenerator
        if "deepseek" in model.lower()
        else OpenAIChatGenerator
    )
    return gen_cls(
        api_key=Secret.from_token(api_key),
        model=model,
        api_base_url=api_base_url,
        tools=tools,
        generation_kwargs={"temperature": temperature},
        timeout=timeout,
        max_retries=max_retries,
    )


def build_agent(
    generator: OpenAIChatGenerator,
    tools,
    *,
    max_steps: int = 30,
) -> Agent:
    """Construct and warm up the Haystack Agent."""
    agent = Agent(
        chat_generator=generator,
        tools=tools,
        max_agent_steps=max_steps,
    )
    agent.warm_up()
    return agent


# ── Candidate extraction and validation-parity reading ───────────────────

_PRUSSIAN_RE = re.compile(r"^PRUSSIAN:\s*(.+?)\s*$", re.MULTILINE)


def extract_candidate(text: str | None) -> str | None:
    """Pull the Prussian sentence out of the model's final response.

    Convention: the model's final line is exactly ``PRUSSIAN: <satz>``.
    We take the LAST multiline match (the model may write ``PRUSSIAN:``
    once per draft, the final one wins).  Fallback: last non-empty line,
    stripped of surrounding quotes/whitespace.

    No fence stripping — the system prompt forbids code fences around the
    PRUSSIAN marker, and we trust the model to obey.
    """
    if not text:
        return None
    matches = list(_PRUSSIAN_RE.finditer(text))
    if matches:
        return matches[-1].group(1).strip().strip('"').strip("'").strip()
    # Fallback: last non-empty line
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s:
            return s.strip('"').strip("'").strip()
    return None


def parse_last_validation(messages: list[ChatMessage]) -> dict | None:
    """Find the LAST ``validate_prussian`` tool result in the message
    history and return its parsed JSON payload.

    The agent's verification verdict is read off the model's own last
    validation call — we do NOT run ``run_validate`` again after the
    loop. Returns ``None`` if the model never called ``validate_prussian``
    or if the tool result could not be parsed.
    """
    # Walk the history, track the latest tool_call id whose tool_name is
    # validate_prussian, then find the tool-result message that references
    # that id.
    last_tc_id = None
    for m in messages:
        for tc in m.tool_calls or []:
            if tc.tool_name == "validate_prussian":
                last_tc_id = tc.id
    if last_tc_id is None:
        return None
    for m in reversed(messages):
        for tr in m.tool_call_results or []:
            if tr.origin and getattr(tr.origin, "id", None) == last_tc_id:
                try:
                    return json.loads(tr.result)
                except (ValueError, TypeError):
                    return None
    return None


# ── RunResult and run_agent ────────────────────────────────────────────────


@dataclass
class RunResult:
    """Outcome of a single ``run_agent`` invocation.

    ``validation`` is the parsed JSON of the LAST ``validate_prussian``
    tool call the model made during the run (NOT a fresh validation we
    run after the loop).  It is ``None`` if the model never validated.
    """

    input: str
    final: str | None
    output: str
    validation: dict | None
    tool_calls: int
    latency_s: float
    model: str


def run_agent(
    sentence: str,
    *,
    agent: Agent,
    system_prompt: str,
    stream_cb: Callable | None = None,
    trace: bool = False,
) -> RunResult:
    """Run the agent once and package the outcome.

    A single ``agent.run()`` — the correction cycle happens inside the
    model's own tool-calling rounds (``validate_prussian`` is one of the
    tools, the system prompt mandates validation before the final
    ``PRUSSIAN:`` line).  ``max_agent_steps`` on the Agent bounds the
    rounds.
    """
    messages = [
        ChatMessage.from_system(system_prompt),
        ChatMessage.from_user(f"Translate to Prussian: {sentence}"),
    ]
    t0 = time.perf_counter()
    res = agent.run(messages=messages, streaming_callback=stream_cb)
    msgs = res["messages"]
    if trace:
        trace_messages(0, msgs)
    final_msg = msgs[-1] if msgs else None
    final_text = (final_msg.text if final_msg else "") or ""
    return RunResult(
        input=sentence,
        final=extract_candidate(final_text),
        output=final_text,
        validation=parse_last_validation(msgs),
        tool_calls=count_tool_calls(msgs),
        latency_s=round(time.perf_counter() - t0, 3),
        model=getattr(agent.chat_generator, "model", ""),
    )