"""Agent runner for ``prussian-agent`` — smolagents edition.

Minimal glue between smolagents ToolCallingAgent and the CLI.
The model self-corrects by calling validate_prussian on its draft.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from smolagents import OpenAIModel, ToolCallingAgent
from smolagents.agents import AgentLogger, LogLevel
from smolagents.memory import ActionStep

_err = Console(file=sys.stderr, highlight=False, force_terminal=True)


def _print_step(step: ActionStep) -> None:
    for tc in (step.tool_calls or []):
        args = json.dumps(tc.arguments, ensure_ascii=False) if tc.arguments else ""
        if len(args) > 120:
            args = args[:117] + "…"
        _err.print(Panel(f"[bold]{tc.name}[/bold]({args})", border_style="blue", expand=False))
    obs = (step.observations or "").strip()
    if obs:
        # Truncate very long observations
        if len(obs) > 500:
            obs = obs[:497] + "…"
        _err.print(f"  → {obs}", style="dim")


def build_model(*, model: str, api_key: str, api_base_url: str, temperature: float = 0.2) -> OpenAIModel:
    return OpenAIModel(model_id=model, api_key=api_key, api_base=api_base_url, temperature=temperature)


def build_agent(model: OpenAIModel, tools: list, *, max_steps: int = 30, instructions: str | None = None) -> ToolCallingAgent:
    logger = AgentLogger(level=LogLevel.OFF, console=Console(file=sys.stderr, highlight=False))
    return ToolCallingAgent(tools=tools, model=model, max_steps=max_steps, instructions=instructions, stream_outputs=False, logger=logger, step_callbacks=[_print_step])


_PRUSSIAN_RE = re.compile(r"^PRUSSIAN:\s*(.+?)\s*$", re.MULTILINE)


def extract_candidate(text: str | None) -> str | None:
    """Pull the Prussian sentence from the model's output.

    Looks for ``PRUSSIAN: <sentence>`` lines (last match wins).
    Fallback: last non-empty line.
    """
    if not text:
        return None
    matches = list(_PRUSSIAN_RE.finditer(text))
    if matches:
        return matches[-1].group(1).strip().strip('"').strip("'")
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s:
            return s.strip('"').strip("'")
    return None


def parse_last_validation(agent: ToolCallingAgent) -> dict | None:
    """Read the LAST validate_prussian result from agent memory.

    Walks ``agent.memory.steps`` backwards. Each ActionStep has
    ``tool_calls: list[ToolCall]`` and ``observations: str``.
    """
    for step in reversed(agent.memory.steps):
        if not isinstance(step, ActionStep):
            continue
        if not step.tool_calls:
            continue
        if any(tc.name == "validate_prussian" for tc in step.tool_calls):
            try:
                return json.loads((step.observations or "").strip())
            except (ValueError, TypeError):
                return None
    return None


def count_tool_calls(agent: ToolCallingAgent) -> int:
    n = 0
    for step in agent.memory.steps:
        if isinstance(step, ActionStep) and step.tool_calls:
            n += len(step.tool_calls)
    return n


@dataclass
class RunResult:
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
    agent: ToolCallingAgent,
    system_prompt: str,
) -> RunResult:
    task = f"Translate to Prussian: {sentence}"
    t0 = time.perf_counter()

    final_answer = agent.run(task, stream=False)

    # Extract final model_output from last ActionStep
    final_text = ""
    for step in reversed(agent.memory.steps):
        if isinstance(step, ActionStep) and step.model_output:
            final_text = step.model_output
            break

    model_id = getattr(agent.model, "model_id", "")
    return RunResult(
        input=sentence,
        final=extract_candidate(str(final_answer) if final_answer else None),
        output=final_text,
        validation=parse_last_validation(agent),
        tool_calls=count_tool_calls(agent),
        latency_s=round(time.perf_counter() - t0, 3),
        model=model_id,
    )
