"""Unit-tests for the agent loop – no real LLM or MCP calls.

The core loop is:

    1. LLM generates text (``model_output``).
    2. ``extract_candidate`` pulls the ``PRUSSIAN:`` line from the
       model's text.
    3. ``validate_prussian`` is called on the candidate (tool-call
       result lands in ``observations``).
    4. ``parse_last_validation`` reads the validation verdict from
       ``agent.memory.steps``.

Two integration-style tests exercise the full ``run_agent`` path with
a mock that simulates tool-calling + final-answer steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from smolagents.memory import ActionStep, ToolCall
from smolagents.monitoring import Timing

from agents.runner import (
    build_model,
    extract_candidate,
    parse_last_validation,
    RunResult,
    count_tool_calls,
)


def _tc(name: str, arguments: Any = None, id: str = "call_0") -> ToolCall:
    return ToolCall(name=name, arguments=arguments or {}, id=id)


def _step(step_number: int, tool_calls=None, observations="", model_output=""):
    return ActionStep(
        step_number=step_number,
        timing=Timing(start_time=0.0),
        tool_calls=tool_calls,
        observations=observations,
        model_output=model_output,
    )


# ── Pure-function tests (no agent required) ───────────────────────────────


class TestExtractCandidate:
    """extract_candidate must pull the PRUSSIAN: line from arbitrary text."""

    def test_standard_case(self):
        t = "Let me check the dictionary.\nPRUSSIAN: As wīda gaīlan berzin"
        assert extract_candidate(t) == "As wīda gaīlan berzin"

    def test_last_wins_when_multiple_pru_lines(self):
        t = (
            "PRUSSIAN: As wīda gaīlan berzin\n"
            "validate_prussian returned violations_found\n"
            "PRUSSIAN: As wīda galan berzin"
        )
        assert extract_candidate(t) == "As wīda galan berzin"

    def test_fallback_to_last_nonempty_line(self):
        t = "Some preamble\nvalidate_prussian …\nAs wīda galan berzin"
        assert extract_candidate(t) == "As wīda galan berzin"

    def test_none_and_empty(self):
        assert extract_candidate(None) is None
        assert extract_candidate("") is None

    def test_strips_quotes(self):
        t = "PRUSSIAN: \"As wīda galan berzin\""
        assert extract_candidate(t) == "As wīda galan berzin"


# ── parse_last_validation reads from agent memory steps ──────────────────


class TestParseLastValidation:
    """parse_last_validation walks ``agent.memory.steps`` backwards."""

    @staticmethod
    def _make_agent_with_steps(steps):
        class _Agent:
            pass
        a = _Agent()
        a.memory = type("M", (), {"steps": steps})()
        return a

    def test_returns_last_validation_json(self):
        ok_result = json.dumps({
            "overall": {"status": "verified_in_coverage", "n_sentences": 1, "n_violations": 0},
            "sentences": [{"sent_id": "s1", "text": "As wīda galan berzin", "status": "verified_in_coverage", "violations": [], "coverage": {}}],
        })
        bad_result = json.dumps({
            "overall": {"status": "violations_found", "n_sentences": 1, "n_violations": 2},
            "sentences": [{"sent_id": "s1", "text": "test", "status": "violations_found", "violations": [{"rule": "NomSg"}], "coverage": {}}],
        })
        steps = [
            _step(1, tool_calls=[_tc("lookup_tool", {"lemma": "test"})], observations='{"some": "other"}', model_output="draft1"),
            _step(2, tool_calls=[_tc("validate_prussian", {"text": "test"})], observations=bad_result, model_output="draft2"),
            _step(3, tool_calls=[_tc("validate_prussian", {"text": "corrected"})], observations=ok_result, model_output="PRUSSIAN: corrected"),
        ]
        agent = self._make_agent_with_steps(steps)
        v = parse_last_validation(agent)
        assert v is not None
        assert v["overall"]["status"] == "verified_in_coverage"
        assert v["overall"]["n_violations"] == 0

    def test_none_when_no_validate_call(self):
        steps = [
            _step(1, tool_calls=[_tc("lookup_tool", {"lemma": "x"})], observations="{}", model_output=""),
        ]
        agent = self._make_agent_with_steps(steps)
        assert parse_last_validation(agent) is None

    def test_none_on_unparseable_observations(self):
        steps = [
            _step(1, tool_calls=[_tc("validate_prussian", {"text": "t"})], observations="NOT JSON AT ALL", model_output=""),
        ]
        agent = self._make_agent_with_steps(steps)
        assert parse_last_validation(agent) is None

    def test_returns_none_for_final_step_without_validate(self):
        steps = [
            _step(1, tool_calls=[_tc("validate_prussian", {"text": "ok"})],
                  observations='{"overall": {"status": "verified_in_coverage"}}', model_output=""),
            _step(2, tool_calls=[_tc("lookup_tool", {"lemma": "x"})],
                  observations="{}", model_output="PRUSSIAN: foo bar"),
        ]
        agent = self._make_agent_with_steps(steps)
        v = parse_last_validation(agent)
        assert v is not None
        assert v["overall"]["status"] == "verified_in_coverage"


class TestCountToolCalls:
    def test_counts_all_tool_calls(self):
        steps = [
            _step(1, tool_calls=[_tc("a"), _tc("b", id="call_1")]),
            _step(2, tool_calls=[_tc("a")]),
        ]

        class _Agent:
            pass
        a = _Agent()
        a.memory = type("M", (), {"steps": steps})()
        assert count_tool_calls(a) == 3

    def test_zero_on_empty(self):
        class _Agent:
            pass
        a = _Agent()
        a.memory = type("M", (), {"steps": []})()
        assert count_tool_calls(a) == 0


# ── build_model sanity ────────────────────────────────────────────────────


class TestBuildModel:
    def test_returns_openai_model(self):
        from smolagents import OpenAIModel
        m = build_model(
            model="test-model",
            api_key="sk-test",
            api_base_url="http://localhost:8000",
            temperature=0.5,
        )
        assert isinstance(m, OpenAIModel)


# ── Integration: run_agent with mock LLM ──────────────────────────────────


class TestRunAgent:
    """Exercise ``run_agent`` with a mock ``ToolCallingAgent`` that
    simulates two ActionSteps in memory."""

    def test_final_result_fields(self):
        ok_result = json.dumps({
            "overall": {"status": "verified_in_coverage", "n_sentences": 1, "n_violations": 0},
            "sentences": [{"sent_id": "s1", "text": "As wīda galan berzin", "status": "verified_in_coverage", "violations": [], "coverage": {}}],
        })
        steps = [
            _step(1,
                  tool_calls=[_tc("validate_prussian", {"text": "As wīda galan berzin"})],
                  observations=ok_result,
                  model_output="draft"),
            _step(2,
                  tool_calls=[_tc("validate_prussian", {"text": "As wīda galan berzin"})],
                  observations=ok_result,
                  model_output="PRUSSIAN: As wīda galan berzin"),
        ]

        class _Mem:
            pass
        _Mem.steps = steps
        class _Agent:
            model = type("M", (), {"model_id": "mock-model"})()
            def __init__(self, mem):
                self.memory = mem
            def run(self, task, stream=False):
                return "PRUSSIAN: As wīda galan berzin"
        agent = _Agent(_Mem())

        from agents.runner import run_agent as _run_agent
        result = _run_agent(
            "Ich sehe eine weiße Birke",
            agent=agent,
            system_prompt="test",
        )
        assert isinstance(result, RunResult)
        assert result.input == "Ich sehe eine weiße Birke"
        assert result.final == "As wīda galan berzin"
        assert result.tool_calls == 2
        assert result.model == "mock-model"
        assert result.validation["overall"]["status"] == "verified_in_coverage"
        assert result.latency_s >= 0
