"""Offline tests for the ``prussian-agent`` loop.

No LLM, no SearchEngine.  The ``extract_candidate`` and
``parse_last_validation`` helpers are tested directly; ``run_agent`` is
exercised against a fake agent that returns a scripted message history
including an assistant ``validate_prussian`` tool call and a tool result
message — so we verify the verdict is read off the model's own history
(no second validation run).

The validate-only CLI path is exercised through subprocess invocations
of the real CLI; those require the FST/CG3 artifacts to be built
(``make -C ../prussian-fst all cg3-check``) and are skipped if
``prussian_fst`` cannot be imported.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prussian_mcp.agent.runner import (
    RunResult,
    extract_candidate,
    parse_last_validation,
    run_agent,
)
from haystack.dataclasses import ChatMessage, ToolCall


# ── extract_candidate ─────────────────────────────────────────────────────


class TestExtractCandidate:
    def test_simple_marker(self):
        text = "Some reasoning.\nPRUSSIAN: As wīda gaīlan berzin"
        assert extract_candidate(text) == "As wīda gaīlan berzin"

    def test_last_marker_wins(self):
        text = (
            "Draft 1.\nPRUSSIAN: foo\n"
            "Draft 2.\nPRUSSIAN: bar\n"
        )
        # last Multiline match wins
        assert extract_candidate(text) == "bar"

    def test_marker_with_extra_whitespace(self):
        text = "PRUSSIAN:   kelānts  "
        assert extract_candidate(text) == "kelānts"

    def test_marker_with_surrounding_quotes_is_stripped(self):
        text = 'PRUSSIAN: "As wīda gaīlan berzin"'
        # inner quote pair is stripped (matches spec: quotes stripped)
        assert extract_candidate(text) == 'As wīda gaīlan berzin'

    def test_no_marker_fallback_last_nonempty_line(self):
        text = "line1\nline2 with content\n   \n"
        assert extract_candidate(text) == "line2 with content"

    def test_no_marker_strips_fences_NOT_done(self):
        # The spec is explicit: no fence stripping.  A fenced block
        # without a PRUSSIAN: line falls back to the last non-empty
        # line, which is the closing ``` marker (the model is told not
        # to use fences; if it does, this is a known limitation).
        text = "```\nas wīda\n```"
        # Fallback path: last non-empty line is "```" — strip quotes/WS;
        # backticks are NOT stripped per spec.
        assert extract_candidate(text) == "```"

    def test_empty_text(self):
        assert extract_candidate("") is None
        assert extract_candidate(None) is None

    def test_marker_inside_word_not_matched_at_end_of_line(self):
        # The regex requires PRUSSIAN: at start of a line, so embedded
        # text doesn't match.
        text = "Some text PRUSSIAN: foo bar"
        # No match: PRUSSIAN: isn't at start of a line — fallback to last non-empty line
        assert extract_candidate(text) == "Some text PRUSSIAN: foo bar"


# ── parse_last_validation ──────────────────────────────────────────────────


def _assistant_with_validate_call(call_id: str, args: dict) -> ChatMessage:
    return ChatMessage.from_assistant(
        text="",
        tool_calls=[ToolCall(id=call_id, tool_name="validate_prussian",
                             arguments=json.dumps(args))],
    )


def _tool_result_message(call_id: str, payload: dict) -> ChatMessage:
    """Construct a tool-result message tied to a ToolCall id."""
    call = ToolCall(id=call_id, tool_name="validate_prussian",
                    arguments="{}")
    msg = ChatMessage.from_tool(tool_result=json.dumps(payload), origin=call)
    return msg


class TestParseLastValidation:
    def test_no_validate_in_history(self):
        msgs = [ChatMessage.from_system("sys"), ChatMessage.from_user("u")]
        assert parse_last_validation(msgs) is None

    def test_single_validate_call(self):
        payload = {"overall": {"status": "verified_in_coverage",
                                "n_sentences": 1, "n_violations": 0},
                    "sentences": []}
        msgs = [
            _assistant_with_validate_call("c1", {"text": "As wīda"}),
            _tool_result_message("c1", payload),
            ChatMessage.from_assistant(text="PRUSSIAN: As wīda"),
        ]
        out = parse_last_validation(msgs)
        assert out is not None
        assert out["overall"]["status"] == "verified_in_coverage"

    def test_last_validate_call_wins(self):
        p1 = {"overall": {"status": "violations_found"}}
        p2 = {"overall": {"status": "verified_in_coverage"}}
        msgs = [
            _assistant_with_validate_call("c1", {"text": "draft"}),
            _tool_result_message("c1", p1),
            _assistant_with_validate_call("c2", {"text": "fixed"}),
            _tool_result_message("c2", p2),
            ChatMessage.from_assistant(text="PRUSSIAN: fixed"),
        ]
        out = parse_last_validation(msgs)
        assert out["overall"]["status"] == "verified_in_coverage"

    def test_unparseable_result_returns_none(self):
        msgs = [
            _assistant_with_validate_call("c1", {"text": "draft"}),
            ChatMessage.from_tool(tool_result="not json",
                                  origin=ToolCall(id="c1",
                                                  tool_name="validate_prussian",
                                                  arguments="{}")),
        ]
        assert parse_last_validation(msgs) is None

    def test_tool_result_missing_returns_none(self):
        # Model called validate_prussian but the agent loop ended
        # before the tool result landed in history.
        msgs = [_assistant_with_validate_call("c1", {"text": "draft"})]
        assert parse_last_validation(msgs) is None


# ── run_agent with a fake Agent ────────────────────────────────────────────


def _build_fake_agent_messages(
    final_text: str,
    validation_payload: dict | None,
    more_tool_calls: int = 0,
) -> list[ChatMessage]:
    """Construct a scripted message history as if Agent.run() produced it.

    Layout: system, user, assistant(validate_call), tool_result,
    [optional more lookup tool calls], assistant(final_text).
    """
    msgs = [
        ChatMessage.from_system("sys"),
        ChatMessage.from_user("Translate to Prussian: ..."),
    ]
    if validation_payload is not None:
        msgs.append(_assistant_with_validate_call(
            "validate", {"text": "draft sentence"}))
        msgs.append(_tool_result_message("validate", validation_payload))
    for i in range(more_tool_calls):
        msgs.append(ChatMessage.from_assistant(
            text="",
            tool_calls=[ToolCall(id=f"tc_{i}",
                                 tool_name="lookup_prussian_word",
                                 arguments=json.dumps({"word": "foo"}))],
        ))
        msgs.append(ChatMessage.from_tool(
            tool_result="[]",
            origin=ToolCall(id=f"tc_{i}",
                            tool_name="lookup_prussian_word",
                            arguments="{}"),
        ))
    msgs.append(ChatMessage.from_assistant(text=final_text))
    return msgs


def _fake_agent(messages: list[ChatMessage], *, model: str = "fake-model") -> MagicMock:
    """Agent stub whose .run() returns the scripted messages."""
    agent = MagicMock()
    agent.run.return_value = {"messages": messages}
    agent.chat_generator = MagicMock()
    agent.chat_generator.model = model
    return agent


class TestRunAgent:
    def test_verified_exit_path(self):
        payload = {"overall": {"status": "verified_in_coverage",
                                "n_sentences": 1, "n_violations": 0}}
        msgs = _build_fake_agent_messages(
            final_text="Some analysis.\nPRUSSIAN: As wīda gaīlan berzin",
            validation_payload=payload,
        )
        agent = _fake_agent(msgs)
        result = run_agent("Ich sehe eine weiße Birke",
                           agent=agent, system_prompt="sys")
        assert isinstance(result, RunResult)
        assert result.final == "As wīda gaīlan berzin"
        assert result.validation == payload
        assert result.tool_calls == 1
        assert result.model == "fake-model"
        assert result.input == "Ich sehe eine weiße Birke"

    def test_violations_exit_path(self):
        payload = {"overall": {"status": "violations_found",
                                "n_sentences": 1, "n_violations": 2}}
        msgs = _build_fake_agent_messages(
            final_text="PRUSSIAN: As wīda gailā berzin",
            validation_payload=payload,
        )
        agent = _fake_agent(msgs)
        result = run_agent("test", agent=agent, system_prompt="sys")
        assert result.validation == payload
        # Exit-code mapping is exercised in TestCLIViaSubprocess below;
        # here we just confirm the verdict is parsed.

    def test_no_validation_in_run_returns_none(self):
        # Model never called validate_prussian — validation is None.
        msgs = _build_fake_agent_messages(
            final_text="PRUSSIAN: foo", validation_payload=None,
        )
        agent = _fake_agent(msgs)
        result = run_agent("test", agent=agent, system_prompt="sys")
        assert result.validation is None
        assert result.final == "foo"

    def test_no_candidate_extracted(self):
        payload = {"overall": {"status": "verified_in_coverage"}}
        msgs = _build_fake_agent_messages(
            final_text="",  # empty final assistant text
            validation_payload=payload,
        )
        agent = _fake_agent(msgs)
        result = run_agent("test", agent=agent, system_prompt="sys")
        assert result.final is None

    def test_tool_call_count_complex(self):
        payload = {"overall": {"status": "verified_in_coverage"}}
        msgs = _build_fake_agent_messages(
            final_text="PRUSSIAN: foo",
            validation_payload=payload,
            more_tool_calls=3,
        )
        agent = _fake_agent(msgs)
        result = run_agent("test", agent=agent, system_prompt="sys")
        assert result.tool_calls == 4  # 1 validate + 3 lookups


# ── validate-only CLI subprocess tests ─────────────────────────────────────


def _fst_available() -> bool:
    try:
        import prussian_fst  # noqa: F401
        return True
    except ImportError:
        return False


PROJECT_DIR = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(not _fst_available(),
                    reason="prussian_fst not importable — FST artifacts not built")
class TestCLIValidateOnly:
    def _run_cli(self, *args: str) -> subprocess.CompletedProcess:
        venv_py = PROJECT_DIR / ".venv" / "bin" / "python"
        return subprocess.run(
            [str(venv_py), "-m", "prussian_mcp.cli",
             "--validate-only", *args],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_DIR),
        )

    def test_verified_sentence(self):
        r = self._run_cli("As wīda gaīlan berzin")
        assert r.returncode == 0, r.stderr
        assert "verified_in_coverage" in r.stdout

    def test_violated_sentence(self):
        # Missing macronisation or wrong agreement — should NOT be
        # verified.  The exact status (out_of_coverage or
        # violations_found) depends on the analyser; both are non-zero
        # exit codes.
        r = self._run_cli("As wīda gailā berzin")
        assert r.returncode in (2, 3), r.stderr

    def test_json_output(self):
        r = self._run_cli("--json", "As wīda gaīlan berzin")
        assert r.returncode == 0
        parsed = json.loads(r.stdout)
        assert parsed["overall"]["status"] == "verified_in_coverage"