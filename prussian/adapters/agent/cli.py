"""argparse CLI for ``prussian-agent``.

Examples::

    prussian-agent "Ich sehe eine weiße Birke"
    prussian-agent "…" --json
    prussian-agent --validate-only "As wīda gaīlan berzin"
    prussian-agent "…" --mcp-url https://strfry.org/prussian-mcp/mcp
    prussian-agent "…" --system-prompt prompts/agent_system_en.md \\
        --extra-prompt A --extra-prompt B

Exit codes (derived from ``validation.overall.status`` when present):

* 0 — verified_in_coverage
* 2 — out_of_coverage
* 3 — violations_found
* 4 — no candidate / no validate_prussian tool call in the run
* 1 — runtime error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Resolve the default prompt from the source tree (works for editable
# installs and direct from-source runs; for wheel installs pass
# --system-prompt explicitly).
DEFAULT_PROMPT = Path(__file__).resolve().parent.parent / "prompts" / "agent_system_en.md"

# Maps validation overall.status -> exit code.
STATUS_TO_EXIT = {
    "verified_in_coverage": 0,
    "out_of_coverage": 2,
    "violations_found": 3,
}


def load_system_prompt(path: Path) -> str:
    """Load the system prompt from a text/markdown file."""
    return path.read_text(encoding="utf-8").strip()


def _exit_code_from_validation(validation: dict | None) -> int:
    if not validation:
        return 4
    overall = validation.get("overall") or {}
    status = overall.get("status")
    return STATUS_TO_EXIT.get(status, 4)


def _build_env_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="prussian-agent",
        description=(
            "Translate a German sentence into Old Prussian via an LLM "
            "agent with dictionary/FST/CG3 tools.  The model self-corrects "
            "by calling validate_prussian on its draft before the final "
            "PRUSSIAN: line."
        ),
    )
    ap.add_argument("sentence", help="German sentence to translate.")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", ""),
                    help="LLM model name (default: $OPENAI_MODEL).")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""),
                    help="OpenAI-compatible base URL (default: $OPENAI_BASE_URL).")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                    help="API key (default: $OPENAI_API_KEY).")
    ap.add_argument("--max-steps", type=int, default=30,
                    help="max steps for the agent (default: 30).")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="LLM sampling temperature (default: 0.2).")
    ap.add_argument("--json", action="store_true",
                    help="emit a JSON result on stdout instead of plain text.")
    ap.add_argument("--mcp-url", default=None,
                    help="use a remote MCP server instead of the in-process "
                         "tools.")
    ap.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT,
                    help="path to the system-prompt markdown file (default: "
                         "prompts/agent_system_en.md).")
    ap.add_argument("--extra-prompt", type=Path, action="append",
                    default=[], help="additional prompt file appended to the "
                    "system prompt (repeatable).")
    ap.add_argument("--validate-only", action="store_true",
                    help="skip the LLM; run only the FST/CG3 grammar check "
                         "on the given Prussian text and exit.")
    ap.add_argument("--include-conllu", action="store_true",
                    help="with --validate-only, also include the CoNLL-U "
                         "dependency analysis in the JSON output.")
    ap.add_argument("--verbose", action="store_true",
                    help="print the full Python traceback on runtime errors.")
    return ap


def _run_validate_only(args: argparse.Namespace) -> int:
    from prussian.engine.fst.validate import run_validate  # lazy
    try:
        raw = run_validate(args.sentence, include_conllu=args.include_conllu)
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        parsed = None
    if args.json:
        sys.stdout.write(raw if isinstance(raw, str) else json.dumps(raw))
        if not sys.stdout.write("\n"):
            pass
    else:
        if parsed and "overall" in parsed:
            overall = parsed["overall"]
            print(f"overall: {overall.get('status')} "
                  f"({overall.get('n_sentences')} sentences, "
                  f"{overall.get('n_violations')} violations)")
            for s in parsed.get("sentences", []):
                print(f"  [{s.get('sent_id')}] {s.get('status')}: "
                      f"{s.get('text')}")
        else:
            print(raw)
    return _exit_code_from_validation(parsed)


def _build_toolset(args: argparse.Namespace):
    """Return (toolset, is_remote).  In-process is default; --mcp-url
    switches to smolagents MCPClient."""
    if args.mcp_url:
        try:
            from smolagents import ToolCollection
        except ImportError as e:
            raise SystemExit(
                f"--mcp-url requires smolagents: {e}\n"
                "  uv sync"
            )
        tool_collection = ToolCollection.from_mcp(
            {"url": args.mcp_url, "transport": "streamable-http"},
            trust_remote_code=True,
        )
        return list(tool_collection.tools), True
    # In-process default — lazy SearchEngine import inside the factory.
    from .tools import build_local_toolset
    return build_local_toolset(), False


def main(argv: list[str] | None = None) -> int:
    ap = _build_env_argparser()
    args = ap.parse_args(argv)

    # --validate-only short-circuits before any LLM/Agent setup.
    if args.validate_only:
        if args.mcp_url:
            ap.error("--validate-only and --mcp-url are mutually exclusive")
        return _run_validate_only(args)

    # Resolve model/base-url/api-key with sane fallbacks.
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("warning: OPENAI_API_KEY not set — most providers will reject "
              "the request", file=sys.stderr)
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "")
    model = args.model or os.environ.get("OPENAI_MODEL", "")
    if not model:
        ap.error("no model configured — set OPENAI_MODEL or pass --model")

    # Build system prompt.
    try:
        system_prompt = load_system_prompt(args.system_prompt)
    except (OSError, RuntimeError) as e:
        ap.error(f"could not load system prompt: {e}")
    for ep in args.extra_prompt:
        system_prompt = system_prompt + "\n\n" + ep.read_text(encoding="utf-8")

    # Toolset (in-process default, remote via --mcp-url).
    try:
        toolset, is_remote = _build_toolset(args)
    except SystemExit:
        raise
    except Exception as e:
        print(f"error building toolset: {type(e).__name__}: {e}",
              file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    # Model + Agent.
    from .runner import build_agent, build_model, run_agent

    llm_model = build_model(
        model=model,
        api_key=api_key,
        api_base_url=base_url,
        temperature=args.temperature,
    )
    agent = build_agent(llm_model, toolset, max_steps=args.max_steps,
                        instructions=system_prompt)

    try:
        result = run_agent(
            args.sentence,
            agent=agent,
            system_prompt=system_prompt,
        )
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    # Output.
    if args.json:
        out = {
            "input": result.input,
            "final": result.final,
            "status": (result.validation or {}).get("overall", {}).get("status"),
            "validation": result.validation,
            "tool_calls": result.tool_calls,
            "latency_s": result.latency_s,
            "model": result.model,
        }
        sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write((result.final or result.output) + "\n")

    return _exit_code_from_validation(result.validation)


if __name__ == "__main__":
    sys.exit(main())
