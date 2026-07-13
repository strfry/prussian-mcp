# prussian-mcp — agent and dictionary server

## What this project is

Two products in one repo:

1. **`prussian-agent`** — a CLI that translates a German sentence into
   Old Prussian via a smolagents ToolCallingAgent with access to four
   tools (`search_dictionary`, `lookup_prussian_word`, `get_word_forms`,
   `validate_prussian`).  The model self-corrects by calling
   `validate_prussian` on its draft before emitting the final
   `PRUSSIAN: <sentence>` line — there is no external orchestration
   loop, just `max_steps` on the Agent.
2. **FastMCP server** (`mcp_server.py`) — the same four tools exposed
   via the MCP protocol (stdio or streamable-http), plus a streaming
   LLM proxy.  Drives any MCP-capable client (Claude Code, Claude
   Desktop, OpenCode).

The dictionary/FST engine lives in `prussian_engine/` (see
`prussian_engine/search.py:SearchEngine`, `prussian_engine/fsg_check.py`).
`prussian-fst` (sibling repo `../prussian-fst`) is imported in-process
as an editable uv path-dependency — see `[tool.uv.sources]` in
`pyproject.toml`.

## Layout

| Path | Role |
|---|---|
| `agents/` | The `prussian-agent` runner: `runner.py` (`run_agent`/`RunResult`, `extract_candidate`, `parse_last_validation`), `tools.py` (in-process smolagents tools), `cli.py` (argparse entry point). |
| `prussian_engine/` | Dictionary + FST/CG3 engine (`SearchEngine`, `run_validate`, config from env). |
| `mcp_server.py` | FastMCP server (stdio / streamable-http), streaming LLM proxy, MCP tools + resources. |
| `prompts/` | `agent_system_en.md` is the canonical agent prompt; `plan_prompt.txt` / `final_prompt.txt` for MCP plan/final prompts; `syntax_rules.txt`, `base_vocab.md` as MCP resources. |
| `tests/test_agent_loop.py` | Offline tests: `extract_candidate`, `parse_last_validation`, `run_agent` with a mock agent; `--validate-only` subprocess tests over the real CLI. |

## Running the agent

`uv run prussian-agent …` (or `.venv/bin/prussian-agent …`).  Source
the env file first — `prussian_engine.config` reads env vars at import
time, so they must be set before the SearchEngine is constructed (the
toolset factory does that lazily, so the CLI startup can still import
argparse first).

```bash
source env.hf-voyage.sh        # OPENAI_API_KEY + HF router + Voyage reranker
uv run prussian-agent "Ich sehe eine weiße Birke" --json
uv run prussian-agent --validate-only "As wīda gaīlan berzin"   # FST/CG3 only
uv run prussian-agent "…" --mcp-url https://strfry.org/prussian-mcp/mcp
```

Exit codes from `validation.overall.status`: 0 verified_in_coverage ·
2 out_of_coverage · 3 violations_found · 4 no candidate / no
validate_prussian in the run · 1 runtime error.

## Config and secrets

- All env files (`env.hf.sh`, `env.hf-voyage.sh`, `env.jina.sh`,
  `env.local.sh`, `*.env`) are gitignored.  They set
  `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` plus the
  embedding / reranker env vars that `prussian_engine.config`
  reads.  Source one before invoking the CLI.
- `prussian-fst` is an editable uv path-dependency.  Adjust the path
  in `[tool.uv.sources]` of `pyproject.toml` if the checkout lives
  elsewhere, then `uv sync`.

## Tests

```bash
.venv/bin/python -m pytest tests/test_agent_loop.py -q
```

The subprocess tests for `--validate-only` require the FST/CG3
artifacts to be built: `make -C ../prussian-fst all cg3-check`.

## Important rules

### 1. Bei Unsicherheit fragen statt raten
- Immer das Question-Tool nutzen wenn unsicher
- Keine Annahmen über Kompatibilität machen
- Vor wichtigen Änderungen nachfragen

### 2. Venv beachten
- Python-Programme immer über Projekt-Virtualenv `.venv` benutzen
