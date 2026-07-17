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
2. **FastMCP server** (`prussian.adapters.mcp`) — the same four tools
   exposed via the MCP protocol (stdio or streamable-http).  Drives any
   MCP-capable client (Claude Code, Claude Desktop, OpenCode).  It is a
   thin adapter: nothing but the four tools plus an FST health check.

Everything lives in one package, **`prussian/`**.  The four tool
functions in `prussian/tools/` are the single source of truth; the MCP,
inspect-ai, and CLI adapters wrap them 1:1.  The dictionary/FST engine
is `prussian/engine/` (see `prussian/engine/search.py:SearchEngine`,
`prussian/engine/fst/validate.py`).  `prussian-fst` (sibling repo
`../prussian-fst`) is imported in-process as an editable uv
path-dependency — see `[tool.uv.sources]` in `pyproject.toml`.

## Layout

Single top-level package `prussian/`:

| Path | Role |
|---|---|
| `prussian/tools/` | The four canonical tools (`search_tool`, `lookup_tool`, `wordforms_tool`, `validate_tool`) — the single source of truth every adapter wraps. Also the `validate`/`search`/`lookup`/`wordforms` console scripts. |
| `prussian/engine/` | Dictionary + FST/CG3 engine. `search.py` (`SearchEngine`), `morphology.py` (PGR), `embeddings/` (`backend.py`, `client.py`, `rerank.py`), `fst/` (`tags.py`, `validate.py` = `run_validate`/`validate_with_corrections`/`check_fsg_pipeline`). |
| `prussian/adapters/mcp.py` | FastMCP server (stdio / streamable-http): the four tools + FST health, nothing else. Entry point `prussian-mcp`. |
| `prussian/adapters/agent/` | The `prussian-agent` CLI: `runner.py` (`run_agent`/`RunResult`, `extract_candidate`, `parse_last_validation`), `tools.py` (in-process smolagents tools), `cli.py` (argparse entry point). |
| `prussian/adapters/inspect_tools.py` | inspect-ai `@tool` wrappers for the reconstruction eval. |
| `prussian/config.py` | Env-driven config, read at import time. |
| `evals/` | inspect-ai eval harness (`reconstruction.py`, `corpus_dataset.py`); imports `prussian.adapters.inspect_tools`. Needs the `eval` extra (`inspect-ai`). |
| `prompts/` | `agent_system_en.md` is the canonical agent prompt; `syntax_rules.txt`, `base_vocab.md`. |
| `tests/test_agent_loop.py` | Offline tests: `extract_candidate`, `parse_last_validation`, `run_agent` with a mock agent. |

## Running the agent

`uv run prussian-agent …` (or `.venv/bin/prussian-agent …`).  Source
the env file first — `prussian.config` reads env vars at import
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

## Running the MCP server

Thin adapter exposing the same four tools over MCP.  Source an env file
first (it constructs a `SearchEngine` on startup).

```bash
source env.hf-model2vec.sh
uv run prussian-mcp                 # stdio (Claude Code / Desktop)
uv run prussian-mcp --web          # streamable-http for Claude Web
# or: .venv/bin/python -m prussian.adapters.mcp
```

`--host` / `--port` (or `MCP_HOST` / `MCP_PORT`) configure `--web` mode.
`.mcp.json` / `opencode.json` already point clients at
`python -m prussian.adapters.mcp` / `uv run prussian-mcp`.

## Config and secrets

- All env files (`env.hf.sh`, `env.hf-voyage.sh`, `env.jina.sh`,
  `env.local.sh`, `*.env`) are gitignored.  They set
  `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` plus the
  embedding / reranker env vars that `prussian.config`
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

## Eval Suite Setup

The inspect-ai reconstruction eval (`feature/evals` branch) additionally requires:

1. **CG3 Grammar binaries** (for `validate_prussian` tool):
   ```bash
   make -C ../prussian-fst cg3-sets   # Generate CG3 sets from valence.json
   make -C ../prussian-fst cg3-check  # Compile .cg3 rules → .bin binaries
   ```
   Output: `../prussian-fst/fst/build/cg3/*.bin` (disambiguator, dependency, validator)

2. **ConLLU silver standard** (for focus-token recovery metrics):
   ```bash
   make -C ../prussian-fst conllu
   ```
   Output: `../prussian-fst/data/prussian_silver.conllu` (parsed gold corpus)

3. **Corpus data** (already in `../prussian-corpus` as part of sibling checkouts)

Then run the eval:
```bash
source .venv/bin/activate
inspect eval evals/reconstruction.py --model openai/$OPENAI_MODEL
inspect view  # browse results
```

## Important rules

### 1. Bei Unsicherheit fragen statt raten
- Immer das Question-Tool nutzen wenn unsicher
- Keine Annahmen über Kompatibilität machen
- Vor wichtigen Änderungen nachfragen

### 2. Venv beachten
- Python-Programme immer über Projekt-Virtualenv `.venv` benutzen
