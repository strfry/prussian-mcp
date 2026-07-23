"""FastMCP adapter — the four prussian tools over MCP (stdio / SSE).

A thin wrapper: every tool delegates to :mod:`prussian.tools`, the exact
same functions the inspect eval (:mod:`prussian.adapters.inspect`) and
the CLI agent (:mod:`prussian.adapters.agent`) use.  No LLM proxy, no
prompts, no resources — just the tools plus a startup FST health check.

Run it::

    python -m prussian.adapters.mcp            # stdio (Claude Code / Desktop)
    python -m prussian.adapters.mcp --web      # streamable-http (Claude Web)
    prussian-mcp                               # console-script entry point
"""

import argparse
import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from prussian.engine.fst.validate import check_fsg_pipeline
from prussian.tools import lookup_tool, search_tool, spec, validate_tool, wordforms_tool
from prussian.tools.runtime import get_engine, get_reranker

# ── Server ────────────────────────────────────────────────────────────────────

# Allow strfry.org for remote access via SSH tunnel.
_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=["127.0.0.1:*", "localhost:*", "strfry.org:*", "strfry.org"],
    allowed_origins=[
        "http://127.0.0.1:*",
        "http://localhost:*",
        "https://strfry.org",
        "https://strfry.org:*",
    ],
)
mcp = FastMCP(
    "Prussian Dictionary",
    transport_security=_security,
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8001")),
)

# The engine and reranker are the shared lazy singletons from
# ``prussian.tools.runtime`` so importing this module stays cheap and
# side-effect free (see the ``get_engine`` / ``get_reranker`` docstrings).


# ── Tools (1:1 with prussian.tools) ──────────────────────────────────────────
#
# Tool descriptions come from ``prussian.tools.spec`` (the single source of
# truth): each function's ``__doc__`` is set from the canonical spec before
# FastMCP registers it, so the MCP text can never drift from the other
# adapters.


def search_dictionary(
    query: str,
    top_k: int = 10,
    filter_tags: str | None = None,
    context: str = "",
) -> list[dict[str, Any]]:
    return search_tool(
        get_engine(),
        query,
        top_k=top_k,
        filter_tags=filter_tags,
        reranker=get_reranker() if context else None,
        context=context or None,
    )


def lookup_prussian_word(text: str, fuzzy: bool = False) -> list[dict[str, Any]]:
    return lookup_tool(get_engine(), text, fuzzy=fuzzy)


def get_word_forms(lemma: str, features: str | None = None) -> list[dict[str, Any]]:
    return wordforms_tool(get_engine(), lemma, features=features)


def validate_prussian(text: str, include_conllu: bool = False) -> str:
    return validate_tool(text, include_conllu=include_conllu)


# Attach the canonical docstring, then register with FastMCP.  ``mcp.tool()``
# reads ``__doc__`` for the tool description and the ``Args:`` block for the
# per-parameter schema, so the docstring must be set first.
for _fn, _spec in (
    (search_dictionary, spec.SEARCH),
    (lookup_prussian_word, spec.LOOKUP),
    (get_word_forms, spec.WORDFORMS),
    (validate_prussian, spec.VALIDATE),
):
    _fn.__doc__ = spec.docstring(_spec)
    mcp.tool()(_fn)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prussian Dictionary MCP server (four tools + FST health).",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        default=os.getenv("MCP_TRANSPORT") == "sse",
        help="Serve over streamable-http (Claude Web) instead of stdio.",
    )
    parser.add_argument("--host", default=None, help="Host for --web mode.")
    parser.add_argument("--port", type=int, default=None, help="Port for --web mode.")
    args = parser.parse_args()

    if args.host:
        mcp.settings.host = args.host
    if args.port:
        mcp.settings.port = args.port

    print("Loading Prussian Dictionary search engine...")
    get_engine()
    print("Search engine loaded successfully!")

    # Quick health check of the FST/CG3 grammar pipeline.
    _ok, fsg_msg = check_fsg_pipeline()
    print(fsg_msg)

    if args.web:
        print(
            f"Starting MCP server (streamable-http) on "
            f"http://{mcp.settings.host}:{mcp.settings.port}"
        )
        mcp.run(transport="streamable-http")
    else:
        print("Starting MCP server (stdio)")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
