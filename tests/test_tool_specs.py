"""Tool descriptions come from one source (prussian.tools.spec).

Guards "einheitliche Toolbeschreibungen": the FastMCP server, the inspect-ai
eval and the smolagents CLI must all present the canonical description text
from ``prussian.tools.spec``.  Runs offline with a mock engine (no embeddings,
no network).  The inspect-ai adapter is only checked when the ``eval`` extra
(inspect_ai) is installed.
"""

import asyncio
import unittest
from unittest.mock import MagicMock

from prussian.tools import spec

_SPECS = {s.name: s for s in spec.ALL}


def _mcp_tools():
    import prussian.adapters.mcp as mcpmod

    tools = asyncio.new_event_loop().run_until_complete(mcpmod.mcp.list_tools())
    return {t.name: t for t in tools}


def _smolagents_tools():
    from prussian.adapters.agent.tools import build_local_toolset

    return {t.name: t for t in build_local_toolset(engine=MagicMock())}


class TestToolSpecSingleSource(unittest.TestCase):
    """Every adapter derives its tool text from prussian.tools.spec."""

    def test_specs_cover_the_four_tools(self):
        self.assertEqual(
            set(_SPECS),
            {"search_dictionary", "lookup_prussian_word",
             "get_word_forms", "validate_prussian"},
        )

    def test_mcp_uses_spec(self):
        mcp = _mcp_tools()
        self.assertEqual(set(mcp), set(_SPECS))
        for name, s in _SPECS.items():
            # FastMCP uses the whole docstring (description + Args) as the
            # tool description.
            self.assertEqual(mcp[name].description, spec.docstring(s),
                             f"MCP {name} description")

    def test_smolagents_uses_spec(self):
        sa = _smolagents_tools()
        self.assertEqual(set(sa), set(_SPECS))
        for name, s in _SPECS.items():
            self.assertEqual(sa[name].description, s.description,
                             f"smolagents {name} description")
            for arg, doc in s.args.items():
                self.assertEqual(sa[name].inputs[arg]["description"], doc,
                                 f"smolagents {name}.{arg} arg doc")

    def test_inspect_uses_spec(self):
        try:
            from inspect_ai.tool._tool_def import ToolDef
            import prussian.adapters.inspect_tools as it
        except ImportError:
            self.skipTest("inspect_ai not installed (eval extra)")

        for name, s in _SPECS.items():
            td = ToolDef(getattr(it, name)())
            self.assertEqual(td.description, s.description,
                             f"inspect {name} description")
            for arg, doc in s.args.items():
                p = td.parameters.properties.get(arg)
                self.assertIsNotNone(p, f"inspect {name}.{arg} missing")
                self.assertEqual(p.description, doc,
                                 f"inspect {name}.{arg} arg doc")


if __name__ == "__main__":
    unittest.main()
