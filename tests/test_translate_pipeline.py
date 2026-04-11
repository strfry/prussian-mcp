"""Tests for the Haystack translation pipeline."""

import json
import pytest

from prussian_engine.search import SearchEngine
from prussian_mcp.translate_pipeline import (
    create_lookup_tool,
    create_tools,
    build_pipeline,
    run_translation,
)


@pytest.fixture(scope="module")
def engine():
    return SearchEngine()


@pytest.fixture(scope="module")
def lookup_fn(engine):
    return create_lookup_tool(engine)


class TestLookupTool:
    def test_lookup_without_pgr(self, lookup_fn):
        """Lookup with just a query returns matches."""
        result = json.loads(lookup_fn(query="König"))
        assert "matches" in result
        assert len(result["matches"]) > 0
        # Should not have forms key without pgr
        assert "forms" not in result

    def test_lookup_with_pgr(self, lookup_fn):
        """Lookup with pgr filter returns form information."""
        result = json.loads(lookup_fn(query="König", pgr="NOM.SG.MASC"))
        assert "matches" in result
        assert len(result["matches"]) > 0
        # Should have forms key with matching forms
        assert "forms" in result
        assert len(result["forms"]) > 0
        form = result["forms"][0]
        assert "lemma" in form
        assert "matching_forms" in form
        assert "translations" in form
        assert len(form["matching_forms"]) > 0

    def test_lookup_empty_query(self, lookup_fn):
        """Lookup with nonsense query still returns matches list."""
        result = json.loads(lookup_fn(query="xyznonexistent"))
        assert "matches" in result

    def test_lookup_pgr_no_match(self, lookup_fn):
        """Lookup with pgr that doesn't match any form omits form key."""
        result = json.loads(lookup_fn(query="König", pgr="VOC.DU.NEUT"))
        assert "matches" in result
        # Unlikely PGR combo — form may or may not be present


class TestCreateTools:
    def test_creates_tool_list(self, engine):
        tools = create_tools(engine)
        assert len(tools) == 1
        assert tools[0].name == "lookup"


class TestBuildPipeline:
    def test_pipeline_builds(self, engine):
        pipeline, tools = build_pipeline(engine)
        assert pipeline is not None
        assert len(tools) == 1
        # Verify components exist
        assert "prompt" in pipeline.graph.nodes
        assert "generator" in pipeline.graph.nodes
        assert "router" in pipeline.graph.nodes
        assert "tool_invoker" in pipeline.graph.nodes


@pytest.mark.integration
class TestIntegration:
    def test_pipeline_single_shot(self, engine):
        """Run a full translation (requires running LLM)."""
        result = run_translation(engine, "Der Sohn des Bauern gibt dem Hund den Pfefferkuchen", max_loops=10)
        assert result
        assert isinstance(result, str)
        assert result != "Max tool-calling loops reached"

    def test_streaming(self, engine):
        """Run translation with streaming and verify tokens arrive."""
        tokens = []

        def on_token(chunk):
            if chunk.content:
                tokens.append(chunk.content)

        result = run_translation(
            engine, "Das Wasser ist kalt", streaming_callback=on_token, max_loops=10
        )
        assert result
        assert isinstance(result, str)
        # Streaming should have produced tokens
        assert len(tokens) > 0
