"""Integration tests for the full Deep Research pipeline.

These tests require API keys and make real API calls.
Run with: pytest tests/integration/ -v --run-integration
"""

import json
import os
from pathlib import Path

import pytest


# Skip all tests in this module if no API key
pytestmark = pytest.mark.integration


@pytest.fixture
def fixtures_dir():
    """Path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


class TestFullPipeline:
    """Integration tests for the complete research pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_produces_verified_quotes(self):
        """Full pipeline should produce report with verified quotes section."""
        from langchain_core.messages import HumanMessage
        from open_deep_research.graph import deep_researcher

        result = await deep_researcher.ainvoke(
            {"messages": [HumanMessage(content="What is RAG?")]},
            config={"configurable": {"preset": "fast", "allow_clarification": False, "test_mode": True}}
        )

        # Structure assertions
        assert result.get("source_store"), "Should have sources"
        assert result.get("final_report"), "Should have report"

        # Verification ran (not disabled)
        report = result["final_report"]
        assert "Verification was disabled" not in report, "Verification should be enabled"

        # Report should have Verified Findings section
        assert "## Verified Findings" in report, "Report should have Verified Findings section"

    @pytest.mark.asyncio
    async def test_pipeline_collects_extract_api_sources(self):
        """Pipeline should collect sources via Extract API."""
        from langchain_core.messages import HumanMessage
        from open_deep_research.graph import deep_researcher

        result = await deep_researcher.ainvoke(
            {"messages": [HumanMessage(content="What is machine learning?")]},
            config={"configurable": {"preset": "fast", "allow_clarification": False, "test_mode": True}}
        )

        sources = result.get("source_store", [])
        assert len(sources) > 0, "Should have sources"

        # Check extraction methods
        methods = {s.get("extraction_method") for s in sources}
        assert "extract_api" in methods, "Should have Extract API sources"

        # Extract API sources should have substantial content
        extract_api_sources = [s for s in sources if s.get("extraction_method") == "extract_api"]
        if extract_api_sources:
            avg_content_len = sum(len(s.get("content", "")) for s in extract_api_sources) / len(extract_api_sources)
            assert avg_content_len > 1000, f"Extract API sources should have substantial content (avg: {avg_content_len})"

    @pytest.mark.asyncio
    async def test_verification_pass_rate(self):
        """At least some quotes should pass verification."""
        from langchain_core.messages import HumanMessage
        from open_deep_research.graph import deep_researcher

        result = await deep_researcher.ainvoke(
            {"messages": [HumanMessage(content="What are the benefits of RAG?")]},
            config={"configurable": {"preset": "fast", "allow_clarification": False, "test_mode": True}}
        )

        # Note: evidence_snippets is cleared after report generation
        # but we can check the report content
        report = result.get("final_report", "")

        # Should have some verified quotes in the report
        # Look for the markdown format: "quote" — [Source](url)
        assert '"' in report and "](http" in report, "Report should contain quoted sources"

    @pytest.mark.asyncio
    async def test_save_fixture_for_offline_tests(self, fixtures_dir):
        """Generate and save a fixture for offline testing."""
        from langchain_core.messages import HumanMessage
        from open_deep_research.graph import deep_researcher

        result = await deep_researcher.ainvoke(
            {"messages": [HumanMessage(content="What is RAG?")]},
            config={"configurable": {"preset": "fast", "allow_clarification": False, "test_mode": True}}
        )

        # Save fixture
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        fixture_path = fixtures_dir / "rag_query_result.json"

        fixture_data = {
            "source_store": result.get("source_store", []),
            "final_report": result.get("final_report", ""),
            "research_brief": result.get("research_brief", ""),
        }

        with open(fixture_path, "w") as f:
            json.dump(fixture_data, f, indent=2, default=str)

        assert fixture_path.exists(), "Fixture should be saved"
        print(f"\nFixture saved to: {fixture_path}")


class TestPresets:
    """Test different preset configurations."""

    @pytest.mark.asyncio
    async def test_fast_preset_runs(self):
        """Fast preset should complete successfully."""
        from langchain_core.messages import HumanMessage
        from open_deep_research.graph import deep_researcher

        result = await deep_researcher.ainvoke(
            {"messages": [HumanMessage(content="What is AI?")]},
            config={"configurable": {"preset": "fast", "allow_clarification": False, "test_mode": True}}
        )

        assert result.get("final_report"), "Should produce report"
        assert len(result.get("final_report", "")) > 1000, "Report should have substance"

    @pytest.mark.asyncio
    async def test_balanced_preset_enables_brief_context(self):
        """Balanced preset should enable brief context pre-search."""
        from open_deep_research.configuration import Configuration

        config = Configuration(preset="balanced")
        config = config.apply_preset()

        assert config.enable_brief_context is True, "Balanced should enable brief context"
        assert config.use_council is False, "Balanced should not enable council"
