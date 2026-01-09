"""Unit tests for synthesis module.

Uses mock data only - no API calls.
"""

import pytest
from open_deep_research.pointer_extract import Pointer, Extraction
from open_deep_research.synthesis import (
    SynthesisBlock,
    SynthesizedReport,
    format_facts_for_synthesis,
    parse_synthesis_response,
    synthesize_report,
    render_report_markdown,
    render_report_plain,
)


@pytest.fixture
def sample_extractions():
    """Sample verified extractions for testing."""
    return [
        Extraction(
            pointer=Pointer("src_001", ["RAND", "security"], "RAND report"),
            status="verified",
            extracted_text="The RAND Corporation released a security report in October 2025.",
            match_score=1.0,
            source_url="https://rand.org/report"
        ),
        Extraction(
            pointer=Pointer("src_002", ["OpenAI", "safety"], "OpenAI initiatives"),
            status="verified",
            extracted_text="OpenAI announced the Frontier Risk Council for safe deployment.",
            match_score=0.9,
            source_url="https://openai.com/safety"
        ),
    ]


@pytest.fixture
def sample_synthesis():
    """Sample synthesis data from LLM."""
    return {
        "intro": "This report covers recent AI safety developments.",
        "transitions": [
            "Among the key findings,",
            "Building on these recommendations,"
        ],
        "conclusion": "These developments show progress in AI safety."
    }


class TestFormatFactsForSynthesis:
    """Tests for formatting facts for synthesis prompt."""

    def test_formats_verified_facts(self, sample_extractions):
        output = format_facts_for_synthesis(sample_extractions)

        assert "FACT 1:" in output
        assert "FACT 2:" in output
        assert "RAND Corporation" in output
        assert "OpenAI" in output

    def test_includes_source_context(self, sample_extractions):
        output = format_facts_for_synthesis(sample_extractions)

        assert "RAND report" in output
        assert "OpenAI initiatives" in output

    def test_skips_non_verified(self):
        extractions = [
            Extraction(
                pointer=Pointer("src_001", ["test"], "Test"),
                status="not_found",
                match_score=0.0
            )
        ]

        output = format_facts_for_synthesis(extractions)

        assert "FACT" not in output


class TestParseSynthesisResponse:
    """Tests for parsing LLM synthesis response."""

    def test_valid_json(self):
        response = '''{
            "intro": "Introduction here",
            "transitions": ["First transition", "Second transition"],
            "conclusion": "Conclusion here"
        }'''

        result = parse_synthesis_response(response, num_facts=2)

        assert result["intro"] == "Introduction here"
        assert len(result["transitions"]) == 2
        assert result["conclusion"] == "Conclusion here"

    def test_json_with_surrounding_text(self):
        response = '''Here is the synthesis:
        {"intro": "Test intro", "transitions": ["T1"], "conclusion": "End"}
        Done.'''

        result = parse_synthesis_response(response, num_facts=1)

        assert result["intro"] == "Test intro"

    def test_invalid_json_returns_defaults(self):
        response = "This is not valid JSON"

        result = parse_synthesis_response(response, num_facts=3)

        assert result["intro"] == ""
        assert len(result["transitions"]) == 3
        assert result["conclusion"] == ""


class TestSynthesizeReport:
    """Tests for building synthesized report."""

    def test_builds_report_structure(self, sample_extractions, sample_synthesis):
        report = synthesize_report(
            sample_extractions,
            "Test Report",
            sample_synthesis
        )

        assert report.title == "Test Report"
        assert len(report.blocks) > 0

    def test_includes_intro_block(self, sample_extractions, sample_synthesis):
        report = synthesize_report(
            sample_extractions,
            "Test",
            sample_synthesis
        )

        intro_blocks = [b for b in report.blocks if b.type == "intro"]
        assert len(intro_blocks) == 1
        assert "AI safety" in intro_blocks[0].content

    def test_includes_verified_blocks(self, sample_extractions, sample_synthesis):
        report = synthesize_report(
            sample_extractions,
            "Test",
            sample_synthesis
        )

        verified_blocks = [b for b in report.blocks if b.type == "verified"]
        assert len(verified_blocks) == 2

    def test_includes_transitions(self, sample_extractions, sample_synthesis):
        report = synthesize_report(
            sample_extractions,
            "Test",
            sample_synthesis
        )

        transition_blocks = [b for b in report.blocks if b.type == "transition"]
        assert len(transition_blocks) >= 1

    def test_includes_conclusion(self, sample_extractions, sample_synthesis):
        report = synthesize_report(
            sample_extractions,
            "Test",
            sample_synthesis
        )

        conclusion_blocks = [b for b in report.blocks if b.type == "conclusion"]
        assert len(conclusion_blocks) == 1

    def test_verified_count_property(self, sample_extractions, sample_synthesis):
        report = synthesize_report(
            sample_extractions,
            "Test",
            sample_synthesis
        )

        assert report.verified_count == 2

    def test_synthesis_count_property(self, sample_extractions, sample_synthesis):
        report = synthesize_report(
            sample_extractions,
            "Test",
            sample_synthesis
        )

        # intro + transitions + conclusion
        assert report.synthesis_count >= 3


class TestRenderReportMarkdown:
    """Tests for markdown rendering."""

    def test_includes_title(self, sample_extractions, sample_synthesis):
        report = synthesize_report(sample_extractions, "My Title", sample_synthesis)
        output = render_report_markdown(report, use_color=False)

        assert "# My Title" in output

    def test_includes_verified_content(self, sample_extractions, sample_synthesis):
        report = synthesize_report(sample_extractions, "Test", sample_synthesis)
        output = render_report_markdown(report, use_color=False)

        assert "RAND Corporation" in output
        assert "OpenAI" in output

    def test_includes_source_links(self, sample_extractions, sample_synthesis):
        report = synthesize_report(sample_extractions, "Test", sample_synthesis)
        output = render_report_markdown(report, use_color=False)

        assert "https://rand.org" in output
        assert "https://openai.com" in output

    def test_color_mode_adds_styling(self, sample_extractions, sample_synthesis):
        report = synthesize_report(sample_extractions, "Test", sample_synthesis)
        output = render_report_markdown(report, use_color=True)

        assert "style=" in output
        assert "#dcfce7" in output  # Green background for verified

    def test_includes_stats_footer(self, sample_extractions, sample_synthesis):
        report = synthesize_report(sample_extractions, "Test", sample_synthesis)
        output = render_report_markdown(report, use_color=False)

        assert "Verified facts:" in output
        assert "Synthesized sections:" in output


class TestRenderReportPlain:
    """Tests for plain text rendering."""

    def test_includes_title(self, sample_extractions, sample_synthesis):
        report = synthesize_report(sample_extractions, "Plain Test", sample_synthesis)
        output = render_report_plain(report)

        assert "Plain Test" in output

    def test_marks_verified_blocks(self, sample_extractions, sample_synthesis):
        report = synthesize_report(sample_extractions, "Test", sample_synthesis)
        output = render_report_plain(report)

        assert "[VERIFIED]" in output

    def test_marks_intro_and_conclusion(self, sample_extractions, sample_synthesis):
        report = synthesize_report(sample_extractions, "Test", sample_synthesis)
        output = render_report_plain(report)

        assert "[INTRO]" in output
        assert "[CONCLUSION]" in output


class TestSynthesisBlock:
    """Tests for SynthesisBlock dataclass."""

    def test_verified_block(self):
        block = SynthesisBlock(
            type="verified",
            content="Test content",
            source_url="https://example.com",
            context="Test context"
        )

        assert block.type == "verified"
        assert block.source_url == "https://example.com"

    def test_transition_block(self):
        block = SynthesisBlock(
            type="transition",
            content="This leads to..."
        )

        assert block.type == "transition"
        assert block.source_url is None


class TestSynthesizedReport:
    """Tests for SynthesizedReport dataclass."""

    def test_empty_report(self):
        report = SynthesizedReport(title="Empty")

        assert report.verified_count == 0
        assert report.synthesis_count == 0
        assert len(report.blocks) == 0

    def test_counts_block_types(self):
        report = SynthesizedReport(
            title="Test",
            blocks=[
                SynthesisBlock(type="verified", content="Fact 1"),
                SynthesisBlock(type="verified", content="Fact 2"),
                SynthesisBlock(type="transition", content="And then"),
                SynthesisBlock(type="intro", content="Intro"),
                SynthesisBlock(type="conclusion", content="End"),
            ]
        )

        assert report.verified_count == 2
        assert report.synthesis_count == 3  # transition + intro + conclusion
