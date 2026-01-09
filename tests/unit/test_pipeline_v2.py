"""Unit tests for pipeline_v2 module.

Uses mock data only - no API calls.
"""

import sys
import importlib.util
from pathlib import Path

import pytest

# Direct module loading to avoid package __init__.py dependencies
src_dir = Path(__file__).parent.parent.parent / "src" / "open_deep_research"

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load pointer_extract first (no dependencies)
pointer_extract = load_module("pointer_extract", src_dir / "pointer_extract.py")
Pointer = pointer_extract.Pointer
Extraction = pointer_extract.Extraction

# Patch it into sys.modules so pipeline_v2 can import it
sys.modules["open_deep_research.pointer_extract"] = pointer_extract

# Load pipeline_v2
pipeline_v2 = load_module("pipeline_v2", src_dir / "pipeline_v2.py")
ThemeGroup = pipeline_v2.ThemeGroup
CuratedFacts = pipeline_v2.CuratedFacts
ThemedSection = pipeline_v2.ThemedSection
HybridReport = pipeline_v2.HybridReport
batch_sources = pipeline_v2.batch_sources
format_facts_for_arranger = pipeline_v2.format_facts_for_arranger
parse_arranger_response = pipeline_v2.parse_arranger_response
format_theme_facts = pipeline_v2.format_theme_facts
render_hybrid_report = pipeline_v2.render_hybrid_report


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_sources():
    """Sample sources for testing."""
    return {
        f"src_{i:03d}": {
            "content": f"This is content for source {i}. It contains important facts.",
            "url": f"https://example.com/source/{i}",
            "title": f"Source {i} Title"
        }
        for i in range(50)
    }


@pytest.fixture
def sample_verified_extractions():
    """Sample verified extractions."""
    return [
        Extraction(
            pointer=Pointer("src_001", ["RAND", "security"], "RAND security report"),
            status="verified",
            extracted_text="RAND released a security report recommending multi-layered approaches.",
            match_score=1.0,
            source_url="https://rand.org/report"
        ),
        Extraction(
            pointer=Pointer("src_002", ["OpenAI", "safety"], "OpenAI initiatives"),
            status="verified",
            extracted_text="OpenAI announced the Frontier Risk Council for safe deployment.",
            match_score=0.95,
            source_url="https://openai.com/safety"
        ),
        Extraction(
            pointer=Pointer("src_003", ["EU", "regulation"], "EU AI Act"),
            status="verified",
            extracted_text="The EU AI Act became enforceable in February 2025.",
            match_score=0.9,
            source_url="https://eu.gov/ai-act"
        ),
        Extraction(
            pointer=Pointer("src_004", ["benchmark", "evaluation"], "Safety benchmarks"),
            status="verified",
            extracted_text="New safety benchmarks were released for evaluating frontier models.",
            match_score=0.88,
            source_url="https://safety.org/benchmarks"
        ),
        Extraction(
            pointer=Pointer("src_005", ["Google", "DeepMind"], "Google safety"),
            status="verified",
            extracted_text="Google DeepMind published research on scalable oversight techniques.",
            match_score=0.92,
            source_url="https://deepmind.com/safety"
        ),
    ]


@pytest.fixture
def sample_themed_sections(sample_verified_extractions):
    """Sample themed sections."""
    return [
        ThemedSection(
            theme="Governance & Regulation",
            intro="Several regulatory developments shaped AI governance in 2025.",
            facts=[sample_verified_extractions[2]],
            transitions=["Among the key developments,"]
        ),
        ThemedSection(
            theme="Industry Initiatives",
            intro="Major AI labs announced safety initiatives.",
            facts=[sample_verified_extractions[0], sample_verified_extractions[1], sample_verified_extractions[4]],
            transitions=["First,", "Building on this,", "Additionally,"]
        ),
        ThemedSection(
            theme="Research & Evaluation",
            intro="Evaluation methods advanced significantly.",
            facts=[sample_verified_extractions[3]],
            transitions=["In the evaluation space,"]
        ),
    ]


# =============================================================================
# Test: batch_sources
# =============================================================================

class TestBatchSources:
    """Tests for source batching."""

    def test_single_batch(self):
        sources = {f"src_{i}": {"content": f"Content {i}"} for i in range(5)}
        batches = batch_sources(sources, batch_size=10)
        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_multiple_batches(self):
        sources = {f"src_{i}": {"content": f"Content {i}"} for i in range(50)}
        batches = batch_sources(sources, batch_size=20)
        assert len(batches) == 3
        assert len(batches[0]) == 20
        assert len(batches[1]) == 20
        assert len(batches[2]) == 10

    def test_exact_batch_boundary(self):
        sources = {f"src_{i}": {"content": f"Content {i}"} for i in range(40)}
        batches = batch_sources(sources, batch_size=20)
        assert len(batches) == 2
        assert len(batches[0]) == 20
        assert len(batches[1]) == 20

    def test_empty_sources(self):
        batches = batch_sources({}, batch_size=20)
        assert len(batches) == 0


# =============================================================================
# Test: format_facts_for_arranger
# =============================================================================

class TestFormatFactsForArranger:
    """Tests for arranger formatting."""

    def test_formats_verified_facts(self, sample_verified_extractions):
        output = format_facts_for_arranger(sample_verified_extractions)
        assert "[1]" in output
        assert "[2]" in output
        assert "RAND" in output
        assert "OpenAI" in output

    def test_includes_source_context(self, sample_verified_extractions):
        output = format_facts_for_arranger(sample_verified_extractions)
        assert "RAND security report" in output
        assert "OpenAI initiatives" in output

    def test_truncates_long_facts(self):
        long_extraction = Extraction(
            pointer=Pointer("src_001", ["test"], "Test"),
            status="verified",
            extracted_text="A" * 500,  # Very long text
            match_score=1.0
        )
        output = format_facts_for_arranger([long_extraction])
        assert "..." in output
        assert len(output) < 600  # Should be truncated

    def test_skips_non_verified(self):
        extractions = [
            Extraction(
                pointer=Pointer("src_001", ["test"], "Test"),
                status="not_found",
                match_score=0.0
            )
        ]
        output = format_facts_for_arranger(extractions)
        assert "[1]" not in output


# =============================================================================
# Test: parse_arranger_response
# =============================================================================

class TestParseArrangerResponse:
    """Tests for parsing arranger LLM response."""

    def test_valid_json(self, sample_verified_extractions):
        response = '''{
            "groups": [
                {"theme": "Governance", "fact_ids": [1, 3]},
                {"theme": "Industry", "fact_ids": [2, 4, 5]}
            ],
            "excluded": [6, 7]
        }'''
        result = parse_arranger_response(response, sample_verified_extractions)
        assert len(result.groups) == 2
        assert result.groups[0].theme == "Governance"
        assert result.groups[0].fact_ids == [1, 3]
        assert result.excluded_ids == [6, 7]

    def test_json_with_surrounding_text(self, sample_verified_extractions):
        response = '''Here is the organization:
        {"groups": [{"theme": "Test", "fact_ids": [1, 2]}], "excluded": []}
        Done.'''
        result = parse_arranger_response(response, sample_verified_extractions)
        assert len(result.groups) == 1
        assert result.groups[0].theme == "Test"

    def test_excluded_as_objects(self, sample_verified_extractions):
        response = '''{
            "groups": [{"theme": "All", "fact_ids": [1, 2, 3]}],
            "excluded": [{"id": 4, "reason": "redundant"}, {"id": 5, "reason": "off-topic"}]
        }'''
        result = parse_arranger_response(response, sample_verified_extractions)
        assert result.excluded_ids == [4, 5]

    def test_invalid_json_fallback(self, sample_verified_extractions):
        response = "This is not valid JSON at all"
        result = parse_arranger_response(response, sample_verified_extractions)
        # Should fallback to single group with all facts
        assert len(result.groups) == 1
        assert result.groups[0].theme == "Key Findings"
        assert result.groups[0].fact_ids == [1, 2, 3, 4, 5]

    def test_empty_groups(self, sample_verified_extractions):
        response = '{"groups": [], "excluded": []}'
        result = parse_arranger_response(response, sample_verified_extractions)
        assert len(result.groups) == 0


# =============================================================================
# Test: format_theme_facts
# =============================================================================

class TestFormatThemeFacts:
    """Tests for formatting facts within a theme."""

    def test_formats_selected_facts(self, sample_verified_extractions):
        output = format_theme_facts([], [1, 3], sample_verified_extractions)
        assert "FACT 1 (ID 1)" in output
        assert "FACT 2 (ID 3)" in output
        assert "RAND" in output
        assert "EU AI Act" in output

    def test_handles_invalid_ids(self, sample_verified_extractions):
        output = format_theme_facts([], [1, 100], sample_verified_extractions)
        assert "FACT 1 (ID 1)" in output
        # ID 100 doesn't exist, should be skipped

    def test_preserves_order(self, sample_verified_extractions):
        output = format_theme_facts([], [3, 1, 2], sample_verified_extractions)
        # Should maintain the order given
        eu_pos = output.find("EU AI Act")
        rand_pos = output.find("RAND")
        openai_pos = output.find("OpenAI")
        assert eu_pos < rand_pos < openai_pos


# =============================================================================
# Test: HybridReport
# =============================================================================

class TestHybridReport:
    """Tests for HybridReport dataclass."""

    def test_verified_count(self, sample_themed_sections):
        report = HybridReport(
            title="Test Report",
            executive_summary="Summary",
            sections=sample_themed_sections,
            analysis="Analysis",
            conclusion="Conclusion"
        )
        # 1 + 3 + 1 = 5 facts
        assert report.verified_count == 5

    def test_empty_report(self):
        report = HybridReport(
            title="Empty",
            executive_summary="",
            sections=[],
            analysis="",
            conclusion=""
        )
        assert report.verified_count == 0


# =============================================================================
# Test: render_hybrid_report
# =============================================================================

class TestRenderHybridReport:
    """Tests for markdown rendering."""

    def test_includes_title(self, sample_themed_sections):
        report = HybridReport(
            title="AI Safety Report 2025",
            executive_summary="This is the summary.",
            sections=sample_themed_sections,
            analysis="This is the analysis.",
            conclusion="This is the conclusion."
        )
        output = render_hybrid_report(report, use_color=False)
        assert "# AI Safety Report 2025" in output

    def test_includes_executive_summary(self, sample_themed_sections):
        report = HybridReport(
            title="Test",
            executive_summary="Executive summary here.",
            sections=sample_themed_sections,
            analysis="Analysis",
            conclusion="Conclusion"
        )
        output = render_hybrid_report(report, use_color=False)
        assert "## Executive Summary" in output
        assert "Executive summary here" in output

    def test_includes_themed_sections(self, sample_themed_sections):
        report = HybridReport(
            title="Test",
            executive_summary="Summary",
            sections=sample_themed_sections,
            analysis="Analysis",
            conclusion="Conclusion"
        )
        output = render_hybrid_report(report, use_color=False)
        assert "### Governance & Regulation" in output
        assert "### Industry Initiatives" in output
        assert "### Research & Evaluation" in output

    def test_includes_verified_facts(self, sample_themed_sections):
        report = HybridReport(
            title="Test",
            executive_summary="Summary",
            sections=sample_themed_sections,
            analysis="Analysis",
            conclusion="Conclusion"
        )
        output = render_hybrid_report(report, use_color=False)
        assert "EU AI Act" in output
        assert "RAND" in output
        assert "OpenAI" in output

    def test_includes_analysis_section(self, sample_themed_sections):
        report = HybridReport(
            title="Test",
            executive_summary="Summary",
            sections=sample_themed_sections,
            analysis="This is the detailed analysis.",
            conclusion="Conclusion"
        )
        output = render_hybrid_report(report, use_color=False)
        assert "## Analysis & Implications" in output
        assert "detailed analysis" in output

    def test_includes_conclusion(self, sample_themed_sections):
        report = HybridReport(
            title="Test",
            executive_summary="Summary",
            sections=sample_themed_sections,
            analysis="Analysis",
            conclusion="Final conclusion here."
        )
        output = render_hybrid_report(report, use_color=False)
        assert "## Conclusion" in output
        assert "Final conclusion" in output

    def test_includes_stats_footer(self, sample_themed_sections):
        report = HybridReport(
            title="Test",
            executive_summary="Summary",
            sections=sample_themed_sections,
            analysis="Analysis",
            conclusion="Conclusion",
            total_extracted=100,
            total_verified=70,
            total_used=50
        )
        output = render_hybrid_report(report, use_color=False)
        assert "Sources processed: 100" in output
        assert "Verified facts: 70" in output
        assert "Facts in report: 50" in output

    def test_color_mode_styling(self, sample_themed_sections):
        report = HybridReport(
            title="Test",
            executive_summary="Summary",
            sections=sample_themed_sections,
            analysis="Analysis",
            conclusion="Conclusion"
        )
        output = render_hybrid_report(report, use_color=True)
        assert "style=" in output
        assert "#dcfce7" in output  # Green for verified
        assert "#6b7280" in output  # Gray for synthesis


# =============================================================================
# Test: ThemeGroup
# =============================================================================

class TestThemeGroup:
    """Tests for ThemeGroup dataclass."""

    def test_basic_group(self):
        group = ThemeGroup(
            theme="Governance",
            fact_ids=[1, 2, 3]
        )
        assert group.theme == "Governance"
        assert len(group.fact_ids) == 3

    def test_with_intro(self):
        group = ThemeGroup(
            theme="Industry",
            fact_ids=[4, 5],
            intro="This section covers industry developments."
        )
        assert group.intro == "This section covers industry developments."


# =============================================================================
# Test: CuratedFacts
# =============================================================================

class TestCuratedFacts:
    """Tests for CuratedFacts dataclass."""

    def test_basic_curation(self):
        curated = CuratedFacts(
            groups=[
                ThemeGroup("A", [1, 2]),
                ThemeGroup("B", [3, 4])
            ],
            excluded_ids=[5, 6]
        )
        assert len(curated.groups) == 2
        assert len(curated.excluded_ids) == 2

    def test_no_exclusions(self):
        curated = CuratedFacts(
            groups=[ThemeGroup("All", [1, 2, 3, 4, 5])],
            excluded_ids=[]
        )
        assert len(curated.excluded_ids) == 0


# =============================================================================
# Test: ThemedSection
# =============================================================================

class TestThemedSection:
    """Tests for ThemedSection dataclass."""

    def test_basic_section(self, sample_verified_extractions):
        section = ThemedSection(
            theme="Test Theme",
            intro="Introduction text",
            facts=sample_verified_extractions[:2],
            transitions=["First,", "Second,"]
        )
        assert section.theme == "Test Theme"
        assert len(section.facts) == 2
        assert len(section.transitions) == 2

    def test_empty_transitions(self, sample_verified_extractions):
        section = ThemedSection(
            theme="Test",
            intro="Intro",
            facts=sample_verified_extractions[:1],
            transitions=[]
        )
        assert len(section.transitions) == 0
