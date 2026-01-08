"""Unit tests for claim_gate.py.

Tests for:
- extract_key_terms: Key term extraction from claims
- verify_claim_in_sources: Claim verification against sources
"""
import pytest

from open_deep_research.nodes.claim_gate import extract_key_terms, verify_claim_in_sources


class TestExtractKeyTerms:
    """Tests for extract_key_terms function."""

    def test_extracts_acronyms(self):
        """Should extract standard acronyms."""
        terms = extract_key_terms("The AEF-1 standard and LITHOS system")
        assert "AEF-1" in terms
        assert "LITHOS" in terms

    def test_extracts_single_letter_codes(self):
        """Should extract single-letter codes like A-1."""
        terms = extract_key_terms("Form A-1 and Schedule B-2 are required")
        assert "A-1" in terms
        assert "B-2" in terms

    def test_extracts_complex_codes(self):
        """Should extract complex hyphenated codes."""
        terms = extract_key_terms("NIST-SP-800-53 and M-25-22 compliance")
        assert "NIST-SP-800-53" in terms or "NIST" in terms
        assert "M-25-22" in terms

    def test_extracts_proper_nouns(self):
        """Should extract multi-word proper nouns."""
        # Note: Sentence-initial "The" gets included as part of proper noun
        terms = extract_key_terms("The Digital Trust Centre issued guidance")
        # Check that some form of the proper noun is captured
        assert any("Digital Trust Centre" in t for t in terms)

    def test_extracts_quoted_strings(self):
        """Should extract quoted strings."""
        terms = extract_key_terms('The report called it "highly effective"')
        assert "highly effective" in terms

    def test_extracts_percentages_symbol(self):
        """Should extract percentages with % symbol."""
        terms = extract_key_terms("Accuracy improved by 40.5%")
        assert "40.5%" in terms

    def test_extracts_percentages_word(self):
        """Should extract 'X percent' format."""
        terms = extract_key_terms("Reduced errors by 50 percent")
        assert "50 percent" in terms

    def test_extracts_dollar_amounts_simple(self):
        """Should extract simple dollar amounts."""
        terms = extract_key_terms("Budget of $5,000 was allocated")
        assert "$5,000" in terms

    def test_extracts_dollar_amounts_with_suffix(self):
        """Should extract dollar amounts with B/M/T suffix."""
        terms = extract_key_terms("Revenue reached $1.2B and costs were $25T")
        assert "$1.2B" in terms or "$1.2b" in terms
        assert "$25T" in terms or "$25t" in terms

    def test_extracts_dollar_amounts_spelled_out(self):
        """Should extract 'X billion/million' format."""
        terms = extract_key_terms("Investment of $5 billion announced")
        assert "$5 billion" in terms or "$5" in terms

    def test_vague_claim_returns_empty(self):
        """Vague claims without specific terms should return empty list."""
        terms = extract_key_terms("Research shows this is important")
        # No acronyms, no proper nouns, no numbers
        assert terms == []

    def test_mixed_claim(self):
        """Should extract multiple types of terms."""
        terms = extract_key_terms('The NIST report shows 40% improvement worth $1.2B')
        assert any("NIST" in t for t in terms)
        assert any("40%" in t for t in terms)
        assert any("$1.2B" in t or "$1.2b" in t for t in terms)


class TestVerifyClaimInSources:
    """Tests for verify_claim_in_sources function."""

    def test_verifies_when_terms_found(self):
        """Should return True when key terms are in sources."""
        claim = "NIST-SP-800-53 compliance achieved"
        key_terms = ["NIST-SP-800-53"]
        sources = [{"content": "We implemented NIST-SP-800-53 controls"}]

        is_verified, reason = verify_claim_in_sources(claim, key_terms, sources)
        assert is_verified is True
        assert "found: 1/1" in reason

    def test_fails_when_terms_missing(self):
        """Should return False when key terms not in sources."""
        claim = "FAKE-123 was implemented"
        key_terms = ["FAKE-123"]
        sources = [{"content": "We implemented something else entirely"}]

        is_verified, reason = verify_claim_in_sources(claim, key_terms, sources)
        assert is_verified is False
        assert "missing" in reason

    def test_passes_with_partial_match(self):
        """Should pass when more than half of terms are found."""
        claim = "ABC and DEF and GHI mentioned"
        key_terms = ["ABC", "DEF", "GHI"]
        sources = [{"content": "The ABC and DEF standards apply"}]

        is_verified, reason = verify_claim_in_sources(claim, key_terms, sources)
        assert is_verified is True
        assert "found: 2/3" in reason

    def test_fails_with_majority_missing(self):
        """Should fail when more than half of terms are missing."""
        claim = "ABC and DEF and GHI mentioned"
        key_terms = ["ABC", "DEF", "GHI"]
        sources = [{"content": "Only ABC is mentioned here"}]

        is_verified, reason = verify_claim_in_sources(claim, key_terms, sources)
        assert is_verified is False
        assert "missing" in reason

    def test_empty_key_terms_returns_true(self):
        """Empty key terms should return True (defensive case)."""
        is_verified, reason = verify_claim_in_sources("claim", [], [])
        assert is_verified is True
        assert reason == "no_key_terms"

    def test_case_insensitive_matching(self):
        """Matching should be case-insensitive."""
        key_terms = ["NIST"]
        sources = [{"content": "The nist guidelines specify..."}]

        is_verified, reason = verify_claim_in_sources("NIST compliance", key_terms, sources)
        assert is_verified is True


class TestVagueClaimsTracking:
    """Tests for tracking vague claims that can't be verified."""

    def test_vague_claim_no_key_terms(self):
        """Vague claims should extract no key terms."""
        vague_claims = [
            "Research shows this is important",
            "Many experts believe this works",
            "Studies indicate positive results",
            "The evidence suggests improvement",
        ]

        for claim in vague_claims:
            terms = extract_key_terms(claim)
            assert terms == [], f"Expected no terms for: {claim}, got: {terms}"

    def test_specific_claim_has_key_terms(self):
        """Specific claims should extract key terms."""
        specific_claims = [
            ("The GDPR requires consent", ["GDPR"]),
            ("Form A-1 must be filed", ["A-1"]),
            ("Accuracy of 95% achieved", ["95%"]),
            ("Budget of $1.2B approved", ["$1.2B"]),
        ]

        for claim, expected in specific_claims:
            terms = extract_key_terms(claim)
            for exp in expected:
                assert any(exp in t or exp.lower() in t.lower() for t in terms), \
                    f"Expected {exp} in terms for: {claim}, got: {terms}"
