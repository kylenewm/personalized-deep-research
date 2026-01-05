"""Shared pytest fixtures and configuration for Deep Research tests."""

import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables for integration tests
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires API keys)",
    )
    parser.addoption(
        "--run-evaluation",
        action="store_true",
        default=False,
        help="Run evaluation tests (slow, generates reports)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "evaluation: mark test as evaluation test")


def pytest_collection_modifyitems(config, items):
    """Skip integration/evaluation tests unless explicitly requested."""
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="need --run-integration option")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    if not config.getoption("--run-evaluation"):
        skip_evaluation = pytest.mark.skip(reason="need --run-evaluation option")
        for item in items:
            if "evaluation" in item.keywords:
                item.add_marker(skip_evaluation)


@pytest.fixture
def sample_source():
    """Provide a sample source for testing."""
    return {
        "url": "https://example.com/rag-article",
        "title": "Understanding RAG Systems",
        "content": """Retrieval-Augmented Generation (RAG) is a technique that enhances large language models
        by retrieving relevant information from external knowledge bases. RAG reduces hallucinations by 40%
        in healthcare applications according to recent studies. The technique involves two main components:
        a retriever that finds relevant documents and a generator that produces responses based on those documents.""",
        "extraction_method": "extract_api",
        "timestamp": "2024-01-01T00:00:00"
    }


@pytest.fixture
def sample_sources():
    """Provide multiple sample sources for testing."""
    return [
        {
            "url": "https://example.com/rag-overview",
            "title": "RAG Overview",
            "content": "RAG combines retrieval and generation for better accuracy. It significantly reduces hallucination rates.",
            "extraction_method": "extract_api"
        },
        {
            "url": "https://example.com/llm-guide",
            "title": "LLM Best Practices",
            "content": "Large language models can benefit from external knowledge sources to improve factual accuracy.",
            "extraction_method": "search_raw"
        }
    ]


@pytest.fixture
def sample_snippet():
    """Provide a sample evidence snippet for testing."""
    return {
        "snippet_id": "abc123",
        "source_id": "https://example.com/rag-article",
        "url": "https://example.com/rag-article",
        "source_title": "Understanding RAG Systems",
        "quote": "RAG reduces hallucinations by 40% in healthcare applications",
        "status": "PENDING"
    }
