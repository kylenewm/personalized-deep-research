# Deep Research Agent

AI-powered research agent that searches the web, gathers sources, and generates verified research reports.

## What it does

1. Takes a research query
2. Searches the web using Tavily
3. Extracts and analyzes content from sources
4. Verifies claims against source material
5. Generates a research report with citations

## Setup

```bash
# Clone and install
git clone https://github.com/kylenewm/personalized-deep-research.git
cd personalized-deep-research

python -m venv venv
source venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm

# Configure API keys
cp .env.example .env
# Add your keys to .env
```

**Required API keys:**
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

## Usage

```bash
# Run a research query
python scripts/test_e2e_quick.py "What is quantum computing?"

# Test verification logic
python scripts/mock_verification_test.py
```

## Project Structure

```
src/open_deep_research/
├── graph.py           # Main workflow
├── configuration.py   # Settings
├── verification.py    # Claim verification
└── nodes/             # Pipeline stages
    ├── brief.py       # Research planning
    ├── supervisor.py  # Coordinates researchers
    ├── researcher.py  # Web research
    ├── report.py      # Report generation
    └── verify.py      # Quote verification
```
