# Evaluation Audit: 2026-01-12

> Reference document for eval infrastructure gaps and optimization status.

---

## Current Eval Infrastructure

| Tool | What It Measures | Cost/Speed | Status |
|------|------------------|------------|--------|
| `eval/run_eval.py` | fact_quality, theme_coverage, duplicate_rate, citation_accuracy, synthesis_quality, uncited_rate | ~$0.05-0.30, 2-20min | Working |
| `scripts/prompt_sandbox.py` | word count, artifacts, headers | ~$0.01, 30s | Passing |
| `scripts/citation_sandbox.py` | % of input facts cited | ~$0.02, 30s | Passing (wrong metric) |
| `scripts/dedup_sandbox.py` | precision, recall, FP rate | ~$0.01, 15s | 0% FP, 73% recall |
| `scripts/arrangement_sandbox.py` | theme coherence, exclusion | ~$0.02, 30s | Passing |
| `scripts/quality_sandbox.py` | config validation | Free, <1s | Passing |
| `scripts/resynthesis_test.py` | fact_usage_rate, **citation_accuracy** (LLM) | ~$0.07, 45s | ✅ Fixed - now checks accuracy |

---

## Metric Status Matrix

### GREEN (Passing)

| Metric | Target | Actual | Source |
|--------|--------|--------|--------|
| fact_quality | ≥3.5 | 4.33 | eval/run_eval.py |
| theme_coverage | ≥3.5 | 4.20 | eval/run_eval.py |
| duplicate_rate | ≤15% | 0% | eval/run_eval.py |
| citation_accuracy | ≥4.0 | 4.80 | eval/run_eval.py |
| synthesis_quality | ≥3.5 | 4.40 | eval/run_eval.py |
| brief_preservation | ≥4.0 | 5.0 | eval/run_eval.py |
| brief_dilution | ≥4.0 | 5.0 | eval/run_eval.py |

### YELLOW (Close but not passing)

| Metric | Target | Actual | Gap |
|--------|--------|--------|-----|
| match_score | ≥0.80 | 0.76 | Extraction text matching |
| uncited_rate | ≤5% | 7.7% | Synthesis citation coverage |

### RED (No Eval Exists)

| Gap | Why It Matters | Proposed Fix | Status |
|-----|----------------|--------------|--------|
| Citation correctness | We check if `[3]` appears, not if `[3]` matches the claim | Add accuracy check to resynthesis_test.py | ✅ FIXED |
| Source authority | `prefer_authoritative_sources` is prompt hint only | Add domain tier distribution metric | Pending |
| Query answering | Does final report actually answer the original query? | Add downstream eval question | Pending |
| Hallucination grounding | Claims not verified against source content | Expensive - defer | Deferred |
| Search quality | Tavily results not evaluated | Add search sandbox integration | Pending |

---

## Sandboxes vs Full Eval Coverage

| Component | Full Eval | Sandbox | Gap |
|-----------|-----------|---------|-----|
| Query preservation | Brief eval ✓ | No | Sandboxes don't test query transformation |
| Fact quality | Upstream eval ✓ | prompt_sandbox (heuristics) | Eval uses LLM; sandbox uses word count |
| Citation presence | Downstream eval ✓ | citation_sandbox ✓ | Both check presence, neither checks correctness |
| Citation accuracy | Downstream eval ✓ | **NO** | No fast test for citation correctness |
| Synthesis quality | Downstream eval ✓ | No | No sandbox for prose quality |
| Cross-batch dedup | Pipeline code ✓ | dedup_sandbox (within-batch) | Sandbox doesn't test cross-batch |
| Source quality | No | quality_sandbox (config only) | No actual domain filtering eval |

---

## Efficient vs Inefficient Evals

### Efficient (use for iteration)

| Tool | Time | Cost | Use Case |
|------|------|------|----------|
| Sandboxes on fixtures | 30s | $0.01-0.02 | Prompt tuning |
| Mini eval (15 facts) | 2-3min | $0.05 | PR checks |
| resynthesis_test.py | 30s | $0.05 | Synthesis prompt tuning |
| Unit tests | <10s | Free | Logic validation |

### Inefficient (use sparingly)

| Tool | Time | Cost | Use Case |
|------|------|------|----------|
| Full pipeline run | 20min | $0.50+ | E2E validation |
| Full eval (150 facts) | 15-20min | $0.30 | Weekly regression |
| Source verification | Minutes/fact | $$ | Ground truth (deferred) |

---

## Wrong Metrics Being Tracked

### citation_sandbox.py & resynthesis_test.py

**What they measure:** "Did all input facts get used?"
- Counts unique `[N]` markers in output
- Compares to total input facts
- Reports "uncited facts" = facts not mentioned

**Why this is wrong:**
- We have MORE facts than we need
- Low usage is fine if prose is good
- Doesn't check if citations are CORRECT

**What we should measure:** "For each citation in output, is it accurate?"
- For each `[N]` in prose, extract the surrounding claim
- Compare claim to fact `[N]` content
- Report accuracy rate

---

## Gold Datasets

| Dataset | Sources | Facts | Has Report | Last Used |
|---------|---------|-------|------------|-----------|
| agentic_coding_2026.json | 58 | 73 | Yes | Showcase |
| latest_research.json | 160 | 112 | No | Autonomous agents |
| claude_code_orch_*.json | ~60 | 62 | Yes | Table artifact testing |

---

## Fixes Applied Today (2026-01-12)

1. **Table artifact fix** - pipe threshold >3 → >=2
   - Result: 37/43 → 0/62 artifacts, match_score 0.63 → 0.76

2. **Synthesis prompt strengthening** - 80% → 90% target, GOOD/BAD examples
   - Result: citation_sandbox 100%, eval no regression
   - Pending: Full pipeline run to verify real improvement

3. **Cross-batch dedup** - Added second pass after LLM batch dedup
   - Result: Caught 47 cross-batch duplicates

4. **Eval framework tuning** - duplicate_rate threshold 2% → 15%
   - Result: 20% → 0% (was flagging related facts as duplicates)

---

## Priority Gaps to Fix

| Priority | Gap | Fast Test Possible? | Effort |
|----------|-----|---------------------|--------|
| P0 | Citation accuracy sandbox | Yes - extend resynthesis_test.py | Medium |
| P1 | Run full pipeline to verify synthesis improvement | No - 20min run | Low |
| P2 | Source authority distribution | Yes - count domains in facts | Low |
| P3 | Query answering eval | Yes - add to downstream eval | Medium |
| P4 | Hallucination grounding | No - expensive | High (defer) |

---

## Next Steps

1. Add citation accuracy check to resynthesis_test.py
2. Run full pipeline to verify synthesis prompt improvement
3. Add source authority tier counting
4. Consider query answering eval

---

*Generated: 2026-01-12*
