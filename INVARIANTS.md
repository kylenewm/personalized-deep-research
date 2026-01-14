# INVARIANTS.md

> Contracts the system must obey. Append-only — never delete or weaken existing invariants.

---

## Current Invariants

### I1: Verified Findings Integrity

**Contract:** Verified findings MUST include a verbatim quote and a source URL. Verified Findings also MUST be derived only from evidence_snippets with status == PASS (or show disabled message).

**Enforcement:** The final report generation prompt must use a "Selector" pattern that only allows selecting from the provided `PASS` evidence snippets. It must not invent or paraphrase quotes in the Verified section.

### I2: Deterministic Quote Verification

**Contract:** Verification of **quotes** (S04) must be deterministic and code-based, NOT LLM-based.

**Scope:** This applies to `verify_evidence` node (S04) which checks extracted quotes against source content. It does NOT apply to upstream claim extraction (S03), which may use LLM for best-effort identification.

**Enforcement:** `verify_evidence` node must use substring matching (strict) or Jaccard similarity (fuzzy > 0.8) against the stored raw text.

### I3: Canonical Store

**Contract:** The LangGraph Store (namespaced by `thread_id`) is the single source of truth for raw content.

**Enforcement:**
- Search tools WRITE raw content to `(thread_id, "raw")`.
- Extraction nodes READ from `(thread_id, "raw")`.
- Raw content is never passed in the main graph state (too large).

### I4: Fail-Safe Gating

**Contract:** If the Store is unavailable or unconfigured, the Verified section must be explicitly disabled. check_store also must set both verified_disabled and verified_disabled_reason; report must display the reason when disabled.

**Enforcement:** `check_store` node runs at the start of the graph. If it fails, `verified_disabled` is set to `True`, and downstream nodes (extract/verify) short-circuit.

### I5: CLI Scope

**Contract:** V0 is a CLI tool.

**Enforcement:** No web server, no complex authentication flows, no multi-user database requirements beyond local SQLite.

### I6: No Secrets in Repo

**Contract:** No API keys, tokens, or credentials may be committed to the repository.

**Allowed:**
- `.env.example` with placeholder values only
- environment variables provided at runtime

### I7: No Sweeping Refactors

**Contract:** Changes must be scoped to the current task. Do not refactor unrelated code.

**Allowed:**
- Fixing immediate dependencies of the change
- Renaming if explicitly required by the task

**Not Allowed:**
- "While I'm here" cleanups
- Restructuring code not mentioned in the task

### I8: Deterministic and Scoped Writes

**Contract:** All file writes must be deterministic and scoped to the current step.

**Rules:**
- Touch only files required by the task
- Prefer editing existing files over creating new ones
- Generated outputs must be reproducible given the same inputs
- Large lists must not grow unbounded via reducers; evidence_snippets uses replace/override semantics

---

## History

### 2026-01-07 — Initial migration

- Migrated I1-I8 from `docs/archive/invariants_v0.md` to repo root
- Established append-only invariants file
- Original file preserved in archive for reference

### 2026-01-07 — Violation audit

**Violations discovered during trust audit:**

| Invariant | Status | Evidence |
|-----------|--------|----------|
| I1 | VIOLATED | Diversity not enforced in extract.py:135; verified section can be modified post-generation |
| I2 | CLARIFIED | claim_gate.py:141 LLM usage is NOT an I2 violation (I2 covers quote verification, not claim extraction). Tokenization in verify.py:24-35 still needs fix. |
| I3 | OK | Analyzed: dual storage is intentional architecture, not a bug (see LOG.md) |
| I4 | VIOLATED | verify.py:144 returns `{}` on disabled, leaves snippets in PENDING state |

**Action required:** Fix I1, I2 tokenization, and I4 violations.

### 2026-01-13 — Phase 8: Wire Up Dead Code

Added invariants for infrastructure that was written but not wired up:

### I9: Span Verification Required

**Contract:** All extractions with status="verified" MUST pass `verify_span()` check before being used in reports.

**Enforcement:**
- `extract_batch()` and `extract_from_source_chunked()` must call `verify_span()` after extraction
- Extractions failing span verification must be downgraded to status="span_mismatch"
- Reports must not include extractions with span_mismatch status in "Verified" sections

**Rationale:** Span offsets enable deterministic reverification. Without calling verify_span(), the spans are untested and could be invalid after text cleaning.

### I10: Run Artifacts Persistence

**Contract:** Every pipeline run SHOULD produce a saved artifact file when `artifacts_dir` is provided, containing:
- `run_id` and `timestamp`
- Source content hashes (for reproducibility)
- Prompt version hashes (for regression attribution)
- Final report hash

**Enforcement:**
- `run_pipeline_v2()` must call `save_run_artifacts()` when `artifacts_dir` is provided
- Artifact files must be JSON-serializable and loadable via `load_run_artifacts()`
- Tests must verify artifact files are created and contain required fields

**Rationale:** Without persisted artifacts, runs cannot be replayed or compared for regressions.

### I11: Checkpoint Persistence

**Contract:** Pipeline checkpoints SHOULD be persisted to disk when `checkpoint_dir` is provided, not just attached to in-memory HybridReport.

**Enforcement:**
- `run_pipeline_v2()` must save checkpoints to JSON file when `checkpoint_dir` is provided
- Checkpoint files must contain: `pre_dedup`, `post_dedup`, `pre_arrangement`, `post_arrangement`
- Tests must verify checkpoint files are created with all required keys

**Rationale:** In-memory checkpoints are lost when the process ends. Persisted checkpoints enable post-hoc analysis and fixture extraction.
