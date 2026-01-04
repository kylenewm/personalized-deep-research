# Invariants (V0)

These are contracts the build must obey. V0 uses them primarily for human review.
Automated enforcement can be added later.

## I1: Verified Findings Integrity
**Contract:** Verified findings MUST include a verbatim quote and a source URL. Verified Findings also MUST be derived only from evidence_snippets with status == PASS (or show disabled message).
**Enforcement:** The final report generation prompt must use a "Selector" pattern that only allows selecting from the provided `PASS` evidence snippets. It must not invent or paraphrase quotes in the Verified section.

## I2: Deterministic Verification
**Contract:** Verification of quotes must be deterministic and code-based, NOT LLM-based.
**Enforcement:** `verify_evidence` node must use substring matching (strict) or Jaccard similarity (fuzzy > 0.8) against the stored raw text.

## I3: Canonical Store
**Contract:** The LangGraph Store (namespaced by `thread_id`) is the single source of truth for raw content.
**Enforcement:**
- Search tools WRITE raw content to `(thread_id, "raw")`.
- Extraction nodes READ from `(thread_id, "raw")`.
- Raw content is never passed in the main graph state (too large).

## I4: Fail-Safe Gating
**Contract:** If the Store is unavailable or unconfigured, the Verified section must be explicitly disabled. check_store also must set both verified_disabled and verified_disabled_reason; report must display the reason when disabled.
**Enforcement:** `check_store` node runs at the start of the graph. If it fails, `verified_disabled` is set to `True`, and downstream nodes (extract/verify) short-circuit.

## I5: CLI Scope
**Contract:** V0 is a CLI tool.
**Enforcement:** No web server, no complex authentication flows, no multi-user database requirements beyond local SQLite.

## I6: No Secrets in Repo
**Contract:** No API keys, tokens, or credentials may be committed to the repository.
**Allowed:**
- `.env.example` with placeholder values only
- environment variables provided at runtime

## I7: No Sweeping Refactors
**Contract:** Changes must be scoped to the current task. Do not refactor unrelated code.
**Allowed:**
- Fixing immediate dependencies of the change
- Renaming if explicitly required by the task
**Not Allowed:**
- "While I'm here" cleanups
- Restructuring code not mentioned in the task

## I8: Deterministic and Scoped Writes
**Contract:** All file writes must be deterministic and scoped to the current step.
**Rules:**
- Touch only files required by the task
- Prefer editing existing files over creating new ones
- Generated outputs must be reproducible given the same inputs
- Large lists must not grow unbounded via reducers; evidence_snippets uses replace/override semantics
