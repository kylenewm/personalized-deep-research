# Deep Research Trust Architecture (V0)

## 1. Trust Logic Data Flow

This diagram illustrates the "Verify-First" data pipeline. The key innovation is that **quotes are extracted deterministically from raw HTML**, not hallucinated by an LLM.

```mermaid
graph TD
    %% Nodes
    Store[(LangGraph Store)]
    HTML[Raw Source HTML]
    Sanitizer[HTML Sanitizer]
    Extractor[Evidence Extractor]
    Verifier[Substring Verifier]
    LLM[Report Generator]
    Report[Final Report]

    %% Flow
    Store -->|Fetch Raw Content| HTML
    HTML -->|Strip Tags, Keep Paras| Sanitizer
    Sanitizer -->|Clean Text| Extractor
    Extractor -->|Filter (15-60 words)| Candidates[Candidate Quotes]
    Candidates -->|Exact/Fuzzy Match| Verifier
    Verifier -->|Status: PASS/FAIL| VerifiedList[Verified Evidence List]
    
    VerifiedList -->|Input| LLM
    LLM -->|Select & Format (No Invention)| Report

    %% Styling
    style Store fill:#f9f,stroke:#333
    style Verifier fill:#bbf,stroke:#333
    style LLM fill:#bfb,stroke:#333
```

---

## 2. Detailed Implementation Outline

### A. Store Gating (`check_store`)
**Purpose:** Fail fast if we can't guarantee verification.
*   **Input:** `RunnableConfig` (checks for `store` and `thread_id`).
*   **Logic:**
    *   If `store` is None OR `thread_id` is missing:
        *   Set `state["verified_disabled"] = True`.
        *   Log warning: "Store unavailable; Verified section disabled."
    *   Else:
        *   Set `state["verified_disabled"] = False`.

### B. Evidence Extraction (`extract_evidence`)
**Purpose:** Mine raw ore (text) from the mountain (HTML).
*   **Input:** `state["source_store"]` (list of URLs/IDs).
*   **Process:**
    1.  Iterate through all sources.
    2.  `store.get((thread_id, "raw"), source_id)` to retrieve HTML.
    3.  `sanitize_for_quotes(html)`:
        *   Regex strip `<script>`, `<style>`, and all tags.
        *   **Crucial:** Replace block-level tags (`</div>`, `</p>`) with `\n\n` to preserve paragraph structure.
    4.  Split by `\n\n`.
    5.  Filter paragraphs:
        *   Length: 15 to 60 words.
        *   Content: Must contain at least one noun phrase or number (simple heuristic).
*   **Output:** `state["evidence_snippets"]` (list of dicts).

### C. Verification (`verify_evidence`)
**Purpose:** The "Audit" layer.
*   **Input:** `state["evidence_snippets"]`.
*   **Process:**
    1.  For each snippet:
        *   Reload raw HTML (or use cached clean text).
        *   **Check 1 (Strict):** `if snippet["quote"] in clean_text`. -> `PASS`.
        *   **Check 2 (Fuzzy):** Tokenize both into sets of words. Calculate Jaccard similarity. If > 0.8 -> `PASS`.
        *   Else -> `FAIL`.
*   **Output:** Updated `state["evidence_snippets"]` with status fields.

### D. Report Generation (Selector Mode)
**Purpose:** Turn data into a document without hallucinations.
*   **Prompt Adjustment:**
    *   Old: "Write a report using these notes."
    *   New: "Section 1: **Verified Findings**. Select 3-5 quotes from the `AVAILABLE_QUOTES` list. Copy them exactly. Format as: `* Claim - [Source Title](url) - "Quote"`."
*   **Constraint:** If `verified_disabled` is True, this section is skipped with a placeholder message.

---

## 3. System Invariants (V0)

These contracts guarantee the integrity of the refactor and the trust system.

### I1 — No Secrets in Repo
**Contract**
No API keys, tokens, or credentials may be committed to the repository.
**Allowed**
- `.env.example` with placeholder values only
- environment variables provided at runtime

### I2 — No Sweeping Refactors
**Contract**
Changes must be scoped to the `src/open_deep_research` directory and specifically the "Trust Logic" nodes.
**Not Allowed**
- Rewriting unrelated tools (e.g., standard `web_search`).
- Changing the `langgraph.json` configuration structure.

### I3 — Deterministic Verification
**Contract**
The `verify_evidence` node must be deterministic.
**Rules**
- It must not use an LLM.
- It must rely solely on string manipulation and math (Jaccard similarity).
- Running it twice on the same store data must yield the exact same `PASS/FAIL` results.

### I4 — Store Namespace Isolation
**Contract**
Raw content must be stored in a namespaced key structure to prevent collisions.
**Rule**
- Key format: `(thread_id, "raw")` -> `source_id`.
- This ensures concurrent runs (different threads) do not overwrite each other's raw source cache.

### I5 — Read-Only Source Extraction
**Contract**
The extraction process must never modify the stored raw content.
**Rule**
- `extract_evidence` reads from Store but never writes to it.
- Only the `tavily_search` tool is authorized to write to the "raw" namespace.

