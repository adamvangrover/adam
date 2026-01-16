# 🧠 Spec-Driven Agent Protocol (SDAP)

**Context:**
*   **Repository:** `adamvangrover/adam`
*   **Purpose:** A rigorous protocol for working with AI Agents to ensure high-quality, maintainable, and safe code generation.
*   **Philosophy:** **"Plan first in read-only mode, then execute and iterate continuously."**

---

## 📜 The 5 Core Principles

### 1. Start with Vision, Let AI Draft Details
Don't over-engineer upfront. Provide a high-level goal (the "Product Brief") and use the agent's "Plan Mode" (read-only) to draft the detailed specification.
*   *Action:* "Draft a detailed spec for [Goal] covering objectives, features, and constraints."
*   *Why:* Leverages AI's elaboration strength while keeping you in control.

### 2. Structure the Spec (PRD/SRS)
The spec is the "Source of Truth". It must be structured, not a stream of consciousness.
*   **Core Areas:**
    1.  **Commands:** Executable commands (`pytest`, `npm test`).
    2.  **Testing:** Test plan, file locations, coverage requirements.
    3.  **Structure:** Explicit file paths (`src/`, `tests/`).
    4.  **Style:** Coding conventions and snippets.
    5.  **Git:** Branching and commit message format.
    6.  **Boundaries:** Strict constraints.

### 3. Modular Context & Execution
Do not feed the entire project context into one prompt.
*   **Divide & Conquer:** Break the spec into independent modules (e.g., Database, API, Frontend).
*   **Spec Summaries:** Use an "Extended Table of Contents" to give the agent a map without the weight of the full text.
*   **One Task Focus:** "Implement Section 3.1" -> Verify -> "Implement Section 3.2".

### 4. Explicit Boundaries (The 3 Tiers)
Define clear rules for what the agent can and cannot do.
*   ✅ **Always:** Run tests before commits. Log errors. Follow style guides.
*   ⚠️ **Ask First:** Database schema changes. Adding dependencies. Modifying CI/CD.
*   🚫 **Never:** Commit secrets. Edit `node_modules/` or build artifacts. Remove failing tests.

### 5. Verification & Iteration Loop
The spec is a living document.
*   **Test-Driven:** Define success criteria (tests) in the spec.
*   **Self-Correction:** Force the agent to review its output against the spec.
*   **Update Loop:** If the plan changes, update the spec first, then the code.

---

## 📝 Master Spec Template (`SPEC.md`)

Copy this template to creating a new task specification.

```markdown
# Spec: [Project/Feature Name]

## 1. Overview & Objectives
*   **Goal:** [One sentence summary]
*   **User Story:** As a [Role], I want to [Action], so that [Benefit].
*   **Success Metrics:** [Measurable outcomes, e.g., "Latency < 200ms", "100% Test Pass"]

## 2. Technical Context
*   **Stack:** Python 3.10+, [Frameworks], [Libraries]
*   **Existing Components:** `core/engine/...`, `tests/...`
*   **Architecture:** [Brief description or diagram reference]

## 3. Implementation Plan
### Phase 1: Foundation
*   [ ] Task 1.1: [Description]
*   [ ] Task 1.2: [Description]

### Phase 2: Core Logic
*   [ ] Task 2.1: [Description]

### Phase 3: Integration & UI
*   [ ] Task 3.1: [Description]

## 4. Commands & Development
*   **Install:** `pip install -e .`
*   **Test:** `pytest tests/path/to/test.py`
*   **Lint:** `flake8 src/`
*   **Run:** `python scripts/run.py`

## 5. Verification & Testing Strategy
*   **Unit Tests:** Must cover all new utility functions.
*   **Integration Tests:** Verify end-to-end flow for [Feature].
*   **Conformance:** Output must match JSON Schema [Link].

## 6. Constraints & Boundaries
*   ✅ **Always:**
    *   Add docstrings to every function (Google Style).
    *   Type-hint every argument and return value.
    *   Handle `requests.exceptions.Timeout`.
*   🚫 **Never:**
    *   Hardcode API keys.
    *   Use `print()` for logging (use `logging.getLogger()`).
    *   Modify `core/critical_infrastructure/`.
```

---

## 🔄 The Workflow (Step-by-Step)

### Phase 1: Specify & Plan (Read-Only)
**Prompt:**
> "You are a Senior Architect. I want to build [Goal].
> 1. Explore the codebase (read-only) to understand existing patterns.
> 2. Draft a `SPEC.md` using the Standard Template.
> 3. Do not write any code yet. Just plan."

*   **Human Action:** Review `SPEC.md`. Correct assumptions. Refine boundaries. **Approve.**

### Phase 2: Modular Execution
**Prompt:**
> "We are executing **Phase 1: Foundation** from `SPEC.md`.
> 1. Create the file `src/module/base.py`.
> 2. Implement the `BaseClass` as defined in Section 3.
> 3. Adhere to Constraints in Section 6.
> 4. Verify by creating a test file `tests/test_base.py` and running it."

*   **Human Action:** Review code. Check test results. **Commit.**

### Phase 3: Verification & Review (LLM-as-a-Judge)
**Prompt:**
> "Review the file `src/module/base.py` against `SPEC.md`.
> *   Does it meet all objectives in Section 1?
> *   Are all Constraints in Section 6 followed?
> *   Are there potential edge cases missing?
> Output a strict 'PASS' or 'FAIL' with a list of remediation steps."

---

## 🛡️ Anti-Patterns to Avoid

*   **The "Mega-Prompt":** Dumping the entire codebase and 50 requirements into one message. *Result: Hallucinations.*
*   **Vague Instructions:** "Make it better" or "Fix the bug". *Result: Unpredictable changes.*
*   **Vibe Coding:** Skipping tests because "it looks right". *Result: "House of Cards" code.*
*   **Ignoring Context:** Failing to tell the agent about existing utilities or patterns. *Result: Duplicate code.*

---

## 🗂️ Advanced Tactics

### Spec Summaries (for Large Contexts)
If the spec is huge, give the agent a map:
> "Refer to `SPEC.md`. Summary:
> *   **§1 Auth:** JWT implementation details.
> *   **§2 DB:** PostgreSQL schema definitions.
> *   **§3 API:** REST endpoints for User resource.
> Read the full section only when working on that specific module."

### Self-Correction Hook
Append this to every coding prompt:
> "**After generating the code, list 3 ways this implementation verifies it meets the spec constraints.**"
