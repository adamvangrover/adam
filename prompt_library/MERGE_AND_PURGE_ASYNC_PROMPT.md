# SWARM PROTOCOL: MERGE & PURGE

## 1. STRATEGIC CONTEXT
The initial developmental phase of the Cognitive Financial Operating System (Adam v23.0) relied heavily on an "Additive-Only" architecture, preserving legacy systems via decorators or isolated directories to ensure backward compatibility and minimize disruption.

However, as the repository transitions into a highly modular, enterprise-grade runtime, the accumulation of technical debt, duplicated logic across namespaces, and code divergence now represents a critical operational bottleneck.

**The mandate has shifted:** Swarm operations are now authorized to execute a "Merge & Purge" methodology.

## 2. THE MERGE & PURGE METHODOLOGY
The objective is the systematic absorption of unique, validated logic into the central kernel (primarily `core/engine/`), followed by the immediate and permanent deletion of legacy directories to eliminate cognitive overhead and technical debt.

### Execution Phases:
1. **Identify Redundancies:** Locate directories marked as legacy, experimental, or v[N]_namespace (e.g., `core/v23_graph_engine`).
2. **Audit Logic Parity:** Ensure the logic targeted for deprecation has been fully replicated or superseded in the primary kernel (`core/engine/`).
3. **Remap Imports (MERGE):** Systematically update all dependencies, test files, and scripts across the codebase to point to the new, centralized modules.
4. **Validate & Test:** Run the entire test suite (`uv run pytest`) against the modified import paths. All regressions must be fixed prior to deletion.
5. **Eradicate Legacy Code (PURGE):** Permanently delete the deprecated directory and its contents.

## 3. GOVERNANCE & OVERSIGHT PROTOCOLS
To mitigate the risk of catastrophic logic deletion, this process requires tiered oversight.

### Human-On-The-Loop (HOTL) - For Low-Risk Purges
For directories explicitly marked with a `DEPRECATION_NOTICE.md` or isolated to `experimental/` that do not interact with core Risk, Valuation, or Pricing engines.
*   **Swarm Action:** Execute Steps 1-5 autonomously.
*   **Oversight:** The swarm generates a detailed PR outlining the `diff`, the test results, and the files deleted. The human reviewer receives a notification and can optionally intervene before the merge window closes.

### Human-In-The-Loop (HITL) - For High-Risk Purges
For deep architectural consolidation involving active Risk Engines, HNASP protocols, or localized Storage components.
*   **Swarm Action:** Execute Steps 1-4. The swarm MUST halt before Step 5 (Deletion).
*   **Oversight:** The swarm drafts an "Architectural Consolidation Memo" detailing exactly what logic is being merged and requests explicit authorization to execute the Purge phase. The human Chief Risk Officer / Lead Architect must provide a cryptographic or explicit textual approval to proceed.

## 4. ASYNC EXECUTION PROMPT (For Swarm Orchestrator)

```markdown
**System Role:** You are the Refactoring Swarm Orchestrator. Your current task is to execute the "Merge & Purge" protocol on the target directory: [TARGET_DIRECTORY].

**Directives:**
1. Trace all dependencies for `[TARGET_DIRECTORY]` using `grep` across the repository.
2. Update all import paths to point to the consolidated target: `[NEW_KERNEL_DIRECTORY]`.
3. Run `uv run pytest` to verify the merge. If failures occur, patch the import errors immediately.
4. If this is a HOTL operation, run `rm -rf [TARGET_DIRECTORY]` and submit the PR.
5. If this is a HITL operation, prepare the "Architectural Consolidation Memo" and await human authorization.

**Failure State:** If test failures cannot be resolved within 3 iterations, execute a git reset and escalate to the human architect with the error trace.
```

## 5. EVOLUTION LOG
*See `docs/SWARM_EVOLUTION_LOG.md` for historical application of this protocol.*
