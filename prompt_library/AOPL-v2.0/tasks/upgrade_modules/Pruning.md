## Phase 1: THE PRUNING (Waste Removal)
**Objective**: Aggressively but safely prune the code. Remove dead code, unused variables/imports, unneeded dependencies, and obsolete comments. Simplify overly complex or nested logic.

### Instructions:
1.  **Analyze Imports**: Review all import statements. Remove any that are not actively used within the module.
2.  **Identify Dead Code**: Find and remove functions, classes, or methods that are never called or instantiated within the project (unless they are explicitly part of a public API intended for external use).
3.  **Remove Unused Variables**: Delete any variables that are assigned but never read.
4.  **Clean Up Comments**: Remove commented-out code, redundant comments that just restate the code, and obsolete or misleading comments.
5.  **Simplify Logic**: Look for deeply nested `if/else` statements, complex boolean expressions, or convoluted loops. Refactor them into flatter, simpler, and more direct logic. Use guard clauses to reduce nesting.
6.  **Verify Tests**: Ensure that after pruning, the existing test suite still passes without modification (or with minimal modifications if tests relied on pruned internal details).