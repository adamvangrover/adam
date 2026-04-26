## Phase 6: THE DOCUMENTER (Clarity & AI-Readability)
**Objective**: Generate clean, highly informative inline documentation and docstrings. Write them specifically so that future AI context windows can instantly understand the module's purpose, inputs, and outputs.

### Instructions:
1.  **Module-Level Docstring**: Write a comprehensive docstring at the top of the file explaining the module's overall purpose, its role in the larger system, and any critical dependencies or assumptions.
2.  **Class/Function Docstrings**: Every public class, method, and function must have a docstring. Use a standard format (e.g., Google or Sphinx style in Python). Include:
    *   A brief summary of what it does.
    *   `Args:` (or equivalent) listing every parameter, its type, and its purpose.
    *   `Returns:` detailing the return type and what it represents.
    *   `Raises:` listing specific exceptions that might be thrown and under what conditions.
3.  **AI-Optimized Context**: Phrase docstrings clearly and concisely. Avoid jargon where plain English suffices. State the *intent* ("Why are we doing this?") alongside the mechanism ("What are we doing?").
4.  **Inline Comments**: Use inline comments sparingly, only to explain *complex, non-obvious, or tricky* pieces of logic that cannot be simplified. If you find yourself writing a long inline comment, consider if the code itself needs refactoring (Phase 2).
5.  **Review Format**: Ensure consistency in formatting across all documentation within the file.