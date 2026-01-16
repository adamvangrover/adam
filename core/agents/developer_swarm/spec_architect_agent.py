# spec_architect_agent.py
"""
This module defines the SpecArchitectAgent, a specialized agent responsible for
generating comprehensive technical specifications (SPEC.md) adhering to the
Spec-Driven Agent Protocol (SDAP).
"""

from typing import Any, Dict, List, Optional
from core.agents.agent_base import AgentBase

class SpecArchitectAgent(AgentBase):
    """
    The SpecArchitectAgent is the 'Architect' in the Spec-Driven Development workflow.
    Its sole purpose is to take a high-level vision or goal and produce a rigorous,
    structured SPEC.md file. It operates in 'Plan Mode' (read-only) and does not
    write application code.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        # self.name is read-only in AgentBase, but we can set it during initialization if allowed
        # or just rely on the config. Ideally, we just use the class identity.
        # self.name = "SpecArchitectAgent"

    def _load_prompt_template(self) -> str:
        """
        Loads the SDAP Prompt Template.
        In a real scenario, this would load from a file.
        """
        return (
            "You are the SpecArchitectAgent. Your job is to draft a rigorous technical specification "
            "(SPEC.md) for the following goal. You must strictly follow the Spec-Driven Agent Protocol (SDAP).\n"
            "Reference the 5 Core Principles: Vision, Structure, Modularity, Boundaries, Verification.\n"
            "Do NOT write code. Output only the Markdown spec."
        )

    async def execute(self, goal: str, context_files: Optional[List[str]] = None) -> str:
        """
        Takes a high-level goal and generates a SPEC.md content string.

        :param goal: A string describing the high-level objective (e.g., "Add User Auth").
        :param context_files: Optional list of file paths to read for context.
        :return: A string containing the Markdown content of the generated SPEC.md.
        """

        system_instruction = self._load_prompt_template()

        # 2. Simulate Context Gathering (Read-Only)
        # In a full implementation, this agent would use tools to read `context_files`.
        codebase_context = ""
        if context_files:
            codebase_context = f"\nAnalyzed context from: {', '.join(context_files)}"

        # 3. Generate the Spec (Simulated LLM Call)
        # We return a structured template filled with the goal for this demonstration.

        spec_content = f"""# Spec: {goal}

## 1. Overview & Objectives
*   **Goal:** Implement {goal}
*   **User Story:** As a user, I want {goal} so that I can utilize new capabilities.
*   **Success Metrics:** All tests pass, Feature is usable.

## 2. Technical Context
*   **Stack:** Python 3.10+, Pytest
*   **Context:** {codebase_context or "General Repository Context"}

## 3. Implementation Plan
### Phase 1: Foundation
*   [ ] Define data models and schemas.
*   [ ] Create basic scaffolding.

### Phase 2: Core Logic
*   [ ] Implement main logic for {goal}.

### Phase 3: Integration & UI
*   [ ] Wire up API endpoints or UI components.

## 4. Commands & Development
*   **Test:** `pytest tests/`
*   **Lint:** `flake8 src/`

## 5. Verification & Testing Strategy
*   **Unit Tests:** Cover all new functions.
*   **Integration Tests:** Verify end-to-end flow.

## 6. Constraints & Boundaries
*   ✅ **Always:** Type-hint all functions, Add docstrings.
*   🚫 **Never:** Commit secrets, Break existing public APIs.
"""
        return spec_content

    def get_skill_schema(self) -> Dict[str, Any]:
        """
        Defines the skills of the SpecArchitectAgent.
        """
        schema = super().get_skill_schema()
        schema["skills"].append(
            {
                "name": "generate_spec",
                "description": "Generates a SPEC.md file from a high-level goal.",
                "parameters": [
                    {"name": "goal", "type": "string", "description": "The high-level objective."},
                    {"name": "context_files", "type": "array", "items": {"type": "string"}, "description": "List of files to analyze."}
                ]
            }
        )
        return schema
