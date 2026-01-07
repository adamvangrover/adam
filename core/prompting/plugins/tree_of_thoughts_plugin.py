from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from core.prompting.base_prompt_plugin import BasePromptPlugin, PromptMetadata
# from core.schemas.cognitive_state import CognitiveState, ThoughtNode, StrategicPlan # Unused but available for future expansion

# --- Schemas ---

class TreeOfThoughtsInput(BaseModel):
    problem_statement: str = Field(..., description="The core problem to solve.")
    context_data: str = Field(..., description="Background information or data context.")
    max_steps: int = Field(3, description="Maximum depth of the thought tree.")
    num_proposals: int = Field(3, description="Number of thoughts to generate per step.")

class TreeOfThoughtsOutput(BaseModel):
    final_answer: str = Field(..., description="The best solution found.")
    reasoning_trace: List[str] = Field(..., description="The path of thoughts leading to the solution.")
    confidence_score: float = Field(..., ge=0, le=1)

# --- Plugin ---

class TreeOfThoughtsPlugin(BasePromptPlugin[TreeOfThoughtsOutput]):
    """
    Implements a Tree of Thoughts (ToT) reasoning engine using the Prompt-as-Code framework.

    This plugin does NOT perform the search itself (BFS/DFS) in the `render` method
    (which would require multiple LLM calls). Instead, it provides the structured
    prompting interface that an agent loop would use to 'Generate' and 'Evaluate' thoughts.

    For this 'Prompt-as-Code' demonstration, we implement a 'One-Shot ToT' where the
    LLM is instructed to simulate the tree search internally and output the best path.
    """

    def get_input_schema(self):
        return TreeOfThoughtsInput

    def get_output_schema(self):
        return TreeOfThoughtsOutput

    def render(self, inputs: Dict[str, Any]) -> str:
        """
        Renders a complex 'System 2' prompt that instructs the model to perform
        an internal Tree of Thoughts search.
        """
        validated_data = self.validate_inputs(inputs)

        # We construct a specialized template here rather than reading from a file,
        # to demonstrate logic-heavy prompt construction.

        prompt = f"""
# TREE OF THOUGHTS REASONING PROTOCOL

## ROLE
You are a Super-Forecaster and Strategic Planner capable of exploring multiple future scenarios.

## TASK
Solve the following problem by explicitly generating multiple "thought branches", evaluating them, and selecting the best path.

**Problem:** {validated_data.problem_statement}
**Context:** {validated_data.context_data}

## SEARCH PARAMETERS
- Max Depth: {validated_data.max_steps}
- Width: {validated_data.num_proposals} branches per step.

## PROTOCOL
1. **Root Phase:** Analyze the problem and propose {validated_data.num_proposals} initial approaches.
2. **Exploration Phase:** For each promising approach, simulate the next logical steps.
3. **Evaluation Phase:** Critically assess each path for feasibility and risk.
4. **Selection Phase:** Prune the weak paths and converge on the single best solution.

## OUTPUT FORMAT
You must return a JSON object matching this schema:
{{
  "final_answer": "The concise solution.",
  "reasoning_trace": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
  "confidence_score": 0.95
}}

Provide **only** the valid JSON.
"""
        return prompt.strip()
