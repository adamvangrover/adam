from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from core.prompting.base_prompt_plugin import BasePromptPlugin, PromptMetadata

class CoVeInput(BaseModel):
    query: str
    draft_response: Optional[str] = None # If None, the plugin generates it first

class CoVeOutput(BaseModel):
    final_response: str
    verified_facts: List[str]
    corrections_made: List[str]

    def get_summary(self) -> str:
        """
        Returns a summary of the verification process.
        """
        return f"Verified {len(self.verified_facts)} facts. Corrections: {len(self.corrections_made)}."

class ChainOfVerificationPlugin(BasePromptPlugin[CoVeOutput]):
    """
    Implements the Chain of Verification (CoVe) pattern.
    Draft -> Verify -> Revise.
    """

    def get_input_schema(self):
        return CoVeInput

    def get_output_schema(self):
        return CoVeOutput

    def render(self, inputs: Dict[str, Any]) -> str:
        data = self.validate_inputs(inputs)

        prompt = f"""
# CHAIN OF VERIFICATION PROTOCOL

## 1. USER QUERY
{data.query}

"""
        if data.draft_response:
             prompt += f"""
## 2. DRAFT RESPONSE TO AUDIT
{data.draft_response}

## 3. VERIFICATION TASK
The draft above may contain hallucinations or errors.
1. Identify all factual claims in the draft.
2. Verify each claim against your internal knowledge or provided context.
3. List any corrections needed.
4. Rewrite the response to be fully accurate.

"""
        else:
            prompt += f"""
## 2. INSTRUCTIONS
1. **Draft**: Generate a preliminary answer to the query.
2. **Verify**: Critique your own draft. Identify potential inaccuracies.
3. **Revise**: Produce a final, verified response.

"""

        prompt += """
## 4. OUTPUT FORMAT (JSON)
{
  "final_response": "The revised, accurate answer.",
  "verified_facts": ["Fact 1", "Fact 2"],
  "corrections_made": ["Changed X to Y", "Removed claim Z"]
}
"""
        return prompt.strip()
