import asyncio
import logging
import json
import uuid
from typing import Optional, Dict, Any, List

from core.agents.agent_base import AgentBase
try:
    from semantic_kernel import Kernel, KernelArguments
except ImportError:
    Kernel = Any
    KernelArguments = Any

from core.agents.governance.repo_guardian.schemas import (
    PullRequest, FileDiff, ReviewDecision, CodeReviewParams, ReviewDecisionStatus, ReviewComment
)
from core.agents.governance.repo_guardian.prompts import SYSTEM_PROMPT, REVIEW_PROMPT_TEMPLATE
from core.agents.governance.repo_guardian.tools import GitTools, StaticAnalyzer
from core.prompting.base_prompt_plugin import BasePromptPlugin

# Initialize logger
logger = logging.getLogger(__name__)

class RepoGuardianAgent(AgentBase):
    """
    The RepoGuardian Agent serves as an automated code reviewer and gatekeeper.
    It analyzes proposed changes against repository standards and provides
    structured feedback.
    """

    def __init__(self, config: Dict[str, Any], constitution: Optional[Dict[str, Any]] = None, kernel: Optional[Kernel] = None):
        super().__init__(config, constitution, kernel)
        self.tools = GitTools()
        self.analyzer = StaticAnalyzer()

        # Override system persona if needed, or rely on prompt injection
        self.system_prompt = SYSTEM_PROMPT

    async def execute(self, **kwargs: Any) -> ReviewDecision:
        """
        Main execution method.
        Expected kwargs:
        - pr (dict or PullRequest): The pull request data.
        - params (dict or CodeReviewParams, optional): Review parameters.
        """
        try:
            # 1. Parse Inputs
            pr_data = kwargs.get("pr")
            if not pr_data:
                raise ValueError("Missing 'pr' in execution arguments.")

            if isinstance(pr_data, dict):
                pr = PullRequest(**pr_data)
            elif isinstance(pr_data, PullRequest):
                pr = pr_data
            else:
                raise TypeError(f"Invalid type for 'pr': {type(pr_data)}")

            params_data = kwargs.get("params", {})
            if isinstance(params_data, dict):
                params = CodeReviewParams(**params_data)
            elif isinstance(params_data, CodeReviewParams):
                params = params_data
            else:
                params = CodeReviewParams()

            logger.info(f"RepoGuardian starting review for PR {pr.pr_id} by {pr.author}")

            # 2. Heuristic Analysis (Pre-LLM)
            heuristic_comments = self._run_heuristics(pr)

            # 3. LLM Review
            decision = await self._llm_review(pr, params)

            # 4. Merge Heuristics into Decision
            decision.comments.extend(heuristic_comments)

            # Adjust score/status based on critical heuristic failures
            critical_issues = [c for c in heuristic_comments if c.severity == "critical"]
            if critical_issues and decision.status == ReviewDecisionStatus.APPROVE:
                decision.status = ReviewDecisionStatus.REQUEST_CHANGES
                decision.summary += "\n\n[AUTOMATED] Decision downgraded due to critical heuristic failures."
                decision.score = max(0, decision.score - 20)

            logger.info(f"Review complete for PR {pr.pr_id}: {decision.status}")
            return decision

        except Exception as e:
            logger.error(f"Error in RepoGuardian execution: {e}", exc_info=True)
            # Fallback decision
            return ReviewDecision(
                pr_id=kwargs.get("pr", {}).get("pr_id", "unknown") if kwargs.get("pr") else "unknown",
                status=ReviewDecisionStatus.REJECT,
                summary=f"Internal Agent Error: {str(e)}",
                score=0,
                comments=[ReviewComment(
                    filepath="general",
                    severity="critical",
                    message=f"Agent crashed during review: {str(e)}"
                )]
            )

    def _run_heuristics(self, pr: PullRequest) -> List[ReviewComment]:
        """Runs deterministic checks."""
        comments = []

        for file in pr.files:
            # Check for large files
            if len(file.diff_content) > 50000:
                comments.append(ReviewComment(
                    filepath=file.filepath,
                    severity="warning",
                    message="File diff is unusually large (>50KB). Consider breaking this into smaller commits."
                ))

            # Check for Pydantic V2 usage in schemas
            if "pydantic" in file.diff_content and "BaseModel" in file.diff_content:
                if "validator" in file.diff_content and "field_validator" not in file.diff_content:
                     comments.append(ReviewComment(
                        filepath=file.filepath,
                        severity="warning",
                        message="Detected potential legacy Pydantic V1 'validator'. Use V2 'field_validator' if possible."
                    ))

            # Check for strict typing
            if file.filepath.endswith(".py") and file.change_type != "delete":
                if "def " in file.diff_content and "->" not in file.diff_content:
                     comments.append(ReviewComment(
                        filepath=file.filepath,
                        severity="suggestion",
                        message="Some function definitions appear to be missing return type hints."
                    ))

        return comments

    async def _llm_review(self, pr: PullRequest, params: CodeReviewParams) -> ReviewDecision:
        """Delegates the deep understanding to the LLM."""

        # Prepare the Prompt using simple Jinja2 rendering (simulated here or via existing tools)
        # For robustness, we'll do simple f-string/replace if Jinja isn't handy, but we assume Jinja2 is installed per memory.
        from jinja2 import Template

        template = Template(REVIEW_PROMPT_TEMPLATE)
        user_prompt = template.render(pr=pr, params=params)

        # Mock LLM call if no kernel
        if not self.kernel:
            logger.warning("No Semantic Kernel present. Using mock response.")
            return ReviewDecision(
                pr_id=pr.pr_id,
                status=ReviewDecisionStatus.APPROVE,
                summary="[MOCK] No Kernel. Changes look structurally okay based on heuristics.",
                score=80,
                comments=[]
            )

        # Call LLM via Kernel (Assuming we have a suitable function registered or we use the prompt directly)
        try:
            # This relies on the agent being configured with a chat completion service
            # We construct a full prompt with System + User
            full_prompt = f"{self.system_prompt}\n\n{user_prompt}"

            # Note: This is a simplification. In a real v23/v22 setup, we'd use a specific prompt function.
            # Assuming 'kernel.invoke_prompt' or similar exists for ad-hoc queries.
            # If not, we fall back to a registered function if available.

            # Attempting standard SK invocation
            # Using a hypothetical "ChatPlugin" or "WriterPlugin"

            # For now, let's assume we use the 'Prompt-as-Code' style wrapper if we were fully integrated.
            # Here I will try to use `kernel.invoke_prompt` if it exists (SK > 1.0)

            result_str = ""
            if hasattr(self.kernel, 'invoke_prompt'):
                 # SK Python v1.0+
                 result = await self.kernel.invoke_prompt(prompt=full_prompt)
                 result_str = str(result)
            else:
                # Fallback or SK < 1.0
                # We might need to register the function first
                sk_func = self.kernel.create_semantic_function(full_prompt, max_tokens=2000, temperature=0.2)
                result = await self.kernel.invoke(sk_func)
                result_str = str(result)

            # Parse JSON from result
            # Clean up markdown code blocks if present
            cleaned_result = result_str.replace("```json", "").replace("```", "").strip()

            try:
                data = json.loads(cleaned_result)
                return ReviewDecision(**data)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {result_str}")
                return ReviewDecision(
                    pr_id=pr.pr_id,
                    status=ReviewDecisionStatus.REQUEST_CHANGES,
                    summary="Agent failed to parse LLM review. Please check logs.",
                    score=50,
                    comments=[ReviewComment(filepath="meta", severity="warning", message="LLM output was not valid JSON.")]
                )

        except Exception as e:
            logger.error(f"LLM Interaction failed: {e}")
            raise
