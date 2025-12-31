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
    PullRequest, FileDiff, ReviewDecision, CodeReviewParams, ReviewDecisionStatus, ReviewComment, AnalysisResult
)
from core.agents.governance.repo_guardian.prompts import SYSTEM_PROMPT, REVIEW_PROMPT_TEMPLATE
from core.agents.governance.repo_guardian.tools import GitTools, StaticAnalyzer, SecurityScanner
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
        self.scanner = SecurityScanner()

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
            heuristic_comments, analysis_results = self._run_heuristics(pr)

            # 3. LLM Review
            decision = await self._llm_review(pr, params, analysis_results)

            # 4. Merge Heuristics into Decision
            decision.comments.extend(heuristic_comments)
            decision.analysis_results = analysis_results

            # Adjust score/status based on critical heuristic failures
            critical_issues = [c for c in heuristic_comments if c.severity == "critical"]
            if critical_issues:
                decision.status = ReviewDecisionStatus.REJECT if any(c.message.startswith("SECURITY") for c in critical_issues) else ReviewDecisionStatus.REQUEST_CHANGES
                decision.summary += "\n\n[AUTOMATED] Decision downgraded due to critical heuristic failures."
                decision.score = max(0, decision.score - (25 * len(critical_issues)))

            logger.info(f"Review complete for PR {pr.pr_id}: {decision.status}")
            return decision

        except Exception as e:
            logger.error(f"Error in RepoGuardian execution: {e}", exc_info=True)
            # Fallback decision
            return ReviewDecision(
                pr_id=kwargs.get("pr", {}).get("pr_id", "unknown") if isinstance(kwargs.get("pr"), dict) else "unknown",
                status=ReviewDecisionStatus.REJECT,
                summary=f"Internal Agent Error: {str(e)}",
                score=0,
                comments=[ReviewComment(
                    filepath="general",
                    severity="critical",
                    message=f"Agent crashed during review: {str(e)}"
                )]
            )

    def _run_heuristics(self, pr: PullRequest) -> tuple[List[ReviewComment], Dict[str, AnalysisResult]]:
        """Runs deterministic checks."""
        comments = []
        results = {}

        for file in pr.files:
            file_results = AnalysisResult()

            # Check for large files
            if len(file.diff_content) > 50000:
                comments.append(ReviewComment(
                    filepath=file.filepath,
                    severity="warning",
                    message="File diff is unusually large (>50KB). Consider breaking this into smaller commits."
                ))

            # Security Scan
            findings = self.scanner.scan_content(file.diff_content)
            if findings:
                file_results.security_findings = findings
                for finding in findings:
                    comments.append(ReviewComment(
                        filepath=file.filepath,
                        severity="critical",
                        message=f"SECURITY: Potential {finding['type']} detected: {finding['snippet']}"
                    ))

            # Python-specific AST Analysis
            if file.filepath.endswith(".py") and file.change_type != "delete":
                # Only analyze if we have content. For diffs, this is tricky as valid python might not be in diff chunks.
                # Ideally, we reconstruct the full file, but for now we try to parse the new_content if provided,
                # or just look for patterns in diff if that fails.

                content_to_analyze = file.new_content if file.new_content else file.diff_content
                # Heuristic: if analyzing diff content, AST might fail.
                # We can try to clean it (remove +/-, skip headers) or just use naive checks for diffs.

                # If we have the full new content, use AST
                if file.new_content:
                    report = self.analyzer.analyze_python_code(file.new_content, file.filepath)

                    file_results.missing_docstrings = report["missing_docstrings"]
                    file_results.missing_type_hints = report["missing_type_hints"]
                    file_results.dangerous_functions = report["dangerous_functions"]

                    for issue in report["dangerous_functions"]:
                        comments.append(ReviewComment(
                            filepath=file.filepath,
                            severity="warning",
                            message=f"Usage of dangerous function detected: {issue}"
                        ))

                # Fallback Naive Checks on Diff Content if AST didn't run (e.g. no new_content)
                else:
                    if "def " in file.diff_content and "->" not in file.diff_content:
                         file_results.missing_type_hints.append("Potentially missing return annotation (heuristic)")
                         comments.append(ReviewComment(
                            filepath=file.filepath,
                            severity="suggestion",
                            message="Some function definitions appear to be missing return type hints."
                        ))

            # Pydantic V2 Check (Naive)
            if "pydantic" in file.diff_content and "BaseModel" in file.diff_content:
                if "validator" in file.diff_content and "field_validator" not in file.diff_content:
                     comments.append(ReviewComment(
                        filepath=file.filepath,
                        severity="warning",
                        message="Detected potential legacy Pydantic V1 'validator'. Use V2 'field_validator' if possible."
                    ))

            results[file.filepath] = file_results

        return comments, results

    async def _llm_review(self, pr: PullRequest, params: CodeReviewParams, analysis_results: Dict[str, AnalysisResult]) -> ReviewDecision:
        """Delegates the deep understanding to the LLM."""

        # Prepare the Prompt using simple Jinja2 rendering (simulated here or via existing tools)
        from jinja2 import Template

        template = Template(REVIEW_PROMPT_TEMPLATE)
        # Convert Pydantic objects to dicts for Jinja
        analysis_dicts = {k: v.model_dump() for k, v in analysis_results.items()}
        user_prompt = template.render(pr=pr, params=params, analysis_results=analysis_dicts)

        # Mock LLM call if no kernel
        if not self.kernel:
            logger.warning("No Semantic Kernel present. Using mock response.")
            return ReviewDecision(
                pr_id=pr.pr_id,
                status=ReviewDecisionStatus.APPROVE,
                summary="[MOCK] No Kernel. Changes look structurally okay based on heuristics.",
                score=80,
                comments=[],
                analysis_results=analysis_results
            )

        # Call LLM via Kernel
        try:
            full_prompt = f"{self.system_prompt}\n\n{user_prompt}"

            result_str = ""
            if hasattr(self.kernel, 'invoke_prompt'):
                 # SK Python v1.0+
                 result = await self.kernel.invoke_prompt(prompt=full_prompt)
                 result_str = str(result)
            else:
                # Fallback or SK < 1.0
                sk_func = self.kernel.create_semantic_function(full_prompt, max_tokens=2000, temperature=0.2)
                result = await self.kernel.invoke(sk_func)
                result_str = str(result)

            # Parse JSON from result
            cleaned_result = result_str.replace("```json", "").replace("```", "").strip()

            try:
                data = json.loads(cleaned_result)
                # Ensure analysis results are passed through if LLM didn't include them
                if "analysis_results" not in data:
                    data["analysis_results"] = analysis_results
                return ReviewDecision(**data)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {result_str}")
                return ReviewDecision(
                    pr_id=pr.pr_id,
                    status=ReviewDecisionStatus.REQUEST_CHANGES,
                    summary="Agent failed to parse LLM review. Please check logs.",
                    score=50,
                    comments=[ReviewComment(filepath="meta", severity="warning", message="LLM output was not valid JSON.")],
                    analysis_results=analysis_results
                )

        except Exception as e:
            logger.error(f"LLM Interaction failed: {e}")
            raise
