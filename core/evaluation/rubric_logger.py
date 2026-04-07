import json
import os
from datetime import datetime
from typing import Dict, Any

class EvaluationMarkdownLogger:
    """
    Formats evaluation and refinement results into a structured Markdown document
    for both human review and machine ingestion.
    """
    def __init__(self, log_dir: str = "logs/evaluations"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_refinement_session(self, session_data: Dict[str, Any], drift_report: Dict[str, Any] = None) -> str:
        """
        Creates a markdown log for a full iterative refinement session.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.log_dir, f"refinement_session_{timestamp}.md")

        final_prompt = session_data.get("final_prompt", "")
        iterations = session_data.get("iterations", 0)
        history = session_data.get("history", [])

        md_content = []
        md_content.append(f"# Prompt Refinement Session: {timestamp}")
        md_content.append(f"**Total Iterations:** {iterations}")

        if drift_report:
            md_content.append("## Model Drift & Performance Report")
            md_content.append(f"- **Status:** {drift_report.get('status', 'unknown')}")
            md_content.append(f"- **Score Degradation:** {drift_report.get('score_degradation_pct', 0.0)}%")
            md_content.append(f"- **Avg Efficiency:** {drift_report.get('avg_efficiency', 0.0)} tokens/ms")

        md_content.append("\n## Final Prompt")
        md_content.append("```text\n" + final_prompt + "\n```\n")

        md_content.append("## Iteration History\n")

        for iteration in history:
            md_content.append(f"### Iteration {iteration['iteration']}")

            # System Metrics
            sys_metrics = iteration.get("system_metrics", {})
            md_content.append("#### System Metrics (System as Judge)")
            md_content.append(f"- **Latency:** {sys_metrics.get('latency_ms', 0):.2f} ms")
            md_content.append(f"- **Token Efficiency:** {sys_metrics.get('token_efficiency', 0):.2f} tokens/ms")
            md_content.append(f"- **Format Valid:** {sys_metrics.get('format_valid', False)}")
            md_content.append(f"- **Token Usage:** {json.dumps(sys_metrics.get('token_usage', {}))}")

            # LLM Metrics
            llm_metrics = iteration.get("llm_metrics", {})
            md_content.append("\n#### LLM Review (LLM as Judge)")
            md_content.append(f"- **Overall Score:** {llm_metrics.get('overall_score', 0):.2f}")

            md_content.append("\n**Critique:**")
            md_content.append(f"> {llm_metrics.get('critique', 'None').replace(chr(10), chr(10) + '> ')}")

            md_content.append("\n**Improvement Suggestions:**")
            for sugg in llm_metrics.get('improvement_suggestions', []):
                md_content.append(f"- {sugg}")

            md_content.append("\n---\n")

        # Machine-readable JSON block
        md_content.append("## Machine Readable Data")
        md_content.append("```json\n" + json.dumps(session_data, indent=2) + "\n```")

        full_markdown = "\n".join(md_content)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_markdown)

        return filename
