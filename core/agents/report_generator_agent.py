# core/agents/report_generator_agent.py

import logging
from typing import Any, Dict, Optional

from core.agents.agent_base import AgentBase
from semantic_kernel import Kernel


class ReportGeneratorAgent(AgentBase):
    """
    An agent responsible for generating final reports by synthesizing
    analysis from other agents.
    """

    def __init__(self, config: Dict[str, Any], constitution: Optional[Dict[str, Any]] = None, kernel: Optional[Kernel] = None):
        super().__init__(config, constitution, kernel)
        logging.info(f"ReportGeneratorAgent initialized with config: {config}")

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Executes the report generation process.

        It expects a dictionary of analysis results in kwargs.
        Example kwargs:
        {
            'user_query': 'What is the credit risk of Apple Inc.?',
            'fundamental_analysis': {'revenue': '...', 'pe_ratio': '...'},
            'market_sentiment': {'score': 'Positive', 'summary': '...'},
            'report_format': 'markdown'
        }
        """
        logging.info("Executing ReportGeneratorAgent...")

        report_parts = []

        # 1. Add a title based on the user query
        user_query = kwargs.get('user_query', 'Analysis Report')
        report_parts.append(f"# Report for: {user_query}\n")

        # 2. Iterate through the analysis results and add them to the report
        for analysis_type, analysis_result in kwargs.items():
            # Skip non-analysis keys
            if analysis_type in ['user_query', 'report_format']:
                continue

            if isinstance(analysis_result, dict):
                report_parts.append(f"## {analysis_type.replace('_', ' ').title()}\n")
                for key, value in analysis_result.items():
                    report_parts.append(f"- **{key.replace('_', ' ').title()}:** {value}")
                report_parts.append("\n")
            else:
                report_parts.append(f"## {analysis_type.replace('_', ' ').title()}\n{analysis_result}\n")

        # 3. Join the parts into a single report string
        final_report = "\n".join(report_parts)

        logging.info(f"Generated report:\n{final_report}")

        # For now, we return the report as a string.
        # In the future, this could return a file path or a more structured object.
        return final_report

    def get_skill_schema(self) -> Dict[str, Any]:
        """
        Defines the agent's skills for the MCP service registry.
        """
        return {
            "name": type(self).__name__,
            "description": "Generates a final, human-readable report from structured analysis data.",
            "skills": [
                {
                    "name": "generate_report",
                    "description": "Takes a dictionary of analysis results and formats them into a report.",
                    "parameters": [
                        {
                            "name": "analysis_data",
                            "description": "A dictionary where keys are analysis types and values are the results.",
                            "type": "dict",
                            "required": True
                        }
                    ]
                }
            ]
        }
