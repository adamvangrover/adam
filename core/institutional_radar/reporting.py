import json
import logging
import pandas as pd
from typing import Dict
from core.llm_plugin import LLMPlugin
from core.utils.logging_utils import get_logger

logger = get_logger("institutional_radar.reporting")

class InstitutionalRadarReporter:
    def __init__(self, llm_plugin: LLMPlugin = None):
        self.llm = llm_plugin or LLMPlugin()

    def generate_report(self,
                        crowding_df: pd.DataFrame,
                        sector_flow_df: pd.DataFrame,
                        cluster_buys_df: pd.DataFrame,
                        quarter_label: str) -> str:
        """
        Synthesizes the analytical data into a narrative report.
        """
        # Data Serialization
        context = {
            "quarter": quarter_label,
            "sector_flows": sector_flow_df.to_dict(orient='records') if not sector_flow_df.empty else [],
            "top_crowded_trades": crowding_df.sort_values('crowding_score', ascending=False).head(5).to_dict(orient='records') if not crowding_df.empty else [],
            "cluster_buys": cluster_buys_df.head(5).to_dict(orient='records') if not cluster_buys_df.empty else []
        }

        json_context = json.dumps(context, indent=2)

        # Prompt Engineering
        prompt = f"""
You are a Senior Portfolio Strategist. Analyze the following JSON data regarding {quarter_label} 13F flows.
Identify the primary rotation trend.
Write a 300-word executive summary using sophisticated financial terminology.
Focus on the divergence between 'Fundamental' and 'Quant' flows.
Do not mention short positions as facts, only as possibilities.

Context Data:
{json_context}

Executive Summary:
"""
        logger.info("Sending prompt to LLM...")
        try:
            report = self.llm.generate_text(prompt, task="default")
            return report
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return "Error generating narrative report."

    def format_html_report(self, narrative: str, context_data: Dict) -> str:
        """
        Helper to format the output as HTML snippet for the UI.
        """
        # This could be more elaborate, but for now just wrapping in divs
        html = f"""
        <div class="radar-report">
            <h2 class="text-xl font-bold text-cyan-400 mb-4">Quarterly Trend Monitor: {context_data.get('quarter', 'N/A')}</h2>
            <div class="narrative prose prose-invert mb-6">
                {narrative.replace(chr(10), '<br>')}
            </div>
            <!-- Additional visualization data could be embedded here -->
        </div>
        """
        return html
