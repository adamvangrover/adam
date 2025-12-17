import logging
import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional

# Core imports
from core.agents.agent_base import AgentBase
from core.vertical_risk_agent.ingestion.sec_13f_handler import Sec13FHandler

logger = logging.getLogger(__name__)

class InstitutionalTrendAgent(AgentBase):
    """
    Agent responsible for monitoring institutional capital flows via 13F filings
    and generating strategic market intelligence reports.
    
    Architecture:
    1. Ingestion Layer (Hard Logic): Fetches raw 13F data via Sec13FHandler.
    2. Processing Layer (Pandas): Calculates deltas (New/Exits/Increases).
    3. Cognitive Layer (LLM): Synthesizes quantitative moves into qualitative strategy.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        if config is None:
            config = {}
        super().__init__(config, **kwargs)
        
        # --- HARD SKILL COMPONENTS (Data Fetching) ---
        self.handler = Sec13FHandler()
        # Default watchlist "The Old Guard & The Quants"
        self.watchlist = config.get("watchlist", {
            "Berkshire Hathaway": "0001067983",
            "Renaissance Technologies": "0001037389",
            "Bridgewater Associates": "0001350694",
            "Citadel Advisors": "0001423053"
        })

        # --- SOFT SKILL COMPONENTS (Cognition) ---
        self.persona = "Chief Investment Strategist"
        # Prompt-as-Code file path
        self.prompt_path = config.get(
            "prompt_path", 
            "prompt_library/AOPL-v1.0/professional_outcomes/LIB-PRO-010_quarterly_trend_monitor.md"
        )

    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Orchestrates the pipeline: Fetch -> Calculate Delta -> Synthesize.
        """
        logger.info("Executing Institutional Trend Analysis...")
        
        # 1. Context Extraction
        task = kwargs.get("task", "analyze")
        period = kwargs.get("period", "2025-Q3")
        
        # 2. Hard Logic Execution (The Quant Layer)
        if task in ["analyze", "synthesis"]:
            try:
                quantitative_data = self._run_quantitative_analysis(period)
            except Exception as e:
                logger.error(f"Failed to run quantitative analysis: {e}")
                return {"error": str(e)}
        else:
            return {"error": f"Unknown task: {task}"}

        # 3. Cognitive Execution (The LLM Layer)
        # We convert the structured quantitative data into a context string for the LLM
        report_content = await self._synthesize_intelligence(quantitative_data, period)

        return {
            "report_type": "Quarterly Trend Monitor",
            "period": period,
            "quantitative_metrics": quantitative_data, # Return raw data for UI/Charts
            "synthesis": report_content # Return LLM narrative
        }

    def _run_quantitative_analysis(self, period: str) -> Dict[str, Any]:
        """
        Executes the specific pandas logic to find deltas between quarters.
        (Preserved from Feature Branch)
        """
        previous_period = self._get_previous_period(period)
        analysis_results = {
            "period": period,
            "compared_to": previous_period,
            "funds": []
        }

        for fund_name, cik in self.watchlist.items():
            logger.debug(f"Processing {fund_name} ({period})...")
            
            # Fetch Holdings
            curr_holdings = self.handler.fetch_holdings(cik, period)
            prev_holdings = self.handler.fetch_holdings(cik, previous_period)

            # Calculate Delta
            if curr_holdings.empty and prev_holdings.empty:
                logger.warning(f"No data for {fund_name}")
                continue
                
            delta = self.handler.calculate_delta(curr_holdings, prev_holdings)

            # Filter for high-signal moves (New Entries & Complete Exits)
            new_positions = delta[delta['action_calculated'] == 'NEW']
            exits = delta[delta['action_calculated'] == 'EXIT']
            
            # Identify Top Weightings (if available in handler)
            # Assuming 'pct_portfolio' exists or derived from value
            
            fund_summary = {
                "name": fund_name,
                "cik": cik,
                "top_buys": new_positions.nlargest(5, 'value_curr')[['ticker', 'name', 'value_curr']].to_dict('records') if not new_positions.empty else [],
                "top_sells": exits.nlargest(5, 'value_prev')[['ticker', 'name', 'value_prev']].to_dict('records') if not exits.empty else [],
                "total_aum": delta['value_curr'].sum()
            }
            analysis_results["funds"].append(fund_summary)

        return analysis_results

    async def _synthesize_intelligence(self, quantitative_data: Dict[str, Any], period: str) -> str:
        """
        Injects the calculated data into the System Prompt and invokes the Kernel.
        (Preserved and Expanded from Main Branch)
        """
        # Convert dict to a formatted string for the LLM to read easily
        data_context = json.dumps(quantitative_data, indent=2)

        # Load Prompt
        try:
            with open(self.prompt_path, 'r') as f:
                system_prompt_template = f.read()
        except FileNotFoundError:
            logger.warning(f"Prompt file missing at {self.prompt_path}. Using fallback.")
            system_prompt_template = "You are a Wall Street Strategist. Analyze the following 13F data trends."

        # Construct Prompt
        full_prompt = f"""
        {system_prompt_template}

        CURRENT PERIOD: {period}
        
        === QUANTITATIVE DATA INPUT ===
        {data_context}
        ===============================
        """

        # Invoke Kernel
        if self.kernel:
            try:
                result = await self.kernel.invoke_prompt(prompt=full_prompt)
                return str(result)
            except Exception as e:
                logger.error(f"LLM Invocation failed: {e}")
                return self._mock_generation(data_context) # Fallback to mock
        else:
            return self._mock_generation(data_context)

    def _get_previous_period(self, period: str) -> str:
        """
        Utility to calculate previous quarter.
        """
        if "Q3" in period: return period.replace("Q3", "Q2")
        elif "Q2" in period: return period.replace("Q2", "Q1")
        elif "Q1" in period:
            year = int(period.split('-')[0]) - 1
            return f"{year}-Q4"
        elif "Q4" in period: return period.replace("Q4", "Q3")
        return period

    def _mock_generation(self, raw_data: str) -> str:
        """
        Fallback generator if LLM is offline.
        """
        return f"""
        ## Offline Analysis Mode
        System could not connect to Cognitive Kernel. 
        
        Data processed successfully:
        {raw_data[:500]}...
        
        (Please check LLM configuration)
        """