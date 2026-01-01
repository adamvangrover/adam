import logging
import asyncio
import json
from typing import Dict, Any, List, Optional

from core.interfaces.financial_analyzer import (
    BaseFinancialAnalyzer,
    FinancialAnalysisResult,
    RiskFactor,
    StrategicInsight,
    ESGMetric,
    CompetitorDynamic,
    ForwardGuidance,
    SupplyChainNode,
    GeopoliticalExposure,
    TechnologicalMoat
)
from core.llm_plugin import LLMPlugin, LLMAPIError

logger = logging.getLogger(__name__)

class GeminiFinancialReportAnalyzer(BaseFinancialAnalyzer):
    """
    Alphabet Ecosystem Analyzer using Google's Gemini 1.5 Pro.
    Leverages huge context window, native multimodal capabilities,
    and structured generation for deep qualitative analysis.
    """

    def __init__(self, llm_plugin: LLMPlugin):
        self.llm = llm_plugin
        # Ensure we are using a Gemini model for best results, but allow fallback
        if "gemini" not in self.llm.get_model_name().lower() and "mock" not in self.llm.get_model_name().lower():
             logger.warning(f"GeminiFinancialReportAnalyzer initialized with non-Gemini model: {self.llm.get_model_name()}. Features may be limited.")

    async def analyze_report(self, report_text: str, context: Dict[str, Any] = None) -> FinancialAnalysisResult:
        """
        Asynchronously analyzes a financial report using Chain-of-Thought prompting
        and structured JSON output.
        """
        context = context or {}
        ticker = context.get("ticker", "UNKNOWN")
        period = context.get("period", "UNKNOWN")

        logger.info(f"Starting Gemini analysis for {ticker} ({period})...")

        # 1. Construct the Chain-of-Thought Prompt
        # We force the model to reason first before emitting JSON.
        prompt = f"""
        You are an expert Principal Financial Analyst for 'Adam', an advanced AI investment system.
        Your task is to perform a deep qualitative analysis of the following financial report for {ticker}.

        Do not just summarize. Extract actionable intelligence.

        REPORT TEXT (Snippet):
        {report_text[:100000]}... (truncated if too long, though Gemini 1.5 handles 1M+ tokens)

        ---
        ANALYSIS INSTRUCTIONS:

        1. **Risk Factors**: Identify specific risks. Categorize by severity (Low/Medium/High/Critical) and time horizon. Look for "hidden" risks not explicitly flagged as risk factors.
        2. **Strategic Insights**: What is management's core strategy? How confident are they? (Score 0.0-1.0).
        3. **ESG Metrics**: Evaluate Environmental, Social, and Governance factors qualitatively.
        4. **Competitor Dynamics**: Who are they worried about? What is their market positioning?
        5. **Forward Guidance**: Extract concrete numbers and predictions.

        DEEP RESEARCH TOPICS (v24.0):
        6. **Supply Chain Mapping**: Extract key suppliers, manufacturing hubs, and logistics partners. Assess their risk.
        7. **Geopolitical Exposure**: Identify revenue or assets exposed to specific regions (China, Russia, Middle East, etc.) and the nature of the risk.
        8. **Technological Moat**: Analyze the durability of their IP and technology stack. Is it a wide or narrow moat?

        Thinking Process:
        First, silently reason through the text. Connect dots between the CEO's letter and the Risk Factors section.
        Then, generate the output strictly in the requested JSON format.
        """

        # 2. Async Execution Wrapper
        # Since LLMPlugin is synchronous (standard python `requests` or `google-generativeai` blocking calls),
        # we offload to a thread to keep the agent loop non-blocking.
        try:
            loop = asyncio.get_running_loop()
            # We use a helper to call the synchronous generate_structured
            result, metadata = await loop.run_in_executor(
                None,
                lambda: self.llm.generate_structured(
                    prompt=prompt,
                    response_schema=FinancialAnalysisResult,
                    thinking_level="high", # Hint to the underlying plugin if supported
                    thought_signature=f"analyze_{ticker}_{period}"
                )
            )

            # Post-processing: inject known context if missing
            if result.ticker == "UNKNOWN": result.ticker = ticker
            if result.period == "UNKNOWN": result.period = period

            logger.info(f"Gemini Analysis completed for {ticker}. Thought Signature: {metadata.get('thought_signature')}")
            return result

        except Exception as e:
            logger.error(f"Gemini Analysis Failed: {e}")
            # Return empty/safe result on failure
            return FinancialAnalysisResult(
                ticker=ticker,
                period=period,
                risk_factors=[],
                strategic_insights=[],
                esg_metrics=[],
                competitor_dynamics=[],
                forward_guidance=[],
                supply_chain_map=[],
                geopolitical_exposure=[],
                technological_moat=[],
                summary_narrative=f"Analysis failed: {str(e)}"
            )

    async def analyze_multimodal(self, image_paths: List[str], prompt: str) -> str:
        """
        Uses Gemini Vision to analyze charts or tables in the report.
        """
        logger.info(f"Starting Gemini Multimodal Analysis on {len(image_paths)} images...")

        combined_analysis = []
        loop = asyncio.get_running_loop()

        for path in image_paths:
            try:
                # Parallelize? For now, sequential async
                response = await loop.run_in_executor(
                    None,
                    lambda p=path: self.llm.generate_multimodal(prompt, p)
                )
                combined_analysis.append(f"Image {path}: {response}")
            except Exception as e:
                logger.error(f"Failed to analyze image {path}: {e}")
                combined_analysis.append(f"Image {path}: Error - {e}")

        return "\n\n".join(combined_analysis)

# --- Integration Helper ---

def get_gemini_analyzer() -> GeminiFinancialReportAnalyzer:
    """Factory to get a configured analyzer instance."""
    # Assumes LLMPlugin is configured via env vars or default config
    plugin = LLMPlugin()
    return GeminiFinancialReportAnalyzer(plugin)
