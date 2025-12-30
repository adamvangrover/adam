from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from core.llm_plugin import LLMPlugin
from core.analysis.base_analyzer import BaseFinancialAnalyzer
import logging
import asyncio
import json

# Configure logging
logger = logging.getLogger(__name__)

# --- Expanded Schema for Comprehensive Analysis ---

class RiskFactor(BaseModel):
    description: str = Field(..., description="Description of the risk factor.")
    severity: str = Field(..., description="Severity of the risk (High, Medium, Low).")
    time_horizon: str = Field(..., description="Estimated time horizon (Short, Medium, Long).")
    mitigation_strategy: Optional[str] = Field(None, description="Potential mitigation strategy mentioned or inferred.")

class FinancialInsight(BaseModel):
    category: str = Field(..., description="Category (Strategy, Operations, Market, Legal).")
    observation: str = Field(..., description="The specific observation or fact.")
    sentiment: str = Field(..., description="Sentiment (Positive, Negative, Neutral).")
    confidence: float = Field(..., description="Confidence score (0.0 - 1.0) in this insight.")

class CompetitorAnalysis(BaseModel):
    name: str = Field(..., description="Name of the competitor mentioned.")
    relationship: str = Field(..., description="Context of mention (e.g., gaining share, losing share, pricing pressure).")

class ESGMetrics(BaseModel):
    environmental_score: str = Field(..., description="Qualitative score (High/Med/Low) based on mentions of green initiatives vs violations.")
    social_score: str = Field(..., description="Qualitative score based on labor, diversity, and community mentions.")
    governance_score: str = Field(..., description="Qualitative score based on board structure, transparency, and ethics.")
    key_issues: List[str] = Field(default_factory=list, description="Specific ESG issues identified.")

class ReportAnalysis(BaseModel):
    executive_summary: str = Field(..., description="A concise executive summary of the report.")
    risk_factors: List[RiskFactor] = Field(default_factory=list, description="List of identified risk factors.")
    strategic_insights: List[FinancialInsight] = Field(default_factory=list, description="List of key strategic insights.")
    competitor_mentions: List[CompetitorAnalysis] = Field(default_factory=list, description="Competitors mentioned and the context.")
    esg_analysis: Optional[ESGMetrics] = Field(None, description="Environmental, Social, and Governance analysis.")
    management_sentiment: str = Field(..., description="Overall assessment of management sentiment/tone.")
    forward_looking_statements: List[str] = Field(default_factory=list, description="Key predictions or guidance provided by management.")

class GeminiFinancialReportAnalyzer(BaseFinancialAnalyzer):
    """
    Analyzer that leverages Google Gemini's long-context capabilities
    to extract deep insights from financial texts.

    Implements BaseFinancialAnalyzer for future alignment.
    """

    def __init__(self, llm_plugin: Optional[LLMPlugin] = None):
        """
        Initialize with an existing LLMPlugin or create a new one.
        """
        if llm_plugin:
            self.llm = llm_plugin
        else:
            # specialized config for Gemini if not provided
            config = {
                "provider": "gemini",
                "gemini_model_name": "gemini-1.5-pro"
            }
            try:
                self.llm = LLMPlugin(config=config)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini for analysis: {e}. Falling back to default config.")
                self.llm = LLMPlugin() # Fallback to whatever is default

    async def analyze_report(self, report_text: str, context: str = "") -> ReportAnalysis:
        """
        Asynchronously analyzes a financial report text.
        """
        if not report_text:
            return ReportAnalysis(
                executive_summary="No report text provided.",
                management_sentiment="Unknown"
            )

        # Advanced Prompting: Chain of Thought
        prompt = (
            f"You are a senior financial analyst. Your task is to perform a comprehensive 'Deep Dive' analysis "
            f"of the following financial report text.\n\n"
            f"Context: {context}\n\n"
            f"Report Text:\n{report_text}\n\n"
            f"Please follow this reasoning process:\n"
            f"1. Read the text thoroughly to understand the macro and micro context.\n"
            f"2. Identify specific risk factors, categorizing their severity and time horizon.\n"
            f"3. Extract strategic insights, assessing the sentiment and your confidence in them.\n"
            f"4. Look for mentions of competitors and analyze the competitive dynamic.\n"
            f"5. Evaluate ESG (Environmental, Social, Governance) factors mentioned.\n"
            f"6. Synthesize the management's tone and extract key forward-looking statements.\n"
            f"7. Summarize the findings into a structured format.\n\n"
            f"Provide the final output strictly adhering to the JSON schema for ReportAnalysis."
        )

        try:
            # Use structured generation
            # Note: generate_structured is currently synchronous in LLMPlugin.
            # We wrap it in asyncio.to_thread to prevent blocking the event loop
            # if the underlying API call is blocking (which it likely is in the sync plugin).

            loop = asyncio.get_running_loop()
            analysis, metadata = await loop.run_in_executor(
                None,
                lambda: self.llm.generate_structured(
                    prompt=prompt,
                    response_schema=ReportAnalysis,
                    thinking_level="high", # Hint to Gemini if supported
                    thought_signature="financial_deep_dive"
                )
            )

            logger.info(f"Gemini Analysis complete. Usage: {metadata.get('usage')}")
            return analysis

        except Exception as e:
            logger.error(f"Error during Gemini analysis: {e}")
            # Return empty/safe result with error note in summary
            return ReportAnalysis(
                executive_summary=f"Analysis failed due to an error: {str(e)}",
                management_sentiment="Error"
            )

    async def analyze_image(self, image_path: str, context: str = "") -> Dict[str, Any]:
        """
        Asynchronously analyzes a financial image (chart, table).
        For future multimodal alignment.
        """
        # Placeholder for actual implementation using Gemini Vision
        logger.info(f"Multimodal analysis requested for {image_path}. Feature pending.")
        return {"status": "not_implemented", "description": "Image analysis feature coming in v24.0"}

if __name__ == "__main__":
    # Test block
    logging.basicConfig(level=logging.INFO)
    print("--- Testing GeminiFinancialReportAnalyzer (Async/Comprehensive) ---")

    # Mock text
    mock_report = """
    Q3 2024 Earnings Call.
    We are pleased to report a 20% increase in revenue. However, supply chain headwinds persist in the Asia-Pacific region.
    We are doubling down on our AI strategy, investing $500M into new data centers.
    We see significant risk from regulatory changes in the EU, which could impact 10% of our business over the next 2 years.
    Our competitor, TechCorp, has aggressively lowered prices, putting pressure on our margins.
    We are committed to reducing our carbon footprint by 30% by 2030.
    Management remains cautiously optimistic about the fiscal year end and expects 15% growth next year.
    """

    async def run_test():
        analyzer = GeminiFinancialReportAnalyzer()
        # Note: This will likely use MockLLM if no API key is set in env
        result = await analyzer.analyze_report(mock_report, context="Tech Sector")
        print("\nResult:")
        print(result.model_dump_json(indent=2))

    asyncio.run(run_test())
