import logging
import asyncio
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from core.llm_plugin import LLMPlugin
from core.interfaces.financial_analyzer import BaseFinancialAnalyzer, FinancialAnalysisResult

logger = logging.getLogger(__name__)

class ChartDataPoint(BaseModel):
    label: str = Field(..., description="The label on the x-axis or legend")
    value: float = Field(..., description="The numerical value")
    unit: str = Field(..., description="Unit of measurement (e.g., USD, Percentage)")

class ChartAnalysis(BaseModel):
    title: str = Field(..., description="Title of the chart")
    chart_type: str = Field(..., description="Type of chart (Bar, Line, Pie, Scatter)")
    key_trend: str = Field(..., description="The primary trend or insight visible")
    data_points: List[ChartDataPoint] = Field(..., description="Extracted raw data points")
    anomalies: List[str] = Field(default_factory=list, description="Any visual anomalies or outliers")

class MultimodalAnalyst(BaseFinancialAnalyzer):
    """
    Specialized analyst for extracting structured data from financial visuals.
    Part of the Adam v24.0 'Vision' suite.
    """

    def __init__(self, llm_plugin: LLMPlugin = None):
        self.llm = llm_plugin or LLMPlugin()

    async def analyze_report(self, report_text: str, context: Dict[str, Any] = None) -> FinancialAnalysisResult:
        """
        Stub implementation to satisfy BaseFinancialAnalyzer interface.
        MultimodalAnalyst primarily focuses on images, not full text reports.
        """
        return FinancialAnalysisResult(
            ticker=context.get("ticker", "UNKNOWN"),
            period=context.get("period", "UNKNOWN"),
            risk_factors=[],
            strategic_insights=[],
            esg_metrics=[],
            competitor_dynamics=[],
            forward_guidance=[],
            summary_narrative="MultimodalAnalyst does not process text reports."
        )

    async def analyze_multimodal(self, image_paths: List[str], prompt: str) -> str:
        """
        Base interface implementation. Returns string summary of images.
        """
        results = []
        for path in image_paths:
            chart_data = await self.analyze_chart(path, context=prompt)
            results.append(f"Chart: {chart_data.title}\nTrend: {chart_data.key_trend}")
        return "\n".join(results)

    async def analyze_chart(self, image_path: str, context: str = "") -> ChartAnalysis:
        """
        Extracts structured data from a chart image.
        """
        prompt = f"""
        Analyze this financial chart.
        Context: {context}

        Task:
        1. Identify the chart type and title.
        2. Describe the key trend.
        3. EXTRACT the raw data points as accurately as possible.
        4. Note any visual anomalies.

        Output strictly as JSON matching the ChartAnalysis schema.
        """

        try:
            # We use the generate_multimodal method
            response_text = self.llm.generate_multimodal(
                prompt=prompt + "\n\nResponse must be valid JSON.",
                image_path=image_path
            )

            # Robust Parsing
            clean_text = response_text.strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()

            data = json.loads(clean_text)
            return ChartAnalysis(**data)

        except Exception as e:
            logger.error(f"Chart Analysis failed for {image_path}: {e}")
            # Return a fallback object
            return ChartAnalysis(
                title="Analysis Failed",
                chart_type="Unknown",
                key_trend=f"Error: {e}",
                data_points=[],
                anomalies=[]
            )

    async def analyze_satellite_image(self, image_path: str, target_type: str) -> Dict[str, Any]:
        """
        Experimental: Analyzes satellite imagery for economic indicators.
        e.g., Counting cars in a retail parking lot.
        """
        prompt = f"""
        Analyze this satellite image.
        Target: Count the number of {target_type} visible.
        Estimate the utilization rate of the facility.

        Output JSON: {{ "count": int, "utilization_rate": float, "confidence": float }}
        """

        try:
            response_text = self.llm.generate_multimodal(prompt, image_path)

            clean_text = response_text.strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()

            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"Satellite Analysis failed: {e}")
            return {"error": str(e)}
