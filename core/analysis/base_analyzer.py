from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel

class BaseFinancialAnalyzer(ABC):
    """
    Abstract Base Class for Financial Analyzers.
    Ensures future alignment and interchangeability of analysis engines
    (e.g., swapping Gemini for GPT-4 or a local SLM).
    """

    @abstractmethod
    async def analyze_report(self, report_text: str, context: str = "") -> BaseModel:
        """
        Asynchronously analyzes a financial report text.

        Args:
            report_text (str): The raw text of the report.
            context (str): Additional context (e.g., company name, sector).

        Returns:
            BaseModel: A Pydantic model containing the structured analysis.
        """
        pass

    @abstractmethod
    async def analyze_image(self, image_path: str, context: str = "") -> Dict[str, Any]:
        """
        Asynchronously analyzes a financial image (chart, table).
        For future multimodal alignment.
        """
        pass
