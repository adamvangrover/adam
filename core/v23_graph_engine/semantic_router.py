from typing import List, Dict, Optional, Callable, Any
from enum import Enum
import re

class IntentCategory(Enum):
    """
    Categories of user intent.
    """
    FINANCIAL_ANALYSIS = "financial_analysis"
    MARKET_DATA = "market_data"
    GENERAL_KNOWLEDGE = "general_knowledge"
    UNKNOWN = "unknown"

class SemanticRouter:
    """
    Routes user queries to the appropriate agent or tool based on semantic intent.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the SemanticRouter.
        """
        self.routes: Dict[IntentCategory, Any] = {}
        # In a real implementation, we would load the model here
        # self.model = SentenceTransformer(model_name)
        self.model = None

    def add_route(self, category: IntentCategory, handler: Callable):
        """
        Adds a route for a specific intent category.
        """
        self.routes[category] = handler

    def route(self, query: str) -> Callable:
        """
        Determines the intent of the query and returns the appropriate handler.
        """
        intent = self._determine_intent(query)
        handler = self.routes.get(intent)

        if handler:
            return handler
        else:
            return self._default_handler

    def _determine_intent(self, query: str) -> IntentCategory:
        """
        Uses heuristics or a model to determine the intent of the query.
        """
        query_lower = query.lower()

        if any(keyword in query_lower for keyword in ["price", "stock", "market", "trade", "volume"]):
            return IntentCategory.MARKET_DATA
        elif any(keyword in query_lower for keyword in ["analyze", "report", "fundamental", "financials", "balance sheet"]):
            return IntentCategory.FINANCIAL_ANALYSIS
        elif any(keyword in query_lower for keyword in ["what is", "who is", "explain"]):
            return IntentCategory.GENERAL_KNOWLEDGE
        else:
            return IntentCategory.UNKNOWN

    def _default_handler(self, query: str) -> str:
        """
        Default handler for unknown intents.
        """
        return f"I'm not sure how to handle that query. (Intent: UNKNOWN)"

# Example usage
if __name__ == "__main__":
    router = SemanticRouter()

    def market_data_handler(query):
        return f"Handling market data query: {query}"

    def analysis_handler(query):
        return f"Handling analysis query: {query}"

    router.add_route(IntentCategory.MARKET_DATA, market_data_handler)
    router.add_route(IntentCategory.FINANCIAL_ANALYSIS, analysis_handler)

    print(router.route("What is the stock price of AAPL?")("What is the stock price of AAPL?"))
    print(router.route("Analyze the Q3 earnings report for TSLA")("Analyze the Q3 earnings report for TSLA"))
