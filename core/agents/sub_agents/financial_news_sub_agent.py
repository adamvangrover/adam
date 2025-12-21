import requests

from core.agents.agent_base import AgentBase


class FinancialNewsSubAgent(AgentBase):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        self.base_url = self.config.get("base_url")

    def execute(self, query):
        params = {
            "q": query,
            "apiKey": self.api_key,
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            return self._to_structured_output(data)
        except requests.exceptions.RequestException as e:
            self.log_error(f"Error fetching financial news: {e}")
            return self._to_error_output(e)

    def _to_structured_output(self, data):
        # Convert the raw API response to the standard metadata schema
        articles = []
        for article in data.get("articles", []):
            articles.append({
                "title": article.get("title"),
                "url": article.get("url"),
                "source": article.get("source", {}).get("name"),
            })
        return {
            "source_agent": self.__class__.__name__,
            "confidence_score": 0.9,
            "data": articles,
        }

    def _to_error_output(self, error):
        return {
            "source_agent": self.__class__.__name__,
            "confidence_score": 0.0,
            "error": str(error),
        }
