from core.agents.agent_base import AgentBase
from textblob import TextBlob

class SentimentAnalysisMetaAgent(AgentBase):
    def __init__(self, config):
        super().__init__(config)

    def execute(self, sub_agent_output):
        if sub_agent_output.get("error"):
            return self._to_error_output(sub_agent_output.get("error"))

        articles = sub_agent_output.get("data", [])
        if not articles:
            return self._to_error_output("No articles found")

        sentiments = []
        for article in articles:
            sentiment = self._analyze_sentiment(article.get("title"))
            sentiments.append({
                "title": article.get("title"),
                "sentiment": sentiment,
            })
        return self._to_structured_output(sentiments)

    def _analyze_sentiment(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def _to_structured_output(self, sentiments):
        return {
            "source_agent": self.__class__.__name__,
            "confidence_score": 0.8,
            "data": sentiments,
        }

    def _to_error_output(self, error):
        return {
            "source_agent": self.__class__.__name__,
            "confidence_score": 0.0,
            "error": str(error),
        }
