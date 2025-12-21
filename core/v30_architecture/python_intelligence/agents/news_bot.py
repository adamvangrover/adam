import logging

logger = logging.getLogger("NewsBot")

class NewsBotAgent:
    """
    Appendix B: NewsBot Sentiment Analysis Pipeline
    """
    def __init__(self):
        self.watchlist = ["AAPL", "TSLA", "MSFT"]

    def run_cycle(self):
        """
        Ingestion -> Filtering -> Scoring -> Correlation -> Action
        """
        logger.info("NewsBot: Scanning RSS feeds and APIs...")

        # 1. Ingestion (Mock)
        raw_feed = [
            {"title": "Apple releases revolutionary AR glasses", "content": "...", "source": "TechCrunch"},
            {"title": "Tesla faces supply chain delays in Berlin", "content": "...", "source": "Reuters"}
        ]

        # 2. Filtering & Scoring (Mock FinBERT)
        processed_items = []
        for item in raw_feed:
            # 3. Scoring
            sentiment = self._mock_finbert(item['title'])

            # 4. Correlation (NER)
            symbols = self._extract_symbols(item['title'])

            for sym in symbols:
                if sym in self.watchlist:
                    logger.info(f"NewsBot: Processed {sym} sentiment: {sentiment}")
                    if sentiment < -0.6:
                         # 5. Action
                         logger.warning(f"ALERT: High Negative Sentiment for {sym}: {item['title']}")
                         processed_items.append({"symbol": sym, "sentiment": sentiment, "alert": True})

        return processed_items

    def _mock_finbert(self, text):
        if "delay" in text.lower(): return -0.75
        if "release" in text.lower(): return 0.8
        return 0.0

    def _extract_symbols(self, text):
        symbols = []
        if "Apple" in text: symbols.append("AAPL")
        if "Tesla" in text: symbols.append("TSLA")
        return symbols
