# core/data_sources/prediction_market_api.py
import logging


class SimulatedPredictionMarketAPI:
    def get_market_data(self, keywords=None):
        logging.info(f"Fetching SIMULATED prediction market data. Keywords: {keywords}")
        return [
            {"market": f"Market for {keywords or 'an event'}", "probability": 0.6},
        ]

    def get_market_sentiment(self):
        logging.info("Fetching SIMULATED prediction market sentiment.")
        return 0.6
