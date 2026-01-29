import random
from typing import List, Dict

class BSLPortfolioGenerator:
    """
    Generates a simulated Broadly Syndicated Loan (BSL) portfolio
    representing the "Market" for consensus analysis.
    """

    def __init__(self):
        self.sectors = [
            "Technology", "Healthcare", "Industrials", "Consumer Discretionary",
            "Energy", "Real Estate", "Financials", "Utilities"
        ]
        self.ratings = ["BB+", "BB", "BB-", "B+", "B", "B-", "CCC+"]

    def generate_portfolio(self, size: int = 50) -> List[Dict]:
        """
        Generates N simulated loan assets.
        """
        portfolio = []
        for i in range(size):
            sector = random.choice(self.sectors)
            rating = random.choices(self.ratings, weights=[10, 20, 20, 20, 15, 10, 5])[0]

            # Correlate spread with rating
            base_spread = 250
            rating_risk = self.ratings.index(rating) * 75
            spread = base_spread + rating_risk + random.randint(-25, 50)

            # Generate metrics
            ebitda = round(random.uniform(50, 500), 1) # $M
            leverage = round(random.uniform(3.5, 6.5), 1)
            # Adjust leverage based on sector (Tech/Real Estate higher)
            if sector in ["Technology", "Real Estate"]:
                leverage += 1.0

            debt_amt = round(ebitda * leverage, 1)

            asset = {
                "id": f"{sector[:3].upper()}-{random.randint(100, 999)}",
                "name": f"Simulated {sector} Co. {i+1}",
                "sector": sector,
                "rating": rating,
                "market_value": debt_amt, # Assuming par for simplicity of generating 'value'
                "leverage": leverage,
                "spread_bps": spread,
                "volatility": 0.1 + (self.ratings.index(rating) * 0.05) # Worse rating = higher vol
            }
            portfolio.append(asset)

        return portfolio
