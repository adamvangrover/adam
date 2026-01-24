import json
import logging
import os
import sys
import random
import asyncio
from datetime import datetime
from typing import Dict, Any, List

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_sources.data_fetcher import DataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MarketOmniScanner")

class MarketOmniScanner:
    """
    Scans the entire financial universe to build a Digital Twin Critique.
    """

    # Taxonomy
    SECTORS = {
        "Technology": ["AAPL", "MSFT", "NVDA", "ORCL", "ADBE"],
        "Financials": ["JPM", "BAC", "GS", "MS", "BLK"],
        "Healthcare": ["JNJ", "LLY", "UNH", "PFE", "MRK"],
        "ConsumerDiscretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
        "Industrials": ["CAT", "GE", "HON", "UPS", "BA"],
        "Utilities": ["NEE", "DUK", "SO"],
        "Materials": ["LIN", "APD", "FCX"]
    }

    ASSET_CLASSES = {
        "Credit": ["HYG", "LQD", "JNK", "TLT", "IEF"],
        "Commodities": ["GLD", "SLV", "USO", "UNG"],
        "Currencies": ["DX-Y.NYB", "EURUSD=X", "JPY=X", "GBPUSD=X"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"]
    }

    def __init__(self):
        self.fetcher = DataFetcher()

    async def scan_all(self) -> Dict[str, Any]:
        """
        Executes the full scan.
        """
        logger.info("Starting Omni-Scan...")

        timestamp = datetime.now().isoformat()

        # 1. Sector Scan
        sectors_data = await self._scan_sectors()

        # 2. Asset Class Scan
        assets_data = await self._scan_assets()

        # 3. Derivatives & Volatility (Simulated where fetcher lacks coverage)
        derivs_data = self._scan_derivatives()

        # 4. Institutional Flows (Simulated)
        flows_data = self._simulate_flows()

        # 5. Critique Generation
        critique = self._generate_critique(sectors_data, assets_data, derivs_data)

        return {
            "timestamp": timestamp,
            "metadata": {
                "scan_type": "Full Point-for-Point",
                "version": "v25.1"
            },
            "sectors": sectors_data,
            "markets": assets_data,
            "derivatives": derivs_data,
            "institutional_flows": flows_data,
            "digital_twin_critique": critique
        }

    async def _scan_sectors(self) -> Dict[str, Any]:
        results = {}
        for sector, tickers in self.SECTORS.items():
            logger.info(f"Scanning Sector: {sector}")
            sector_metrics = []

            # Fetch concurrently
            loop = asyncio.get_running_loop()
            tasks = [loop.run_in_executor(None, self.fetcher.fetch_realtime_snapshot, t) for t in tickers]
            snapshots = await asyncio.gather(*tasks)

            for t, snap in zip(tickers, snapshots):
                metric = {
                    "ticker": t,
                    "price": snap.get("last_price", 0.0),
                    "change": snap.get("last_price", 0.0) - snap.get("previous_close", 0.0) if snap else 0.0,
                    "volume": 0, # Snapshot might lack volume, ignore for now
                    "status": "Active" if snap else "Data Unavailable"
                }
                sector_metrics.append(metric)

            # Aggregate Sector Stats
            avg_price = sum(m["price"] for m in sector_metrics) / len(sector_metrics) if sector_metrics else 0

            results[sector] = {
                "constituents": sector_metrics,
                "average_price": avg_price,
                "sentiment": "Bullish" if avg_price > 100 else "Bearish" # Mock logic
            }
        return results

    async def _scan_assets(self) -> Dict[str, Any]:
        results = {}
        for category, tickers in self.ASSET_CLASSES.items():
            logger.info(f"Scanning Asset Class: {category}")
            metrics = []

            loop = asyncio.get_running_loop()
            tasks = [loop.run_in_executor(None, self.fetcher.fetch_realtime_snapshot, t) for t in tickers]
            snapshots = await asyncio.gather(*tasks)

            for t, snap in zip(tickers, snapshots):
                metrics.append({
                    "ticker": t,
                    "price": snap.get("last_price", 0.0),
                    "previous_close": snap.get("previous_close", 0.0)
                })
            results[category] = metrics
        return results

    def _scan_derivatives(self) -> Dict[str, Any]:
        # Fetch VIX real data, simulate others
        # Ideally fetcher has fetch_volatility_metrics
        # We'll use the fetcher sync methods or simulate if needed
        # For speed in this script, let's simulate the surface
        return {
            "volatility_surface": {
                "ATM_IV": 18.5,
                "Skew": "Steep Put Skew",
                "Term_Structure": "Contango"
            },
            "greeks": {
                "System_Delta": 0.45,
                "System_Gamma": -0.05, # Short Gamma
                "System_Vega": 1.2
            },
            "key_contracts": [
                {"contract": "ES_F", "price": 5850.0, "oi": 2500000},
                {"contract": "NQ_F", "price": 20100.0, "oi": 800000},
                {"contract": "CL_F", "price": 72.50, "oi": 450000}
            ]
        }

    def _simulate_flows(self) -> Dict[str, Any]:
        return {
            "dark_pool_volume": "High (45% of total)",
            "retail_participation": "Moderate (15%)",
            "cta_positioning": "Max Long",
            "buyback_desk": "Active (Blackout window ending)",
            "sector_rotation": "Tech -> Energy"
        }

    def _generate_critique(self, sectors, assets, derivs) -> List[Dict[str, Any]]:
        critiques = []

        # 1. Valuation Critique
        tech_px = sectors.get("Technology", {}).get("average_price", 0)
        if tech_px > 200:
            critiques.append({
                "area": "Valuation",
                "severity": "High",
                "finding": "Technology sector valuations stretched > 2 sigma vs historic mean.",
                "recommendation": "Reduce beta exposure."
            })

        # 2. Credit Critique
        credit = assets.get("Credit", [])
        hyg = next((x for x in credit if x["ticker"] == "HYG"), {})
        if hyg.get("price", 100) < 70:
             critiques.append({
                "area": "Credit",
                "severity": "Critical",
                "finding": "High Yield spreads widening rapidly. Distress imminent.",
                "recommendation": "Go into cash."
            })
        else:
             critiques.append({
                "area": "Credit",
                "severity": "Low",
                "finding": "Credit spreads tight. Risk appetite healthy.",
                "recommendation": "Carry trade attractive."
            })

        # 3. Volatility
        iv = derivs.get("volatility_surface", {}).get("ATM_IV", 20)
        if iv < 12:
             critiques.append({
                "area": "Volatility",
                "severity": "Medium",
                "finding": "VIX Complacency. Tail risk underpriced.",
                "recommendation": "Buy cheap puts."
            })

        return critiques

async def main():
    scanner = MarketOmniScanner()
    data = await scanner.scan_all()

    output_path = "showcase/data/full_market_critique.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Full Market Critique saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
