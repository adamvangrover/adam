# scripts/generate_credit_memos.py

import json
import argparse
from typing import Dict, Any, List
import os
import sys
import logging

# Add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data.realtime_fetcher import RealtimeFetcher

from core.agents.orchestrators.credit_memo_orchestrator import CreditMemoOrchestrator

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MemoFactory")

# --- 1. Comprehensive Data Library (Mock) ---

MOCK_LIBRARY = {
    "Apple_Inc": {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "description": "Global leader in consumer electronics, software, and services.",
        "historical": {
            "revenue": [365000, 394000, 383000],  # 2021, 2022, 2023
            "ebitda": [120000, 130000, 125000],
            "net_income": [94000, 99000, 96000],
            "total_assets": [351000, 352000, 352000],
            "total_liabilities": [287000, 302000, 290000],
            "total_debt": [120000, 110000, 105000],
            "cash": [60000, 50000, 60000],
            "interest_expense": [2800, 3000, 3200],
            "capex": [11000, 12000, 10000],
            "year": [2021, 2022, 2023],
        },
        "forecast_assumptions": {
            "revenue_growth": 0.05,
            "ebitda_margin": 0.32,
            "discount_rate": 0.09,
            "terminal_growth_rate": 0.03,
        },
        "market_data": {
            "share_price": 185.00,
            "market_cap": 2800000,
            "beta": 1.2,
            "pe_ratio": 29.5,
            "price_data": [150, 155, 160, 175, 180, 185],  # Simplified for vol calc
        },
        "docs": {
            "10-K": "Competition is intense. Global supply chain risks are significant. We rely on third-party manufacturing. Regulatory scrutiny in EU and US is increasing regarding App Store practices.",
            "Credit_Agreement": "Standard LSTA terms. Negative Pledge on all significant assets. Cross-Default threshold $100M. Change of Control trigger at 50% ownership change. No financial maintenance covenants.",
        },
    },
    "Tesla_Inc": {
        "ticker": "TSLA",
        "name": "Tesla Inc.",
        "sector": "Consumer Cyclical",
        "description": "Electric vehicle and clean energy company.",
        "historical": {
            "revenue": [53000, 81000, 96000],
            "ebitda": [9000, 17000, 15000],
            "net_income": [5500, 12500, 10000],
            "total_assets": [62000, 82000, 90000],
            "total_liabilities": [30000, 36000, 38000],
            "total_debt": [5000, 4000, 3000],
            "cash": [17000, 22000, 25000],
            "interest_expense": [300, 200, 150],
            "capex": [6000, 7000, 8000],
            "year": [2021, 2022, 2023],
        },
        "forecast_assumptions": {
            "revenue_growth": 0.15,
            "ebitda_margin": 0.18,
            "discount_rate": 0.12,
            "terminal_growth_rate": 0.04,
        },
        "market_data": {
            "share_price": 240.00,
            "market_cap": 750000,
            "beta": 2.1,
            "pe_ratio": 65.0,
            "price_data": [200, 210, 250, 220, 230, 240],
        },
        "docs": {
            "10-K": "Highly dependent on key personnel (Elon Musk). Regulatory risks regarding FSD technology. Raw material price volatility (Lithium, Nickel). Competition from legacy auto increasing.",
            "Credit_Agreement": "Asset Sale Sweep: 100% of net proceeds must prepay term loan. Negative Pledge. Restricted Payments basket limited to $500M per annum.",
        },
    },
    "Carvana_Co": {
        "ticker": "CVNA",
        "name": "Carvana Co.",
        "sector": "Consumer Cyclical (Distressed)",
        "description": "E-commerce platform for buying and selling used cars.",
        "historical": {
            "revenue": [12800, 13600, 11500],
            "ebitda": [-500, -800, 100],  # Turned corner slightly?
            "net_income": [-1000, -1500, -500],
            "total_assets": [8000, 7000, 6500],
            "total_liabilities": [9000, 9500, 9200],
            "total_debt": [6000, 6500, 6300],
            "cash": [500, 300, 400],
            "interest_expense": [400, 600, 650],
            "capex": [200, 100, 50],
            "year": [2021, 2022, 2023],
        },
        "forecast_assumptions": {
            "revenue_growth": 0.02,
            "ebitda_margin": 0.03,
            "discount_rate": 0.15,
            "terminal_growth_rate": 0.01,
        },
        "market_data": {
            "share_price": 45.00,
            "market_cap": 8000,
            "beta": 3.5,
            "pe_ratio": -15.0,
            "price_data": [30, 35, 40, 38, 42, 45],
        },
        "docs": {
            "10-K": "Substantial doubt about ability to continue as going concern. High debt load. Interest rates impact demand significantly. Inventory valuation risks.",
            "Credit_Agreement": "Financial Covenant Breach likely if EBITDA drops below $50M. Cross-Default trigger. Springing maturity if 2025 notes not refinanced. PIK toggle available on unsecured notes.",
        },
    },
    "JPMorgan_Chase": {
        "ticker": "JPM",
        "name": "JPMorgan Chase & Co.",
        "sector": "Financial Services",
        "description": "Leading global financial services firm.",
        "historical": {
            "revenue": [121000, 128000, 150000],
            "ebitda": [45000, 40000, 50000],  # PPNR approx
            "net_income": [48000, 37000, 49000],
            "total_assets": [3700000, 3700000, 3900000],
            "total_liabilities": [3400000, 3400000, 3600000],
            "total_debt": [300000, 350000, 400000],  # Long term debt
            "cash": [500000, 600000, 550000],  # Cash & securities
            "interest_expense": [10000, 20000, 40000],
            "capex": [0, 0, 0],  # N/A for banks usually
            "year": [2021, 2022, 2023],
        },
        "forecast_assumptions": {
            "revenue_growth": 0.03,
            "ebitda_margin": 0.35,
            "discount_rate": 0.10,
            "terminal_growth_rate": 0.02,
        },
        "market_data": {
            "share_price": 170.00,
            "market_cap": 490000,
            "beta": 1.1,
            "pe_ratio": 10.5,
            "price_data": [140, 150, 160, 165, 168, 170],
        },
        "docs": {
            "10-K": "Regulatory capital requirements (Basel III endgame). Credit risk in CRE portfolio. Cyber security threats. Geopolitical instability impacting global markets.",
            "Credit_Agreement": "Standard bank debt. No financial maintenance covenants. TLAC requirements applicable.",
        },
    },
    "NVIDIA_Corp": {
        "ticker": "NVDA",
        "name": "NVIDIA Corp.",
        "sector": "Technology",
        "description": "Leader in GPU and AI computing.",
        "historical": {
            "revenue": [26900, 27000, 60900],
            "ebitda": [11000, 7000, 35000],
            "net_income": [9700, 4300, 29700],
            "total_assets": [44000, 41000, 65000],
            "total_liabilities": [17000, 19000, 22000],
            "total_debt": [10900, 11000, 10000],
            "cash": [21000, 13000, 26000],
            "interest_expense": [230, 270, 250],
            "capex": [1000, 1500, 2000],
            "year": [2021, 2022, 2023],
        },
        "forecast_assumptions": {
            "revenue_growth": 0.25,
            "ebitda_margin": 0.55,
            "discount_rate": 0.11,
            "terminal_growth_rate": 0.04,
        },
        "market_data": {
            "share_price": 850.00,
            "market_cap": 2100000,
            "beta": 1.8,
            "pe_ratio": 70.0,
            "price_data": [500, 600, 700, 750, 800, 850],
        },
        "docs": {
            "10-K": "Export controls to China affect sales. Supply chain constraints at TSMC. AI regulation risks. High valuation expectations.",
            "Credit_Agreement": "Revolving credit facility $3B. Unsecured. Negative Pledge. Cross-Default $250M.",
        },
    },
    "Exxon_Mobil": {
        "ticker": "XOM",
        "name": "Exxon Mobil Corp.",
        "sector": "Energy",
        "description": "International energy and chemical company.",
        "historical": {
            "revenue": [285000, 413000, 344000],
            "ebitda": [45000, 80000, 60000],
            "net_income": [23000, 55000, 36000],
            "total_assets": [338000, 369000, 376000],
            "total_liabilities": [163000, 168000, 165000],
            "total_debt": [47000, 40000, 37000],
            "cash": [6800, 29000, 31000],
            "interest_expense": [1500, 1400, 1200],
            "capex": [16000, 22000, 25000],
            "year": [2021, 2022, 2023],
        },
        "forecast_assumptions": {
            "revenue_growth": 0.02,
            "ebitda_margin": 0.18,
            "discount_rate": 0.08,
            "terminal_growth_rate": 0.01,
        },
        "market_data": {
            "share_price": 105.00,
            "market_cap": 420000,
            "beta": 0.9,
            "pe_ratio": 11.5,
            "price_data": [100, 102, 98, 104, 106, 105],
        },
        "docs": {
            "10-K": "Commodity price volatility. Climate change regulations and transition risks. Geopolitical risks in operation zones.",
            "Credit_Agreement": "Clean balance sheet. No significant restrictive covenants. Access to commercial paper markets.",
        },
    },
    "Meta_Platforms": {
        "ticker": "META",
        "name": "Meta Platforms",
        "sector": "Technology",
        "description": "Social technology company.",
        "historical": {
            "revenue": [117000, 116000, 134000],
            "ebitda": [54000, 40000, 60000],
            "net_income": [39000, 23000, 39000],
            "total_assets": [165000, 185000, 229000],
            "total_liabilities": [41000, 60000, 76000],
            "total_debt": [14000, 26000, 37000],  # Increasing debt for buybacks/capex
            "cash": [48000, 40000, 65000],
            "interest_expense": [0, 100, 500],
            "capex": [19000, 32000, 28000],
            "year": [2021, 2022, 2023],
        },
        "forecast_assumptions": {
            "revenue_growth": 0.10,
            "ebitda_margin": 0.45,
            "discount_rate": 0.10,
            "terminal_growth_rate": 0.03,
        },
        "market_data": {
            "share_price": 480.00,
            "market_cap": 1200000,
            "beta": 1.4,
            "pe_ratio": 28.0,
            "price_data": [350, 400, 450, 460, 470, 480],
        },
        "docs": {
            "10-K": "Regulatory headwinds (GDPR, antitrust). Reliance on advertising revenue. Reality Labs losses.",
            "Credit_Agreement": "Investment grade documentation. Negative Pledge. Priority Debt basket.",
        },
    },
    "AMC_Entertainment": {
        "ticker": "AMC",
        "name": "AMC Entertainment",
        "sector": "Communication Services (Distressed)",
        "description": "Theatrical exhibition company.",
        "historical": {
            "revenue": [2500, 3900, 4800],
            "ebitda": [-300, 46, 400],
            "net_income": [-1200, -900, -300],
            "total_assets": [10000, 9500, 9000],
            "total_liabilities": [12000, 11800, 11500],
            "total_debt": [5500, 5200, 4800],
            "cash": [1500, 600, 800],
            "interest_expense": [400, 380, 350],
            "capex": [100, 150, 200],
            "year": [2021, 2022, 2023],
        },
        "forecast_assumptions": {
            "revenue_growth": 0.01,
            "ebitda_margin": 0.10,
            "discount_rate": 0.14,
            "terminal_growth_rate": 0.00,
        },
        "market_data": {
            "share_price": 4.50,
            "market_cap": 1500,
            "beta": 2.8,
            "pe_ratio": -5.0,
            "price_data": [6, 5, 4, 4.2, 4.4, 4.5],
        },
        "docs": {
            "10-K": "Going concern doubts. Negative working capital. Dilution risk from equity raises. Box office recovery uncertain.",
            "Credit_Agreement": "Heavily encumbered assets. First Lien/Second Lien structure. Restricted Payments completely blocked. Financial Covenants suspended but springing back soon.",
        },
    },
    "Pfizer_Inc": {
        "ticker": "PFE",
        "name": "Pfizer Inc.",
        "sector": "Healthcare",
        "description": "Biopharmaceutical company.",
        "historical": {
            "revenue": [81000, 100000, 58000],  # Covid cliff
            "ebitda": [30000, 40000, 18000],
            "net_income": [22000, 31000, 2100],
            "total_assets": [181000, 197000, 226000],  # Seagen acquisition
            "total_liabilities": [105000, 101000, 137000],
            "total_debt": [33000, 35000, 65000],  # Debt up for M&A
            "cash": [31000, 22000, 12000],
            "interest_expense": [1200, 1300, 2500],
            "capex": [2000, 3000, 4000],
            "year": [2021, 2022, 2023],
        },
        "forecast_assumptions": {
            "revenue_growth": 0.03,
            "ebitda_margin": 0.35,
            "discount_rate": 0.08,
            "terminal_growth_rate": 0.02,
        },
        "market_data": {
            "share_price": 28.00,
            "market_cap": 158000,
            "beta": 0.6,
            "pe_ratio": 15.0,
            "price_data": [35, 32, 30, 29, 28.5, 28],
        },
        "docs": {
            "10-K": "Patent cliffs on key drugs. Integration risks of recent acquisitions. Pricing pressure from government regulation (IRA).",
            "Credit_Agreement": "Acquisition financing bridge loan. Mandatory prepayment from bond issuance. Leverage covenant < 4.5x temporarily stepped up.",
        },
    },
    "Walmart_Inc": {
        "ticker": "WMT",
        "name": "Walmart Inc.",
        "sector": "Consumer Defensive",
        "description": "Retail giant.",
        "historical": {
            "revenue": [572000, 611000, 648000],
            "ebitda": [36000, 33000, 38000],
            "net_income": [13000, 11000, 15000],
            "total_assets": [244000, 243000, 250000],
            "total_liabilities": [152000, 153000, 155000],
            "total_debt": [45000, 49000, 46000],
            "cash": [14000, 8000, 9000],
            "interest_expense": [2200, 2500, 2700],
            "capex": [13000, 16000, 18000],
            "year": [2021, 2022, 2023],
        },
        "forecast_assumptions": {
            "revenue_growth": 0.03,
            "ebitda_margin": 0.06,
            "discount_rate": 0.07,
            "terminal_growth_rate": 0.02,
        },
        "market_data": {
            "share_price": 60.00,
            "market_cap": 480000,
            "beta": 0.5,
            "pe_ratio": 25.0,
            "price_data": [50, 52, 55, 58, 59, 60],
        },
        "docs": {
            "10-K": "Labor cost inflation. Supply chain efficiency. Competition from Amazon. FX headwinds.",
            "Credit_Agreement": "High grade. Very flexible terms.",
        },
    },
}

# --- 2. Pipeline Wrapper ---


class CreditMemoPipeline:
    """Wrapper for the Credit Memo Orchestrator to run the full library."""

    def __init__(self, output_dir="showcase/data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.orchestrator = CreditMemoOrchestrator(
            mock_library=MOCK_LIBRARY, output_dir=output_dir
        )
        self.library_index = []
        self.interaction_logs = {}

    def fetch_live_data(self, ticker: str, base_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetches live data while preserving the hand-crafted description and docs."""
        fetcher = RealtimeFetcher()
        logger.info(f"Fetching live data for {ticker}...")

        hist = fetcher.fetch_historical_data(ticker)
        market = fetcher.fetch_market_data(ticker)
        forecast = fetcher.fetch_forecast_assumptions(ticker)

        # Create a new dict preserving base details
        live_data = {
            "ticker": ticker,
            "name": base_data.get("name", ticker),
            "sector": base_data.get("sector", "Unknown"),
            "description": base_data.get("description", "No description available."),
            "docs": base_data.get("docs", {}),
        }

        if hist:
            # Drop auxiliary keys like _price_history for ICAT
            if "_price_history" in hist:
                del hist["_price_history"]
            live_data["historical"] = hist
        else:
            live_data["historical"] = base_data.get("historical", {})

        if market:
            live_data["market_data"] = market
        else:
            live_data["market_data"] = base_data.get("market_data", {})

        if forecast:
            live_data["forecast_assumptions"] = forecast
        else:
            live_data["forecast_assumptions"] = base_data.get(
                "forecast_assumptions", {}
            )

        return live_data

    def run_pipeline(self, live: bool = False, target_ticker: str = None):
        logger.info("Starting Credit Memo Generation Pipeline...")

        # Load existing library index if it exists to merge rather than overwrite
        index_path = os.path.join(self.output_dir, "credit_memo_library.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                try:
                    self.library_index = json.load(f)
                except json.JSONDecodeError:
                    self.library_index = []

        # Load existing interaction logs
        logs_path = os.path.join(self.output_dir, "risk_legal_interaction.json")
        if os.path.exists(logs_path):
            with open(logs_path, "r") as f:
                try:
                    self.interaction_logs = json.load(f)
                except json.JSONDecodeError:
                    self.interaction_logs = {}

        entities_to_process = MOCK_LIBRARY
        if target_ticker:
            # Filter if a specific ticker is provided
            entities_to_process = {
                k: v for k, v in MOCK_LIBRARY.items() if v["ticker"] == target_ticker
            }
            if not entities_to_process:
                logger.error(f"Ticker {target_ticker} not found in MOCK_LIBRARY.")
                return

        for key, data in entities_to_process.items():
            process_key = key
            process_data = data

            if live:
                process_key = f"{key}_Live"
                process_data = self.fetch_live_data(data["ticker"], data)

            result = self.orchestrator.process_entity(process_key, process_data)
            if not result:
                continue

            memo = result["memo"]
            logs = result["interaction_log"]

            # Save Memo
            filename = f"credit_memo_{process_key}.json"
            with open(os.path.join(self.output_dir, filename), "w") as f:
                json.dump(memo, f, indent=2)

            # Update Index (merge by ID)
            existing_idx = next(
                (
                    i
                    for i, item in enumerate(self.library_index)
                    if item["id"] == process_key
                ),
                None,
            )

            index_entry = {
                "id": process_key,
                "borrower_name": process_data["name"],
                "ticker": process_data["ticker"],
                "sector": process_data["sector"],
                "report_date": memo["report_date"],
                "risk_score": memo["risk_score"],
                "file": filename,
                "summary": f"{process_data['name']} ({process_data['sector']}). Risk Score: {memo['risk_score']}.",
            }

            if existing_idx is not None:
                self.library_index[existing_idx] = index_entry
            else:
                self.library_index.append(index_entry)

            # Update Interaction Logs
            self.interaction_logs[process_key] = logs

        self.save_library_index()
        self.save_interaction_logs()
        logger.info("Pipeline Complete.")

    def save_library_index(self):
        with open(os.path.join(self.output_dir, "credit_memo_library.json"), "w") as f:
            json.dump(self.library_index, f, indent=2)

    def save_interaction_logs(self):
        with open(
            os.path.join(self.output_dir, "risk_legal_interaction.json"), "w"
        ) as f:
            json.dump(self.interaction_logs, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Credit Memos")
    parser.add_argument(
        "--live", action="store_true", help="Use live data to generate memos"
    )
    parser.add_argument(
        "--ticker", type=str, help="Specific ticker to run (e.g., AAPL)"
    )
    args = parser.parse_args()

    pipeline = CreditMemoPipeline()
    pipeline.run_pipeline(live=args.live, target_ticker=args.ticker)
