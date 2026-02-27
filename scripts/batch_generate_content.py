
import sys
import os
import logging
import json
import random
from datetime import date

# Add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.orchestrators.credit_memo_orchestrator import CreditMemoOrchestrator
# Try importing other generators if they are modules, otherwise we might execute them via subprocess
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BatchContentGen")

# --- 1. New Credit Memos ---

NEW_CREDIT_LIBRARY = {
    "Goldman_Sachs": {
        "ticker": "GS",
        "name": "Goldman Sachs Group Inc.",
        "sector": "Financial Services",
        "description": "Leading global investment banking, securities and investment management firm.",
        "historical": {
            "revenue": [59000, 47000, 46000],
            "ebitda": [25000, 15000, 14000],
            "net_income": [21000, 11000, 8500],
            "total_assets": [1400000, 1500000, 1600000],
            "total_liabilities": [1300000, 1400000, 1500000],
            "total_debt": [250000, 260000, 270000],
            "cash": [150000, 160000, 170000],
            "interest_expense": [5000, 8000, 12000],
            "capex": [0, 0, 0],
            "year": [2021, 2022, 2023]
        },
        "forecast_assumptions": {
            "revenue_growth": [0.04, 0.05, 0.04],
            "ebitda_margin": [0.30, 0.31, 0.32],
            "discount_rate": 0.10,
            "terminal_growth_rate": 0.02
        },
        "market_data": {
            "share_price": 385.00,
            "market_cap": 125000,
            "beta": 1.4,
            "pe_ratio": 14.5,
            "price_data": [350, 360, 375, 380, 390, 385]
        },
        "docs": {
            "10-K": "Cyclicality of investment banking revenues. Regulatory capital constraints. Intense competition for talent.",
            "Credit_Agreement": "Standard ISDA master agreements. unsecured senior debt. No maintenance covenants."
        }
    },
    "Bank_of_America": {
        "ticker": "BAC",
        "name": "Bank of America Corp.",
        "sector": "Financial Services",
        "description": "One of the world's leading financial institutions.",
        "historical": {
            "revenue": [89000, 95000, 99000],
            "ebitda": [35000, 30000, 32000],
            "net_income": [32000, 27000, 26000],
            "total_assets": [3100000, 3050000, 3180000],
            "total_liabilities": [2800000, 2800000, 2900000],
            "total_debt": [280000, 290000, 300000],
            "cash": [350000, 300000, 320000],
            "interest_expense": [6000, 12000, 25000],
            "capex": [0, 0, 0],
            "year": [2021, 2022, 2023]
        },
        "forecast_assumptions": {
            "revenue_growth": [0.02, 0.03, 0.02],
            "ebitda_margin": [0.32, 0.33, 0.34],
            "discount_rate": 0.09,
            "terminal_growth_rate": 0.02
        },
        "market_data": {
            "share_price": 34.00,
            "market_cap": 270000,
            "beta": 1.0,
            "pe_ratio": 10.0,
            "price_data": [30, 31, 32, 33, 33.5, 34]
        },
        "docs": {
            "10-K": "Interest rate sensitivity (HTM securities portfolio). Credit quality of consumer loan book. Digital banking adoption.",
            "Credit_Agreement": "G-SIB surcharge requirements. TLAC compliance. Stable funding profile."
        }
    },
    "Chevron_Corp": {
        "ticker": "CVX",
        "name": "Chevron Corporation",
        "sector": "Energy",
        "description": "Integrated energy company with upstream and downstream operations.",
        "historical": {
            "revenue": [162000, 246000, 200000],
            "ebitda": [30000, 55000, 40000],
            "net_income": [15000, 35000, 21000],
            "total_assets": [240000, 257000, 260000],
            "total_liabilities": [100000, 100000, 105000],
            "total_debt": [31000, 23000, 20000],
            "cash": [5000, 17000, 8000],
            "interest_expense": [600, 500, 450],
            "capex": [8000, 12000, 15000],
            "year": [2021, 2022, 2023]
        },
        "forecast_assumptions": {
            "revenue_growth": [0.01, 0.02, 0.01],
            "ebitda_margin": [0.20, 0.21, 0.20],
            "discount_rate": 0.08,
            "terminal_growth_rate": 0.01
        },
        "market_data": {
            "share_price": 150.00,
            "market_cap": 280000,
            "beta": 0.8,
            "pe_ratio": 12.0,
            "price_data": [140, 145, 142, 148, 152, 150]
        },
        "docs": {
            "10-K": "Permian basin production growth. LNG project execution. Carbon capture investments. Oil price volatility.",
            "Credit_Agreement": "Top tier credit rating. Access to commercial paper. No material covenants."
        }
    },
    "Coca_Cola_Co": {
        "ticker": "KO",
        "name": "The Coca-Cola Company",
        "sector": "Consumer Defensive",
        "description": "Global beverage company.",
        "historical": {
            "revenue": [38000, 43000, 45000],
            "ebitda": [12000, 13000, 14000],
            "net_income": [9000, 9500, 10000],
            "total_assets": [94000, 92000, 97000],
            "total_liabilities": [69000, 67000, 70000],
            "total_debt": [38000, 36000, 35000],
            "cash": [10000, 11000, 13000],
            "interest_expense": [900, 800, 1000],
            "capex": [1500, 1400, 1800],
            "year": [2021, 2022, 2023]
        },
        "forecast_assumptions": {
            "revenue_growth": [0.04, 0.05, 0.04],
            "ebitda_margin": [0.31, 0.32, 0.31],
            "discount_rate": 0.07,
            "terminal_growth_rate": 0.02
        },
        "market_data": {
            "share_price": 60.00,
            "market_cap": 260000,
            "beta": 0.6,
            "pe_ratio": 24.0,
            "price_data": [58, 59, 58.5, 59.5, 60.5, 60]
        },
        "docs": {
            "10-K": "FX headwinds. Commodity cost inflation (sugar, aluminum). Tax litigation with IRS.",
            "Credit_Agreement": "High grade issuer. Minimal covenants."
        }
    },
    "PepsiCo_Inc": {
        "ticker": "PEP",
        "name": "PepsiCo Inc.",
        "sector": "Consumer Defensive",
        "description": "Global food and beverage leader.",
        "historical": {
            "revenue": [79000, 86000, 91000],
            "ebitda": [13000, 14000, 15000],
            "net_income": [7000, 8900, 9000],
            "total_assets": [92000, 92000, 100000],
            "total_liabilities": [75000, 75000, 82000],
            "total_debt": [36000, 39000, 44000],
            "cash": [5000, 6000, 5000],
            "interest_expense": [1000, 1500, 1800],
            "capex": [4000, 5000, 5500],
            "year": [2021, 2022, 2023]
        },
        "forecast_assumptions": {
            "revenue_growth": [0.04, 0.05, 0.04],
            "ebitda_margin": [0.16, 0.17, 0.16],
            "discount_rate": 0.07,
            "terminal_growth_rate": 0.02
        },
        "market_data": {
            "share_price": 165.00,
            "market_cap": 227000,
            "beta": 0.6,
            "pe_ratio": 25.0,
            "price_data": [160, 162, 164, 166, 168, 165]
        },
        "docs": {
            "10-K": "Supply chain efficiency. Consumer shift to healthier options. Quaker Oats recall impact.",
            "Credit_Agreement": "Standard high grade terms."
        }
    },
    "McDonalds_Corp": {
        "ticker": "MCD",
        "name": "McDonald's Corp.",
        "sector": "Consumer Cyclical",
        "description": "Global food service retailer.",
        "historical": {
            "revenue": [23000, 23000, 25000],
            "ebitda": [12000, 11000, 13000],
            "net_income": [7500, 6000, 8000],
            "total_assets": [53000, 50000, 55000],
            "total_liabilities": [58000, 56000, 60000], # Negative equity due to buybacks
            "total_debt": [35000, 36000, 37000],
            "cash": [4000, 2000, 4000],
            "interest_expense": [1200, 1300, 1500],
            "capex": [2000, 1900, 2300],
            "year": [2021, 2022, 2023]
        },
        "forecast_assumptions": {
            "revenue_growth": [0.05, 0.06, 0.05],
            "ebitda_margin": [0.52, 0.53, 0.52],
            "discount_rate": 0.08,
            "terminal_growth_rate": 0.02
        },
        "market_data": {
            "share_price": 270.00,
            "market_cap": 195000,
            "beta": 0.7,
            "pe_ratio": 24.0,
            "price_data": [280, 285, 275, 260, 265, 270]
        },
        "docs": {
            "10-K": "Franchisee relationships. Inflationary pressure on menu prices. Geopolitical impact in Middle East.",
            "Credit_Agreement": "Revolver $3.5B. Leverage Ratio < 3.5x (Net Debt / EBITDA)."
        }
    },
    "Walt_Disney_Co": {
        "ticker": "DIS",
        "name": "The Walt Disney Company",
        "sector": "Communication Services",
        "description": "Diversified international family entertainment and media enterprise.",
        "historical": {
            "revenue": [67000, 82000, 88000],
            "ebitda": [10000, 12000, 14000],
            "net_income": [2000, 3000, 2300],
            "total_assets": [203000, 203000, 205000],
            "total_liabilities": [110000, 107000, 105000],
            "total_debt": [54000, 48000, 46000],
            "cash": [15000, 11000, 14000],
            "interest_expense": [1400, 1500, 1800],
            "capex": [3000, 4900, 5000],
            "year": [2021, 2022, 2023]
        },
        "forecast_assumptions": {
            "revenue_growth": [0.05, 0.06, 0.05],
            "ebitda_margin": [0.18, 0.19, 0.20],
            "discount_rate": 0.09,
            "terminal_growth_rate": 0.02
        },
        "market_data": {
            "share_price": 110.00,
            "market_cap": 200000,
            "beta": 1.2,
            "pe_ratio": 40.0,
            "price_data": [90, 95, 100, 105, 115, 110]
        },
        "docs": {
            "10-K": "Streaming profitability (Disney+, Hulu, ESPN). Linear TV decline. Park attendance resilience. Succession planning.",
            "Credit_Agreement": "Standard high grade. Interest Coverage Ratio > 3.0x."
        }
    }
}

def run_credit_memos():
    logger.info("Generating New Credit Memos...")
    output_dir = "showcase/data"
    orchestrator = CreditMemoOrchestrator(mock_library=NEW_CREDIT_LIBRARY, output_dir=output_dir)

    # Load existing logs if possible to append
    log_path = os.path.join(output_dir, "risk_legal_interaction.json")
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                interaction_logs = json.load(f)
            except:
                interaction_logs = {}
    else:
        interaction_logs = {}

    for key, data in NEW_CREDIT_LIBRARY.items():
        result = orchestrator.process_entity(key, data)
        if result:
            memo = result["memo"]
            logs = result["interaction_log"]

            filename = f"credit_memo_{key}.json"
            with open(os.path.join(output_dir, filename), 'w') as f:
                json.dump(memo, f, indent=2)

            interaction_logs[key] = logs
            logger.info(f"Generated {filename}")

    with open(log_path, 'w') as f:
        json.dump(interaction_logs, f, indent=2)

# --- 2. Execute Other Generators ---

def run_other_scripts():
    scripts = [
        "scripts/generate_fraud_cases.py",
        "scripts/generate_war_room_data.py",
        "scripts/generate_sector_outlook.py",
        "scripts/generate_deep_dive_reports.py" # Runs for hardcoded list, maybe we edit it later
    ]

    for script in scripts:
        if os.path.exists(script):
            logger.info(f"Running {script}...")
            subprocess.run([sys.executable, script])
        else:
            logger.warning(f"Script not found: {script}")

if __name__ == "__main__":
    run_credit_memos()
    run_other_scripts()
