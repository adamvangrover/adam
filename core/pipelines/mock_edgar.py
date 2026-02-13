import random
from typing import Dict, Any, List, Optional

class MockEdgar:
    """
    Simulates retrieval from SEC EDGAR database.
    
    Capabilities:
    1. specific_history: Provides realistic hardcoded historical data for major tech companies.
    2. synthetic_generation: Generates random 10-K artifacts (BBoxes, chunks) for any ticker,
       populating them with realistic data if available, or synthetic data if not.
    """

    # --- Source 1: Narrative Templates (Synthetic) ---
    SECTOR_DATA = {
        "Technology": {
            "risks": ["Supply chain disruption", "Rapid technological obsolescence", "Data privacy regulation"],
            "narratives": ["Revenue driven by cloud adoption", "AI integration boosting margins", "Hardware sales normalizing"]
        },
        "Financial": {
            "risks": ["Interest rate volatility", "Credit default rise", "Regulatory capital requirements"],
            "narratives": ["Net interest income expanded", "Investment banking fees muted", "Trading revenue outperformed"]
        },
        "Consumer": {
            "risks": ["Inflationary pressure", "Supply chain costs", "Shift in consumer spending"],
            "narratives": ["E-commerce channel growth", "Direct-to-consumer pivot", "Inventory levels elevated"]
        }
    }

    # --- Source 2: Hardcoded Real Data (Realistic) ---
    FINANCIALS_DB = {
        "AAPL": {
            "company_name": "Apple Inc.",
            "history": [
                {"fiscal_year": 2021, "revenue": 365817, "ebitda": 120233, "total_debt": 124719, "interest_expense": 2645, "total_assets": 351002, "total_liabilities": 287912, "total_equity": 63090},
                {"fiscal_year": 2022, "revenue": 394328, "ebitda": 130541, "total_debt": 120069, "interest_expense": 2931, "total_assets": 352755, "total_liabilities": 302083, "total_equity": 50672},
                {"fiscal_year": 2023, "revenue": 383285, "ebitda": 114301, "total_debt": 111088, "interest_expense": 3933, "total_assets": 352583, "total_liabilities": 290437, "total_equity": 62146}
            ]
        },
        "MSFT": {
            "company_name": "Microsoft Corporation",
            "history": [
                {"fiscal_year": 2021, "revenue": 168088, "ebitda": 80816, "total_debt": 58120, "interest_expense": 2346, "total_assets": 333779, "total_liabilities": 191791, "total_equity": 141988},
                {"fiscal_year": 2022, "revenue": 198270, "ebitda": 97843, "total_debt": 49751, "interest_expense": 2063, "total_assets": 364840, "total_liabilities": 198298, "total_equity": 166542},
                {"fiscal_year": 2023, "revenue": 211915, "ebitda": 102384, "total_debt": 47204, "interest_expense": 1968, "total_assets": 411976, "total_liabilities": 205753, "total_equity": 206223}
            ]
        },
        "GOOGL": {
            "company_name": "Alphabet Inc.",
            "history": [
                {"fiscal_year": 2021, "revenue": 257637, "ebitda": 91155, "total_debt": 14817, "interest_expense": 346, "total_assets": 359268, "total_liabilities": 107633, "total_equity": 251635},
                {"fiscal_year": 2022, "revenue": 282836, "ebitda": 74842, "total_debt": 14701, "interest_expense": 357, "total_assets": 365264, "total_liabilities": 109120, "total_equity": 256144},
                {"fiscal_year": 2023, "revenue": 307394, "ebitda": 88164, "total_debt": 13253, "interest_expense": 321, "total_assets": 402392, "total_liabilities": 119048, "total_equity": 283344}
            ]
        },
        "AMZN": {
            "company_name": "Amazon.com, Inc.",
            "history": [
                {"fiscal_year": 2021, "revenue": 469822, "ebitda": 59175, "total_debt": 48744, "interest_expense": 1809, "total_assets": 420549, "total_liabilities": 282304, "total_equity": 138245},
                {"fiscal_year": 2022, "revenue": 513983, "ebitda": 54169, "total_debt": 67150, "interest_expense": 2367, "total_assets": 462675, "total_liabilities": 316632, "total_equity": 146043},
                {"fiscal_year": 2023, "revenue": 574785, "ebitda": 85515, "total_debt": 58316, "interest_expense": 3178, "total_assets": 527854, "total_liabilities": 326084, "total_equity": 201770}
            ]
        },
        "NVDA": {
            "company_name": "NVIDIA Corporation",
            "history": [
                {"fiscal_year": 2022, "revenue": 26914, "ebitda": 11216, "total_debt": 10946, "interest_expense": 236, "total_assets": 44187, "total_liabilities": 17575, "total_equity": 26612},
                {"fiscal_year": 2023, "revenue": 26974, "ebitda": 5600, "total_debt": 11130, "interest_expense": 272, "total_assets": 41182, "total_liabilities": 19081, "total_equity": 22101},
                {"fiscal_year": 2024, "revenue": 60922, "ebitda": 34480, "total_debt": 8461, "interest_expense": 257, "total_assets": 65728, "total_liabilities": 22750, "total_equity": 42978}
            ]
        }
    }

    @classmethod
    def get_financial_history(cls, ticker: str) -> Dict[str, Any]:
        """
        Returns the historical financials for the given ticker if available.
        """
        data = cls.FINANCIALS_DB.get(ticker.upper())
        if not data:
            raise ValueError(f"Ticker {ticker} not found in Mock EDGAR Source.")
        return data

    @staticmethod
    def _generate_random_financials(ticker: str) -> Dict[str, float]:
        """Internal helper to generate synthetic numbers for unknown tickers."""
        random.seed(ticker)
        base = random.randint(50, 500)
        
        # Calculate derived values
        liabilities = float(int(base * random.uniform(0.4, 0.8)))
        assets = float(base)
        
        return {
            "total_assets": assets,
            "total_liabilities": liabilities,
            "total_equity": round(assets - liabilities, 2),
            "ebitda": float(int(base * random.uniform(0.1, 0.2))),
            "total_debt": float(int(base * random.uniform(0.2, 0.5))),
            "interest_expense": float(round(base * 0.02, 2))
        }

    @classmethod
    def get_latest_financials(cls, ticker: str, sector: str = None) -> Dict[str, float]:
        """
        Smart retrieval: 
        1. Checks hardcoded database (FINANCIALS_DB) for realistic data.
        2. Falls back to synthetic generation if ticker is unknown.
        """
        ticker = ticker.upper()
        
        # 1. Try Realistic Data
        if ticker in cls.FINANCIALS_DB:
            latest_year = cls.FINANCIALS_DB[ticker]["history"][-1]
            return {
                "total_assets": float(latest_year.get("total_assets", 0)),
                "total_liabilities": float(latest_year.get("total_liabilities", 0)),
                "total_equity": float(latest_year.get("total_equity", 0)),
                "ebitda": float(latest_year.get("ebitda", 0)),
                "total_debt": float(latest_year.get("total_debt", 0)),
                "interest_expense": float(latest_year.get("interest_expense", 0))
            }
        
        # 2. Fallback to Synthetic
        return cls._generate_random_financials(ticker)

    @classmethod
    def generate_10k(cls, ticker: str, name: str, sector: str) -> Dict[str, Any]:
        """
        Generates a synthetic 10-K document structure (chunks, bboxes).
        Populates the Financial Table chunk with data from get_latest_financials.
        """
        random.seed(ticker)
        ticker = ticker.upper()
        doc_id = f"doc_{ticker}_10K_2026"

        # Resolve Company Name (Use real name if available, else use provided)
        if ticker in cls.FINANCIALS_DB:
            name = cls.FINANCIALS_DB[ticker]["company_name"]

        # 1. Financials (Smart Fetch)
        fin = cls.get_latest_financials(ticker, sector)

        # 2. Chunks Generation
        chunks = []
        chunk_id = 1

        # Header
        chunks.append({
            "chunk_id": f"chunk_{chunk_id:03d}",
            "type": "header",
            "page": 1,
            "bbox": [50, 30, 400, 60],
            "content": f"{name} ({ticker}) - FORM 10-K"
        })
        chunk_id += 1

        # Narratives
        templates = cls.SECTOR_DATA.get(sector, cls.SECTOR_DATA["Technology"])["narratives"]
        for t in templates:
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:03d}",
                "type": "narrative",
                "page": random.randint(3, 10),
                "bbox": [50, random.randint(100, 600), 550, random.randint(650, 750)],
                "content": f"{t} ({ticker} specific context)."
            })
            chunk_id += 1

        # Table
        chunks.append({
            "chunk_id": f"chunk_{chunk_id:03d}",
            "type": "financial_table",
            "page": 25,
            "bbox": [50, 200, 550, 500],
            "content": "Consolidated Financial Statements",
            "content_json": fin
        })
        chunk_id += 1

        # Risks
        risks = cls.SECTOR_DATA.get(sector, cls.SECTOR_DATA["Technology"])["risks"]
        for r in risks:
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:03d}",
                "type": "risk_factor",
                "page": random.randint(15, 20),
                "bbox": [50, random.randint(100, 600), 550, random.randint(650, 750)],
                "content": f"Risk Factor: {r}"
            })
            chunk_id += 1

        return {
            "borrower_details": {
                "name": name,
                "ticker": ticker,
                "sector": sector,
                "rating": random.choice(["AAA", "AA", "A", "BBB+", "BBB"])
            },
            "documents": [{
                "doc_id": doc_id,
                "title": f"{ticker} 10-K",
                "page_count": 50,
                "chunks": chunks
            }],
            "market_data": {
                "sentiment": random.uniform(0.4, 0.9),
                "trend": random.choice(["Bullish", "Neutral", "Bearish"])
            }
        }

    def list_tickers(self):
        """Returns the list of available hardcoded tickers."""
        return list(self.FINANCIALS_DB.keys())