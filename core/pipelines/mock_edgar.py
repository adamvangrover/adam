import random
from typing import Dict, Any

class MockEdgar:
    """
    Simulates retrieval from SEC EDGAR database.
    Generates synthetic 10-K artifacts including BBoxes and Financial Tables.
    """

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

    @staticmethod
    def get_financials(ticker: str, sector: str) -> Dict[str, float]:
        # Deterministic random based on ticker
        random.seed(ticker)
        base = random.randint(50, 500)

        return {
            "total_assets": float(base),
            "total_liabilities": float(int(base * random.uniform(0.4, 0.8))),
            "total_equity": float(int(base * random.uniform(0.2, 0.6))), # Will be fixed to balance later
            "ebitda": float(int(base * random.uniform(0.1, 0.2))),
            "total_debt": float(int(base * random.uniform(0.2, 0.5))),
            "interest_expense": float(round(base * 0.02, 2))
        }

    @classmethod
    def generate_10k(cls, ticker: str, name: str, sector: str) -> Dict[str, Any]:
        random.seed(ticker)
        doc_id = f"doc_{ticker}_10K_2026"

        # 1. Financials
        fin = cls.get_financials(ticker, sector)
        # Force Balance Sheet Equation
        fin["total_equity"] = round(fin["total_assets"] - fin["total_liabilities"], 2)

        # 2. Chunks
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
