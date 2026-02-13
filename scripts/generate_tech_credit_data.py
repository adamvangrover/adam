import json
import random
import os

# Configuration
COMPANIES = [
    {
        "name": "Apple Inc.",
        "ticker": "AAPL",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "rating": "AA+",
        "revenue_growth": "5.2%",
        "assets": 352.0,
        "liabilities": 290.0,
        "equity": 62.0,
        "ebitda": 120.0,
        "debt": 100.0,
        "risk_factors": [
            "Global supply chain disruptions could impact iPhone production.",
            "Antitrust regulatory scrutiny in EU and US markets.",
            "Dependence on China for manufacturing and assembly."
        ],
        "narrative_chunks": [
            "Services revenue reached an all-time high, driven by App Store and Apple Music subscriptions.",
            "Wearables, Home, and Accessories segment grew 10% YoY.",
            "Gross margin expansion due to favorable commodity pricing and mix shift."
        ]
    },
    {
        "name": "Microsoft Corp",
        "ticker": "MSFT",
        "sector": "Technology",
        "industry": "Software - Infrastructure",
        "rating": "AAA",
        "revenue_growth": "12.5%",
        "assets": 480.0,
        "liabilities": 220.0,
        "equity": 260.0,
        "ebitda": 150.0,
        "debt": 80.0,
        "risk_factors": [
            "Intense competition in cloud computing from AWS and Google Cloud.",
            "Cybersecurity threats targeting Azure infrastructure.",
            "Integration risks associated with large acquisitions (e.g., Activision)."
        ],
        "narrative_chunks": [
            "Azure revenue increased by 25%, outpacing the broader cloud market.",
            "Office 365 Commercial revenue grew 15% due to seat growth and ARPU expansion.",
            "Gaming revenue declined slightly due to console hardware cyclicality."
        ]
    },
    {
        "name": "NVIDIA Corp",
        "ticker": "NVDA",
        "sector": "Technology",
        "industry": "Semiconductors",
        "rating": "A-",
        "revenue_growth": "85.0%",
        "assets": 65.0,
        "liabilities": 25.0,
        "equity": 40.0,
        "ebitda": 35.0,
        "debt": 10.0,
        "risk_factors": [
            "Export controls on AI chips to China could materially impact revenue.",
            "Supply constraints at TSMC for CoWoS packaging.",
            "Cyclicality in the gaming GPU market."
        ],
        "narrative_chunks": [
            "Data Center revenue tripled YoY, driven by H100 demand for generative AI training.",
            "Automotive revenue grew 20% as EV adoption increases.",
            "Gaming segment normalized after the crypto-mining boom."
        ]
    },
    {
        "name": "Alphabet Inc.",
        "ticker": "GOOGL",
        "sector": "Technology",
        "industry": "Internet Content & Information",
        "rating": "AA+",
        "revenue_growth": "9.0%",
        "assets": 390.0,
        "liabilities": 110.0,
        "equity": 280.0,
        "ebitda": 110.0,
        "debt": 30.0,
        "risk_factors": [
            "Regulatory challenges regarding Search dominance and AdTech.",
            "AI disruption to traditional Search business model.",
            "Traffic acquisition costs (TAC) continue to rise."
        ],
        "narrative_chunks": [
            "Google Cloud reached profitability for the first time.",
            "YouTube ad revenue stabilized after a decline in the previous quarter.",
            "Search & Other revenue grew 8%, reflecting resilience in core ads."
        ]
    }
]

def generate_bbox(page_width=600, page_height=800):
    x0 = random.randint(50, page_width - 200)
    y0 = random.randint(50, page_height - 100)
    x1 = x0 + random.randint(100, 200)
    y1 = y0 + random.randint(20, 100)
    return [x0, y0, x1, y1]

def generate_tech_credit_data():
    library = {}

    for company in COMPANIES:
        ticker = company["ticker"]
        doc_id = f"doc_{ticker}_10K_2025"

        # 1. Generate Chunks
        chunks = []
        chunk_id_counter = 1

        # Header
        chunks.append({
            "chunk_id": f"chunk_{chunk_id_counter:03d}",
            "type": "header",
            "page": 1,
            "bbox": [50, 30, 400, 60],
            "content": f"{company['name']} - Annual Report on Form 10-K (2025)"
        })
        chunk_id_counter += 1

        # Narrative Chunks (MD&A)
        for narrative in company["narrative_chunks"]:
            chunks.append({
                "chunk_id": f"chunk_{chunk_id_counter:03d}",
                "type": "narrative",
                "page": random.randint(5, 15),
                "bbox": generate_bbox(),
                "content": narrative
            })
            chunk_id_counter += 1

        # Risk Factors
        for risk in company["risk_factors"]:
            chunks.append({
                "chunk_id": f"chunk_{chunk_id_counter:03d}",
                "type": "risk_factor",
                "page": random.randint(20, 30),
                "bbox": generate_bbox(),
                "content": risk
            })
            chunk_id_counter += 1

        # Financial Table
        chunks.append({
            "chunk_id": f"chunk_{chunk_id_counter:03d}",
            "type": "financial_table",
            "page": 45,
            "bbox": [50, 200, 550, 500],
            "content": "Consolidated Balance Sheet",
            "content_json": {
                "total_assets": company["assets"],
                "total_liabilities": company["liabilities"],
                "total_equity": company["equity"],
                "ebitda": company["ebitda"],
                "total_debt": company["debt"],
                "interest_expense": round(company["debt"] * 0.05, 2) # Assume 5% cost of debt
            }
        })

        # 2. Structure Data
        library[company["name"]] = {
            "borrower_details": {
                "name": company["name"],
                "ticker": ticker,
                "industry": company["industry"],
                "sector": company["sector"],
                "rating": company["rating"]
            },
            "documents": [
                {
                    "doc_id": doc_id,
                    "title": f"{company['name']} 10-K (2025)",
                    "page_count": 80,
                    "chunks": chunks
                }
            ],
            "market_data": {
                "revenue_growth": company["revenue_growth"],
                "news_sentiment": random.uniform(0.6, 0.95),
                "sector_trend": "Bullish" if company["sector"] == "Technology" else "Neutral"
            },
            "knowledge_graph": {
                "nodes": [
                    {"id": company["name"], "label": "Borrower", "group": "corporation"},
                    {"id": "USA", "label": "Jurisdiction", "group": "country"},
                    {"id": company["industry"], "label": "Industry", "group": "industry"}
                ],
                "edges": [
                    {"from": company["name"], "to": "USA", "label": "DOMICILED_IN"},
                    {"from": company["name"], "to": company["industry"], "label": "OPERATES_IN"}
                ]
            }
        }

    # Write to file
    output_path = "showcase/data/tech_credit_data.json"
    with open(output_path, 'w') as f:
        json.dump(library, f, indent=2)

    print(f"Generated tech credit data for {len(COMPANIES)} companies at {output_path}")

if __name__ == "__main__":
    generate_tech_credit_data()
