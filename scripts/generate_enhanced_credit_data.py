import json
import os
import random
from datetime import datetime

LIBRARY_PATH = 'showcase/data/credit_memo_library.json'
DATA_DIR = 'showcase/data/'

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def generate_synthetic_memo(item):
    # Create a basic memo structure from the library item
    name = item.get('borrower_name', 'Unknown')
    ticker = item.get('ticker', 'UNK')
    sector = item.get('sector', 'General')

    # Generate synthetic financials based on sector
    if sector == 'Technology':
        rev = random.randint(50000, 300000)
        margin = 0.35
    elif sector == 'Financial':
        rev = random.randint(80000, 200000)
        margin = 0.40
    else:
        rev = random.randint(40000, 150000)
        margin = 0.15

    ebitda = int(rev * margin)

    return {
        "borrower_name": name,
        "ticker": ticker,
        "sector": sector,
        "report_date": datetime.now().isoformat(),
        "risk_score": item.get('risk_score', 75),
        "historical_financials": [
            {"period": "FY2025", "revenue": rev, "ebitda": ebitda, "total_assets": rev * 2, "total_liabilities": rev * 1, "total_equity": rev * 1},
            {"period": "FY2024", "revenue": int(rev * 0.9), "ebitda": int(ebitda * 0.9)},
            {"period": "FY2023", "revenue": int(rev * 0.8), "ebitda": int(ebitda * 0.8)}
        ],
        "financial_ratios": {
            "leverage_ratio": random.uniform(1.5, 4.0),
            "ebitda": ebitda,
            "revenue": rev,
            "dscr": random.uniform(2.0, 10.0)
        },
        "sections": [
            {
                "title": "Executive Summary",
                "content": item.get('summary', f"Analysis of {name}. Strong market position with solid fundamentals.")
            },
            {
                "title": "Business Overview",
                "content": f"{name} operates in the {sector} sector. The company has shown resilient growth."
            },
            {
                "title": "Risk Factors",
                "content": "Key risks include macroeconomic headwinds and competitive pressures. [Ref: chunk_006]"
            }
        ],
        "documents": item.get('documents', [])
    }

def enhance_memo(memo):
    # Enhance Outlook
    score = memo.get('risk_score', 50)
    rating = "STRONG BUY" if score > 80 else ("BUY" if score > 60 else ("HOLD" if score > 40 else "SELL"))

    # Base price on DCF if exists, else random
    dcf = memo.get('dcf_analysis', {})
    base_price = dcf.get('share_price', random.uniform(100, 300))

    memo['outlook'] = {
        "rating": rating,
        "conviction": random.randint(70, 99),
        "price_target_base": round(base_price * 1.1, 2),
        "price_target_bull": round(base_price * 1.35, 2),
        "price_target_bear": round(base_price * 0.85, 2),
        "last_updated": datetime.now().isoformat()
    }

    # Enhance Forecasts
    if 'historical_financials' in memo and memo['historical_financials']:
        hist = memo['historical_financials']
        # Sort desc by period just to be sure we get latest
        hist.sort(key=lambda x: x['period'], reverse=True)
        last = hist[0]

        last_rev = last.get('revenue', 100000)
        last_ebitda = last.get('ebitda', 30000)

        memo['forecast_financials'] = [
            {"period": "FY2026E", "revenue": round(last_rev * 1.10), "ebitda": round(last_ebitda * 1.12)},
            {"period": "FY2027E", "revenue": round(last_rev * 1.25), "ebitda": round(last_ebitda * 1.28)},
            {"period": "FY2028E", "revenue": round(last_rev * 1.45), "ebitda": round(last_ebitda * 1.50)}
        ]

    # Enhance DCF if missing or simple
    if 'dcf_analysis' not in memo:
        memo['dcf_analysis'] = {
            "share_price": round(base_price, 2),
            "enterprise_value": round(base_price * 1000000 * random.uniform(0.8, 1.2)), # Mock EV
            "wacc": 0.085,
            "growth_rate": 0.03
        }
    else:
        # Ensure it has all fields needed for sensitivity
        if 'wacc' not in memo['dcf_analysis']: memo['dcf_analysis']['wacc'] = 0.085
        if 'growth_rate' not in memo['dcf_analysis']: memo['dcf_analysis']['growth_rate'] = 0.03

    return memo

def main():
    print(f"Loading library from {LIBRARY_PATH}...")
    try:
        library = load_json(LIBRARY_PATH)
    except Exception as e:
        print(f"Error loading library: {e}")
        return

    items = library if isinstance(library, list) else library.values()

    for item in items:
        item_id = item.get('id')
        if not item_id: continue

        filename = f"credit_memo_{item_id}.json"
        filepath = os.path.join(DATA_DIR, filename)

        if os.path.exists(filepath):
            print(f"Enhancing existing file: {filepath}")
            try:
                memo = load_json(filepath)
                enhanced_memo = enhance_memo(memo)
                save_json(filepath, enhanced_memo)
            except Exception as e:
                print(f"  -> Failed to enhance: {e}")
        else:
            print(f"Creating new file: {filepath}")
            try:
                memo = generate_synthetic_memo(item)
                enhanced_memo = enhance_memo(memo)
                save_json(filepath, enhanced_memo)
            except Exception as e:
                print(f"  -> Failed to create: {e}")

if __name__ == "__main__":
    main()
