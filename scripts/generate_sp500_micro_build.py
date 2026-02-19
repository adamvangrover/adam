import os
import json
import random
from datetime import datetime, timedelta

# Configuration
OUTPUT_DIR = "showcase/data"
EQUITY_DIR = os.path.join(OUTPUT_DIR, "equity_reports")
CREDIT_DIR = os.path.join(OUTPUT_DIR, "credit_reports")
DOCS_DIR = os.path.join(OUTPUT_DIR, "documents")
MARKET_DATA_FILE = os.path.join(OUTPUT_DIR, "sp500_market_data.json")
MARKET_DATA_JS_FILE = os.path.join(OUTPUT_DIR, "sp500_market_data.js")

# Companies
COMPANIES = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "price": 185.0, "risk_score": 92},
    {"ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Technology", "price": 405.0, "risk_score": 95},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "price": 175.0, "risk_score": 88},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Discretionary", "price": 180.0, "risk_score": 85},
    {"ticker": "NVDA", "name": "NVIDIA Corp.", "sector": "Technology", "price": 950.0, "risk_score": 75},
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Discretionary", "price": 170.0, "risk_score": 60},
    {"ticker": "META", "name": "Meta Platforms Inc.", "sector": "Technology", "price": 500.0, "risk_score": 80},
    {"ticker": "BRK.B", "name": "Berkshire Hathaway", "sector": "Financials", "price": 410.0, "risk_score": 98},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financials", "price": 200.0, "risk_score": 90},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "price": 155.0, "risk_score": 94},
    {"ticker": "V", "name": "Visa Inc.", "sector": "Financials", "price": 280.0, "risk_score": 93},
    {"ticker": "PG", "name": "Procter & Gamble", "sector": "Consumer Staples", "price": 165.0, "risk_score": 96},
    {"ticker": "MA", "name": "Mastercard Inc.", "sector": "Financials", "price": 470.0, "risk_score": 92},
    {"ticker": "UNH", "name": "UnitedHealth Group", "sector": "Healthcare", "price": 480.0, "risk_score": 89},
    {"ticker": "HD", "name": "Home Depot", "sector": "Consumer Discretionary", "price": 350.0, "risk_score": 88},
    {"ticker": "CVX", "name": "Chevron Corp.", "sector": "Energy", "price": 160.0, "risk_score": 85},
    {"ticker": "MRK", "name": "Merck & Co.", "sector": "Healthcare", "price": 130.0, "risk_score": 91},
    {"ticker": "ABBV", "name": "AbbVie Inc.", "sector": "Healthcare", "price": 175.0, "risk_score": 87},
    {"ticker": "KO", "name": "Coca-Cola Co.", "sector": "Consumer Staples", "price": 60.0, "risk_score": 95},
    {"ticker": "PEP", "name": "PepsiCo Inc.", "sector": "Consumer Staples", "price": 170.0, "risk_score": 94}
]

def ensure_dirs():
    os.makedirs(EQUITY_DIR, exist_ok=True)
    os.makedirs(CREDIT_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

def generate_equity_report(company):
    content = f"""
    <html>
    <head>
        <title>{company['name']} ({company['ticker']}) - Equity Research</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; padding: 20px; line-height: 1.6; color: #333; }}
            h1 {{ color: #004e98; border-bottom: 2px solid #004e98; padding-bottom: 10px; }}
            .header {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
            .metric {{ background: #f0f0f0; padding: 10px; border-radius: 4px; text-align: center; }}
            .metric h3 {{ margin: 0; font-size: 14px; color: #666; }}
            .metric p {{ margin: 5px 0 0; font-size: 18px; font-weight: bold; }}
            .section {{ margin-bottom: 20px; }}
            .chart-placeholder {{ width: 100%; height: 200px; background: #eee; display: flex; align-items: center; justify-content: center; color: #888; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <h1>{company['name']} ({company['ticker']})</h1>
                <p>Sector: {company['sector']} | Date: {datetime.now().strftime('%B %d, %Y')}</p>
            </div>
            <div style="text-align: right;">
                <h2>Rating: BUY</h2>
                <p>Target Price: ${company['price'] * 1.2:.2f}</p>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px;">
            <div class="metric"><h3>Price</h3><p>${company['price']}</p></div>
            <div class="metric"><h3>P/E Ratio</h3><p>{random.uniform(15, 35):.1f}x</p></div>
            <div class="metric"><h3>EPS (TTM)</h3><p>${random.uniform(2, 10):.2f}</p></div>
            <div class="metric"><h3>Div Yield</h3><p>{random.uniform(0.5, 3.5):.1f}%</p></div>
        </div>

        <div class="section">
            <h2>Investment Thesis</h2>
            <p>{company['name']} remains a dominant player in the {company['sector']} sector. Our analysis suggests that the market is underestimating the company's growth potential in key areas such as AI, cloud computing, and emerging markets. Strong free cash flow generation and a robust balance sheet provide a margin of safety.</p>
        </div>

        <div class="section">
            <h2>Financial Performance</h2>
            <div class="chart-placeholder">[Revenue Growth Chart]</div>
            <p>Revenue has grown at a CAGR of {random.uniform(5, 15):.1f}% over the past 5 years. Margins remain healthy despite inflationary pressures.</p>
        </div>

        <div class="section">
            <h2>Valuation</h2>
            <p>Trading at {random.uniform(15, 30):.1f}x forward earnings, the stock offers an attractive entry point relative to historical averages and peers.</p>
        </div>

        <div class="footer" style="margin-top: 50px; border-top: 1px solid #ccc; padding-top: 10px; font-size: 12px; color: #888;">
            Generated by Office Nexus Equity Research Module. strictly Confidential.
        </div>
    </body>
    </html>
    """
    with open(os.path.join(EQUITY_DIR, f"{company['ticker']}_Equity_Report.html"), 'w') as f:
        f.write(content)

def generate_credit_report(company):
    rating = "AAA" if company['risk_score'] > 95 else "AA" if company['risk_score'] > 90 else "A" if company['risk_score'] > 80 else "BBB"
    content = f"""
    <html>
    <head>
        <title>{company['name']} ({company['ticker']}) - Credit Memo</title>
        <style>
            body {{ font-family: 'Times New Roman', serif; padding: 40px; line-height: 1.5; color: #000; max-width: 800px; margin: auto; background: #fff; }}
            h1 {{ text-align: center; text-decoration: underline; margin-bottom: 30px; }}
            h2 {{ font-size: 18px; border-bottom: 1px solid #000; padding-bottom: 5px; margin-top: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            td, th {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
            th {{ background: #f9f9f9; }}
            .risk-score {{ font-size: 24px; font-weight: bold; color: {'green' if company['risk_score'] > 80 else 'orange'}; }}
        </style>
    </head>
    <body>
        <h1>CREDIT APPROVAL MEMORANDUM</h1>

        <table>
            <tr>
                <th>Borrower:</th>
                <td>{company['name']}</td>
                <th>Ticker:</th>
                <td>{company['ticker']}</td>
            </tr>
            <tr>
                <th>Sector:</th>
                <td>{company['sector']}</td>
                <th>Date:</th>
                <td>{datetime.now().strftime('%Y-%m-%d')}</td>
            </tr>
            <tr>
                <th>Internal Rating:</th>
                <td>{rating}</td>
                <th>Score:</th>
                <td class="risk-score">{company['risk_score']}/100</td>
            </tr>
        </table>

        <h2>Executive Summary</h2>
        <p>This memorandum recommends approval of the proposed credit facility for {company['name']}. The company exhibits strong credit fundamentals, including robust cash flow generation, manageable leverage ratios, and a leading market position in the {company['sector']} industry.</p>

        <h2>Financial Analysis</h2>
        <p><strong>Leverage:</strong> Net Debt/EBITDA stands at {random.uniform(0.5, 2.5):.1f}x, well within covenant levels.</p>
        <p><strong>Liquidity:</strong> The company maintains a cash balance of ${random.randint(5, 50)} billion and has access to an undrawn revolver.</p>
        <p><strong>Cash Flow:</strong> Free Cash Flow conversion remains strong at >{random.randint(80, 100)}%.</p>

        <h2>Risk Factors</h2>
        <ul>
            <li>Macroeconomic headwinds impacting consumer spending.</li>
            <li>Regulatory scrutiny in key jurisdictions.</li>
            <li>Technological disruption from competitors.</li>
        </ul>

        <h2>Recommendation</h2>
        <p><strong>Approve.</strong> The risk/reward profile is favorable.</p>

        <br><br>
        <p>__________________________<br>Credit Officer</p>
    </body>
    </html>
    """
    with open(os.path.join(CREDIT_DIR, f"{company['ticker']}_Credit_Memo.html"), 'w') as f:
        f.write(content)

def generate_documents(company):
    # Thesis
    thesis = f"""
    INVESTMENT THESIS: {company['name']}
    Date: {datetime.now().strftime('%Y-%m-%d')}
    Analyst: Office Nexus AI

    1. Competitive Advantage
    {company['name']} has a wide economic moat driven by network effects and brand loyalty.

    2. Growth Drivers
    - International expansion
    - New product lines
    - Margin expansion

    3. Risks
    - Regulation
    - FX headwinds

    Conclusion: LONG
    """
    with open(os.path.join(DOCS_DIR, f"{company['ticker']}_Thesis.txt"), 'w') as f:
        f.write(thesis)

def generate_market_data():
    data = []
    for company in COMPANIES:
        # Generate some mock price history (last 30 days)
        history = []
        price = company['price']
        for i in range(30):
            change = random.uniform(-0.02, 0.02)
            price = price * (1 + change)
            history.append(round(price, 2))

        data.append({
            "ticker": company['ticker'],
            "name": company['name'],
            "sector": company['sector'],
            "current_price": company['price'],
            "change_pct": round(random.uniform(-1.5, 1.5), 2),
            "market_cap": f"{random.uniform(0.1, 3.0):.1f}T",
            "pe_ratio": round(random.uniform(15, 40), 1),
            "dividend_yield": round(random.uniform(0, 4), 2),
            "price_history": history,
            "risk_score": company['risk_score']
        })

    with open(MARKET_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    with open(MARKET_DATA_JS_FILE, 'w') as f:
        json_str = json.dumps(data, indent=2)
        f.write(f"window.MARKET_DATA = {json_str};")

def main():
    print("Generating S&P 500 Micro Build Data...")
    ensure_dirs()

    for company in COMPANIES:
        generate_equity_report(company)
        generate_credit_report(company)
        generate_documents(company)
        print(f"Generated artifacts for {company['ticker']}")

    generate_market_data()
    print(f"Market data saved to {MARKET_DATA_FILE} and {MARKET_DATA_JS_FILE}")
    print("Done.")

if __name__ == "__main__":
    main()
