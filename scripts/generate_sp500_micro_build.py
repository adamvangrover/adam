import os
import json
import random
import math
from datetime import datetime, timedelta

# Configuration
OUTPUT_DIR = "showcase/data"
EQUITY_DIR = os.path.join(OUTPUT_DIR, "equity_reports")
CREDIT_DIR = os.path.join(OUTPUT_DIR, "credit_reports")
DOCS_DIR = os.path.join(OUTPUT_DIR, "documents")
MARKET_DATA_FILE = os.path.join(OUTPUT_DIR, "sp500_market_data.json")
MARKET_DATA_JS_FILE = os.path.join(OUTPUT_DIR, "sp500_market_data.js")

# Companies List (~50 Major S&P 500 Tickers)
COMPANIES = [
    # Technology
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "price": 185.0, "risk_score": 92},
    {"ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Technology", "price": 405.0, "risk_score": 95},
    {"ticker": "NVDA", "name": "NVIDIA Corp.", "sector": "Technology", "price": 950.0, "risk_score": 75},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "price": 175.0, "risk_score": 88},
    {"ticker": "META", "name": "Meta Platforms Inc.", "sector": "Technology", "price": 500.0, "risk_score": 80},
    {"ticker": "AVGO", "name": "Broadcom Inc.", "sector": "Technology", "price": 1300.0, "risk_score": 82},
    {"ticker": "ORCL", "name": "Oracle Corp.", "sector": "Technology", "price": 125.0, "risk_score": 85},
    {"ticker": "ADBE", "name": "Adobe Inc.", "sector": "Technology", "price": 480.0, "risk_score": 88},
    {"ticker": "CRM", "name": "Salesforce Inc.", "sector": "Technology", "price": 300.0, "risk_score": 84},
    {"ticker": "AMD", "name": "Advanced Micro Devices", "sector": "Technology", "price": 180.0, "risk_score": 70},
    {"ticker": "INTC", "name": "Intel Corp.", "sector": "Technology", "price": 35.0, "risk_score": 65},
    {"ticker": "CSCO", "name": "Cisco Systems", "sector": "Technology", "price": 48.0, "risk_score": 89},
    {"ticker": "QCOM", "name": "Qualcomm Inc.", "sector": "Technology", "price": 170.0, "risk_score": 81},
    {"ticker": "TXN", "name": "Texas Instruments", "sector": "Technology", "price": 175.0, "risk_score": 90},
    {"ticker": "AMAT", "name": "Applied Materials", "sector": "Technology", "price": 210.0, "risk_score": 78},
    {"ticker": "INTU", "name": "Intuit Inc.", "sector": "Technology", "price": 650.0, "risk_score": 87},
    {"ticker": "IBM", "name": "IBM", "sector": "Technology", "price": 190.0, "risk_score": 86},
    {"ticker": "NOW", "name": "ServiceNow", "sector": "Technology", "price": 750.0, "risk_score": 83},

    # Consumer Discretionary
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Discretionary", "price": 180.0, "risk_score": 85},
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Discretionary", "price": 175.0, "risk_score": 60},
    {"ticker": "HD", "name": "Home Depot", "sector": "Consumer Discretionary", "price": 350.0, "risk_score": 88},
    {"ticker": "MCD", "name": "McDonald's Corp.", "sector": "Consumer Discretionary", "price": 270.0, "risk_score": 91},
    {"ticker": "NKE", "name": "Nike Inc.", "sector": "Consumer Discretionary", "price": 95.0, "risk_score": 86},
    {"ticker": "SBUX", "name": "Starbucks Corp.", "sector": "Consumer Discretionary", "price": 85.0, "risk_score": 84},
    {"ticker": "LOW", "name": "Lowe's Cos.", "sector": "Consumer Discretionary", "price": 230.0, "risk_score": 87},
    {"ticker": "BKNG", "name": "Booking Holdings", "sector": "Consumer Discretionary", "price": 3600.0, "risk_score": 85},

    # Communication Services
    {"ticker": "NFLX", "name": "Netflix Inc.", "sector": "Communication Services", "price": 620.0, "risk_score": 78},
    {"ticker": "DIS", "name": "Walt Disney Co.", "sector": "Communication Services", "price": 115.0, "risk_score": 82},
    {"ticker": "CMCSA", "name": "Comcast Corp.", "sector": "Communication Services", "price": 40.0, "risk_score": 80},
    {"ticker": "T", "name": "AT&T Inc.", "sector": "Communication Services", "price": 17.0, "risk_score": 75},
    {"ticker": "VZ", "name": "Verizon Communications", "sector": "Communication Services", "price": 40.0, "risk_score": 78},

    # Financials
    {"ticker": "BRK.B", "name": "Berkshire Hathaway", "sector": "Financials", "price": 410.0, "risk_score": 98},
    {"ticker": "JPM", "name": "JPMorgan Chase", "sector": "Financials", "price": 200.0, "risk_score": 90},
    {"ticker": "V", "name": "Visa Inc.", "sector": "Financials", "price": 280.0, "risk_score": 93},
    {"ticker": "MA", "name": "Mastercard Inc.", "sector": "Financials", "price": 470.0, "risk_score": 92},
    {"ticker": "BAC", "name": "Bank of America", "sector": "Financials", "price": 38.0, "risk_score": 85},
    {"ticker": "WFC", "name": "Wells Fargo", "sector": "Financials", "price": 58.0, "risk_score": 82},
    {"ticker": "MS", "name": "Morgan Stanley", "sector": "Financials", "price": 95.0, "risk_score": 84},
    {"ticker": "GS", "name": "Goldman Sachs", "sector": "Financials", "price": 400.0, "risk_score": 83},
    {"ticker": "C", "name": "Citigroup", "sector": "Financials", "price": 60.0, "risk_score": 78},
    {"ticker": "AXP", "name": "American Express", "sector": "Financials", "price": 220.0, "risk_score": 88},
    {"ticker": "SPGI", "name": "S&P Global", "sector": "Financials", "price": 420.0, "risk_score": 94},

    # Healthcare
    {"ticker": "LLY", "name": "Eli Lilly & Co.", "sector": "Healthcare", "price": 750.0, "risk_score": 90},
    {"ticker": "UNH", "name": "UnitedHealth Group", "sector": "Healthcare", "price": 480.0, "risk_score": 89},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "price": 155.0, "risk_score": 94},
    {"ticker": "MRK", "name": "Merck & Co.", "sector": "Healthcare", "price": 130.0, "risk_score": 91},
    {"ticker": "ABBV", "name": "AbbVie Inc.", "sector": "Healthcare", "price": 175.0, "risk_score": 87},
    {"ticker": "TMO", "name": "Thermo Fisher Scientific", "sector": "Healthcare", "price": 580.0, "risk_score": 92},
    {"ticker": "PFE", "name": "Pfizer Inc.", "sector": "Healthcare", "price": 28.0, "risk_score": 85},
    {"ticker": "DHR", "name": "Danaher Corp.", "sector": "Healthcare", "price": 250.0, "risk_score": 90},
    {"ticker": "ABT", "name": "Abbott Laboratories", "sector": "Healthcare", "price": 110.0, "risk_score": 93},
    {"ticker": "AMGN", "name": "Amgen Inc.", "sector": "Healthcare", "price": 270.0, "risk_score": 88},

    # Industrials
    {"ticker": "CAT", "name": "Caterpillar Inc.", "sector": "Industrials", "price": 360.0, "risk_score": 86},
    {"ticker": "GE", "name": "GE Aerospace", "sector": "Industrials", "price": 170.0, "risk_score": 84},
    {"ticker": "UNP", "name": "Union Pacific", "sector": "Industrials", "price": 240.0, "risk_score": 89},
    {"ticker": "HON", "name": "Honeywell Int.", "sector": "Industrials", "price": 200.0, "risk_score": 91},
    {"ticker": "UPS", "name": "UPS", "sector": "Industrials", "price": 150.0, "risk_score": 85},
    {"ticker": "BA", "name": "Boeing Co.", "sector": "Industrials", "price": 180.0, "risk_score": 65},

    # Energy
    {"ticker": "XOM", "name": "Exxon Mobil", "sector": "Energy", "price": 115.0, "risk_score": 88},
    {"ticker": "CVX", "name": "Chevron Corp.", "sector": "Energy", "price": 160.0, "risk_score": 85},
    {"ticker": "COP", "name": "ConocoPhillips", "sector": "Energy", "price": 120.0, "risk_score": 84},
    {"ticker": "SLB", "name": "Schlumberger", "sector": "Energy", "price": 50.0, "risk_score": 80},

    # Consumer Staples
    {"ticker": "PG", "name": "Procter & Gamble", "sector": "Consumer Staples", "price": 165.0, "risk_score": 96},
    {"ticker": "COST", "name": "Costco Wholesale", "sector": "Consumer Staples", "price": 750.0, "risk_score": 94},
    {"ticker": "KO", "name": "Coca-Cola Co.", "sector": "Consumer Staples", "price": 60.0, "risk_score": 95},
    {"ticker": "PEP", "name": "PepsiCo Inc.", "sector": "Consumer Staples", "price": 170.0, "risk_score": 94},
    {"ticker": "WMT", "name": "Walmart Inc.", "sector": "Consumer Staples", "price": 60.0, "risk_score": 93},
    {"ticker": "PM", "name": "Philip Morris Int.", "sector": "Consumer Staples", "price": 95.0, "risk_score": 88},

    # Utilities
    {"ticker": "NEE", "name": "NextEra Energy", "sector": "Utilities", "price": 65.0, "risk_score": 90}
]

# Company Specific Content Dictionary
COMPANY_DETAILS = {
    "AAPL": {
        "analysis": "Apple continues to demonstrate robust pricing power despite inflationary pressures. Services revenue is a key growth driver, outpacing hardware sales.",
        "risk": "Regulatory scrutiny in the EU and US regarding the App Store ecosystem remains a primary overhang. Supply chain concentration in Asia poses geopolitical risks."
    },
    "MSFT": {
        "analysis": "Microsoft's Azure cloud division is gaining market share, driven by AI integration. Copilot adoption across the Office suite provides a new recurring revenue stream.",
        "risk": "Enterprise spending slowdowns could impact shorter-term bookings. AI infrastructure costs are significant and may weigh on margins initially."
    },
    "NVDA": {
        "analysis": "NVIDIA is the undisputed leader in AI hardware. Data center revenue has exploded, and software margins are expanding.",
        "risk": "Sustainability of demand at current levels is debated. Export controls to China impact a significant portion of potential addressable market."
    },
    "TSLA": {
        "analysis": "Tesla maintains industry-leading EV margins but faces increasing competition from Chinese OEMs. FSD technology remains a high-potential wildcard.",
        "risk": "Price cuts to maintain volume are compressing margins. Key man risk regarding leadership bandwidth is a persistent investor concern."
    },
    "AMZN": {
        "analysis": "AWS stabilization is encouraging. Retail margins are improving due to regional fulfillment network restructuring.",
        "risk": "Regulatory antitrust investigations are ongoing. Consumer discretionary spending headwinds could impact retail volume."
    }
}

DEFAULT_CONTENT = {
    "analysis": "The company exhibits solid fundamentals with stable cash flow generation. Operational efficiency programs are yielding margin improvements.",
    "risk": "Macroeconomic volatility and interest rate sensitivity remain key risks. Competitive pressures in the sector could impact market share."
}

def get_company_content(ticker):
    return COMPANY_DETAILS.get(ticker, DEFAULT_CONTENT)

def ensure_dirs():
    os.makedirs(EQUITY_DIR, exist_ok=True)
    os.makedirs(CREDIT_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

def generate_financials(company):
    # History: 2022, 2023, 2024
    # Forecast: 2025, 2026, 2027
    years = [2022, 2023, 2024, 2025, 2026, 2027]

    base_revenue = company['price'] * random.uniform(0.5, 2.0) * 1000
    growth_rate = random.uniform(0.05, 0.15) if company['sector'] == 'Technology' else random.uniform(0.02, 0.08)

    data = {
        "years": years,
        "revenue": [],
        "eps": [],
        "ebitda": [],
        "consensus_revenue": [],
        "system_revenue": []
    }

    # Generate continuous stream
    current_rev = base_revenue

    for i in range(len(years)):
        # Apply growth
        year_growth = growth_rate + random.uniform(-0.02, 0.02)
        current_rev = current_rev * (1 + year_growth)

        margin = random.uniform(0.15, 0.35)
        ni = current_rev * margin
        ebitda = ni * 1.3
        eps = ni / (base_revenue / 50)

        data['revenue'].append(round(current_rev / 1000, 1))
        data['eps'].append(round(eps, 2))
        data['ebitda'].append(round(ebitda / 1000, 1))

        # Forecast divergence (last 3 years)
        if i >= 3:
            divergence = random.uniform(-0.05, 0.05)
            sys_divergence = random.uniform(-0.02, 0.08) # System slightly more optimistic/different

            data['consensus_revenue'].append(round((current_rev * (1+divergence)) / 1000, 1))
            data['system_revenue'].append(round((current_rev * (1+sys_divergence)) / 1000, 1))
        else:
            data['consensus_revenue'].append(None)
            data['system_revenue'].append(None)

    return data

def generate_credit_metrics(company):
    score = company['risk_score']
    leverage = random.uniform(0.5, 1.5) if score > 90 else random.uniform(1.5, 3.5) if score > 80 else random.uniform(3.5, 5.0)
    coverage = random.uniform(15, 30) if score > 90 else random.uniform(8, 15) if score > 80 else random.uniform(3, 8)
    liquidity_score = score + random.uniform(-5, 5)

    # PD Rating logic
    if score > 90: pd = "0.01% - Minimal"
    elif score > 80: pd = "0.05% - Very Low"
    elif score > 70: pd = "0.15% - Low"
    else: pd = "0.50% - Moderate"

    # Regulatory Rating
    if score > 75: reg = "Pass"
    elif score > 60: reg = "Special Mention"
    else: reg = "Substandard"

    return {
        "leverage": round(leverage, 1),
        "interest_coverage": round(coverage, 1),
        "liquidity_score": round(min(100, liquidity_score), 1),
        "rating": "AAA" if score > 95 else "AA" if score > 90 else "A" if score > 80 else "BBB" if score > 70 else "BB",
        "pd_rating": pd,
        "regulatory_rating": reg
    }

def generate_outlook(company):
    analysts = random.randint(20, 50)
    buy = int(analysts * (random.uniform(0.5, 0.9) if company['risk_score'] > 80 else random.uniform(0.2, 0.5)))
    hold = int((analysts - buy) * random.uniform(0.5, 0.8))
    sell = analysts - buy - hold

    upside = random.uniform(0.1, 0.3) if company['risk_score'] > 80 else random.uniform(-0.1, 0.1)
    target = company['price'] * (1 + upside)

    return {
        "consensus": "Buy" if buy > hold + sell else "Hold" if hold > buy else "Sell",
        "price_target": round(target, 2),
        "analysts": {"buy": buy, "hold": hold, "sell": sell},
        "conviction": "High" if buy/analysts > 0.7 else "Medium" if buy/analysts > 0.4 else "Low",
        "rationale": "Strong fundamentals and market leadership." if buy/analysts > 0.6 else "Valuation concerns balance growth prospects."
    }

def generate_equity_report(company, financials, outlook, metrics):
    details = get_company_content(company['ticker'])

    content = f"""
    <html>
    <head>
        <title>{company['name']} ({company['ticker']}) - Equity Research</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; padding: 40px; line-height: 1.6; color: #333; max-width: 900px; margin: auto; background: #fff; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
            h1 {{ color: #004e98; border-bottom: 2px solid #004e98; padding-bottom: 10px; margin-top: 0; }}
            h2 {{ color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 30px; }}
            .header {{ display: flex; justify-content: space-between; margin-bottom: 30px; align-items: flex-end; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px; }}
            .metric {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; border: 1px solid #e9ecef; }}
            .metric h3 {{ margin: 0; font-size: 13px; text-transform: uppercase; color: #6c757d; letter-spacing: 0.5px; }}
            .metric p {{ margin: 5px 0 0; font-size: 20px; font-weight: 600; color: #212529; }}
            .table-container {{ margin-top: 20px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px; text-align: right; border-bottom: 1px solid #dee2e6; }}
            th {{ text-align: right; background: #f1f3f5; color: #495057; font-weight: 600; }}
            th:first-child, td:first-child {{ text-align: left; }}
            .buy-rating {{ color: #28a745; font-weight: bold; font-size: 24px; }}
            .hold-rating {{ color: #ffc107; font-weight: bold; font-size: 24px; }}
            .sell-rating {{ color: #dc3545; font-weight: bold; font-size: 24px; }}
            .forecast-val {{ color: #004e98; font-style: italic; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <h1 style="margin-bottom: 5px;">{company['name']} ({company['ticker']})</h1>
                <p style="margin: 0; color: #666;">{company['sector']} | Equity Research | {datetime.now().strftime('%B %d, %Y')}</p>
            </div>
            <div style="text-align: right;">
                <div class="{outlook['consensus'].lower()}-rating">{outlook['consensus'].upper()}</div>
                <p style="margin: 5px 0 0; font-size: 14px;">Target: ${outlook['price_target']}</p>
            </div>
        </div>

        <div class="metric-grid">
            <div class="metric"><h3>Price</h3><p>${company['price']}</p></div>
            <div class="metric"><h3>P/E Ratio</h3><p>{metrics['pe_ratio']}x</p></div>
            <div class="metric"><h3>EPS (2024)</h3><p>${financials['eps'][2]}</p></div>
            <div class="metric"><h3>Div Yield</h3><p>{metrics['dividend_yield']}%</p></div>
        </div>

        <div class="section">
            <h2>Investment Thesis</h2>
            <p><strong>Conviction: {outlook['conviction']}</strong></p>
            <p>{details['analysis']} {company['name']} is positioned to benefit from secular tailwinds in the {company['sector']} industry.</p>
            <p><strong>Risks:</strong> {details['risk']}</p>
        </div>

        <div class="section">
            <h2>Financial Forecast (Billions)</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Year</th>
                            <th>2022 (A)</th>
                            <th>2023 (A)</th>
                            <th>2024 (A)</th>
                            <th class="forecast-val">2025 (E)</th>
                            <th class="forecast-val">2026 (E)</th>
                            <th class="forecast-val">2027 (E)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Revenue</td>
                            <td>${financials['revenue'][0]}</td>
                            <td>${financials['revenue'][1]}</td>
                            <td>${financials['revenue'][2]}</td>
                            <td class="forecast-val">${financials['revenue'][3]}</td>
                            <td class="forecast-val">${financials['revenue'][4]}</td>
                            <td class="forecast-val">${financials['revenue'][5]}</td>
                        </tr>
                        <tr>
                            <td>Consensus Rev</td>
                            <td>-</td>
                            <td>-</td>
                            <td>-</td>
                            <td>${financials['consensus_revenue'][3]}</td>
                            <td>${financials['consensus_revenue'][4]}</td>
                            <td>${financials['consensus_revenue'][5]}</td>
                        </tr>
                        <tr>
                            <td>System Rev</td>
                            <td>-</td>
                            <td>-</td>
                            <td>-</td>
                            <td><strong>${financials['system_revenue'][3]}</strong></td>
                            <td><strong>${financials['system_revenue'][4]}</strong></td>
                            <td><strong>${financials['system_revenue'][5]}</strong></td>
                        </tr>
                        <tr>
                            <td>EBITDA</td>
                            <td>${financials['ebitda'][0]}</td>
                            <td>${financials['ebitda'][1]}</td>
                            <td>${financials['ebitda'][2]}</td>
                            <td class="forecast-val">${financials['ebitda'][3]}</td>
                            <td class="forecast-val">${financials['ebitda'][4]}</td>
                            <td class="forecast-val">${financials['ebitda'][5]}</td>
                        </tr>
                        <tr>
                            <td>EPS</td>
                            <td>${financials['eps'][0]}</td>
                            <td>${financials['eps'][1]}</td>
                            <td>${financials['eps'][2]}</td>
                            <td class="forecast-val">${financials['eps'][3]}</td>
                            <td class="forecast-val">${financials['eps'][4]}</td>
                            <td class="forecast-val">${financials['eps'][5]}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <h2>Analyst Consensus</h2>
            <p><strong>Breakdown:</strong> {outlook['analysts']['buy']} Buy / {outlook['analysts']['hold']} Hold / {outlook['analysts']['sell']} Sell</p>
            <p>The street remains generally {outlook['consensus'].lower()}ish on the name.</p>
        </div>

        <div class="footer" style="margin-top: 50px; border-top: 1px solid #ccc; padding-top: 10px; font-size: 12px; color: #888; text-align: center;">
            Generated by Office Nexus Equity Research Module. Strictly Confidential.
        </div>
    </body>
    </html>
    """
    with open(os.path.join(EQUITY_DIR, f"{company['ticker']}_Equity_Report.html"), 'w') as f:
        f.write(content)

def generate_credit_report(company, credit, financials):
    details = get_company_content(company['ticker'])

    content = f"""
    <html>
    <head>
        <title>{company['name']} ({company['ticker']}) - Credit Memo</title>
        <style>
            body {{ font-family: 'Times New Roman', serif; padding: 50px; line-height: 1.5; color: #000; max-width: 800px; margin: auto; background: #fff; }}
            h1 {{ text-align: center; text-decoration: underline; margin-bottom: 30px; font-size: 24px; }}
            h2 {{ font-size: 16px; border-bottom: 1px solid #000; padding-bottom: 3px; margin-top: 25px; text-transform: uppercase; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 14px; }}
            td, th {{ border: 1px solid #000; padding: 6px 10px; text-align: left; }}
            th {{ background: #eee; font-weight: bold; }}
            .risk-score {{ font-size: 18px; font-weight: bold; color: {'#006400' if company['risk_score'] > 80 else '#8b0000'}; }}
        </style>
    </head>
    <body>
        <h1>CREDIT APPROVAL MEMORANDUM</h1>

        <table>
            <tr>
                <th width="20%">Borrower:</th>
                <td width="30%">{company['name']}</td>
                <th width="20%">Ticker:</th>
                <td width="30%">{company['ticker']}</td>
            </tr>
            <tr>
                <th>Sector:</th>
                <td>{company['sector']}</td>
                <th>Date:</th>
                <td>{datetime.now().strftime('%Y-%m-%d')}</td>
            </tr>
            <tr>
                <th>Internal Rating:</th>
                <td>{credit['rating']}</td>
                <th>Score:</th>
                <td class="risk-score">{company['risk_score']}/100</td>
            </tr>
            <tr>
                <th>PD Rating:</th>
                <td>{credit['pd_rating']}</td>
                <th>Regulatory:</th>
                <td>{credit['regulatory_rating']}</td>
            </tr>
        </table>

        <h2>Executive Summary</h2>
        <p>This memorandum recommends approval of the proposed credit facility for {company['name']}. {details['analysis']}</p>

        <h2>Key Credit Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Guideline/Covenant</th>
            </tr>
            <tr>
                <td>Net Leverage (Debt/EBITDA)</td>
                <td>{credit['leverage']}x</td>
                <td>&lt; 3.5x</td>
            </tr>
            <tr>
                <td>Interest Coverage (EBITDA/Int)</td>
                <td>{credit['interest_coverage']}x</td>
                <td>&gt; 5.0x</td>
            </tr>
            <tr>
                <td>Liquidity Score</td>
                <td>{credit['liquidity_score']}</td>
                <td>&gt; 70</td>
            </tr>
        </table>

        <h2>Financial Analysis</h2>
        <p><strong>Revenue & Earnings:</strong> The company generated ${financials['revenue'][2]}B in revenue in 2024, with EBITDA of ${financials['ebitda'][2]}B. Forecasts indicate continued growth.</p>
        <p><strong>Liquidity:</strong> The company maintains a strong liquidity profile with access to capital markets and revolving credit facilities.</p>
        <p><strong>Cash Flow:</strong> Free Cash Flow conversion remains strong, supporting debt service and capital returns.</p>

        <h2>Risk Factors</h2>
        <ul>
            <li>{details['risk']}</li>
            <li>Macroeconomic headwinds impacting consumer spending.</li>
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

def generate_market_data_and_artifacts():
    print("Generating S&P 500 Micro Build Data...")
    ensure_dirs()

    market_data = []

    for company in COMPANIES:
        # Generate Component Data
        financials = generate_financials(company)
        credit = generate_credit_metrics(company)
        outlook = generate_outlook(company)

        # Calculate derived metrics
        pe_ratio = round(company['price'] / financials['eps'][2], 1)
        dividend_yield = round(random.uniform(0.5, 3.5), 2)

        metrics = {
            "pe_ratio": pe_ratio,
            "dividend_yield": dividend_yield
        }

        # Generate Reports
        generate_equity_report(company, financials, outlook, metrics)
        generate_credit_report(company, credit, financials)
        generate_documents(company)

        # Generate Price History (1 Year)
        history = []
        curr = company['price']
        for i in range(252):
            history.append(round(curr, 2))
            change = random.uniform(-0.02, 0.02)
            curr = curr / (1 + change) # Go back in time
        history.reverse() # Now index 0 is 1 year ago, last index is current price.

        # Aggregate Data
        market_data.append({
            "ticker": company['ticker'],
            "name": company['name'],
            "sector": company['sector'],
            "current_price": company['price'],
            "change_pct": round(random.uniform(-1.5, 1.5), 2),
            "market_cap": f"{random.uniform(0.1, 3.0):.1f}T",
            "pe_ratio": pe_ratio,
            "dividend_yield": dividend_yield,
            "price_history": history,
            "risk_score": company['risk_score'],
            "financials": financials,
            "credit": credit,
            "outlook": outlook
        })

        print(f"Generated artifacts for {company['ticker']}")

    with open(MARKET_DATA_FILE, 'w') as f:
        json.dump(market_data, f, indent=2)

    with open(MARKET_DATA_JS_FILE, 'w') as f:
        json_str = json.dumps(market_data, indent=2)
        f.write(f"window.MARKET_DATA = {json_str};")

    print(f"Market data saved to {MARKET_DATA_FILE} and {MARKET_DATA_JS_FILE}")
    print("Done.")

if __name__ == "__main__":
    generate_market_data_and_artifacts()
