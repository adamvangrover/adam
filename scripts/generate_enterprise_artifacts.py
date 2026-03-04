import json
import os
import random
import datetime
import uuid
import hashlib
from jinja2 import Environment, FileSystemLoader

# Import our new modules (assuming PYTHONPATH is correct or relative imports work)
import sys
sys.path.append('.')
from core.simulation.financial_statement_generator import FinancialStatementGenerator

DATA_DIR = 'showcase/data'
TEMPLATE_DIR = 'showcase/templates'
MODELS_DIR = os.path.join(DATA_DIR, 'models')
CREDIT_REPORTS_DIR = os.path.join(DATA_DIR, 'credit_reports')
EQUITY_REPORTS_DIR = os.path.join(DATA_DIR, 'equity_reports')

# Setup Jinja2
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)

def generate_enterprise_artifacts():
    print("Initializing Enterprise Artifact Generator...")

    # Load market data
    with open(os.path.join(DATA_DIR, 'sp500_market_data.json'), 'r') as f:
        market_data = json.load(f)

    gen = FinancialStatementGenerator(seed=42)

    for item in market_data:
        ticker = item['ticker']
        print(f"Processing {ticker}...")

        # 1. Generate Realistic Financials
        financials = gen.generate_3_statement_model(ticker, item['sector'], item['market_cap'], item['current_price'])

        # 2. Generate Audit Metadata
        audit = gen.generate_audit_trail()
        audit_id = str(uuid.uuid4())
        data_hash = hashlib.sha256(json.dumps(financials).encode()).hexdigest()[:16]

        # 3. Create JSON Model with Audit
        model_data = {
            "ticker": ticker,
            "financials": financials,
            "audit_trail": {
                **audit,
                "audit_id": audit_id,
                "data_hash": data_hash
            }
        }

        with open(os.path.join(MODELS_DIR, f"{ticker}_Financial_Model_v2.json"), 'w') as f:
            json.dump(model_data, f, indent=2)

        # 4. Render Institutional Equity Report
        equity_template = env.get_template('institutional_report.html')

        # Extract metrics for template
        fy23 = financials[2023]["Income Statement"]
        fy24 = financials[2024]["Income Statement"]
        fy25 = financials[2025]["Income Statement"]

        # Mock Risks based on sector
        risks = [
            f"Cyclical downturn in {item['sector']}.",
            "Regulatory changes affecting margins.",
            "Execution risk on new product launches."
        ]

        equity_html = equity_template.render(
            company_name=item['name'],
            ticker=ticker,
            rating=random.choice(["BUY", "HOLD", "SELL"]),
            date=datetime.date.today().strftime("%B %d, %Y"),
            analyst="Nexus Research Team",
            sector=item['sector'],
            target_price=round(item['current_price'] * random.uniform(0.8, 1.4), 2),
            current_price=item['current_price'],
            upside=random.randint(-10, 40),
            risk_rating=random.choice(["Low", "Medium", "High"]),
            thesis=f"{item['name']} is positioned to capitalize on secular trends in {item['sector']}. Our analysis suggests the market is underappreciating the margin expansion story.",
            rev_2023=fy23["Revenue"], rev_2024=fy24["Revenue"], rev_2025=fy25["Revenue"],
            ebitda_2023=fy23["EBITDA"], ebitda_2024=fy24["EBITDA"], ebitda_2025=fy25["EBITDA"],
            ni_2023=fy23["Net Income"], ni_2024=fy24["Net Income"], ni_2025=fy25["Net Income"],
            eps_2023=round(fy23["Net Income"]/1.0, 2), eps_2024=round(fy24["Net Income"]/1.0, 2), eps_2025=round(fy25["Net Income"]/1.0, 2), # Mock share count 1B
            wacc=round(random.uniform(8.0, 11.0), 1),
            terminal_growth=2.5,
            risks=risks,
            model_id=audit_id,
            model_version=audit['model_version'],
            cutoff_date=audit['data_cutoff'],
            compliance_status=audit['compliance_check'],
            processing_time=random.randint(100, 500)
        )

        with open(os.path.join(EQUITY_REPORTS_DIR, f"{ticker}_Institutional_Report_2026.html"), 'w') as f:
            f.write(equity_html)

        # 5. Render Credit Memo Pro
        credit_template = env.get_template('credit_memo_pro.html')

        # Ratios
        lev23 = financials[2023]["Ratios"]["Leverage (Debt/EBITDA)"]
        lev24 = financials[2024]["Ratios"]["Leverage (Debt/EBITDA)"]
        lev25 = financials[2025]["Ratios"]["Leverage (Debt/EBITDA)"]

        debt23 = financials[2023]["Balance Sheet"]["Total Debt"]
        debt24 = financials[2024]["Balance Sheet"]["Total Debt"]
        debt25 = financials[2025]["Balance Sheet"]["Total Debt"]

        cash25 = financials[2025]["Balance Sheet"]["Cash & Equivalents"]

        # Interest Coverage (EBITDA / Interest) - interest approx 5% of debt
        cov23 = round(fy23["EBITDA"] / (debt23 * 0.05), 1) if debt23 > 0 else "N/A"
        cov24 = round(fy24["EBITDA"] / (debt24 * 0.05), 1) if debt24 > 0 else "N/A"
        cov25 = round(fy25["EBITDA"] / (debt25 * 0.05), 1) if debt25 > 0 else "N/A"

        credit_html = credit_template.render(
            company_name=item['name'],
            ticker=ticker,
            date=datetime.date.today().strftime("%Y-%m-%d"),
            sector=item['sector'],
            credit_rating=random.choice(["AAA", "AA", "A", "BBB", "BB"]),
            outlook="Stable",
            executive_summary=f"{item['name']} maintains a robust credit profile with strong liquidity and manageable leverage ratios. Recent financial performance indicates stable cash flow generation capable of supporting current debt obligations.",
            rev_2023=fy23["Revenue"], rev_2024=fy24["Revenue"], rev_2025=fy25["Revenue"],
            ebitda_2023=fy23["EBITDA"], ebitda_2024=fy24["EBITDA"], ebitda_2025=fy25["EBITDA"],
            debt_2023=debt23, debt_2024=debt24, debt_2025=debt25,
            lev_2023=lev23, lev_2024=lev24, lev_2025=lev25,
            cov_2023=cov23, cov_2024=cov24, cov_2025=cov25,
            risks=risks,
            cash=cash25,
            risk_score=item['risk_score'],
            recommendation="MAINTAIN",
            audit_id=audit_id,
            data_hash=data_hash,
            rag_sources_count=random.randint(5, 20)
        )

        with open(os.path.join(CREDIT_REPORTS_DIR, f"{ticker}_Credit_Memo_Pro_2026.html"), 'w') as f:
            f.write(credit_html)

    print("Enterprise artifacts generated successfully.")

if __name__ == "__main__":
    generate_enterprise_artifacts()
