"""
Script to generate mock 10-K text files for testing the RAG pipeline.
"""
import os
import json

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MOCK_DATA = [
    {
        "ticker": "MSFT",
        "name": "Microsoft Corporation",
        "revenue": 211915,
        "net_income": 72361,
        "assets": 364840,
        "liabilities": 184256,
        "debt": 47204,
        "cash": 48366,
        "op_income": 88524,
        "risks": [
            "AI competition intensifying from Google and OpenAI.",
            "Regulatory scrutiny on acquisitions (Activision).",
            "Cybersecurity threats to Azure infrastructure.",
            "Global PC market slowdown affecting Windows revenue."
        ],
        "desc": "Microsoft develops, licenses, and supports software, services, devices, and solutions worldwide. Its Productivity and Business Processes segment offers Office, Exchange, SharePoint, Microsoft Teams, Office 365 Security and Compliance, and Skype for Business."
    },
    {
        "ticker": "GOOGL",
        "name": "Alphabet Inc.",
        "revenue": 282836,
        "net_income": 59972,
        "assets": 365264,
        "liabilities": 109120,
        "debt": 14700,
        "cash": 21879,
        "op_income": 74842,
        "risks": [
            "Antitrust litigation regarding Search dominance.",
            "Ad revenue volatility due to economic downturns.",
            "Generative AI disrupting core search business model.",
            "Data privacy regulations (GDPR, CCPA) increasing compliance costs."
        ],
        "desc": "Alphabet Inc. offers various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America. It operates through Google Services, Google Cloud, and Other Bets segments."
    },
    {
        "ticker": "AMZN",
        "name": "Amazon.com Inc.",
        "revenue": 513983,
        "net_income": -2722,
        "assets": 462675,
        "liabilities": 316632,
        "debt": 67150,
        "cash": 53888,
        "op_income": 12248,
        "risks": [
            "Intense competition in e-commerce and cloud computing.",
            "Supply chain disruptions and logistics costs.",
            "Regulatory scrutiny on marketplace practices.",
            "Labor unionization efforts and workforce management."
        ],
        "desc": "Amazon.com, Inc. engages in the retail sale of consumer products and subscriptions in North America and internationally. The company operates through three segments: North America, International, and Amazon Web Services (AWS)."
    },
    {
        "ticker": "NVDA",
        "name": "NVIDIA Corporation",
        "revenue": 26974,
        "net_income": 4368,
        "assets": 41182,
        "liabilities": 19081,
        "debt": 10950,
        "cash": 3389,
        "op_income": 5577,
        "risks": [
            "Dependency on Taiwan for semiconductor manufacturing.",
            "Cyclical nature of the semiconductor industry.",
            "Geopolitical tensions restricting sales to China.",
            "Competition from custom AI chips by hyperscalers."
        ],
        "desc": "NVIDIA Corporation provides graphics, and compute and networking solutions in the United States, Taiwan, China, and internationally. The company's Graphics segment offers GeForce GPUs for gaming and PCs."
    },
    {
        "ticker": "TSLA",
        "name": "Tesla Inc.",
        "revenue": 81462,
        "net_income": 12556,
        "assets": 82338,
        "liabilities": 36440,
        "debt": 5748,
        "cash": 16253,
        "op_income": 13656,
        "risks": [
            "Production scaling challenges and supply chain constraints.",
            "Intense competition in the EV market from legacy automakers.",
            "Regulatory scrutiny on Autopilot and FSD features.",
            "CEO reputational risk and distraction."
        ],
        "desc": "Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, and energy generation and storage systems in the United States, China, and internationally. The company operates in two segments: Automotive, and Energy Generation and Storage."
    },
    {
        "ticker": "META",
        "name": "Meta Platforms Inc.",
        "revenue": 116609,
        "net_income": 23200,
        "assets": 185727,
        "liabilities": 60014,
        "debt": 9920,
        "cash": 14680,
        "op_income": 28944,
        "risks": [
            "App Tracking Transparency impacting ad targeting efficiency.",
            "Reality Labs investments yielding uncertain returns.",
            "Regulatory scrutiny on content moderation and user privacy.",
            "Competition from TikTok for user attention."
        ],
        "desc": "Meta Platforms, Inc. engages in the development of products that enable people to connect and share with friends and family through mobile devices, personal computers, virtual reality headsets, and in-home devices worldwide."
    },
    {
        "ticker": "JPM",
        "name": "JPMorgan Chase & Co.",
        "revenue": 128695,
        "net_income": 37676,
        "assets": 3665743,
        "liabilities": 3373405,
        "debt": 298450,
        "cash": 526780,
        "op_income": 45000,
        "risks": [
            "Interest rate volatility affecting net interest income.",
            "Credit quality deterioration in commercial real estate.",
            "Cybersecurity threats to financial infrastructure.",
            "Regulatory capital requirements (Basel III endgame)."
        ],
        "desc": "JPMorgan Chase & Co. operates as a financial services company worldwide. It operates through four segments: Consumer & Community Banking (CCB), Corporate & Investment Bank (CIB), Commercial Banking (CB), and Asset & Wealth Management (AWM)."
    },
    {
        "ticker": "GS",
        "name": "The Goldman Sachs Group",
        "revenue": 47365,
        "net_income": 11261,
        "assets": 1441749,
        "liabilities": 1324560,
        "debt": 260500,
        "cash": 150000,
        "op_income": 13500,
        "risks": [
            "Market volatility impacting trading and investment banking revenues.",
            "Regulatory compliance costs and legal risks.",
            "Talent retention in a competitive environment.",
            "Operational risks from complex financial products."
        ],
        "desc": "The Goldman Sachs Group, Inc., a financial institution, delivers a range of financial services for corporations, financial institutions, governments, and individuals worldwide. It operates through Global Banking & Markets, Asset & Wealth Management, and Platform Solutions segments."
    },
    {
        "ticker": "NFLX",
        "name": "Netflix Inc.",
        "revenue": 31616,
        "net_income": 4492,
        "assets": 48595,
        "liabilities": 27810,
        "debt": 14353,
        "cash": 6082,
        "op_income": 5633,
        "risks": [
            "Intense competition from other streaming services.",
            "Content production costs and strike impacts.",
            "Subscriber growth saturation in mature markets.",
            "Password sharing crackdown implementation risks."
        ],
        "desc": "Netflix, Inc. provides entertainment services. It offers TV series, documentaries, feature films, and mobile games across various genres and languages. The company provides members the ability to receive streaming content through a host of internet-connected devices."
    }
]

def generate_text(data):
    # Template matching the regex patterns in run_credit_memo_rag.py
    return f"""
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549
FORM 10-K

ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
For the fiscal year ended December 31, 2024

{data['name']}
(Exact name of registrant as specified in its charter)

Item 1. Business
{data['desc']}

Item 1A. Risk Factors
The following risk factors could materially affect our business, financial condition, or results of operations:
{chr(10).join(['- ' + r for r in data['risks']])}
- Additional general economic risks including inflation and interest rates.

Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
Overview
We are a leading global company...

Results of Operations
Total net sales were ${data['revenue']:,.1f} million for the year ended December 31, 2024.
Net income was ${data['net_income']:,.1f} million.
Operating income: ${data['op_income']:,.1f} million.

Liquidity and Capital Resources
Cash and cash equivalents: ${data['cash']:,.1f} million.
Total debt was ${data['debt']:,.1f} million as of December 31, 2024.

Item 8. Financial Statements and Supplementary Data
Consolidated Balance Sheets
(In millions)
Assets
Total assets: ${data['assets']:,.1f}

Liabilities and Stockholders’ Equity
Total liabilities: ${data['liabilities']:,.1f}
Total stockholders’ equity: ${data['assets'] - data['liabilities']:,.1f}
    """

def main():
    print(f"Generating {len(MOCK_DATA)} mock 10-K files...")
    for item in MOCK_DATA:
        filename = f"10k_sample_{item['ticker'].lower()}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        content = generate_text(item)

        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Generated {filepath}")

if __name__ == "__main__":
    main()
