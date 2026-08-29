import json

data = [
    # Crypto Hashes/Coins
    {"name": "Bitcoin", "ticker": "BTC", "sector": "Crypto L1", "baseEbitda": 500000000, "volatility": 0.45},
    {"name": "Ethereum", "ticker": "ETH", "sector": "Crypto L1", "baseEbitda": 300000000, "volatility": 0.50},
    {"name": "Solana", "ticker": "SOL", "sector": "Crypto L1", "baseEbitda": 150000000, "volatility": 0.65},
    {"name": "Chainlink", "ticker": "LINK", "sector": "Crypto Oracle", "baseEbitda": 80000000, "volatility": 0.55},
    {"name": "Uniswap", "ticker": "UNI", "sector": "DeFi DEX", "baseEbitda": 90000000, "volatility": 0.60},
    
    # Currencies (FX)
    {"name": "US Dollar", "ticker": "USD", "sector": "Fiat Currency", "baseEbitda": 1000000000, "volatility": 0.05},
    {"name": "Euro", "ticker": "EUR", "sector": "Fiat Currency", "baseEbitda": 800000000, "volatility": 0.06},
    {"name": "Japanese Yen", "ticker": "JPY", "sector": "Fiat Currency", "baseEbitda": 600000000, "volatility": 0.08},
    {"name": "British Pound", "ticker": "GBP", "sector": "Fiat Currency", "baseEbitda": 500000000, "volatility": 0.07},
    {"name": "Swiss Franc", "ticker": "CHF", "sector": "Fiat Currency", "baseEbitda": 400000000, "volatility": 0.05},
    
    # Rates & Hedges
    {"name": "US 10-Year Treasury Yield", "ticker": "TNX", "sector": "Sovereign Rates", "baseEbitda": 1000000000, "volatility": 0.08},
    {"name": "US 2-Year Treasury Yield", "ticker": "IRX", "sector": "Sovereign Rates", "baseEbitda": 1000000000, "volatility": 0.10},
    {"name": "SOFR Rate", "ticker": "SOFR", "sector": "Overnight Rates", "baseEbitda": 500000000, "volatility": 0.04},
    {"name": "VIX Index (Hedge)", "ticker": "VIX", "sector": "Volatility Index", "baseEbitda": 200000000, "volatility": 0.80},
    
    # CDS (Credit Default Swaps)
    {"name": "CDX IG Series 40", "ticker": "CDX.IG", "sector": "CDS Index", "baseEbitda": 800000000, "volatility": 0.15},
    {"name": "CDX HY Series 40", "ticker": "CDX.HY", "sector": "CDS Index", "baseEbitda": 500000000, "volatility": 0.25},
    {"name": "iTraxx Europe Main", "ticker": "ITRX.EU", "sector": "CDS Index", "baseEbitda": 700000000, "volatility": 0.14},
    {"name": "iTraxx Crossover", "ticker": "ITRX.XO", "sector": "CDS Index", "baseEbitda": 400000000, "volatility": 0.28},

    # Structured Products & CDOs
    {"name": "CLO AAA Tranche 2023", "ticker": "CLO.AAA", "sector": "Structured Finance", "baseEbitda": 300000000, "volatility": 0.03},
    {"name": "CLO Mezzanine BB Tranche", "ticker": "CLO.BB", "sector": "Structured Finance", "baseEbitda": 150000000, "volatility": 0.18},
    {"name": "CLO Equity Tranche", "ticker": "CLO.EQ", "sector": "Structured Finance", "baseEbitda": 50000000, "volatility": 0.40},
    {"name": "CMBS Conduit Super Senior", "ticker": "CMBS.SS", "sector": "Structured Finance", "baseEbitda": 400000000, "volatility": 0.05},

    # S&P 500 (Equities)
    {"name": "Apple Inc.", "ticker": "AAPL", "sector": "S&P 500 Tech", "baseEbitda": 120000000000, "volatility": 0.20},
    {"name": "Microsoft Corp.", "ticker": "MSFT", "sector": "S&P 500 Tech", "baseEbitda": 95000000000, "volatility": 0.18},
    {"name": "JPMorgan Chase", "ticker": "JPM", "sector": "S&P 500 Financials", "baseEbitda": 50000000000, "volatility": 0.22},
    {"name": "ExxonMobil", "ticker": "XOM", "sector": "S&P 500 Energy", "baseEbitda": 75000000000, "volatility": 0.25},
    {"name": "Johnson & Johnson", "ticker": "JNJ", "sector": "S&P 500 Healthcare", "baseEbitda": 35000000000, "volatility": 0.15},

    # Broadly Syndicated Loans (BSL) / Generic Corporate Loans
    {"name": "Apex Manufacturing Term Loan B", "ticker": "APEX.TLB", "sector": "BSL Industrials", "baseEbitda": 250000000, "volatility": 0.12},
    {"name": "Nova Healthcare Revolver", "ticker": "NOVA.REV", "sector": "BSL Healthcare", "baseEbitda": 180000000, "volatility": 0.15},
    {"name": "Quantum Tech First Lien", "ticker": "QTK.1L", "sector": "BSL Technology", "baseEbitda": 400000000, "volatility": 0.25},
    {"name": "Stellar Media Unitranche", "ticker": "STLR.UNI", "sector": "Direct Lending", "baseEbitda": 85000000, "volatility": 0.22},

    # Sovereigns
    {"name": "United States Sovereign Debt", "ticker": "US.GOV", "sector": "Sovereign", "baseEbitda": 5000000000, "volatility": 0.02},
    {"name": "German Bunds", "ticker": "DE.GOV", "sector": "Sovereign", "baseEbitda": 4000000000, "volatility": 0.02},
    {"name": "Japan JGBs", "ticker": "JP.GOV", "sector": "Sovereign", "baseEbitda": 3500000000, "volatility": 0.03},
    {"name": "Brazil Sovereign Debt", "ticker": "BR.GOV", "sector": "Emerging Market Sovereign", "baseEbitda": 800000000, "volatility": 0.18},
    {"name": "Argentina Sovereign Bonds", "ticker": "AR.GOV", "sector": "Emerging Market Sovereign", "baseEbitda": 100000000, "volatility": 0.45}
]

with open('live_data.json', 'w') as f:
    json.dump(data, f, indent=4)

print("live_data.json has been populated with diverse asset classes.")
