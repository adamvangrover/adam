import json
import random
import string

random.seed(42)

def random_string(length=4):
    return ''.join(random.choices(string.ascii_uppercase, k=length))

def generate_crypto():
    prefix = random.choice(["Bitcoin", "Ethereum", "Solana", "Avalanche", "Cardano", "Polkadot", "Chainlink", "Polygon", "Cosmos", "Near"])
    suffix = random.choice(["Protocol", "Network", "DAO", "Token", "Coin", "Finance", "Labs", "Yield"])
    name = f"{prefix} {suffix} {random.randint(1, 9999)}"
    ticker = random_string(3) + str(random.randint(1, 9))
    return {
        "name": name,
        "ticker": ticker,
        "sector": "Crypto",
        "baseEbitda": random.uniform(10_000_000, 500_000_000),
        "volatility": random.uniform(0.4, 1.2)
    }

def generate_currency():
    base = random.choice(["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "CNY", "INR"])
    quote = random.choice(["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "CNY", "INR"])
    if base == quote: quote = "ZAR"
    name = f"{base}/{quote} FX Spot"
    ticker = f"{base}{quote}"
    return {
        "name": name,
        "ticker": ticker,
        "sector": "Fiat Currency",
        "baseEbitda": random.uniform(1_000_000_000, 10_000_000_000),
        "volatility": random.uniform(0.03, 0.12)
    }

def generate_rates():
    country = random.choice(["US", "UK", "EU", "JP", "CH", "AU", "CA"])
    duration = random.choice(["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"])
    name = f"{country} {duration} Sovereign Rate"
    ticker = f"{country}{duration}R"
    return {
        "name": name,
        "ticker": ticker,
        "sector": "Sovereign Rates",
        "baseEbitda": random.uniform(1_000_000_000, 5_000_000_000),
        "volatility": random.uniform(0.02, 0.15)
    }

def generate_cds():
    index = random.choice(["CDX IG", "CDX HY", "iTraxx Europe", "iTraxx Crossover", "CDX EM", "Single Name Corporate"])
    series = random.randint(10, 50)
    name = f"{index} Series {series} {random_string(2)}"
    ticker = f"CDS.{random_string(4)}"
    return {
        "name": name,
        "ticker": ticker,
        "sector": "CDS",
        "baseEbitda": random.uniform(100_000_000, 1_000_000_000),
        "volatility": random.uniform(0.1, 0.35)
    }

def generate_structured():
    type_ = random.choice(["CLO", "CMBS", "RMBS", "ABS", "CDO"])
    tranche = random.choice(["AAA", "AA", "A", "BBB", "BB", "B", "Equity"])
    year = random.randint(2015, 2024)
    name = f"Apex {type_} {year}-{random.randint(1,9)} {tranche} Tranche"
    ticker = f"{type_}.{tranche}.{year}"
    return {
        "name": name,
        "ticker": ticker,
        "sector": "Structured Finance",
        "baseEbitda": random.uniform(50_000_000, 500_000_000),
        "volatility": random.uniform(0.02, 0.45) # highly dependent on tranche
    }

def generate_equity():
    prefix = random.choice(["Global", "Advanced", "Quantum", "Nexus", "Stellar", "Apex", "Vanguard", "Pinnacle", "Summit"])
    suffix = random.choice(["Tech", "Health", "Energy", "Materials", "Financials", "Industrials", "Consumer", "Utilities"])
    name = f"{prefix} {suffix} Corp {random.randint(100, 9999)}"
    ticker = random_string(4)
    return {
        "name": name,
        "ticker": ticker,
        "sector": "Equities",
        "baseEbitda": random.uniform(100_000_000, 5_000_000_000),
        "volatility": random.uniform(0.15, 0.4)
    }

def generate_bsl():
    company = random_string(5)
    type_ = random.choice(["Term Loan B", "Revolver", "Unitranche", "Delayed Draw", "First Lien", "Second Lien"])
    name = f"{company} {type_} {random.randint(2025, 2030)}"
    ticker = f"{company}.{type_[:3].upper()}"
    return {
        "name": name,
        "ticker": ticker,
        "sector": "BSL",
        "baseEbitda": random.uniform(20_000_000, 400_000_000),
        "volatility": random.uniform(0.1, 0.25)
    }

def generate_sovereign():
    country = random.choice(["Brazil", "Mexico", "India", "Indonesia", "South Africa", "Turkey", "Argentina", "Chile", "Colombia", "Vietnam"])
    issue = random.choice(["Global 2030", "Local Bond 2025", "Eurobond", "Sovereign Green Bond"])
    name = f"{country} {issue} {random.randint(100, 999)}"
    ticker = f"{country[:3].upper()}.GOV"
    return {
        "name": name,
        "ticker": ticker,
        "sector": "Sovereign Debt",
        "baseEbitda": random.uniform(500_000_000, 10_000_000_000),
        "volatility": random.uniform(0.05, 0.5)
    }

generators = [generate_crypto, generate_currency, generate_rates, generate_cds, generate_structured, generate_equity, generate_bsl, generate_sovereign]

data = []
# Generate 16,000 entities
for i in range(16000):
    gen = random.choice(generators)
    entity = gen()
    # ensure unique id context
    entity['name'] = f"{entity['name']} (ID {i})"
    data.append(entity)

with open('scripts/data_generation/live_data.json', 'w') as f:
    json.dump(data, f)

print(f"Generated {len(data)} entities in scripts/data_generation/live_data.json")
