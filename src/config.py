# src/config.py

DEFAULT_ASSUMPTIONS = {
    'tax_rate': 0.21,
    'risk_free_rate': 0.0425,  # 10Y Treasury
    'market_risk_premium': 0.06,
    'terminal_growth_rate': 0.02,
    'beta': 1.2,
    'projection_years': 5
}

# Regulatory Rating Mapping (internal model proxy)
RATING_MAP = {
    1.0: "IG1 (AAA)",
    3.0: "IG3 (BBB-)",
    4.0: "Pass 4 (BB)",
    5.0: "Pass 5 (B+)",
    6.0: "Pass 6 (B-)",
    7.0: "Special Mention (CCC)",
    8.0: "Substandard (D/Default)"
}
