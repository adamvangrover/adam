from pydantic import BaseModel

class MacroScenario(BaseModel):
    name: str
    description: str
    unemployment_rate: float # Percentage (e.g., 10.0 for 10%)
    real_gdp_change: float # Percentage
    equity_market_change: float # Percentage
    home_price_change: float # Percentage
    cre_price_change: float # Percentage (Commercial Real Estate)
    corporate_bond_spread_change_bps: int # Basis Points
    vix_peak: float
    market_regime: str = "Liquidity Crunch"

class CCAR2025SeverelyAdverse(MacroScenario):
    """
    Federal Reserve CCAR 2025 Severely Adverse Scenario Parameters.
    """
    name: str = "CCAR 2025 Severely Adverse"
    description: str = "Severe global recession with heightened stress in commercial real estate and corporate debt markets."
    unemployment_rate: float = 10.0
    real_gdp_change: float = -7.8
    equity_market_change: float = -50.0
    home_price_change: float = -33.0
    cre_price_change: float = -30.0
    corporate_bond_spread_change_bps: int = 390
    vix_peak: float = 65.0

# Instance for easy import
CCAR_2025 = CCAR2025SeverelyAdverse()
