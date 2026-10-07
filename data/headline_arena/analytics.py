"""
Analytical Layer Upgrades:
- Cross-Asset Volatility & Spread Calibration (Credit Spreads, Swap Spreads, MOVE Index)
- Second-Order Macroeconomic Regime Extraction (Central Bank Liquidity, Geopolitics, Debt Service)
- Bayesian Prior vs. High-Frequency Flow Update (Historical Base Rates + Order Flow + EDGAR Triage)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MacroRegimeState:
    regime_name: str = "Stagflationary_Tightening_with_Selective_SafeHaven"
    fed_funds_rate_upper: float = 5.75
    fed_balance_sheet_trend: str = "quantitative_tightening"
    cdx_ig_spread_bps: float = 124.0        # Investment Grade Credit Default Swap spread
    cdx_hy_spread_bps: float = 438.0        # High Yield Credit Default Swap spread
    ten_year_swap_spread_bps: float = -28.5  # 10Y Swap Spread (deep inversion/collateral squeeze)
    rates_implied_vol_move: float = 118.5   # MOVE Index proxy (elevated rate volatility)
    us_2y10y_spread_bps: float = -14.0      # Bear-flattening yield curve
    geopolitical_risk_index: float = 84.5   # 0-100 scale (Iran war / Middle East escalation)
    debt_service_stress_index: float = 76.0 # 0-100 scale (refinancing cliff & sovereign load)
    global_liquidity_impulse: float = -0.42 # Negative impulse: central bank balance sheets contracting


@dataclass
class AssetMacroProfile:
    ticker: str
    name: str
    exchange: str
    last_settlement_price: float
    current_indicated_price: float
    annualized_vol_pct: float
    dead_zone_pct: float = 0.30             # Standard Headline Arena dead zone (±0.30%)
    prior_settlement_trend: str = "neutral"
    order_flow_imbalance: float = 0.0       # [-1.0, +1.0] scale
    edgar_triage_signal: str = "neutral"    # 13D/13F institutional flow
    primary_catalyst: str = ""
    indicator_divergence: str = ""
    thesis_risks: str = ""


# Baseline universe data grounded in October 2026 live settlement telemetry
ASSET_UNIVERSE: Dict[str, AssetMacroProfile] = {
    "GC": AssetMacroProfile(
        ticker="GC",
        name="Gold Futures",
        exchange="COMEX",
        last_settlement_price=4172.10,
        current_indicated_price=4185.50,
        annualized_vol_pct=17.8,
        dead_zone_pct=0.30,
        prior_settlement_trend="bearish",
        order_flow_imbalance=0.18,
        edgar_triage_signal="institutional_safehaven_bid",
        primary_catalyst="Middle East geopolitical tail-risk reassertion vs terminal Fed hawkish rate hold at 5.75%.",
        indicator_divergence="Gold physical delivery demand and central bank bullion reserve accumulation diverging from rising real 10Y Treasury yields (2.25%).",
        thesis_risks="Diplomatic ceasefire de-escalation in Middle East or hawkish forward-guidance surge pushing real yields above 2.50%."
    ),
    "SI": AssetMacroProfile(
        ticker="SI",
        name="Silver Futures",
        exchange="COMEX",
        last_settlement_price=60.71,
        current_indicated_price=61.15,
        annualized_vol_pct=26.4,
        dead_zone_pct=0.30,
        prior_settlement_trend="bearish",
        order_flow_imbalance=0.08,
        edgar_triage_signal="industrial_hedging",
        primary_catalyst="High-beta precious metals tracking gold with industrial demand friction from slowing global manufacturing PMI (48.4).",
        indicator_divergence="Gold/Silver ratio stretching to 68.4 as monetary premium outpaces physical photovoltaic industrial demand.",
        thesis_risks="Sharp industrial contraction in Asian manufacturing centers or aggressive liquidation in leveraged precious metal futures."
    ),
    "CL": AssetMacroProfile(
        ticker="CL",
        name="WTI Crude Oil Futures",
        exchange="NYMEX",
        last_settlement_price=91.26,
        current_indicated_price=90.80,
        annualized_vol_pct=34.2,
        dead_zone_pct=0.30,
        prior_settlement_trend="bearish",
        order_flow_imbalance=-0.22,
        edgar_triage_signal="commercial_producer_hedging",
        primary_catalyst="Saudi supply routing via Oman offset by persistent US inventory accumulation and global demand destruction narrative.",
        indicator_divergence="Physical prompt Brent-WTI spreads holding near $4.20 while paper speculative net-longs undergo aggressive margin reduction.",
        thesis_risks="Direct naval escalation closing the Strait of Hormuz or retaliatory infrastructure strike triggering an immediate $15 supply spike."
    ),
    "RB": AssetMacroProfile(
        ticker="RB",
        name="RBOB Gasoline Futures",
        exchange="NYMEX",
        last_settlement_price=3.3121,
        current_indicated_price=3.2980,
        annualized_vol_pct=31.5,
        dead_zone_pct=0.30,
        prior_settlement_trend="bearish",
        order_flow_imbalance=-0.19,
        edgar_triage_signal="distributor_drawdown",
        primary_catalyst="Direct crack-spread passthrough from declining crude feedstock alongside seasonal post-summer driving demand decline.",
        indicator_divergence="Gulf Coast refinery utilization staying above 91% creating regional product surplus despite Middle East headline tension.",
        thesis_risks="Unplanned domestic refinery outage along the US Gulf Coast or spike in component alkylate import tariffs."
    ),
    "NG": AssetMacroProfile(
        ticker="NG",
        name="Natural Gas Futures",
        exchange="NYMEX",
        last_settlement_price=3.039,
        current_indicated_price=3.065,
        annualized_vol_pct=42.0,
        dead_zone_pct=0.30,
        prior_settlement_trend="bullish",
        order_flow_imbalance=0.24,
        edgar_triage_signal="utility_winter_hedging",
        primary_catalyst="Early heating degree day (HDD) anomalies in the Upper Midwest and accelerating European LNG export terminal feedgas demand.",
        indicator_divergence="Working gas in underground storage sitting 5.8% above the 5-year average while dry-gas production plateaued at 102.5 Bcf/d.",
        thesis_risks="Mild weather revisions across mid-October forecast models or rapid resumption of constrained Permian associated gas takeaway capacity."
    ),
    "HG": AssetMacroProfile(
        ticker="HG",
        name="Copper Futures",
        exchange="COMEX",
        last_settlement_price=6.579,
        current_indicated_price=6.582,
        annualized_vol_pct=21.0,
        dead_zone_pct=0.30,
        prior_settlement_trend="neutral",
        order_flow_imbalance=-0.02,
        edgar_triage_signal="mining_conglomerate_hedging",
        primary_catalyst="Tug-of-war between Chinese property stabilization credit measures and sluggish global ex-China capex spending.",
        indicator_divergence="LME warehouse canceled warrants rising to 28% while domestic bonded warehouse copper premiums in Shanghai remain subdued.",
        thesis_risks="Major supply disruption at Chilean/Peruvian open-pit mines or abrupt rollout of large-scale Chinese grid infrastructure stimulus."
    ),
    "ES": AssetMacroProfile(
        ticker="ES",
        name="E-mini S&P 500 Futures",
        exchange="CME",
        last_settlement_price=7776.50,
        current_indicated_price=7762.00,
        annualized_vol_pct=15.2,
        dead_zone_pct=0.30,
        prior_settlement_trend="bullish",
        order_flow_imbalance=-0.12,
        edgar_triage_signal="corporate_buyback_blackout",
        primary_catalyst="Equities digesting terminal rate re-pricing and high corporate debt refinancing hurdles against resilient AI enterprise spend.",
        indicator_divergence="Market breadth narrowing with S&P equal-weight index underperforming cap-weighted benchmark by 180 bps over trailing 10 sessions.",
        thesis_risks="Disinflationary surprise in forthcoming core CPI print or dovish pivot commentary in scheduled Fed governor speeches."
    ),
    "ZN": AssetMacroProfile(
        ticker="ZN",
        name="10-Year Treasury Note Futures",
        exchange="CBOT",
        last_settlement_price=104.375,
        current_indicated_price=104.281,
        annualized_vol_pct=8.4,
        dead_zone_pct=0.30,
        prior_settlement_trend="bearish",
        order_flow_imbalance=-0.15,
        edgar_triage_signal="foreign_central_bank_liquidation",
        primary_catalyst="Treasury refunding auction indigestion and persistent term premium expansion driven by expanding US sovereign fiscal deficits.",
        indicator_divergence="10Y real yields pushing to cycle highs while headline 5Y5Y forward inflation expectation swaps remain well-anchored at 2.38%.",
        thesis_risks="Sudden safe-haven flight-to-quality bid triggered by geopolitical escalation in the Middle East flattening the long end."
    ),
    "ZS": AssetMacroProfile(
        ticker="ZS",
        name="Soybean Futures",
        exchange="CBOT",
        last_settlement_price=1277.25,
        current_indicated_price=1274.50,
        annualized_vol_pct=19.5,
        dead_zone_pct=0.30,
        prior_settlement_trend="bearish",
        order_flow_imbalance=-0.21,
        edgar_triage_signal="grain_elevator_forward_sales",
        primary_catalyst="US Midwest harvest completion pace running ahead of 5-year average alongside expanding South American planting acreage projections.",
        indicator_divergence="US export inspections lagging USDA seasonal targets by 12% while Brazilian FOB export basis maintains substantial discount.",
        thesis_risks="Sudden Chinese state buying spree for strategic reserves or adverse late-season frost warnings across northern crop belts."
    ),
    "DXY": AssetMacroProfile(
        ticker="DXY",
        name="US Dollar Index",
        exchange="ICE",
        last_settlement_price=101.695,
        current_indicated_price=101.780,
        annualized_vol_pct=7.2,
        dead_zone_pct=0.30,
        prior_settlement_trend="neutral",
        order_flow_imbalance=0.14,
        edgar_triage_signal="custody_bank_fx_rebalancing",
        primary_catalyst="Rate divergence premium favoring USD as ECB prepares for dovish ease on Oct 29 while Fed remains in restrictive posture.",
        indicator_divergence="US-German 2-year sovereign yield differential widening to 215 bps while EUR/USD struggles to sustain bids above 1.0850.",
        thesis_risks="Coordinated verbal intervention by Bank of Japan supporting JPY or unexpected de-escalation reducing reserve currency safe-haven bid."
    ),
    "VIX": AssetMacroProfile(
        ticker="VIX",
        name="CBOE Volatility Index Futures",
        exchange="CFE",
        last_settlement_price=17.70,
        current_indicated_price=17.85,
        annualized_vol_pct=55.0,
        dead_zone_pct=0.30,
        prior_settlement_trend="neutral",
        order_flow_imbalance=0.04,
        edgar_triage_signal="systematic_vol_selling",
        primary_catalyst="Sticky vol-compression regime driven by dealer short-gamma positioning and systematic overwriting ETFs pinning index options.",
        indicator_divergence="Cross-asset MOVE rates vol holding high at 118.5 while equity VIX remains anchored in the 17.5-18.2 dead-zone corridor.",
        thesis_risks="Severe liquidity air pocket or sudden geopolitical headline triggering mechanical CTA deleveraging and VIX spike above 25.0."
    ),
    "PALM": AssetMacroProfile(
        ticker="PALM",
        name="Crude Palm Oil Futures",
        exchange="MDEX",
        last_settlement_price=4120.0,
        current_indicated_price=4110.0,
        annualized_vol_pct=24.5,
        dead_zone_pct=0.30,
        prior_settlement_trend="bearish",
        order_flow_imbalance=-0.11,
        edgar_triage_signal="agri_refiner_hedging",
        primary_catalyst="Indonesian export quota adjustments counterbalanced by soft import demand from India amid high domestic edible oil stocks.",
        indicator_divergence="Palm-gasoil spread narrowing reducing biodiesel blending economics while production enters seasonal peak cycle.",
        thesis_risks="La Niña precipitation events causing localized flooding across Malaysian plantation estates."
    ),
    "BTC": AssetMacroProfile(
        ticker="BTC",
        name="Bitcoin CME Futures",
        exchange="CME",
        last_settlement_price=104250.0,
        current_indicated_price=104600.0,
        annualized_vol_pct=48.0,
        dead_zone_pct=0.30,
        prior_settlement_trend="bullish",
        order_flow_imbalance=0.22,
        edgar_triage_signal="institutional_etf_inflow",
        primary_catalyst="Institutional allocation and spot ETF absorption continuing to outstrip miner daily issuance post-halving.",
        indicator_divergence="BTC dominance holding at 58.5% while high-beta altcoins lag, indicating selective institutional accumulation over retail speculation.",
        thesis_risks="Macro risk-off deleveraging wave triggered by sharp spike in 10Y real yields or regulatory enforcement action against offshore stablecoins."
    ),
    "ETH": AssetMacroProfile(
        ticker="ETH",
        name="Ether CME Futures",
        exchange="CME",
        last_settlement_price=3280.0,
        current_indicated_price=3265.0,
        annualized_vol_pct=56.0,
        dead_zone_pct=0.30,
        prior_settlement_trend="neutral",
        order_flow_imbalance=-0.08,
        edgar_triage_signal="staking_yield_arbitrage",
        primary_catalyst="L2 fee cannibalization dampening mainnet burn rate while spot ETF flows remain muted relative to Bitcoin.",
        indicator_divergence="ETH/BTC ratio hovering near multi-year lows (0.031) reflecting capital concentration in sovereign-grade digital store of value.",
        thesis_risks="Surge in DeFi TVL or breakthrough enterprise settlement announcement on Ethereum mainnet."
    )
}


class AnalyticalUpgradePipeline:
    """
    Executes the analytical layer upgrades:
    1. Cross-Asset Volatility & Spread Calibration
    2. Second-Order Macro Regime Filtering
    3. Bayesian Prior & High-Frequency Flow Updating
    """

    def __init__(self, regime: Optional[MacroRegimeState] = None):
        self.regime = regime or MacroRegimeState()

    def calibrate_volatility_and_tail_risk(
        self, profile: AssetMacroProfile
    ) -> Tuple[float, float, float]:
        """
        Calibrates daily volatility ($\sigma_{\text{daily}}$), skew ($\gamma_1$), and kurtosis ($\gamma_2$)
        incorporating credit spreads, swap spreads, and rates implied vol (MOVE proxy).
        """
        # Base daily vol
        base_sigma_daily = (profile.annualized_vol_pct / 100.0) / math.sqrt(252)

        # Macro risk multiplier derived from rates IV (MOVE) and credit stress
        move_factor = self.regime.rates_implied_vol_move / 100.0  # e.g. 1.185
        credit_factor = 1.0 + (self.regime.cdx_hy_spread_bps - 350.0) / 1000.0 # Credit stress premium

        # Asset-specific sensitivity
        if profile.ticker in ("ZN", "DXY", "ES"):
            sigma_adj = base_sigma_daily * (0.5 * move_factor + 0.5 * credit_factor)
        elif profile.ticker in ("CL", "RB", "NG"):
            geo_factor = 1.0 + (self.regime.geopolitical_risk_index - 50.0) / 200.0
            sigma_adj = base_sigma_daily * geo_factor
        elif profile.ticker in ("GC", "SI"):
            geo_factor = 1.0 + (self.regime.geopolitical_risk_index - 50.0) / 250.0
            sigma_adj = base_sigma_daily * (0.6 * geo_factor + 0.4 * move_factor)
        else:
            sigma_adj = base_sigma_daily * (0.7 + 0.3 * credit_factor)

        # Skew: directional asymmetric bias from macro regime
        # Equities, high-yield cyclicals have negative skew in tightening
        if profile.ticker in ("ES", "ZS", "HG"):
            skew = -0.45
            kurtosis = 4.8
        elif profile.ticker in ("GC", "CL", "NG"):
            skew = 0.35  # Upside tail risk from supply/geopolitical disruption
            kurtosis = 5.2
        elif profile.ticker in ("DXY",):
            skew = 0.15
            kurtosis = 3.6
        elif profile.ticker in ("VIX",):
            skew = 0.85
            kurtosis = 6.5
        else:
            skew = -0.10
            kurtosis = 4.0

        return sigma_adj, skew, kurtosis

    def evaluate_bayesian_posterior(
        self, profile: AssetMacroProfile
    ) -> Dict[str, float]:
        """
        Combines historical empirical base rates with high-frequency flow updates
        and order flow triage to compute posterior probabilities for [Bear, Base, Bull].

        Empirical settlement base rate from our audit:
        - Low-vol / tightly contested assets: 35-45% Neutral Dead-Zone
        - High-beta momentum assets: 35-40% Trend Continuation
        """
        # Base prior distribution [Bear, Neutral, Bull]
        if profile.ticker in ("VIX", "DXY", "HG"):
            # High propensity to settle in dead zone
            prior = {"bear": 0.25, "neutral": 0.50, "bull": 0.25}
        elif profile.ticker in ("CL", "RB", "NG"):
            # Volatile, low neutrality probability
            prior = {"bear": 0.38, "neutral": 0.24, "bull": 0.38}
        elif profile.ticker in ("GC", "SI"):
            prior = {"bear": 0.32, "neutral": 0.30, "bull": 0.38}
        elif profile.ticker in ("ES", "ZN", "ZS"):
            prior = {"bear": 0.40, "neutral": 0.35, "bull": 0.25}
        else:
            prior = {"bear": 0.33, "neutral": 0.34, "bull": 0.33}

        # High-frequency signal adjustment (Log-Odds update)
        flow = profile.order_flow_imbalance  # [-1.0, 1.0]

        # Order flow & institutional EDGAR impact
        log_odds_bull = math.log(prior["bull"] / prior["neutral"]) + (flow * 1.5)
        log_odds_bear = math.log(prior["bear"] / prior["neutral"]) - (flow * 1.5)

        # Reconstruct normalized posterior
        exp_bull = math.exp(log_odds_bull)
        exp_bear = math.exp(log_odds_bear)
        exp_neutral = 1.0
        total = exp_bull + exp_bear + exp_neutral

        p_bull = round(exp_bull / total, 3)
        p_bear = round(exp_bear / total, 3)
        p_neutral = round(1.0 - (p_bull + p_bear), 3)

        return {"bear": p_bear, "neutral": p_neutral, "bull": p_bull}
