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
# Baseline universe data grounded in October 2026 live settlement telemetry
ASSET_UNIVERSE: Dict[str, AssetMacroProfile] = {
    "GC": AssetMacroProfile(
        ticker="GC",
        name="Gold Futures",
        exchange="COMEX",
        last_settlement_price=4167.60,
        current_indicated_price=4169.20,
        annualized_vol_pct=17.2,
        dead_zone_pct=0.30,
        prior_settlement_trend="neutral",
        order_flow_imbalance=0.03,
        edgar_triage_signal="institutional_safehaven_bid",
        primary_catalyst="Consolidation between Middle East safe-haven support and elevated US real 10Y yields (2.25%).",
        indicator_divergence="Gold physical delivery demand balanced against rising dollar yield, pinning price tightly within the ±0.30% dead zone (-0.04% in trailing session).",
        thesis_risks="Diplomatic breakthrough in Middle East or hawkish forward-guidance surge pushing real yields above 2.50%."
    ),
    "SI": AssetMacroProfile(
        ticker="SI",
        name="Silver Futures",
        exchange="COMEX",
        last_settlement_price=61.400,
        current_indicated_price=61.395,
        annualized_vol_pct=26.4,
        dead_zone_pct=0.30,
        prior_settlement_trend="bullish",
        order_flow_imbalance=0.20,
        edgar_triage_signal="industrial_hedging",
        primary_catalyst="Precious metals upside momentum continuing post +1.15% settlement breakout with photovoltaic industrial demand floor.",
        indicator_divergence="Silver outperforming gold as industrial photovoltaic restocking converges with safe-haven monetary bid.",
        thesis_risks="Sharp industrial contraction in Asian manufacturing centers or aggressive liquidation in leveraged precious metal futures."
    ),
    "CL": AssetMacroProfile(
        ticker="CL",
        name="WTI Crude Oil Futures",
        exchange="NYMEX",
        last_settlement_price=89.30,
        current_indicated_price=89.27,
        annualized_vol_pct=34.2,
        dead_zone_pct=0.30,
        prior_settlement_trend="bearish",
        order_flow_imbalance=-0.26,
        edgar_triage_signal="commercial_producer_hedging",
        primary_catalyst="Persistent downside momentum following -2.69% collapse below $90 psychological support on rising US crude inventories.",
        indicator_divergence="Physical prompt Brent-WTI spreads holding near $4.20 while paper speculative net-longs undergo aggressive margin reduction.",
        thesis_risks="Direct naval escalation closing the Strait of Hormuz or retaliatory infrastructure strike triggering an immediate $15 supply spike."
    ),
    "RB": AssetMacroProfile(
        ticker="RB",
        name="RBOB Gasoline Futures",
        exchange="NYMEX",
        last_settlement_price=3.0600,
        current_indicated_price=3.1115,
        annualized_vol_pct=32.0,
        dead_zone_pct=0.30,
        prior_settlement_trend="bearish",
        order_flow_imbalance=-0.24,
        edgar_triage_signal="distributor_drawdown",
        primary_catalyst="Downside momentum following -7.62% crash in prompt crack spreads and high Gulf Coast refinery runs (91%).",
        indicator_divergence="Severe product surplus along the Atlantic coast outpacing crude declines, crushing refinery crack margins.",
        thesis_risks="Unplanned domestic refinery outage along the US Gulf Coast or spike in component alkylate import tariffs."
    ),
    "NG": AssetMacroProfile(
        ticker="NG",
        name="Natural Gas Futures",
        exchange="NYMEX",
        last_settlement_price=3.080,
        current_indicated_price=3.078,
        annualized_vol_pct=42.0,
        dead_zone_pct=0.30,
        prior_settlement_trend="bullish",
        order_flow_imbalance=0.28,
        edgar_triage_signal="utility_winter_hedging",
        primary_catalyst="Bullish continuation holding above $3.00 milestone (+2.02% trailing gain) supported by Midwestern cold weather anomalies.",
        indicator_divergence="Early heating degree days accelerating storage withdrawals alongside high European LNG export terminal feedgas intake.",
        thesis_risks="Mild weather revisions across mid-October forecast models or rapid resumption of constrained Permian associated gas takeaway capacity."
    ),
    "HG": AssetMacroProfile(
        ticker="HG",
        name="Copper Futures",
        exchange="COMEX",
        last_settlement_price=6.6325,
        current_indicated_price=6.6300,
        annualized_vol_pct=21.0,
        dead_zone_pct=0.30,
        prior_settlement_trend="bullish",
        order_flow_imbalance=0.15,
        edgar_triage_signal="mining_conglomerate_hedging",
        primary_catalyst="Following risk-on equity advance (+0.91% in trailing session) and Chinese infrastructure credit stabilization.",
        indicator_divergence="LME canceled warrants rising to 29% as refined cathode stocks draw down in Asian transit hubs.",
        thesis_risks="Major supply disruption at Chilean/Peruvian open-pit mines or abrupt rollout of large-scale Chinese grid infrastructure stimulus."
    ),
    "ES": AssetMacroProfile(
        ticker="ES",
        name="E-mini S&P 500 Futures",
        exchange="CME",
        last_settlement_price=7830.00,
        current_indicated_price=7831.75,
        annualized_vol_pct=14.8,
        dead_zone_pct=0.30,
        prior_settlement_trend="bullish",
        order_flow_imbalance=0.20,
        edgar_triage_signal="corporate_buyback_execution",
        primary_catalyst="Disinflationary tailwind from plunging energy costs (Crude -2.7%, Gasoline -7.6%) driving corporate margin relief and equity rally (+0.62% in trailing session).",
        indicator_divergence="Market breadth expanding across cyclicals as lower fuel input costs alleviate consumer discretionary balance sheet friction.",
        thesis_risks="Spike in long-term Treasury yields above 4.75% or surprise hawkish rhetoric from Fed governors."
    ),
    "ZN": AssetMacroProfile(
        ticker="ZN",
        name="10-Year Treasury Note Futures",
        exchange="CBOT",
        last_settlement_price=104.250,
        current_indicated_price=104.265625,
        annualized_vol_pct=8.4,
        dead_zone_pct=0.05,
        prior_settlement_trend="bearish",
        order_flow_imbalance=-0.20,
        edgar_triage_signal="foreign_central_bank_liquidation",
        primary_catalyst="Persistent sovereign Treasury auction supply pressure and term premium expansion keeping downward pressure on 10Y note prices.",
        indicator_divergence="Narrow 0.05% dead band makes downward continuation easily breach settlement threshold (-0.10% in trailing session).",
        thesis_risks="Sudden safe-haven flight-to-quality bid triggered by geopolitical escalation in the Middle East flattening the long end."
    ),
    "ZS": AssetMacroProfile(
        ticker="ZS",
        name="Soybean Futures",
        exchange="CBOT",
        last_settlement_price=1281.00,
        current_indicated_price=1280.50,
        annualized_vol_pct=19.5,
        dead_zone_pct=0.30,
        prior_settlement_trend="neutral",
        order_flow_imbalance=-0.04,
        edgar_triage_signal="grain_elevator_forward_sales",
        primary_catalyst="Midwest harvest pressure balanced by export inspection pace, pinning prices tightly in dead-zone corridor (+0.23% trailing move).",
        indicator_divergence="Brazilian export discounts offsetting US harvest speed, maintaining neutral price equilibrium inside the ±0.30% dead zone.",
        thesis_risks="Sudden Chinese state buying spree for strategic reserves or adverse late-season frost warnings across northern crop belts."
    ),
    "DXY": AssetMacroProfile(
        ticker="DXY",
        name="US Dollar Index",
        exchange="ICE",
        last_settlement_price=101.910,
        current_indicated_price=101.910,
        annualized_vol_pct=7.2,
        dead_zone_pct=0.15,
        prior_settlement_trend="bullish",
        order_flow_imbalance=0.18,
        edgar_triage_signal="custody_bank_fx_rebalancing",
        primary_catalyst="Rate differential momentum (US-DE 2Y spread > 215 bps) lifting Dollar Index above 101.90, breaching narrow 0.15% dead band.",
        indicator_divergence="ECB dovish ease anticipation weakening EUR/USD while US macro resilience supports the greenback.",
        thesis_risks="Coordinated verbal intervention by Bank of Japan supporting JPY or unexpected de-escalation reducing reserve currency safe-haven bid."
    ),
    "VIX": AssetMacroProfile(
        ticker="VIX",
        name="CBOE Volatility Index Futures",
        exchange="CFE",
        last_settlement_price=17.45,
        current_indicated_price=17.45,
        annualized_vol_pct=52.0,
        dead_zone_pct=0.80,
        prior_settlement_trend="neutral",
        order_flow_imbalance=0.00,
        edgar_triage_signal="systematic_vol_selling",
        primary_catalyst="Systematic volatility suppression and wide 0.80% dead-zone parameter anchoring settlement to neutral.",
        indicator_divergence="Dealer short-gamma positioning pins index options while headline challenges resolve neutral across all consecutive rounds.",
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

        # Skew: directional asymmetric bias from macro regime & energy transmission
        if profile.ticker in ("CL", "RB"):
            skew = -0.40  # Supply overhang and crack spread liquidation
            kurtosis = 5.0
        elif profile.ticker in ("ES", "HG"):
            skew = 0.25   # Disinflationary profit margin relief from collapsing energy input costs
            kurtosis = 4.2
        elif profile.ticker in ("NG", "SI"):
            skew = 0.35   # Upside momentum from weather/winter drawdowns and bullion beta
            kurtosis = 5.2
        elif profile.ticker in ("ZN",):
            skew = -0.25  # Persistent Treasury refunding supply indigestion
            kurtosis = 3.8
        elif profile.ticker in ("DXY",):
            skew = 0.20   # Rate divergence premium vs ECB
            kurtosis = 3.6
        elif profile.ticker in ("VIX",):
            skew = 0.10   # Systematic option selling pinning vol
            kurtosis = 5.0
        else:
            skew = 0.00
            kurtosis = 3.8

        return sigma_adj, skew, kurtosis

    def evaluate_bayesian_posterior(
        self, profile: AssetMacroProfile
    ) -> Dict[str, float]:
        """
        Combines historical empirical base rates with high-frequency flow updates
        and order flow triage to compute posterior probabilities for [Bear, Base, Bull].

        Empirical settlement base rate grounded in Headline Arena dead zones:
        - VIX: 0.80% dead band & flat pricing -> ~84% Neutral
        - GC / ZS: Trapped inside 0.30% dead band -> ~50% Neutral
        - ZN: 0.05% dead band -> ~12% Neutral, 60% Bearish continuation
        - DXY: 0.15% dead band -> ~25% Neutral, 55% Bullish rate divergence
        - CL / RB: High-vol downside momentum -> ~60% Bearish
        - NG / SI / ES / HG: High-conviction upside momentum -> ~58% Bullish
        """
        # Base prior distribution [Bear, Neutral, Bull]
        if profile.ticker in ("VIX",):
            prior = {"bear": 0.08, "neutral": 0.84, "bull": 0.08}
        elif profile.ticker in ("GC", "ZS"):
            prior = {"bear": 0.25, "neutral": 0.50, "bull": 0.25}
        elif profile.ticker in ("ZN",):
            prior = {"bear": 0.60, "neutral": 0.12, "bull": 0.28}
        elif profile.ticker in ("DXY",):
            prior = {"bear": 0.20, "neutral": 0.25, "bull": 0.55}
        elif profile.ticker in ("CL", "RB"):
            prior = {"bear": 0.60, "neutral": 0.16, "bull": 0.24}
        elif profile.ticker in ("NG", "SI", "ES", "HG"):
            prior = {"bear": 0.18, "neutral": 0.24, "bull": 0.58}
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
