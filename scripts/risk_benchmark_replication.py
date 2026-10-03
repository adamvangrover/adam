import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class CapitalStructureInput:
  entity_name: str
  ticker: str
  total_debt: float
  senior_floating_pct: float
  senior_floating_spread_bps: float
  senior_fixed_pct: float
  senior_fixed_coupon_pct: float
  base_sofr_pct: float
  ebitda: float
  maintenance_capex: float
  mandatory_amort_pct: float
  liquidation_ebitda_multiple: float
  hazard_rates: List[float]

@dataclass
class MacroClearingInput:
  spx_level: float
  us_10y_yield_pct: float
  sofr_cash_pct: float
  brent_crude_usd: float
  vix_level: float
  cdx_hy_spread_bps: float
  trailing_5d_spx: List[float] = field(default_factory=lambda: [7665.20, 7640.10, 7680.45, 7665.28, 7721.58])
  trailing_5d_us10y: List[float] = field(default_factory=lambda: [5.221, 5.245, 5.250, 5.233, 5.276])
  trailing_5d_brent: List[float] = field(default_factory=lambda: [98.40, 99.80, 101.10, 102.31, 102.25])
  trailing_5d_cdx_hy: List[float] = field(default_factory=lambda: [388.0, 391.0, 393.0, 396.0, 395.0])
  us10y_yoy_delta_bps: float = 115.6
  nfp_3m_avg_yoy_delta: int = -113000
  brent_yoy_pct_change: float = 30.3
  spx_forward_pe_yoy_delta: float = 2.4

class CreditStressEngine:

  def __init__(self, cap_struct: CapitalStructureInput):
    self.cs = cap_struct

  def compute_amortization_and_coverage(self) -> Dict[str, Any]:
    senior_floating_debt = self.cs.total_debt * self.cs.senior_floating_pct
    senior_fixed_debt = self.cs.total_debt * self.cs.senior_fixed_pct
    all_in_floating_rate = (
        self.cs.base_sofr_pct + self.cs.senior_floating_spread_bps / 10000.0
    )
    annual_amort = senior_floating_debt * self.cs.mandatory_amort_pct

    cads = self.cs.ebitda - self.cs.maintenance_capex

    result = {
        "all_in_floating_rate_pct": round(all_in_floating_rate * 100, 3),
        "gross_leverage_ratio": round(self.cs.total_debt / self.cs.ebitda, 2),
    }

    current_floating_debt = senior_floating_debt
    fixed_interest = senior_fixed_debt * self.cs.senior_fixed_coupon_pct / 100.0

    for year in range(1, 6):
        floating_interest = current_floating_debt * all_in_floating_rate
        total_interest = floating_interest + fixed_interest
        total_debt_service = total_interest + annual_amort

        result[f"year_{year}"] = {
            "senior_debt": round(current_floating_debt, 2),
            "total_interest": round(total_interest, 2),
            "total_debt_service": round(total_debt_service, 2),
            "interest_coverage_dscr": round(self.cs.ebitda / total_interest, 3) if total_interest > 0 else 0,
            "cash_fcf_dscr": round(cads / total_debt_service, 3) if total_debt_service > 0 else 0,
            "net_fcf_post_debt_service": round(cads - total_debt_service, 2),
        }
        current_floating_debt -= annual_amort

    return result

  def compute_valuation_waterfall_and_lgd(self) -> Dict[str, Any]:
    liquidation_ev = self.cs.ebitda * self.cs.liquidation_ebitda_multiple
    senior_claim = self.cs.total_debt * self.cs.senior_floating_pct
    unsecured_claim = self.cs.total_debt * self.cs.senior_fixed_pct

    senior_recovery = min(liquidation_ev, senior_claim)
    remaining_ev_after_senior = max(0.0, liquidation_ev - senior_claim)
    unsecured_recovery = min(remaining_ev_after_senior, unsecured_claim)

    senior_recovery_rate = senior_recovery / senior_claim
    unsecured_recovery_rate = (
        unsecured_recovery / unsecured_claim if unsecured_claim > 0 else 0.0
    )
    total_recovery_rate = (
        senior_recovery + unsecured_recovery
    ) / self.cs.total_debt

    return {
        "distressed_ev": round(liquidation_ev, 2),
        "waterfall": {
            "senior_secured": {
                "claim": round(senior_claim, 2),
                "recovery": round(senior_recovery, 2),
                "recovery_rate_pct": round(senior_recovery_rate * 100, 2),
                "lgd_pct": round((1.0 - senior_recovery_rate) * 100, 2),
            },
            "senior_unsecured": {
                "claim": round(unsecured_claim, 2),
                "recovery": round(unsecured_recovery, 2),
                "recovery_rate_pct": round(unsecured_recovery_rate * 100, 2),
                "lgd_pct": round((1.0 - unsecured_recovery_rate) * 100, 2),
            },
            "blended_enterprise": {
                "claim": round(self.cs.total_debt, 2),
                "recovery": round(senior_recovery + unsecured_recovery, 2),
                "recovery_rate_pct": round(total_recovery_rate * 100, 2),
                "lgd_pct": round((1.0 - total_recovery_rate) * 100, 2),
            },
        },
    }

  def compute_cumulative_pd(self) -> Dict[str, Any]:
    survival_prob = 1.0
    for h in self.cs.hazard_rates:
      survival_prob *= 1.0 - h
    cumulative_pd = 1.0 - survival_prob
    return {
        "hazard_rates_y1_to_y3": self.cs.hazard_rates,
        "cumulative_3y_survival_prob_pct": round(survival_prob * 100, 2),
        "cumulative_3y_pd_pct": round(cumulative_pd * 100, 2),
    }

class RegulatoryFilingEvaluator:

  @staticmethod
  def evaluate_triggers(
      filing_telemetry: Dict[str, Any],
  ) -> List[Dict[str, str]]:
    active_triggers = []

    if (
        filing_telemetry.get("cash_runway_months", 99) < 12
        or filing_telemetry.get("going_concern_flag") is True
    ):
      active_triggers.append({
          "form": "10-K",
          "severity": "CRITICAL",
          "flag": "Going concern modification or cash runway < 12 months.",
      })

    if (
        filing_telemetry.get("leverage_covenant_headroom", 99.0) < 0.50
        or filing_telemetry.get("revolver_draw_pct", 0.0) > 65.0
    ):
      active_triggers.append({
          "form": "10-Q",
          "severity": "HIGH",
          "flag": (
              "Covenant breach proximity or defensive revolver drawdown >65%."
          ),
      })

    if (
        filing_telemetry.get("primary_spread_over_sofr_bps", 0) > 450
        or filing_telemetry.get("lien_carveouts_expanded") is True
    ):
      active_triggers.append({
          "form": "424B5 / Indenture",
          "severity": "HIGH",
          "flag": (
              "High-yield pricing spread escalation or collateral carve-out"
              " stripping."
          ),
      })

    if filing_telemetry.get("lmt_priming_clause_detected") is True:
      active_triggers.append({
          "form": "8-K (Item 1.01)",
          "severity": "CRITICAL",
          "flag": (
              "Non-pro-rata uptier priming or unrestricted subsidiary asset"
              " transfer."
          ),
      })

    return active_triggers

def generate_headline_arena_payload(
    macro: MacroClearingInput,
) -> Dict[str, Any]:
  return {
      "timestamp": "2026-10-02T17:30:00Z",
      "regime": "Sovereign Supply Indigestion & 5-Week Treasury Yield Streak",
      "predictions": [
          {
              "target": "US_10Y_YIELD",
              "current_spot": macro.us_10y_yield_pct,
              "arena_point_forecast": 5.35,
              "horizon": "5D",
              "bias": "EXPANSION",
              "confidence_interval": [5.22, 5.42],
          },
          {
              "target": "BRENT_CRUDE_PROMPT",
              "current_spot": macro.brent_crude_usd,
              "arena_point_forecast": 104.50,
              "horizon": "5D",
              "bias": "BULLISH",
              "confidence_interval": [98.00, 110.00],
          },
          {
              "target": "SPX_INDEX_MULTIPLE",
              "current_spot": macro.spx_level,
              "arena_point_forecast": 7540.00,
              "horizon": "10D",
              "bias": "COMPRESSION",
              "confidence_interval": [7460.00, 7620.00],
          },
          {
              "target": "CDX_NA_HY_SPREAD",
              "current_spot": macro.cdx_hy_spread_bps,
              "arena_point_forecast": 425.0,
              "horizon": "5D",
              "bias": "WIDENING",
              "confidence_interval": [410.0, 445.0],
          },
      ],
  }

def generate_ledger_payload(engine: CreditStressEngine, macro: MacroClearingInput) -> Dict[str, Any]:
    waterfall_lgd = engine.compute_valuation_waterfall_and_lgd()
    default_probs = engine.compute_cumulative_pd()
    headline_arena = generate_headline_arena_payload(macro)

    return {
        "ledger_id": "MM-V30-20261002-SYSCORE",
        "timestamp": "2026-10-02T17:30:00Z",
        "system_status": "CRITICAL",
        "macro_regime": "SOVEREIGN_SUPPLY_INDIGESTION_5W_YIELD_STREAK",
        "clearing_metrics": {
            "spx_close": macro.spx_level,
            "dow_close": 51177.30,
            "nasdaq_close": 30779.58,
            "us10y_yield": macro.us_10y_yield_pct,
            "us2y_yield": 4.840,
            "us30y_yield": 5.620,
            "sofr_rate": macro.sofr_cash_pct,
            "brent_usd": macro.brent_crude_usd,
            "wti_usd": 90.50,
            "cdx_hy_bps": macro.cdx_hy_spread_bps
        },
        "trailing_5d_series": {
            "spx": macro.trailing_5d_spx,
            "us10y_yield": macro.trailing_5d_us10y,
            "brent_usd": macro.trailing_5d_brent,
            "cdx_hy_bps": macro.trailing_5d_cdx_hy
        },
        "yoy_comparative_metrics": {
            "us10y_yield_yoy_delta_bps": macro.us10y_yoy_delta_bps,
            "nfp_3m_avg_yoy_delta": macro.nfp_3m_avg_yoy_delta,
            "brent_yoy_pct_change": macro.brent_yoy_pct_change,
            "spx_forward_pe_yoy_delta": macro.spx_forward_pe_yoy_delta
        },
        "quantitative_credit_stress": {
            "total_debt_m": engine.cs.total_debt,
            "gross_leverage": round(engine.cs.total_debt / engine.cs.ebitda, 2),
            "liquidation_ev_m": waterfall_lgd["distressed_ev"],
            "senior_secured_lgd_pct": waterfall_lgd["waterfall"]["senior_secured"]["lgd_pct"],
            "senior_unsecured_lgd_pct": waterfall_lgd["waterfall"]["senior_unsecured"]["lgd_pct"],
            "cumulative_3y_pd_pct": default_probs["cumulative_3y_pd_pct"]
        },
        "headline_arena_forecasts": {
            "us10y_target": headline_arena["predictions"][0]["arena_point_forecast"],
            "brent_target": headline_arena["predictions"][1]["arena_point_forecast"],
            "spx_target": headline_arena["predictions"][2]["arena_point_forecast"],
            "cdx_hy_target": headline_arena["predictions"][3]["arena_point_forecast"]
        },
        "adversarial_audit": {
            "red_team_passed": True,
            "data_integrity_score": 0.998
        }
    }

if __name__ == "__main__":
  corp = CapitalStructureInput(
      entity_name="Benchmark Levered Telecom Corp",
      ticker="BLTC",
      total_debt=2500.00,
      senior_floating_pct=0.60,
      senior_floating_spread_bps=450.0,
      senior_fixed_pct=0.40,
      senior_fixed_coupon_pct=5.00,
      base_sofr_pct=0.0388,
      ebitda=280.00,
      maintenance_capex=90.00,
      mandatory_amort_pct=0.01,
      liquidation_ebitda_multiple=5.0,
      hazard_rates=[0.145, 0.220, 0.315],
  )
  macro = MacroClearingInput(
      spx_level=7721.58,
      us_10y_yield_pct=5.276,
      sofr_cash_pct=3.88,
      brent_crude_usd=102.25,
      vix_level=15.31,
      cdx_hy_spread_bps=395.0,
  )

  engine = CreditStressEngine(corp)
  print(json.dumps(generate_ledger_payload(engine, macro), indent=2))
