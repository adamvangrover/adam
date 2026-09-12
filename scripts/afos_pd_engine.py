import math
from scipy.stats import norm
import json
import argparse
import sys

def calculate_pd(
    name="Unknown", ticker_or_lei="UNKNOWN",
    E=None, sigma_E=None, s=None, t_recency=0,
    Total_Debt=None, STD=None, EBITDA=None, Interest=None,
    FCF=None, Total_Assets=None, Cash=None,
    active_risk_flags=None, flags_sum=0.0
):
    if active_risk_flags is None:
        active_risk_flags = []

    # Handle missing values where possible
    E = float(E) if E is not None else 0.0
    sigma_E = float(sigma_E) if sigma_E is not None else 0.0
    s = float(s) if s is not None else 0.0
    Total_Debt = float(Total_Debt) if Total_Debt is not None else 0.0
    STD = float(STD) if STD is not None else 0.0
    EBITDA = float(EBITDA) if EBITDA is not None else 0.0
    Interest = float(Interest) if Interest is not None else 0.0
    Total_Assets = float(Total_Assets) if Total_Assets is not None else 0.0
    Cash = float(Cash) if Cash is not None else 0.0

    missing_telemetry_inputs = []
    if FCF is None:
        missing_telemetry_inputs.append("Free Cash Flow (FCF) - Estimated as EBITDA minus Interest Expense")
        FCF = max(0.0, EBITDA - Interest)
    else:
        FCF = float(FCF)

    LTD = max(0.0, Total_Debt - STD)
    D = STD + 0.5 * LTD

    c_e = 1 if E > 0 and sigma_E > 0 else 0
    c_s = 1 if s > 0 else 0
    C = 0.45 * c_e + 0.25 * c_s + 0.30 * math.exp(-max(t_recency - 90, 0)/90)

    # Structural Merton
    if E > 0 and sigma_E > 0:
        V_A = E + D
        sigma_A = sigma_E * (E / max(V_A, 1.0))
        r = 0.045
        DD = (math.log(max(V_A / max(D, 1.0), 1e-4)) + (r - 0.5 * sigma_A**2)) / max(sigma_A, 1e-4)
        PD_struct = norm.cdf(-DD)
    else:
        DD = 0.0
        PD_struct = 0.0

    # Market-Implied
    if s > 0:
        R = 0.40
        lambda_h = (s * 1e-4) / (1 - R)
        PD_market = 1 - math.exp(-lambda_h * 1.0)
    else:
        PD_market = 0.0

    # Fundamental
    Leverage = min(D / max(EBITDA, 1.0), 12.0)
    ICR = EBITDA / max(Interest, 1.0)
    FCF_Debt = FCF / max(D, 1.0)
    Cash_Assets = Cash / max(Total_Assets, 1.0)
    z = -3.85 + 0.85 * math.log(max(Leverage, 0.1)) - 0.35 * ICR - 1.20 * FCF_Debt - 1.45 * Cash_Assets
    PD_fund = 1 / (1 + math.exp(-z))

    if E > 0 and sigma_E > 0 and s > 0:
        PD_champ = 0.55 * PD_struct + 0.45 * PD_market
    elif E > 0 and sigma_E > 0:
        PD_champ = 0.65 * PD_struct + 0.35 * PD_fund
    elif s > 0:
        PD_champ = 0.50 * PD_market + 0.50 * PD_fund
    else:
        PD_champ = PD_fund

    # Challenger World Model
    EBITDA_stress = EBITDA * 0.85
    ICR_stress = EBITDA_stress / max(Interest + (STD * 0.02), 1.0)
    R_refi = STD / max(D, 1.0)
    z_chall = -2.80 + 1.10 * math.log(max(D / max(EBITDA_stress, 1.0), 1e-4)) - 0.50 * ICR_stress + 1.75 * R_refi - 1.20 * (Cash / max(D, 1.0))
    PD_chall = 1 / (1 + math.exp(-z_chall))

    delta = abs(PD_champ - PD_chall) / max((PD_champ + PD_chall)/2, 1e-4)

    if C >= 0.85 and delta <= 0.35:
        state = "CONSENSUS"
        PD_base = 0.50 * PD_champ + 0.50 * PD_chall
    elif C >= 0.85 and delta > 0.35:
        state = "CONTESTED"
        PD_base = 0.50 * PD_champ + 0.50 * PD_chall + 0.30 * max(0, PD_chall - PD_champ)
    elif 0.40 <= C < 0.85:
        state = "DEGRADED_FALLBACK"
        PD_base = 0.30 * PD_champ + 0.70 * PD_chall
    else:
        state = "SUPERVISORY_OVERRIDE"
        PD_base = max(0.05, 0.05)

    logit_PD_base = math.log(max(PD_base, 1e-6) / max(1 - PD_base, 1e-6))
    logit_PD_calibrated = logit_PD_base + flags_sum
    PD_final_pit = 1 / (1 + math.exp(-logit_PD_calibrated))

    PD_final_pit = max(0.0003, min(PD_final_pit, 0.9999))

    rho = 0.12 * ( (1 - math.exp(-50 * PD_final_pit)) / max(1 - math.exp(-50), 1e-6) ) + 0.24 * (1 - ( (1 - math.exp(-50 * PD_final_pit)) / max(1 - math.exp(-50), 1e-6) ))

    term1 = norm.ppf(PD_final_pit)
    term2 = math.sqrt(rho) * norm.ppf(0.999)
    term3 = math.sqrt(1 - rho)
    PD_basel_stressed = norm.cdf((term1 + term2) / max(term3, 1e-6))
    
    directive = "Standard monitoring."
    if state == "CONTESTED":
        directive = "Monitor closely due to SR 11-7 model divergence flag and elevated challenger PD. "
        if flags_sum > 0:
            directive += "Debt restructuring flag applied."
            
    output = {
        "entity_identification": {
            "name": name,
            "ticker_or_lei": ticker_or_lei,
            "rating_target": "OBLIGOR_LEVEL_ONLY"
        },
        "telemetry_audit": {
            "telemetry_confidence_score": round(C, 2),
            "missing_telemetry_inputs": missing_telemetry_inputs,
            "days_since_financials": t_recency
        },
        "model_pillars": {
            "structural_merton_pd": round(float(PD_struct), 4),
            "distance_to_default": round(float(DD), 4),
            "market_spread_pd": round(float(PD_market), 4),
            "fundamental_logit_pd": round(float(PD_fund), 4)
        },
        "dual_world_arbitration": {
            "champion_pit_pd": round(float(PD_champ), 4),
            "challenger_counterfactual_pd": round(float(PD_chall), 4),
            "disparity_ratio": round(float(delta), 4),
            "execution_state": state,
            "adversarial_surcharge_applied": bool(state == "CONTESTED"),
            "sr11_7_divergence_flag": bool(delta > 0.35)
        },
        "qualitative_layer": {
            "active_risk_flags": active_risk_flags,
            "aggregate_log_odds_shock": float(flags_sum)
        },
        "final_capital_assessment": {
            "final_obligor_pit_pd": round(float(PD_final_pit), 4),
            "final_obligor_pit_pd_bps": round(float(PD_final_pit * 10000), 1),
            "basel_vasicek_999_stressed_pd": round(float(PD_basel_stressed), 4),
            "basel_vasicek_999_stressed_pd_bps": round(float(PD_basel_stressed * 10000), 1),
            "asset_correlation_rho": round(float(rho), 4),
            "actionable_supervisory_directive": directive.strip()
        }
    }
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AFOS v30.1 Obligor PD Engine")
    parser.add_argument("--name", type=str, default="Unknown", help="Entity Name")
    parser.add_argument("--ticker_or_lei", type=str, default="UNKNOWN", help="Ticker or LEI")
    parser.add_argument("--E", type=float, help="Market Equity Value")
    parser.add_argument("--sigma_E", type=float, help="Equity Volatility")
    parser.add_argument("--s", type=float, help="Unsecured CDS or Bond Spread (bps)")
    parser.add_argument("--t_recency", type=int, default=0, help="Days since last financial filing")
    parser.add_argument("--Total_Debt", type=float, help="Total Debt")
    parser.add_argument("--STD", type=float, help="Short-Term Debt")
    parser.add_argument("--EBITDA", type=float, help="EBITDA")
    parser.add_argument("--Interest", type=float, help="Interest Expense")
    parser.add_argument("--FCF", type=float, help="Free Cash Flow")
    parser.add_argument("--Total_Assets", type=float, help="Total Assets")
    parser.add_argument("--Cash", type=float, help="Cash & Equivalents")
    parser.add_argument("--flags_sum", type=float, default=0.0, help="Aggregate log-odds shock from flags")
    parser.add_argument("--flags", type=str, nargs="*", default=[], help="List of active risk flags")

    args = parser.parse_args()

    # If no args passed, run the Carvana default test from prompt
    if len(sys.argv) == 1:
        res = calculate_pd(
            name="Carvana Co.",
            ticker_or_lei="CVNA",
            E=18200.0,
            sigma_E=0.68,
            s=420.0,
            t_recency=0,
            Total_Debt=5400.0,
            STD=600.0,
            EBITDA=420.0,
            Interest=380.0,
            FCF=None,
            Total_Assets=7800.0,
            Cash=550.0,
            active_risk_flags=["debt_restructuring_advisors"],
            flags_sum=1.10
        )
    else:
        res = calculate_pd(
            name=args.name,
            ticker_or_lei=args.ticker_or_lei,
            E=args.E,
            sigma_E=args.sigma_E,
            s=args.s,
            t_recency=args.t_recency,
            Total_Debt=args.Total_Debt,
            STD=args.STD,
            EBITDA=args.EBITDA,
            Interest=args.Interest,
            FCF=args.FCF,
            Total_Assets=args.Total_Assets,
            Cash=args.Cash,
            active_risk_flags=args.flags,
            flags_sum=args.flags_sum
        )

    print(json.dumps(res, indent=2))
