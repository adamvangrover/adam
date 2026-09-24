import json
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# --- Mathematical Primitives ---

def phi_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def phi_inv(p: float) -> float:
    """Peter Acklam's rational approximation to the inverse standard normal CDF."""
    p = max(min(p, 1.0 - 1e-15), 1e-15)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    q = min(p, 1.0 - p)
    if q > 0.02425:
        r = q - 0.5
        r2 = r * r
        x = (((((a[0]*r2 + a[1])*r2 + a[2])*r2 + a[3])*r2 + a[4])*r2 + a[5])*r / \
            (((((b[0]*r2 + b[1])*r2 + b[2])*r2 + b[3])*r2 + b[4])*r2 + 1.0)
    else:
        r = math.sqrt(-2.0 * math.log(q))
        # FIX: numerator polynomial must be fully summed BEFORE the division.
        numerator = (((((c[0]*r + c[1])*r + c[2])*r + c[3])*r + c[4])*r + c[5])
        denominator = ((((d[0]*r + d[1])*r + d[2])*r + d[3])*r + 1.0)
        x = numerator / denominator
        # The raw polynomial (built from q = min(p, 1-p)) approximates phi_inv(q)
        # directly, i.e. it is already correct for the LOWER tail (p < 0.5).
        # For the upper tail (p >= 0.5), phi_inv(p) = -phi_inv(1-p) = -raw_x,
        # so the negation belongs on p >= 0.5, not p < 0.5.
        if p >= 0.5:
            x = -x
    return x

# --- Risk Types & Engine ---

class SystemState(str, Enum):
    CONSENSUS = "CONSENSUS"
    CONTESTED = "CONTESTED"
    DEGRADED_FALLBACK = "DEGRADED_FALLBACK"
    SUPERVISORY_OVERRIDE = "SUPERVISORY_OVERRIDE"

@dataclass(frozen=True)
class ObligorTelemetry:
    entity_name: str
    ticker_or_lei: str = "UNKNOWN"
    equity_market_cap: Optional[float] = None
    equity_volatility: Optional[float] = None
    total_debt: float = 1.0
    short_term_debt: float = 0.0
    ebitda: float = 1.0
    interest_expense: float = 1.0
    free_cash_flow: float = 0.0
    cash_and_equivalents: float = 0.0
    total_assets: float = 1.0
    credit_spread_bps: Optional[float] = None
    days_since_financials: int = 45
    sector_ttc_pd: float = 0.0350
    risk_free_rate: float = 0.045
    qualitative_flags: List[str] = field(default_factory=list)

class RealTimeObligorPDEngine:
    FLOOR_PD = 0.0003
    CAP_PD = 0.9999
    LOGIT_WEIGHTS = {
        "beta_0": -3.85, "beta_leverage": 0.85, "beta_icr": -0.35,
        "beta_fcf_debt": -1.20, "beta_cash_assets": -1.45
    }
    QUALITATIVE_SHOCKS = {
        "covenant_breach": 0.80, "going_concern_warning": 1.40,
        "cfo_auditor_turnover": 0.45, "debt_restructuring_advisors": 1.10,
        "liquidity_runway_under_6m": 0.95, "aggressive_capex_burn": 0.65
    }

    def _solve_merton(self, equity: float, sigma_e: float, default_point: float, r: float) -> Tuple[float, float]:
        if equity <= 0 or default_point <= 0 or sigma_e <= 0:
            return 0.50, 0.0
        va_low, va_high = equity, equity + 3.0 * default_point
        va_best = equity + default_point
        sa_best = max(sigma_e * (equity / va_best), 0.05)
        for _ in range(40):
            va_mid = 0.5 * (va_low + va_high)
            sa_guess = max(sigma_e * (equity / va_mid), 0.01)
            d1 = (math.log(va_mid / default_point) + (r + 0.5 * sa_guess**2)) / sa_guess
            d2 = d1 - sa_guess
            diff = (va_mid * phi_cdf(d1) - default_point * math.exp(-r) * phi_cdf(d2)) - equity
            if abs(diff) < 1e-3:
                va_best, sa_best = va_mid, sa_guess
                break
            if diff < 0:
                va_low = va_mid
            else:
                va_high = va_mid
            va_best, sa_best = va_mid, sa_guess
        dd = (math.log(va_best / default_point) + (r - 0.5 * sa_best**2)) / sa_best
        return max(min(phi_cdf(-dd), self.CAP_PD), self.FLOOR_PD), dd

    def evaluate(self, t: ObligorTelemetry) -> Dict[str, Any]:
        has_equity = 1.0 if (t.equity_market_cap and t.equity_volatility) else 0.0
        has_spread = 1.0 if t.credit_spread_bps is not None else 0.0
        recency = math.exp(-max(t.days_since_financials - 90, 0) / 90.0)
        telemetry_conf = (0.45 * has_equity) + (0.25 * has_spread) + (0.30 * min(max(recency, 0.05), 1.0))

        models = {}
        dd_val = None
        if has_equity and t.total_debt > 0:
            dp = t.short_term_debt + 0.5 * max(t.total_debt - t.short_term_debt, 0.0)
            struct_pd, dd = self._solve_merton(t.equity_market_cap, t.equity_volatility, dp, t.risk_free_rate)
            models["structural"] = struct_pd
            dd_val = round(dd, 3)

        if has_spread and t.credit_spread_bps > 0:
            hazard = (t.credit_spread_bps * 1e-4) / 0.60
            models["market_implied"] = max(min(1.0 - math.exp(-hazard), self.CAP_PD), self.FLOOR_PD)

        leverage = min(t.total_debt / max(t.ebitda, 1.0), 12.0) if t.ebitda > 0 else 12.0
        icr = min(max(t.ebitda / max(t.interest_expense, 1.0), -5.0), 25.0)
        fcf_debt = min(max(t.free_cash_flow / max(t.total_debt, 1.0), -1.0), 1.0)
        cash_assets = min(max(t.cash_and_equivalents / max(t.total_assets, 1.0), 0.0), 1.0)
        w = self.LOGIT_WEIGHTS
        z = w["beta_0"] + w["beta_leverage"]*math.log(max(leverage, 0.1)) + w["beta_icr"]*icr + w["beta_fcf_debt"]*fcf_debt + w["beta_cash_assets"]*cash_assets
        models["fundamental"] = max(min(1.0 / (1.0 + math.exp(-z)), self.CAP_PD), self.FLOOR_PD)

        if "structural" in models and "market_implied" in models:
            champ_pd = 0.55 * models["structural"] + 0.45 * models["market_implied"]
        elif "structural" in models:
            champ_pd = 0.65 * models["structural"] + 0.35 * models["fundamental"]
        elif "market_implied" in models:
            champ_pd = 0.50 * models["market_implied"] + 0.50 * models["fundamental"]
        else:
            champ_pd = models["fundamental"]

        st_ebitda = max(t.ebitda * 0.85, 1.0) if t.ebitda > 0 else 1.0
        st_interest = t.interest_expense + (t.short_term_debt * 0.02)
        st_icr = st_ebitda / max(st_interest, 1.0)
        refi_wall = t.short_term_debt / max(t.total_debt, 1.0)
        cash_buf = t.cash_and_equivalents / max(t.total_debt, 1.0)
        z_chall = -2.80 + 1.10*math.log(max(t.total_debt / st_ebitda, 0.1)) - 0.50*min(max(st_icr, -5.0), 20.0) + 1.75*refi_wall - 1.20*min(max(cash_buf, 0.0), 1.0)
        chall_pd = max(min(1.0 / (1.0 + math.exp(-z_chall)), self.CAP_PD), self.FLOOR_PD)

        disparity = abs(champ_pd - chall_pd) / max((champ_pd + chall_pd) / 2.0, 1e-4)
        surcharge, sr11_7 = False, False

        if telemetry_conf < 0.40:
            state = SystemState.SUPERVISORY_OVERRIDE
            base_pd = max(t.sector_ttc_pd, 0.0500)
            directive = "REJECT_AUTOMATION_MANDATE_SECOND_LINE_OVERRIDE"
        elif telemetry_conf < 0.85:
            state = SystemState.DEGRADED_FALLBACK
            base_pd = 0.30 * champ_pd + 0.70 * chall_pd
            directive = "EXECUTE_WITH_DEGRADED_SHADOW_MONITORING"
        elif disparity > 0.35:
            state = SystemState.CONTESTED
            downside_gap = max(0.0, chall_pd - champ_pd)
            base_pd = 0.50*champ_pd + 0.50*chall_pd + (0.30 * downside_gap)
            surcharge = downside_gap > 0
            sr11_7 = True
            directive = "FREEZE_LIMIT_EXPANSION_LOG_SR11_7_EXCEPTION"
        else:
            state = SystemState.CONSENSUS
            base_pd = 0.50 * champ_pd + 0.50 * chall_pd
            directive = "STANDARD_EXECUTION_CONSENSUS_AFFIRMED"

        shock = sum(self.QUALITATIVE_SHOCKS.get(f, 0.0) for f in t.qualitative_flags)
        calib_logit = math.log(base_pd / (1.0 - base_pd)) + shock
        final_pit_pd = max(min(1.0 / (1.0 + math.exp(-calib_logit)), self.CAP_PD), self.FLOOR_PD)

        p_cap = max(min(final_pit_pd, 0.999), self.FLOOR_PD)
        factor = (1.0 - math.exp(-50.0 * p_cap)) / (1.0 - math.exp(-50.0))
        rho = 0.12 * factor + 0.24 * (1.0 - factor)
        stressed_pd = phi_cdf((phi_inv(p_cap) + math.sqrt(rho) * phi_inv(0.999)) / math.sqrt(1.0 - rho))

        return {
            "entity": {"name": t.entity_name, "identifier": t.ticker_or_lei, "rating_scope": "OBLIGOR_LEVEL_ONLY"},
            "governance_audit": {
                "execution_state": state.value,
                "telemetry_confidence": round(telemetry_conf, 4),
                "disparity_ratio": round(disparity, 4),
                "sr11_7_divergence_flag": sr11_7,
                "adversarial_surcharge_applied": surcharge,
                "actionable_directive": directive
            },
            "dual_world_metrics": {
                "champion_pit_pd_bps": round(champ_pd * 10000, 1),
                "challenger_counterfactual_pd_bps": round(chall_pd * 10000, 1),
                "distance_to_default": dd_val,
                "champion_pillars": {k: round(v * 10000, 1) for k, v in models.items()}
            },
            "final_regulatory_capital": {
                "obligor_pit_pd": round(final_pit_pd, 6),
                "obligor_pit_pd_bps": round(final_pit_pd * 10000, 1),
                "basel_999_stressed_pd": round(stressed_pd, 6),
                "basel_999_stressed_pd_bps": round(stressed_pd * 10000, 1),
                "asset_correlation_rho": round(rho, 4)
            }
        }

# --- MCP JSON-RPC Stdio Dispatcher ---

TOOL_SCHEMA = {
    "name": "assess_obligor_pd",
    "description": "Computes institutional Obligor Probability of Default (PD) via Champion-Challenger dual-world arbitration and Basel III/IV IRB stress.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "entity_name": {"type": "string", "description": "Legal entity name of borrower"},
            "ticker_or_lei": {"type": "string", "description": "Public ticker or LEI code"},
            "equity_market_cap": {"type": "number", "description": "Market value of equity ($M)"},
            "equity_volatility": {"type": "number", "description": "Annualized equity volatility (e.g. 0.40)"},
            "total_debt": {"type": "number", "description": "Total balance sheet funded debt ($M)"},
            "short_term_debt": {"type": "number", "description": "Debt due within 1 year ($M)"},
            "ebitda": {"type": "number", "description": "Trailing 12M EBITDA ($M)"},
            "interest_expense": {"type": "number", "description": "Annual interest expense ($M)"},
            "free_cash_flow": {"type": "number", "description": "Trailing 12M Free Cash Flow ($M)"},
            "cash_and_equivalents": {"type": "number", "description": "Unrestricted cash & equivalents ($M)"},
            "total_assets": {"type": "number", "description": "Total balance sheet assets ($M)"},
            "credit_spread_bps": {"type": "number", "description": "Observable CDS or bond spread (bps)"},
            "days_since_financials": {"type": "integer", "description": "Days since last filing"},
            "qualitative_flags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Active risk signals: [aggressive_capex_burn, liquidity_runway_under_6m, debt_restructuring_advisors, covenant_breach]"
            }
        },
        "required": ["entity_name", "total_debt", "ebitda", "interest_expense"]
    }
}

def send_response(response_obj: Dict[str, Any]) -> None:
    body = json.dumps(response_obj)
    sys.stdout.write(f"Content-Length: {len(body)}\r\n\r\n{body}")
    sys.stdout.flush()

def send_error(req_id, code: int, message: str) -> None:
    send_response({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})

def main():
    engine = RealTimeObligorPDEngine()
    buffer = ""
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        buffer += line
        if "\r\n\r\n" in buffer:
            header, rest = buffer.split("\r\n\r\n", 1)
            content_length = None
            for h in header.split("\r\n"):
                if h.lower().startswith("content-length:"):
                    content_length = int(h.split(":")[1].strip())
            if content_length is not None:
                while len(rest) < content_length:
                    rest += sys.stdin.read(content_length - len(rest))
                payload = json.loads(rest[:content_length])
                buffer = rest[content_length:]

                req_id = payload.get("id")
                method = payload.get("method")

                if method == "initialize":
                    send_response({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "afos-pd-engine", "version": "30.1.2"}
                        }
                    })
                elif method == "tools/list":
                    send_response({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"tools": [TOOL_SCHEMA]}
                    })
                elif method == "tools/call":
                    params = payload.get("params", {})
                    if params.get("name") != "assess_obligor_pd":
                        send_error(req_id, -32601, f"Unknown tool: {params.get('name')!r}")
                        continue
                    args = params.get("arguments", {})
                    missing = [f for f in ("entity_name", "total_debt", "ebitda", "interest_expense") if args.get(f) is None]
                    if missing:
                        send_error(req_id, -32602, f"Missing required arguments: {missing}")
                        continue
                    try:
                        t = ObligorTelemetry(
                            entity_name=args.get("entity_name"),
                            ticker_or_lei=args.get("ticker_or_lei", "UNKNOWN"),
                            equity_market_cap=args.get("equity_market_cap"),
                            equity_volatility=args.get("equity_volatility"),
                            total_debt=float(args.get("total_debt", 1.0)),
                            short_term_debt=float(args.get("short_term_debt", 0.0)),
                            ebitda=float(args.get("ebitda", 1.0)),
                            interest_expense=float(args.get("interest_expense", 1.0)),
                            free_cash_flow=float(args.get("free_cash_flow", 0.0)),
                            cash_and_equivalents=float(args.get("cash_and_equivalents", 0.0)),
                            total_assets=float(args.get("total_assets", 1.0)),
                            credit_spread_bps=args.get("credit_spread_bps"),
                            days_since_financials=int(args.get("days_since_financials", 45)),
                            qualitative_flags=args.get("qualitative_flags", [])
                        )
                        output = engine.evaluate(t)
                    except (TypeError, ValueError) as e:
                        send_error(req_id, -32602, f"Invalid params: {e}")
                        continue
                    send_response({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(output, indent=2)}]
                        }
                    })
                elif method == "notifications/initialized":
                    pass
                else:
                    if req_id is not None:
                        send_error(req_id, -32601, f"Method not found: {method}")

if __name__ == "__main__":
    main()
