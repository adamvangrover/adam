"""
HITL Review Dossier Formatter conforming to Phase 3 specifications.
"""

from __future__ import annotations

from typing import List
from .schema import ChallengeForecast


class DossierFormatter:
    """
    Renders ChallengeForecast objects into the structured Operator Review Dossier format.
    """

    @classmethod
    def format_single(cls, forecast: ChallengeForecast) -> str:
        sub = forecast.proposed_submission
        sc = forecast.scenarios
        val = forecast.validation

        # Discrete / Point forecast string
        discrete_forecast_str = f"{sub.direction.value.upper()} (Point Estimate: ${sub.point_forecast:,.2f})"

        # Confidence interval string
        ci = sub.confidence_interval
        ci_str = f"P10: ${ci.p10:,.2f} | P50: ${ci.p50:,.2f} | P90: ${ci.p90:,.2f}"

        # Schema status
        schema_status = "PASSED" if val.jsonlogic_passed else "FAILED"

        lines = [
            "# " + "=" * 80,
            f"CHALLENGE ID: {forecast.challenge_id} / {forecast.challenge_title}",
            f"RESOLUTION HORIZON: {forecast.resolution_horizon} | RESOLUTION METRIC: {forecast.resolution_metric}",
            "",
            "PROPOSED SUBMISSION:",
            f"* Point Estimate / Discrete Forecast: {discrete_forecast_str}",
            f"* Confidence Interval / Distribution: [{ci_str}]",
            f"* Model Confidence Score: {sub.confidence:.2f}",
            "",
            "EXECUTIVE RATIONALE:",
            forecast.executive_rationale,
            "",
            "STRESS & SCENARIO BREAKDOWN:",
            f"* Base Case (Weight: {sc.base_case_weight:.1f}%): {sc.base_case_thesis}",
            f"* Bull / Upside Scenario (Weight: {sc.bull_case_weight:.1f}%): {sc.bull_case_thesis}",
            f"* Bear / Downside Scenario (Weight: {sc.bear_case_weight:.1f}%): {sc.bear_case_thesis}",
            "",
            "VALIDATION & AUDIT LOG:",
            f"* jsonLogic Schema Check: {schema_status}",
            f"* PROV-O Hash / Lineage ID: {val.prov_o_hash}",
            f"* Tail Risk / Outlier Flags: {val.tail_risk_flags}",
            "",
            "# OPERATOR ACTION REQUIRED:",
            "[ ] APPROVE AS-IS",
            "[ ] OVERRIDE VALUE: [ _____________ ]",
            "[ ] REJECT / SUPPRESS SUBMISSION",
            "",
        ]

        return "\n".join(lines)

    @classmethod
    def format_dossier(cls, forecasts: List[ChallengeForecast]) -> str:
        header = [
            "╔══════════════════════════════════════════════════════════════════════════════════╗",
            "║           HEADLINE ARENA — HUMAN-IN-THE-LOOP (HITL) REVIEW DOSSIER              ║",
            "║                 AUTONOMOUS MACRO DISPATCH & UPGRADE HARNESS                      ║",
            "╚══════════════════════════════════════════════════════════════════════════════════╝",
            "",
            f"STAGED FORECAST COUNT: {len(forecasts)} Challenges Evaluated",
            "STATUS: HOLDING AT OPERATOR INTERCEPTION BOUNDARY (ZERO SUBMISSIONS EXECUTED)",
            "",
        ]

        body = [cls.format_single(fc) for fc in forecasts]

        return "\n".join(header) + "\n".join(body)

    @classmethod
    def format_summary_table(cls, forecasts: List[ChallengeForecast]) -> str:
        headers = ["CHALLENGE ID", "ASSET", "DIRECTION", "POINT EST", "CONF", "JSONLOGIC", "STATUS"]
        rows = []
        for fc in forecasts:
            cid_short = fc.challenge_id[:12] + "…" if len(fc.challenge_id) > 13 else fc.challenge_id
            direction = fc.proposed_submission.direction.value.upper()
            pt = f"${fc.proposed_submission.point_forecast:,.2f}"
            conf = f"{fc.proposed_submission.confidence:.2f}"
            jl = "PASS" if fc.validation.jsonlogic_passed else "FAIL"
            status = "STAGED (HITL)"
            rows.append([cid_short, fc.asset, direction, pt, conf, jl, status])

        col_widths = [max(len(row[i]) for row in ([headers] + rows)) for i in range(len(headers))]
        
        def fmt_row(vals):
            return " | ".join(f"{vals[i]:<{col_widths[i]}}" for i in range(len(vals)))

        sep = "-+-".join("-" * w for w in col_widths)

        lines = [
            fmt_row(headers),
            sep,
        ]
        for r in rows:
            lines.append(fmt_row(r))

        return "\n".join(lines)
