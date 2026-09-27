#!/usr/bin/env python3
"""
ADAM-Macro-Sentinel Engine — Main Orchestrator
================================================
Wires together all components into the full forecast lifecycle:

    MarketContext → Arbitration → Rationale → Gate → Calibrate → Submit → Ledger

Usage:
    python -m adam_sentinel.engine [--dry-run] [--context FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .api import HeadlineArenaClient
from .arbitration import ChampionChallengerArbitrator
from .calibration import EpistemicMemoryLedger
from .gate import GateResult, PreSubmissionGate
from .rationale import MarketContext, RationaleGenerator
from .schema import Direction, ForecastSubmission, RationaleBlock


# ─── Default Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
CREDS_PATH = PROJECT_ROOT / ".ha_credentials.json"
LEDGER_PATH = PROJECT_ROOT / "data" / "memory" / "epistemic_ledger.jsonl"
CONTEXT_PATH = PROJECT_ROOT / "data" / "market_data" / "live_context.json"


class SentinelEngine:
    """
    Main orchestrator for the ADAM-Macro-Sentinel forecasting engine.

    Pipeline:
    1. Load market context (from file or API)
    2. For each asset:
       a. Build MarketContext snapshot
       b. Run Champion-Challenger arbitration
       c. Generate structured 4-dimension rationale
       d. Calibrate confidence via Platt scaling
       e. Build ForecastSubmission
       f. Validate through pre-submission gate
    3. Submit passing forecasts to Headline Arena
    4. Record in Epistemic Memory Ledger
    """

    def __init__(
        self,
        creds_path: Path = CREDS_PATH,
        ledger_path: Path = LEDGER_PATH,
        dry_run: bool = False,
    ):
        self.dry_run = dry_run
        self.client = HeadlineArenaClient(creds_path)
        self.ledger = EpistemicMemoryLedger(ledger_path)
        self.arbitrator = ChampionChallengerArbitrator()
        self.rationale_gen = RationaleGenerator()
        self.gate = PreSubmissionGate()

        # Pipeline state
        self._forecasts: list[ForecastSubmission] = []
        self._gate_results: list[GateResult] = []
        self._submission_results: list[dict] = []

    # ─── Pipeline Execution ──────────────────────────────────────────────

    def run(
        self,
        contexts: list[MarketContext],
        auto_submit: bool = True,
    ) -> dict:
        """
        Execute the full forecasting pipeline.

        Args:
            contexts: List of MarketContext snapshots, one per asset.
            auto_submit: If True (and not dry_run), submit passing forecasts.

        Returns:
            Pipeline execution summary.
        """
        print(self._banner())

        # Step 1: Authenticate
        if not self.dry_run:
            token = self.client.authenticate()
            if not token:
                print("  ✗ Authentication failed.")
                return {"error": "auth_failed"}
            print(f"  ✓ Authenticated as {self.client.agent_id}")
        else:
            print("  ℹ DRY RUN — skipping authentication")

        # Step 2: Process each asset
        print(f"\n  Processing {len(contexts)} asset(s)...")
        for ctx in contexts:
            self._process_asset(ctx)

        # Step 3: Gate validation report
        print(self.gate.summary_report(self._gate_results))

        # Step 4: Submit passing forecasts
        passing = [
            (f, g) for f, g in zip(self._forecasts, self._gate_results)
            if g.overall_pass
        ]

        if auto_submit and not self.dry_run and passing:
            print(f"\n  Submitting {len(passing)} forecast(s)...")
            self._submit_forecasts(passing)
        elif self.dry_run:
            print(f"\n  ℹ DRY RUN — {len(passing)} forecast(s) would be submitted")
        elif not passing:
            print("\n  ⚠ No forecasts passed the gate. Review recommendations above.")

        # Step 5: Summary
        summary = self._build_summary(contexts)
        print(self._format_summary(summary))

        # Step 6: Emit JSON if needed
        return summary

    def _process_asset(self, ctx: MarketContext) -> None:
        """Process a single asset through the full pipeline."""
        print(f"\n  ━━━ {ctx.asset_ticker} ({ctx.asset_name}) ━━━")

        # Step 2a: Champion-Challenger Arbitration
        print(f"    [Arbitration] Champion vs Challenger debate...")
        record = self.arbitrator.arbitrate(ctx)
        print(
            f"    [Arbitration] Winner: {record.winner} → "
            f"{record.final_direction.value} @ {record.final_confidence:.0%}"
        )

        # Step 2b: Calibrate confidence via Platt scaling
        raw_confidence = record.final_confidence
        calibrated = self.ledger.calibrate_confidence(raw_confidence)
        print(
            f"    [Calibrate] Raw: {raw_confidence:.0%} → "
            f"Platt-calibrated: {calibrated:.0%}"
        )

        # Step 2c: Generate structured rationale
        rolling_brier = self.ledger.get_rolling_brier()
        rolling_accuracy = self.ledger.get_rolling_accuracy()

        rationale_dict = self.rationale_gen.generate_full_rationale(
            ctx=ctx,
            confidence=raw_confidence,
            calibrated_confidence=calibrated,
            rolling_brier=rolling_brier,
            rolling_accuracy=rolling_accuracy,
        )

        # Step 2d: Build ForecastSubmission
        try:
            forecast = ForecastSubmission(
                challenge_id=f"pending-{ctx.asset_ticker}",  # Will be matched later
                target_asset=ctx.asset_ticker,
                direction=record.final_direction,
                confidence=calibrated,
                point_forecast=ctx.current_price if ctx.current_price else None,
                std_deviation=(
                    ctx.current_price * (ctx.historical_vol_20d / 100 / (252 ** 0.5))
                    if ctx.historical_vol_20d and ctx.current_price
                    else None
                ),
                rationale=RationaleBlock(**rationale_dict),
                summary_statement=self._generate_summary(ctx, record, calibrated),
            )
        except Exception as e:
            print(f"    ✗ Schema construction failed: {e}")
            return

        self._forecasts.append(forecast)

        # Step 2e: Pre-submission gate
        gate_result = self.gate.validate(
            forecast,
            reference_price=ctx.prior_close,
            arbitration_complete=True,
        )
        self._gate_results.append(gate_result)

        status = "✓ PASS" if gate_result.overall_pass else "✗ FAIL"
        s_rat = gate_result.s_rat_verdict.s_rat_score if gate_result.s_rat_verdict else 0
        print(f"    [Gate] {status} — S_rat: {s_rat:.1f}")

        if not gate_result.overall_pass:
            for reason in gate_result.failure_reasons:
                print(f"      ↳ {reason}")

        # Record in ledger (pre-settlement)
        self.ledger.record_forecast(
            forecast,
            raw_confidence=raw_confidence,
            arbitration_winner=record.winner,
        )

    def _submit_forecasts(
        self,
        passing: list[tuple[ForecastSubmission, GateResult]],
    ) -> None:
        """Submit all gate-passing forecasts to the API."""
        # Discover active challenges to match IDs
        challenges = self.client.discover_challenges()
        challenge_map = {}
        for item in challenges:
            ch = item.get("challenge", item)
            cid = ch.get("id", "")
            # Map by asset ticker
            title = str(ch.get("title", "")).upper()
            for ticker in ["GC", "CL", "ZN", "ES", "NG", "DXY", "HG", "RB", "SI",
                           "BTC", "ETH"]:
                if ticker in title:
                    challenge_map[ticker] = cid
                    break

        for forecast, gate_result in passing:
            # Match forecast to challenge ID
            real_cid = challenge_map.get(forecast.target_asset, "")
            if not real_cid:
                print(f"    ⚠ No matching challenge for {forecast.target_asset}")
                continue

            payload = forecast.to_api_payload()
            payload["challenge_id"] = real_cid

            result = self.client.submit_prediction(real_cid, payload)
            scored = result.get("counts_for_score", False)
            self._submission_results.append({
                "asset": forecast.target_asset,
                "challenge_id": real_cid,
                "direction": forecast.direction.value,
                "confidence": forecast.confidence,
                "scored": scored,
            })
            print(
                f"    ✓ {forecast.target_asset} → {forecast.direction.value} "
                f"({forecast.confidence:.0%}) [scored={scored}]"
            )

    def _generate_summary(
        self,
        ctx: MarketContext,
        record,
        calibrated: float,
    ) -> str:
        """Generate the 2-sentence institutional synthesis."""
        direction_word = record.final_direction.value
        return (
            f"{ctx.asset_name} ({ctx.asset_ticker}) is {direction_word} at "
            f"Platt-calibrated confidence {calibrated:.0%}, driven by "
            f"{ctx.primary_catalyst or 'convergent macro signals'}. "
            f"The {record.winner} thesis prevailed through adversarial "
            f"arbitration with explicit counterfactual rebuttal."
        )

    # ─── Summary & Reporting ─────────────────────────────────────────────

    def _build_summary(self, contexts: list[MarketContext]) -> dict:
        """Build the pipeline execution summary."""
        passed = sum(1 for g in self._gate_results if g.overall_pass)
        submitted = len(self._submission_results)

        return {
            "engine": "ADAM-Macro-Sentinel",
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dry_run": self.dry_run,
            "pipeline": {
                "assets_processed": len(contexts),
                "forecasts_generated": len(self._forecasts),
                "gate_passed": passed,
                "gate_failed": len(self._gate_results) - passed,
                "submitted": submitted,
            },
            "calibration": self.ledger.get_ledger_diagnostics(),
            "arbitration": self.arbitrator.get_win_rates(),
            "forecasts": [
                {
                    "asset": f.target_asset,
                    "direction": f.direction.value,
                    "confidence": f.confidence,
                    "gate_pass": g.overall_pass,
                    "s_rat": g.s_rat_verdict.s_rat_score if g.s_rat_verdict else 0,
                    "hash": f.content_hash()[:12],
                }
                for f, g in zip(self._forecasts, self._gate_results)
            ],
            "ledger_chain_hash": self.ledger.export_chain_hash()[:24],
        }

    def _format_summary(self, summary: dict) -> str:
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║  ADAM-Macro-Sentinel Execution Summary                     ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            f"  Timestamp:        {summary['timestamp']}",
            f"  Mode:             {'DRY RUN' if summary['dry_run'] else 'LIVE'}",
            f"  Assets processed: {summary['pipeline']['assets_processed']}",
            f"  Gate passed:      {summary['pipeline']['gate_passed']}",
            f"  Gate failed:      {summary['pipeline']['gate_failed']}",
            f"  Submitted:        {summary['pipeline']['submitted']}",
            "",
            "  ─── Calibration ─────────────────────────────────────────",
            f"  Platt params:     a={summary['calibration']['platt_params']['a']:.4f}, "
            f"b={summary['calibration']['platt_params']['b']:.4f}",
            f"  Rolling Brier:    {summary['calibration']['avg_brier']:.4f}",
            f"  Rolling accuracy: {summary['calibration']['hit_rate']:.0%}",
            "",
            "  ─── Arbitration ─────────────────────────────────────────",
            f"  Champion win rate:  {summary['arbitration']['champion']:.0%}",
            f"  Challenger win rate:{summary['arbitration']['challenger']:.0%}",
            "",
            "  ─── Forecasts ───────────────────────────────────────────",
            f"  {'Asset':<8} {'Dir':<10} {'Conf':>5} {'Gate':>6} {'S_rat':>6} {'Hash'}",
            f"  {'─'*8} {'─'*10} {'─'*5} {'─'*6} {'─'*6} {'─'*12}",
        ]

        for f in summary.get("forecasts", []):
            gate = "✓" if f["gate_pass"] else "✗"
            lines.append(
                f"  {f['asset']:<8} {f['direction']:<10} "
                f"{f['confidence']:>4.0%} {gate:>6} "
                f"{f['s_rat']:>5.1f} {f['hash']}"
            )

        lines.extend([
            "",
            f"  Ledger chain hash: {summary['ledger_chain_hash']}",
            f"  Agent: https://headlinearena.com/agent/{''}", 
            f"  Board: https://headlinearena.com/rankings",
            "",
        ])

        return "\n".join(lines)

    def _banner(self) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return (
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  ADAM-Macro-Sentinel v1.0 — Autonomous Forecasting Engine  ║\n"
            f"║  {now:<58}║\n"
            "║  Champion-Challenger Arbitration · Platt Calibration       ║\n"
            "║  4-Dimension Rationale · Epistemic Memory Ledger          ║\n"
            "╚══════════════════════════════════════════════════════════════╝"
        )


# ─── Context Loading ─────────────────────────────────────────────────────────

def load_contexts_from_file(path: str | Path) -> list[MarketContext]:
    """
    Load MarketContext objects from a JSON file.
    Expected format: list of objects or {"assets": [...]}
    """
    data = json.loads(Path(path).read_text())

    if isinstance(data, dict):
        assets = data.get("assets", data.get("forecasts", [data]))
    elif isinstance(data, list):
        assets = data
    else:
        assets = [data]

    contexts = []
    for asset in assets:
        ctx = MarketContext(
            asset_ticker=asset.get("ticker", asset.get("asset_ticker", "")),
            asset_name=asset.get("name", asset.get("asset_name", "")),
            current_price=asset.get("current_price", asset.get("price", 0.0)),
            prior_close=asset.get("prior_close", asset.get("prev_close", 0.0)),
            daily_change_pct=asset.get("daily_change_pct", asset.get("change_pct", 0.0)),
            implied_vol=asset.get("implied_vol"),
            historical_vol_20d=asset.get("historical_vol_20d", asset.get("vol_20d")),
            daily_range_1sigma=asset.get("daily_range_1sigma"),
            primary_catalyst=asset.get("primary_catalyst", ""),
            catalyst_type=asset.get("catalyst_type", ""),
            catalyst_timestamp=asset.get("catalyst_timestamp", ""),
            cftc_net_speculative=asset.get("cftc_net_speculative"),
            dealer_gamma_exposure=asset.get("dealer_gamma_exposure"),
            crowding_signal=asset.get("crowding_signal"),
            headline_catalysts=asset.get("headline_catalysts", []),
            supporting_signals=asset.get("supporting_signals", []),
            contrary_signals=asset.get("contrary_signals", []),
            rate_differential=asset.get("rate_differential"),
            yield_curve_shape=asset.get("yield_curve_shape"),
            credit_spread_trend=asset.get("credit_spread_trend"),
            prior_direction=asset.get("prior_direction"),
            prior_score=asset.get("prior_score"),
            prior_error_analysis=asset.get("prior_error_analysis"),
        )
        contexts.append(ctx)

    return contexts


def build_demo_contexts() -> list[MarketContext]:
    """
    Build demo MarketContext objects based on the existing ADAM data
    from ha_round2.py analysis (Sep 2026 macro regime).
    """
    return [
        MarketContext(
            asset_ticker="GC",
            asset_name="Gold Futures",
            current_price=4347.20,
            prior_close=4269.00,
            daily_change_pct=1.83,
            implied_vol=18.5,
            historical_vol_20d=16.2,
            daily_range_1sigma=1.2,
            primary_catalyst=(
                "Iran war escalation: UN mission finds reasonable grounds "
                "US committed war crimes in Iran (Axios)"
            ),
            catalyst_type="geopolitical",
            catalyst_timestamp="Sep 17 2026",
            cftc_net_speculative="net_long",
            crowding_signal="moderately_long",
            headline_catalysts=[
                "UN mission: reasonable grounds US committed war crimes in Iran",
                "Trump tells Axios approaching 'major crossroads' in Iran war",
                "Rising oil, rates and yields brew up stagflation cocktail (Reuters)",
            ],
            supporting_signals=[
                "Gold rallied +1.83% DESPITE Fed rate hike — geopolitical premium dominates",
                "Energy disruption hits Bangladesh and Pakistan as Gulf crisis worsens",
            ],
            contrary_signals=[
                "Goldman Sachs calls for October rate hike — higher real rates bearish for gold",
                "Fed hawkish guidance: more tightening ahead raises opportunity cost",
            ],
            rate_differential="US front-end rates rising on Fed hike",
            yield_curve_shape="bear flattening",
            prior_direction="bearish",
            prior_score=14.0,
            prior_error_analysis=(
                "Single-factor thesis (Fed hike) ignored dominant geopolitical bid. "
                "Gold rallied through the hike, proving safe-haven flow overwhelms "
                "rate headwinds in active war regime."
            ),
        ),
        MarketContext(
            asset_ticker="CL",
            asset_name="WTI Crude Oil Futures",
            current_price=96.48,
            prior_close=102.18,
            daily_change_pct=-5.50,
            implied_vol=35.0,
            historical_vol_20d=28.5,
            daily_range_1sigma=2.1,
            primary_catalyst=(
                "Saudi Arabia increasing crude supply via Oman loading, "
                "US inventory build reported"
            ),
            catalyst_type="supply_shock",
            catalyst_timestamp="Sep 17 2026",
            cftc_net_speculative="net_long",
            crowding_signal="crowded_long",
            headline_catalysts=[
                "Oil eases as Saudi offers more crude via Oman loading (Reuters)",
                "US crude inventories post surprise build",
                "Wall St rises as oil slide offers respite from rate jitters",
            ],
            supporting_signals=[
                "Saudi supply increase is ongoing — not a one-day event",
                "Round 1 bearish thesis correct: scored 89/100",
            ],
            contrary_signals=[
                "Iran war: 'Energy disruption hits Bangladesh and Pakistan'",
                "Trump 'crossroads' on Iran — escalation risk = crude spike",
                "After -5.5% drop, mean-reversion probability elevated",
            ],
            rate_differential="Hawkish Fed signals demand destruction ahead",
            prior_direction="bearish",
            prior_score=89.0,
            prior_error_analysis=None,
        ),
        MarketContext(
            asset_ticker="DXY",
            asset_name="US Dollar Index",
            current_price=100.06,
            prior_close=100.13,
            daily_change_pct=-0.07,
            implied_vol=9.5,
            historical_vol_20d=7.8,
            daily_range_1sigma=0.5,
            primary_catalyst="Fed rate hike widens rate differential vs G10 peers",
            catalyst_type="monetary_policy",
            catalyst_timestamp="Sep 16 2026",
            cftc_net_speculative="net_long",
            crowding_signal="moderately_long",
            headline_catalysts=[
                "Fed raises rates with more tightening signaled",
                "Trump demands lower rates — political tension with Fed",
            ],
            supporting_signals=[
                "Rate differential favoring USD vs EUR, GBP, JPY",
            ],
            contrary_signals=[
                "DXY flat (-0.07%) despite rate hike — bullish thesis not confirmed by price",
                "Trump pressure on Fed introduces policy uncertainty",
            ],
            rate_differential="US rates higher than G10 peers post-hike",
            yield_curve_shape="bear flattening",
            prior_direction="bullish",
            prior_score=50.0,
            prior_error_analysis=(
                "Bullish DXY call at 73% confidence yielded flat price action. "
                "The rate hike was fully priced — no incremental USD demand."
            ),
        ),
        MarketContext(
            asset_ticker="ZN",
            asset_name="10-Year Treasury Note Futures",
            current_price=105.69,
            prior_close=105.20,
            daily_change_pct=0.47,
            implied_vol=12.0,
            historical_vol_20d=10.5,
            daily_range_1sigma=0.4,
            primary_catalyst=(
                "Post-Fed-hike bond market rally — 'sell the rumor, buy the fact' dynamic"
            ),
            catalyst_type="monetary_policy",
            catalyst_timestamp="Sep 17 2026",
            headline_catalysts=[
                "BoE pauses gilt market overhaul following bond selloff",
                "Warsh lays out forces driving up bond yields — but ZN rallied anyway",
            ],
            supporting_signals=[
                "ZN rallied +0.47% post-hike — classic buy-the-fact reflex",
                "Duration demand from liability-driven investors at higher yields",
            ],
            contrary_signals=[
                "Fed forward guidance for more tightening should pressure bonds",
                "Goldman October hike call — additional supply of rate expectations",
            ],
            yield_curve_shape="bear flattening",
            prior_direction="bearish",
            prior_score=25.0,
            prior_error_analysis=(
                "Bearish ZN thesis ignored the post-hike rally reflex. "
                "Rate hikes are often followed by short-covering in duration."
            ),
        ),
    ]


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ADAM-Macro-Sentinel Autonomous Forecasting Engine",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate forecasts without submitting to Headline Arena",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Path to JSON file with market context data",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with built-in demo contexts (Sep 2026 macro regime)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON summary output",
    )

    args = parser.parse_args()

    # Load contexts
    if args.context:
        contexts = load_contexts_from_file(args.context)
    elif args.demo:
        contexts = build_demo_contexts()
    else:
        # Try default context file
        if CONTEXT_PATH.exists():
            contexts = load_contexts_from_file(CONTEXT_PATH)
        else:
            print("  ℹ No context file found — using demo contexts")
            contexts = build_demo_contexts()

    # Run engine
    engine = SentinelEngine(dry_run=args.dry_run)
    summary = engine.run(contexts)

    # Write output
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"\n  Summary written to {args.output}")


if __name__ == "__main__":
    main()
