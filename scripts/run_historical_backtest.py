#!/usr/bin/env python3
"""
CLI Runner for Adam Financial OS Historical Backtest & Reporting Framework.

Executes a multi-session historical simulation across 7 valid trading days (Monday Aug 3 to Tuesday Aug 11, 2026),
evaluating 1-day forecasts forward from t to t+1 without look-ahead bias, applying accurate portfolio return attribution
and turnover-based transaction costs, logging foreign-key linked JSONL ledger records, and producing 9-section Daily Briefs.
"""

import argparse
import os
import sys
import glob

from adam_finance.backtest import (
    MarketObservation,
    RegimeAssessment,
    Forecast,
    Decision,
    Evaluation,
    RiskOfficerFeedback,
    MacroSentinelFeedback,
    HistoricalLedger,
    DailyBriefGenerator,
    calculate_yield_spread_bps,
    calculate_brier_score,
    calculate_strategy_attribution,
    calculate_max_drawdown,
    calculate_calibration_by_buckets,
    run_pre_publication_audit,
)


def run_seven_session_backtest(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Clean existing ledger and brief files in output directory for idempotency
    for existing_file in glob.glob(os.path.join(output_dir, "*.jsonl")) + glob.glob(os.path.join(output_dir, "*.md")):
        try:
            os.remove(existing_file)
        except OSError:
            pass

    ledger_path = os.path.join(output_dir, "historical_backtest_ledger.jsonl")
    ledger = HistoricalLedger(ledger_path)

    # 7 valid trading sessions (August 2026, skipping weekends)
    session_market_data = [
        {
            "date": "2026-08-03",
            "sp500": 7610.0,
            "sp500_prev": 7585.0,
            "y10": 4.58,
            "y10_prev": 4.52,
            "y2": 4.43,
            "y2_prev": 4.40,
            "oil": 84.20,
            "oil_prev": 83.40,
            "gold": 3140.0,
            "gold_prev": 3125.0,
            "btc": 76400.0,
            "btc_prev": 75500.0,
            "regime": "Transition",
            "growth": "Slowing",
            "inflation": "Moderating",
        },
        {
            "date": "2026-08-04",
            "sp500": 7580.0,
            "sp500_prev": 7610.0,
            "y10": 4.62,
            "y10_prev": 4.58,
            "y2": 4.45,
            "y2_prev": 4.43,
            "oil": 84.90,
            "oil_prev": 84.20,
            "gold": 3155.0,
            "gold_prev": 3140.0,
            "btc": 75200.0,
            "btc_prev": 76400.0,
            "regime": "Risk-off",
            "growth": "Slowing",
            "inflation": "Sticky",
        },
        {
            "date": "2026-08-05",
            "sp500": 7540.0,
            "sp500_prev": 7580.0,
            "y10": 4.68,
            "y10_prev": 4.62,
            "y2": 4.48,
            "y2_prev": 4.45,
            "oil": 85.80,
            "oil_prev": 84.90,
            "gold": 3170.0,
            "gold_prev": 3155.0,
            "btc": 74100.0,
            "btc_prev": 75200.0,
            "regime": "Risk-off",
            "growth": "Slowing",
            "inflation": "Accelerating",
        },
        {
            "date": "2026-08-06",
            "sp500": 7590.0,
            "sp500_prev": 7540.0,
            "y10": 4.65,
            "y10_prev": 4.68,
            "y2": 4.44,
            "y2_prev": 4.48,
            "oil": 85.10,
            "oil_prev": 85.80,
            "gold": 3160.0,
            "gold_prev": 3170.0,
            "btc": 75800.0,
            "btc_prev": 74100.0,
            "regime": "Mixed",
            "growth": "Expanding",
            "inflation": "Moderating",
        },
        {
            "date": "2026-08-07",
            "sp500": 7640.0,
            "sp500_prev": 7590.0,
            "y10": 4.60,
            "y10_prev": 4.65,
            "y2": 4.41,
            "y2_prev": 4.44,
            "oil": 84.30,
            "oil_prev": 85.10,
            "gold": 3145.0,
            "gold_prev": 3160.0,
            "btc": 77200.0,
            "btc_prev": 75800.0,
            "regime": "Risk-on",
            "growth": "Expanding",
            "inflation": "Moderating",
        },
        {
            "date": "2026-08-10",
            "sp500": 7680.0,
            "sp500_prev": 7640.0,
            "y10": 4.56,
            "y10_prev": 4.60,
            "y2": 4.38,
            "y2_prev": 4.41,
            "oil": 83.50,
            "oil_prev": 84.30,
            "gold": 3135.0,
            "gold_prev": 3145.0,
            "btc": 78100.0,
            "btc_prev": 77200.0,
            "regime": "Risk-on",
            "growth": "Expanding",
            "inflation": "Moderating",
        },
        {
            "date": "2026-08-11",
            "sp500": 7720.0,
            "sp500_prev": 7680.0,
            "y10": 4.52,
            "y10_prev": 4.56,
            "y2": 4.35,
            "y2_prev": 4.38,
            "oil": 82.80,
            "oil_prev": 83.50,
            "gold": 3120.0,
            "gold_prev": 3135.0,
            "btc": 79000.0,
            "btc_prev": 78100.0,
            "regime": "Risk-on",
            "growth": "Expanding",
            "inflation": "Moderating",
        },
    ]

    all_probabilities = []
    all_outcomes = []
    strategy_equity_curve = [100.0]  # Base portfolio index 100.0

    # Store active forecast and decision for t -> t+1 forward evaluation
    active_forecast = None
    active_decision = None
    previous_target_weight = 1.0

    print("=== Starting 7-Session Historical Backtest Cycle ===")

    for idx, s in enumerate(session_market_data):
        dt_str = s["date"]
        ts_str = f"{dt_str} 16:00 UTC"
        spread = calculate_yield_spread_bps(s["y10"], s["y2"])

        # 1. Market Observations
        observations = [
            MarketObservation("SP500", s["sp500"], s["sp500_prev"], "index_points", "S&P Index Services", dt_str),
            MarketObservation("US_10Y_YIELD", s["y10"], s["y10_prev"], "percent", "US Treasury", dt_str),
            MarketObservation("US_2Y_YIELD", s["y2"], s["y2_prev"], "percent", "US Treasury", dt_str),
            MarketObservation("10Y_2Y_SPREAD", spread / 100.0, (s["y10_prev"] - s["y2_prev"]), "percentage_points", "Calculated (10Y-2Y)", dt_str, data_priority=4),
            MarketObservation("CRUDE_OIL", s["oil"], s["oil_prev"], "USD/bbl", "NYMEX WTI", dt_str),
            MarketObservation("GOLD", s["gold"], s["gold_prev"], "USD/oz", "COMEX Spot", dt_str),
            MarketObservation("BITCOIN", s["btc"], s["btc_prev"], "USD", "Coinbase Aggregate", dt_str),
        ]

        for obs in observations:
            ledger.append_record(obs.to_ledger_dict())

        # 2. Regime Assessment
        regime = RegimeAssessment(
            session_date=dt_str,
            growth_evidence=f"Manufacturing PMI indicates {s['growth']} activity.",
            growth_direction=s["growth"],
            inflation_evidence=f"Consumer price indices confirm {s['inflation']} trajectory.",
            inflation_direction=s["inflation"],
            monetary_policy_implied=f"10Y-2Y yield spread at +{spread} bps.",
            liquidity_risk_appetite="Cross-asset confirmation intact.",
            regime_classification=s["regime"],
            classification_rules="Rulebook-v2.1: Yield Curve + Equity Volatility + Commodity Momentum",
            data_confidence=0.95,
            thesis_confidence=0.85,
            forecast_confidence=0.60,
        )
        ledger.append_record(regime.to_ledger_dict())

        # 3. Evaluate Active Forecast from Day t-1 against Day t realized price move (Forward Horizon)
        evaluations_for_brief = []
        if active_forecast is not None and active_decision is not None:
            # Benchmark return from t-1 close to t close
            fwd_benchmark_ret = round(((s["sp500"] - active_decision.execution_price) / active_decision.execution_price) * 100.0, 4)

            # Correct directional call check
            dir_correct = (fwd_benchmark_ret < 0 and active_forecast.direction == "down") or \
                          (fwd_benchmark_ret > 0 and active_forecast.direction == "up")
            outcome_binary = 1 if dir_correct else 0

            # Calculate strategy gross & net returns
            gross_strat_ret, net_strat_ret = calculate_strategy_attribution(
                benchmark_return_pct=fwd_benchmark_ret,
                target_weight=active_decision.target_weight,
                previous_weight=active_decision.previous_weight,
            )

            # Update strategy cumulative equity curve
            prev_equity = strategy_equity_curve[-1]
            new_equity = prev_equity * (1.0 + net_strat_ret / 100.0)
            strategy_equity_curve.append(new_equity)

            current_strat_dd = calculate_max_drawdown(strategy_equity_curve)
            brier_val = round((active_forecast.probability - outcome_binary) ** 2, 6)

            all_probabilities.append(active_forecast.probability)
            all_outcomes.append(outcome_binary)

            evaluation = Evaluation(
                forecast_id=active_forecast.forecast_id,
                decision_id=active_decision.decision_id,
                evaluation_date=dt_str,
                realized_return_pct=fwd_benchmark_ret,
                benchmark_return_pct=fwd_benchmark_ret,
                net_strategy_return_pct=net_strat_ret,
                direction_correct=dir_correct,
                brier_score=brier_val,
                max_drawdown_pct=current_strat_dd,
            )
            ledger.append_record(evaluation.to_ledger_dict())
            evaluations_for_brief.append(evaluation)

            # Update active forecast status
            active_forecast.status = "evaluated"

        # 4. Generate Signal & Forecast for Day t -> t+1
        fc_id = f"F-{dt_str.replace('-', '')}-001"
        dec_id = f"D-{dt_str.replace('-', '')}-001"

        is_bearish = s["regime"] in ["Risk-off", "Transition"]
        direction = "down" if is_bearish else "up"
        prob = 0.65 if is_bearish else 0.60
        target_wt = 0.8 if is_bearish else 1.0

        forecast = Forecast(
            forecast_id=fc_id,
            created_at=ts_str,
            asset="SP500",
            direction=direction,
            horizon_trading_days=1,
            probability=prob,
            expected_return_pct=-0.5 if is_bearish else 0.5,
            strategy_version="Adam-Macro-v1.0",
            thesis=f"Expect SP500 {direction} move due to {s['regime']} environment.",
            invalidation_condition=f"10Y yield reversal > 10 bps or SP500 breach against {direction}.",
        )
        ledger.append_record(forecast.to_ledger_dict())

        dec_action = "hedge" if target_wt < 1.0 else "maintain"
        decision = Decision(
            decision_id=dec_id,
            forecast_id=fc_id,
            decision_timestamp=ts_str,
            action=dec_action,
            target_weight=target_wt,
            previous_weight=previous_target_weight,
            execution_price=s["sp500"],
        )
        ledger.append_record(decision.to_ledger_dict())

        # 5. Governance & Reviewer Feedback
        risk_fb = RiskOfficerFeedback(
            date=dt_str,
            forecast_id=fc_id,
            decision_id=dec_id,
            concentration_risk=f"Portfolio S&P 500 target weight at {target_wt * 100:.0f}%. Within risk mandate.",
            volatility_correlation_risk="Equity-bond correlation neutral.",
            liquidity_execution_risk="S&P liquidity high; execution turnover minimal.",
            drawdown_stress_loss=f"Current strategy peak-to-trough drawdown: {calculate_max_drawdown(strategy_equity_curve):.2f}%.",
            position_limits_stops="Stop-loss fixed at 2% adverse price movement.",
            permission_decision="Approve",
            proposed_risk_adjustments=f"Set S&P 500 target weight to {target_wt}.",
        )
        ledger.append_record(risk_fb.to_ledger_dict())

        if s["regime"] in ["Risk-on", "Mixed"]:
            supp_ev = f"Expanding growth momentum ({s['growth']}) and moderating inflation support equity upside."
            contra_ev = f"Yield curve spread ({spread} bps) warrants monitoring."
            verdict = "Confirm"
        else:
            supp_ev = f"Yield curve spread ({spread} bps) and slowing growth ({s['growth']}) warrant defensive stance."
            contra_ev = f"Oil price move ({s['oil']} USD/bbl) and resilient employment data."
            verdict = "Qualify"

        macro_fb = MacroSentinelFeedback(
            date=dt_str,
            forecast_id=fc_id,
            supporting_evidence=supp_ev,
            contradictory_evidence=contra_ev,
            leading_indicators_to_monitor="High Yield credit spreads & 2Y Treasury yield trajectory.",
            regime_invalidation_conditions="Yield curve steepening above +20 bps or sharp growth deceleration.",
            thesis_verdict=verdict,
        )
        ledger.append_record(macro_fb.to_ledger_dict())

        # Audit checklist
        audit_res = run_pre_publication_audit(
            market_observations=observations,
            regime=regime,
            forecasts=[forecast],
            decisions=[decision],
            risk_feedback=risk_fb,
            macro_feedback=macro_fb,
            evaluations=evaluations_for_brief,
        )

        exec_summary = {
            "market_regime": s["regime"],
            "primary_driver": f"Yield curve dynamics ({spread} bps spread) and {s['growth'].lower()} growth momentum.",
            "portfolio_stance": dec_action.title(),
            "key_uncertainty": "Upcoming central bank policy meeting & inflation revisions.",
        }

        audit_info = {
            "data_sources": "Primary Treasury & Market Index Feeds",
            "model_version": "Adam-Backtest-v1.0",
            "record_hash": f"SHA256-SESSION-{idx+1}",
        }

        brief_gen = DailyBriefGenerator(
            session_date=dt_str,
            generated_at=ts_str,
            data_status="Verified",
            forecast_horizon="1 day",
            exec_summary=exec_summary,
            market_observations=observations,
            regime=regime,
            forecasts=[forecast],
            decisions=[decision],
            evaluations=evaluations_for_brief,
            risk_feedback=risk_fb,
            macro_feedback=macro_fb,
            audit_info=audit_info,
        )

        md_brief = brief_gen.generate_markdown()
        brief_filename = os.path.join(output_dir, f"daily_brief_{dt_str}.md")
        with open(brief_filename, "w", encoding="utf-8") as f:
            f.write(md_brief)

        print(f"Session Day {idx+1} ({dt_str}): Regime={s['regime']}, Forecast={direction.upper()} (p={prob}), Audit Passed={audit_res['all_passed']}")

        # Set active forecast and decision for next session's evaluation
        active_forecast = forecast
        active_decision = decision
        previous_target_weight = target_wt

    # Final Summary Statistics
    avg_brier = calculate_brier_score(all_probabilities, all_outcomes) if all_outcomes else 0.0
    dir_acc = (sum(all_outcomes) / len(all_outcomes) * 100.0) if all_outcomes else 0.0
    cal_buckets = calculate_calibration_by_buckets(all_probabilities, all_outcomes) if all_outcomes else {}

    print("\n=== Backtest Performance Scorecard ===")
    print(f"Total Trading Sessions   : {len(session_market_data)}")
    print(f"Evaluated Forward Horizon: {len(all_outcomes)} sessions")
    print(f"Directional Accuracy     : {dir_acc:.2f}%")
    print(f"Aggregate Brier Score    : {avg_brier:.4f}")
    print(f"Max Strategy Drawdown    : {calculate_max_drawdown(strategy_equity_curve):.2f}%")
    print(f"Calibration Buckets      : {cal_buckets}")
    print(f"JSONL Ledger Saved To    : {ledger_path}")
    print("======================================\n")


def main():
    parser = argparse.ArgumentParser(description="Adam Financial OS — Historical Backtest Framework CLI")
    parser.add_argument("--output-dir", type=str, default="reports/backtest_results", help="Directory to save briefs and ledger")
    args = parser.parse_args()

    run_seven_session_backtest(args.output_dir)


if __name__ == "__main__":
    main()
