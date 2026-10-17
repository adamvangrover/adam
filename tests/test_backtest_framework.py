import pytest
import os
import json
import tempfile
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


def test_yield_spread_calculation():
    # 10Y = 4.05%, 2Y = 4.20% -> spread = -15 bps
    spread = calculate_yield_spread_bps(4.05, 4.20)
    assert spread == pytest.approx(-15.0)

    # 10Y = 4.50%, 2Y = 4.10% -> spread = +40 bps
    spread_pos = calculate_yield_spread_bps(4.50, 4.10)
    assert spread_pos == pytest.approx(40.0)


def test_market_observation_properties():
    obs = MarketObservation(
        asset="SP500",
        latest_close=5500.0,
        previous_close=5000.0,
        unit="index_points",
        source="S&P Index Services",
        observation_date="2026-08-03",
    )
    assert obs.change == pytest.approx(500.0)
    assert obs.return_pct == pytest.approx(0.10)

    yield_obs = MarketObservation(
        asset="US_10Y_YIELD",
        latest_close=4.20,
        previous_close=4.05,
        unit="percent",
        source="US Treasury",
        observation_date="2026-08-03",
    )
    assert yield_obs.change == pytest.approx(0.15)
    assert yield_obs.change_bps == pytest.approx(15.0)


def test_strategy_attribution_and_turnover_cost():
    # Target weight 0.8, previous weight 1.0 -> Turnover = 0.2
    # Benchmark return = -0.3636%
    # Transaction cost = 5 bps, Slippage = 2 bps -> Total = 7 bps
    # Rebalance cost = 0.2 * 0.07% = 0.014%
    # Gross strategy return = 0.8 * (-0.3636%) = -0.29088% -> -0.2909%
    # Net strategy return = -0.29088% - 0.014% = -0.30488% -> -0.3049%
    gross_ret, net_ret = calculate_strategy_attribution(
        benchmark_return_pct=-0.3636,
        target_weight=0.8,
        previous_weight=1.0,
        transaction_costs_bps=5.0,
        slippage_bps=2.0,
    )
    assert gross_ret == pytest.approx(-0.2909)
    assert net_ret == pytest.approx(-0.3049)

    # Maintain position (Turnover = 0.0) -> Rebalance cost = 0.0
    gross_maint, net_maint = calculate_strategy_attribution(
        benchmark_return_pct=0.5405,
        target_weight=1.0,
        previous_weight=1.0,
    )
    assert gross_maint == pytest.approx(0.5405)
    assert net_maint == pytest.approx(0.5405)


def test_brier_score_calculation():
    # Perfect calibration
    probs = [1.0, 0.0, 1.0]
    outcomes = [1, 0, 1]
    assert calculate_brier_score(probs, outcomes) == pytest.approx(0.0)

    # Uncalibrated / wrong predictions
    probs_wrong = [0.8, 0.9]
    outcomes_wrong = [0, 0]
    # (0.8^2 + 0.9^2) / 2 = (0.64 + 0.81) / 2 = 1.45 / 2 = 0.725
    assert calculate_brier_score(probs_wrong, outcomes_wrong) == pytest.approx(0.725)


def test_brier_score_invalid_input():
    with pytest.raises(ValueError):
        calculate_brier_score([0.5], [1, 0])


def test_max_drawdown_calculation():
    equity_curve = [100.0, 110.0, 99.0, 105.0, 88.0, 95.0]
    # Peak = 110.0, Trough = 88.0 -> Drawdown = (110 - 88) / 110 = 22 / 110 = 20%
    dd = calculate_max_drawdown(equity_curve)
    assert dd == pytest.approx(20.0)


def test_calibration_buckets():
    probs = [0.62, 0.68, 0.82]
    outcomes = [1, 0, 1]
    buckets = calculate_calibration_by_buckets(probs, outcomes, bucket_width=0.1)

    assert "0.6-0.7" in buckets
    assert buckets["0.6-0.7"]["count"] == 2
    assert buckets["0.6-0.7"]["mean_predicted_prob"] == pytest.approx(0.65)
    assert buckets["0.6-0.7"]["actual_realized_rate"] == pytest.approx(0.5)


def test_historical_ledger_immutability_and_reading():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_file = os.path.join(tmpdir, "test_ledger.jsonl")
        ledger = HistoricalLedger(ledger_file)

        fc = Forecast(
            forecast_id="F-001",
            created_at="2026-08-03 16:00 UTC",
            asset="SP500",
            direction="up",
            horizon_trading_days=1,
            probability=0.70,
            expected_return_pct=0.5,
            strategy_version="v1.0",
            thesis="Positive earnings momentum.",
            invalidation_condition="Breach below 5400.",
        )
        ledger.append_record(fc.to_ledger_dict())

        dec = Decision(
            decision_id="D-001",
            forecast_id="F-001",
            decision_timestamp="2026-08-03 16:00 UTC",
            action="maintain",
            target_weight=1.0,
            previous_weight=1.0,
            execution_price=5500.0,
        )
        ledger.append_record(dec.to_ledger_dict())

        # Append evaluation forward at t+1 date without overwriting original forecast
        ev = Evaluation(
            forecast_id="F-001",
            decision_id="D-001",
            evaluation_date="2026-08-04",
            realized_return_pct=0.54,
            benchmark_return_pct=0.54,
            net_strategy_return_pct=0.54,
            direction_correct=True,
            brier_score=0.0256,
            max_drawdown_pct=0.0,
        )
        ledger.append_record(ev.to_ledger_dict())

        all_records = ledger.read_records()
        assert len(all_records) == 3

        forecast_records = ledger.read_records(record_type="forecast")
        assert len(forecast_records) == 1
        assert forecast_records[0]["forecast_id"] == "F-001"
        assert forecast_records[0]["status"] == "pending"  # Original forecast unchanged

        eval_records = ledger.read_records(record_type="evaluation")
        assert len(eval_records) == 1
        assert eval_records[0]["forecast_id"] == "F-001"
        assert eval_records[0]["decision_id"] == "D-001"
        assert eval_records[0]["evaluation_date"] == "2026-08-04"


def test_daily_brief_generator_and_audit():
    obs = [
        MarketObservation("SP500", 5500.0, 5480.0, "index_points", "S&P Index Services", "2026-08-03"),
        MarketObservation("US_10Y_YIELD", 4.05, 4.00, "percent", "US Treasury", "2026-08-03"),
        MarketObservation("US_2Y_YIELD", 4.20, 4.18, "percent", "US Treasury", "2026-08-03"),
        MarketObservation("10Y_2Y_SPREAD", -0.15, -0.18, "percent", "Calculated", "2026-08-03"),
        MarketObservation("CRUDE_OIL", 78.50, 77.80, "USD/bbl", "NYMEX WTI", "2026-08-03"),
        MarketObservation("GOLD", 2420.0, 2410.0, "USD/oz", "COMEX Spot", "2026-08-03"),
        MarketObservation("BITCOIN", 65000.0, 64200.0, "USD", "Coinbase Aggregate", "2026-08-03"),
    ]

    regime = RegimeAssessment(
        session_date="2026-08-03",
        growth_evidence="PMI 52.1.",
        growth_direction="Expanding",
        inflation_evidence="CPI +0.2% MoM.",
        inflation_direction="Moderating",
        monetary_policy_implied="Curve inverted at -15 bps.",
        liquidity_risk_appetite="High risk appetite.",
        regime_classification="Risk-on",
        classification_rules="Rule-v1.0",
        data_confidence=0.95,
        thesis_confidence=0.90,
        forecast_confidence=0.65,
    )

    fc = Forecast(
        forecast_id="F-20260803-001",
        created_at="2026-08-03 16:00 UTC",
        asset="SP500",
        direction="up",
        horizon_trading_days=1,
        probability=0.65,
        expected_return_pct=0.5,
        strategy_version="v1.0",
        thesis="Constructive macro backdrop.",
        invalidation_condition="Yield jump > 10bps.",
    )

    dec = Decision(
        decision_id="D-20260803-001",
        forecast_id="F-20260803-001",
        decision_timestamp="2026-08-03 16:00 UTC",
        action="maintain",
        target_weight=1.0,
        previous_weight=1.0,
        execution_price=5500.0,
    )

    ev = Evaluation(
        forecast_id="F-20260803-001",
        decision_id="D-20260803-001",
        evaluation_date="2026-08-04",
        realized_return_pct=0.36,
        benchmark_return_pct=0.36,
        net_strategy_return_pct=0.36,
        direction_correct=True,
        brier_score=0.0841,
        max_drawdown_pct=0.0,
    )

    risk_fb = RiskOfficerFeedback(
        date="2026-08-03",
        forecast_id="F-20260803-001",
        decision_id="D-20260803-001",
        concentration_risk="S&P 500 at 100% weight.",
        volatility_correlation_risk="Correlation normal.",
        liquidity_execution_risk="High liquidity.",
        drawdown_stress_loss="Drawdown within 1% limit.",
        position_limits_stops="Stop-loss at 2%.",
        permission_decision="Approve",
        proposed_risk_adjustments="None.",
    )

    macro_fb = MacroSentinelFeedback(
        date="2026-08-03",
        forecast_id="F-20260803-001",
        supporting_evidence="Stable earnings and yield curve stabilization.",
        contradictory_evidence="Oil price uptick.",
        leading_indicators_to_monitor="Credit spreads.",
        regime_invalidation_conditions="Uninversion above 0 bps.",
        thesis_verdict="Confirm",
    )

    exec_summary = {
        "market_regime": "Risk-on",
        "primary_driver": "Earnings resilience and moderate inflation.",
        "portfolio_stance": "Maintain",
        "key_uncertainty": "Geopolitical risk.",
    }

    audit_info = {
        "data_sources": "Primary Treasury & Market Feeds",
        "model_version": "v1.0",
        "record_hash": "HASH123456789",
    }

    generator = DailyBriefGenerator(
        session_date="2026-08-03",
        generated_at="2026-08-03 16:00 UTC",
        data_status="Verified",
        forecast_horizon="1 day",
        exec_summary=exec_summary,
        market_observations=obs,
        regime=regime,
        forecasts=[fc],
        decisions=[dec],
        evaluations=[ev],
        risk_feedback=risk_fb,
        macro_feedback=macro_fb,
        audit_info=audit_info,
    )

    markdown_report = generator.generate_markdown()
    assert "# Adam Financial Daily Brief" in markdown_report
    assert "## 1. Executive Summary" in markdown_report
    assert "## 2. Market Data Snapshot" in markdown_report
    assert "## 3. Macro and Regime Evaluation" in markdown_report
    assert "## 4. Backtest and Simulated Decisions" in markdown_report
    assert "## 5. Forecast Register" in markdown_report
    assert "## 6. Risk Officer Feedback" in markdown_report
    assert "## 7. Macro Sentinel Feedback" in markdown_report
    assert "## 8. Follow-up and Accountability" in markdown_report
    assert "## 9. Audit Trail" in markdown_report

    audit_res = run_pre_publication_audit(
        market_observations=obs,
        regime=regime,
        forecasts=[fc],
        decisions=[dec],
        risk_feedback=risk_fb,
        macro_feedback=macro_fb,
        evaluations=[ev],
    )
    assert audit_res["all_passed"] is True
