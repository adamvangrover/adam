#!/usr/bin/env python3
"""
CLI Runner for Scrambled Sobol Quasi-Monte Carlo (QMC) Stress Simulations.

Usage:
    uv run python scripts/run_scrambled_sobol_qmc.py [OPTIONS]

Options:
    --simulations INT    Number of paths per QMC simulation batch (default: 1024)
    --dimensions INT     Number of factor dimensions for convergence testing (default: 4)
    --scramblings INT    Number of randomized scramblings for QMC SE (default: 10)
    --json PATH          Optional path to output detailed JSON results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.risk_engine.scrambled_sobol_qmc import ScrambledSobolQMCEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scrambled_sobol_qmc_runner")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Scrambled Sobol Quasi-Monte Carlo Stress Simulations and Benchmarks."
    )
    parser.add_argument("--simulations", type=int, default=1024, help="Number of simulation paths per batch")
    parser.add_argument("--dimensions", type=int, default=4, help="Factor space dimension")
    parser.add_argument("--scramblings", type=int, default=10, help="Number of randomized scramblings")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "merton", "multi-factor", "sobol-sensitivity", "covenant-headroom", "multi-horizon", "benchmark"],
        help="Simulation execution mode (default: all)",
    )
    parser.add_argument("--json", type=str, default=None, help="Path to save JSON output")
    parser.add_argument("--markdown", type=str, default=None, help="Path to save Markdown report")

    args = parser.parse_args()

    engine = ScrambledSobolQMCEngine(default_scramble=True, seed=42)
    combined_output = {}

    run_all = args.mode in ["all", "benchmark"]
    if run_all or args.mode == "benchmark":
        logger.info("Executing Empirical Convergence & Variance Reduction Benchmark...")
        combined_output["benchmark"] = engine.compare_mc_vs_qmc(
            n_simulations=args.simulations,
            dimension=args.dimensions,
            n_runs=20,
            seed=42,
        )

    if args.mode in ["all", "merton"]:
        logger.info("Executing Merton Structural Credit Risk Stress Simulation...")
        combined_output["merton_credit_stress"] = engine.simulate_merton_credit_stress(
            asset_value=100.0,
            debt_face_value=80.0,
            volatility=0.35,
            risk_free_rate=0.045,
            time_horizon=1.0,
            n_simulations=args.simulations,
            scramble=True,
            n_scramblings=args.scramblings,
            seed=42,
        )

    if args.mode in ["all", "multi-factor"]:
        logger.info("Executing Multi-Factor Macro Portfolio Stress Simulation...")
        exposures = [50.0, 30.0, 20.0]  # Tech, Energy, Financials ($M)
        volatilities = [0.28, 0.35, 0.22]
        correlations = [
            [1.0, 0.45, 0.60],
            [0.45, 1.0, 0.30],
            [0.60, 0.30, 1.0],
        ]
        macro_shocks = [-0.10, -0.15, -0.08]

        combined_output["multi_factor_stress"] = engine.simulate_multi_factor_stress(
            portfolio_exposure=exposures,
            asset_volatilities=volatilities,
            correlation_matrix=correlations,
            stress_shocks=macro_shocks,
            n_simulations=args.simulations,
            scramble=True,
            seed=42,
        )

    if args.mode in ["all", "sobol-sensitivity"]:
        logger.info("Executing Sobol Global Sensitivity Analysis (First-Order S_i & Total-Order ST_i)...")
        # Define model: Credit Default Indicator under EBITDA, Volatility, Debt, Rate shocks
        def credit_stress_model(x_arr):
            ebitda = 25.0 * np.exp(0.20 * (x_arr[:, 0] - 0.5))
            vol = 0.30 + 0.15 * (x_arr[:, 1] - 0.5)
            debt = 100.0 + 20.0 * (x_arr[:, 2] - 0.5)
            r = 0.04 + 0.02 * (x_arr[:, 3] - 0.5)
            d_2 = (np.log(100.0 / debt) + (r - 0.5 * vol**2)) / vol
            return (d_2 < 0.2).astype(np.float64)

        params = ["EBITDA Growth", "Asset Volatility", "Funded Debt Level", "Interest Rate"]
        combined_output["sobol_sensitivity"] = engine.compute_sobol_sensitivity_indices(
            model_func=credit_stress_model,
            dimension=4,
            n_samples=max(512, args.simulations),
            param_names=params,
            scramble=True,
            seed=42,
        )

    if args.mode in ["all", "covenant-headroom"]:
        logger.info("Executing Covenant & Financial Headroom Stress Simulation...")
        combined_output["covenant_headroom_stress"] = engine.simulate_covenant_headroom_stress(
            consolidated_ebitda=25.0,
            total_funded_debt=100.0,
            annual_interest_expense=8.0,
            debt_service_due=15.0,
            annual_capex=3.0,
            annual_tax=1.5,
            max_leverage_covenant=4.5,
            min_interest_coverage_covenant=3.0,
            min_dscr_covenant=1.25,
            ebitda_volatility=0.20,
            interest_rate_shock_std=0.015,
            n_simulations=args.simulations,
            scramble=True,
            seed=42,
        )

    if args.mode in ["all", "multi-horizon"]:
        logger.info("Executing Multi-Horizon Credit Term Structure Simulation (1Y, 3Y, 5Y)...")
        combined_output["multi_horizon_credit_stress"] = engine.simulate_multi_horizon_credit_stress(
            asset_value=100.0,
            debt_face_value=80.0,
            volatility=0.35,
            risk_free_rate=0.045,
            horizons=[1.0, 3.0, 5.0],
            n_simulations=args.simulations,
            scramble=True,
            seed=42,
        )

    print("\n" + "=" * 80)
    print("        SCRAMBLED SOBOL QUASI-MONTE CARLO (QMC) STRESS SIMULATION REPORT        ")
    print("=" * 80)

    if "benchmark" in combined_output:
        b_res = combined_output["benchmark"]
        print("\n1. CONVERGENCE & VARIANCE REDUCTION BENCHMARK (MC vs QMC)")
        print(f"   * Standard MC Variance:           {b_res['empirical_variance_MC']:.6e}")
        print(f"   * Scrambled Sobol QMC Variance:   {b_res['empirical_variance_scrambled_QMC']:.6e}")
        print(f"   * Variance Reduction Factor:      {b_res['variance_reduction_vs_MC']:.2f}x Efficiency Gain")
        print(f"   * Standard MC RMSE:              {b_res['rmse_MC']:.6f}")
        print(f"   * Scrambled Sobol QMC RMSE:      {b_res['rmse_scrambled_QMC']:.6f}")

    if "merton_credit_stress" in combined_output:
        m_res = combined_output["merton_credit_stress"]
        print("\n2. MERTON STRUCTURAL CREDIT MODEL STRESS (V0=$100M, D=$80M, Vol=35%)")
        print(f"   * Probability of Default (PD):   {m_res['probability_of_default'] * 100:.2f}%")
        print(f"   * QMC Standard Error (SE):       +/- {m_res['qmc_standard_error_pd'] * 100:.4f}%")
        print(f"   * Standard MC Equivalent SE:     +/- {m_res['standard_mc_equivalent_se'] * 100:.4f}%")
        print(f"   * Portfolio 95% VaR:              ${m_res['VaR_95']:.2f}M")
        print(f"   * Portfolio 95% CVaR (Tail Loss): ${m_res['Expected_Shortfall_95']:.2f}M")

    if "multi_factor_stress" in combined_output:
        mf_res = combined_output["multi_factor_stress"]
        print("\n3. MULTI-FACTOR MACRO PORTFOLIO STRESS")
        print(f"   * Expected Portfolio Loss:       ${mf_res['portfolio_mean_loss']:.2f}M")
        print(f"   * 95% Value-at-Risk (VaR):        ${mf_res['portfolio_VaR_95']:.2f}M")
        print(f"   * 99% Value-at-Risk (VaR):        ${mf_res['portfolio_VaR_99']:.2f}M")
        print(f"   * 95% Tail Expected Loss (CVaR):  ${mf_res['portfolio_CVaR_95']:.2f}M")
        print("   * Component Tail Risk Contributions:")
        sectors = ["Tech Sector ($50M)", "Energy Sector ($30M)", "Financials Sector ($20M)"]
        for sector, contrib in zip(sectors, mf_res["component_risk_contributions"]):
            print(f"     - {sector}: ${contrib:.2f}M")

    if "sobol_sensitivity" in combined_output:
        s_res = combined_output["sobol_sensitivity"]
        print("\n4. SOBOL GLOBAL SENSITIVITY ANALYSIS (SALTELIL SAMPLING)")
        print(f"   * Model Variance: {s_res['model_variance']:.6e}")
        print("   * Parameter Rankings (First-Order S_i / Total-Order ST_i):")
        for rank, p_name in enumerate(s_res["parameter_rankings"], 1):
            s_i = s_res["first_order_indices"][p_name]
            st_i = s_res["total_order_indices"][p_name]
            print(f"     #{rank} {p_name:<20}: S_i = {s_i:.4f} | ST_i = {st_i:.4f}")

    if "covenant_headroom_stress" in combined_output:
        c_res = combined_output["covenant_headroom_stress"]
        print("\n5. COVENANT & FINANCIAL HEADROOM STRESS")
        b_p = c_res["breach_probabilities"]
        print(f"   * Joint Covenant Breach Prob:     {b_p['joint_covenant_breach_probability'] * 100:.2f}%")
        print(f"   * Leverage Breach Prob:           {b_p['leverage_breach_probability'] * 100:.2f}%")
        print(f"   * Interest Coverage Breach Prob:  {b_p['interest_coverage_breach_probability'] * 100:.2f}%")
        print(f"   * DSCR Breach Prob:               {b_p['dscr_breach_probability'] * 100:.2f}%")
        print(f"   * Tail 95% Stressed Leverage:     {c_res['tail_risk']['p95_stressed_leverage']:.2f}x")

    if "multi_horizon_credit_stress" in combined_output:
        mh_res = combined_output["multi_horizon_credit_stress"]
        print("\n6. MULTI-HORIZON CREDIT DEFAULT TERM STRUCTURE")
        cum_pd = mh_res["cumulative_pd_term_structure"]
        marg_pd = mh_res["marginal_pd_term_structure"]
        for horizon in cum_pd:
            print(f"   * Maturity {horizon:<3}: Cumulative PD = {cum_pd[horizon] * 100:6.2f}% | Marginal PD = {marg_pd[horizon] * 100:6.2f}%")

    print("=" * 80 + "\n")

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined_output, f, indent=2)
        logger.info(f"Detailed simulation results saved to {out_path}")

    if args.markdown:
        out_md = Path(args.markdown)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        md_lines = [
            "# Scrambled Sobol Quasi-Monte Carlo (QMC) Stress Simulation Report",
            "",
            "## Summary of Results",
            "",
            "```json",
            json.dumps(combined_output, indent=2),
            "```",
        ]
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        logger.info(f"Markdown report saved to {out_md}")


if __name__ == "__main__":
    main()
