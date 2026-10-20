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
    parser.add_argument("--json", type=str, default=None, help="Path to save JSON output")

    args = parser.parse_args()

    engine = ScrambledSobolQMCEngine(default_scramble=True, seed=42)

    logger.info("Executing Empirical Convergence & Variance Reduction Benchmark...")
    benchmark_results = engine.compare_mc_vs_qmc(
        n_simulations=args.simulations,
        dimension=args.dimensions,
        n_runs=20,
        seed=42,
    )

    logger.info("Executing Merton Structural Credit Risk Stress Simulation...")
    merton_results = engine.simulate_merton_credit_stress(
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

    logger.info("Executing Multi-Factor Macro Portfolio Stress Simulation...")
    exposures = [50.0, 30.0, 20.0]  # Tech, Energy, Financials ($M)
    volatilities = [0.28, 0.35, 0.22]
    correlations = [
        [1.0, 0.45, 0.60],
        [0.45, 1.0, 0.30],
        [0.60, 0.30, 1.0],
    ]
    macro_shocks = [-0.10, -0.15, -0.08]  # Sector return shocks under severe stress

    multi_factor_results = engine.simulate_multi_factor_stress(
        portfolio_exposure=exposures,
        asset_volatilities=volatilities,
        correlation_matrix=correlations,
        stress_shocks=macro_shocks,
        n_simulations=args.simulations,
        scramble=True,
        seed=42,
    )

    combined_output = {
        "benchmark": benchmark_results,
        "merton_credit_stress": merton_results,
        "multi_factor_stress": multi_factor_results,
    }

    print("\n" + "=" * 80)
    print("        SCRAMBLED SOBOL QUASI-MONTE CARLO (QMC) STRESS SIMULATION REPORT        ")
    print("=" * 80)
    print("\n1. CONVERGENCE & VARIANCE REDUCTION BENCHMARK (MC vs QMC)")
    print(f"   * Standard MC Variance:           {benchmark_results['empirical_variance_MC']:.6e}")
    print(f"   * Scrambled Sobol QMC Variance:   {benchmark_results['empirical_variance_scrambled_QMC']:.6e}")
    print(f"   * Variance Reduction Factor:      {benchmark_results['variance_reduction_vs_MC']:.2f}x Efficiency Gain")
    print(f"   * Standard MC RMSE:              {benchmark_results['rmse_MC']:.6f}")
    print(f"   * Scrambled Sobol QMC RMSE:      {benchmark_results['rmse_scrambled_QMC']:.6f}")

    print("\n2. MERTON STRUCTURAL CREDIT MODEL STRESS (V0=$100M, D=$80M, Vol=35%)")
    print(f"   * Probability of Default (PD):   {merton_results['probability_of_default'] * 100:.2f}%")
    print(f"   * QMC Standard Error (SE):       +/- {merton_results['qmc_standard_error_pd'] * 100:.4f}%")
    print(f"   * Standard MC Equivalent SE:     +/- {merton_results['standard_mc_equivalent_se'] * 100:.4f}%")
    print(f"   * Portfolio 95% VaR:              ${merton_results['VaR_95']:.2f}M")
    print(f"   * Portfolio 95% CVaR (Tail Expected Loss): ${merton_results['Expected_Shortfall_95']:.2f}M")

    print("\n3. MULTI-FACTOR MACRO PORTFOLIO STRESS")
    print(f"   * Expected Portfolio Loss:       ${multi_factor_results['portfolio_mean_loss']:.2f}M")
    print(f"   * 95% Value-at-Risk (VaR):        ${multi_factor_results['portfolio_VaR_95']:.2f}M")
    print(f"   * 99% Value-at-Risk (VaR):        ${multi_factor_results['portfolio_VaR_99']:.2f}M")
    print(f"   * 95% Tail Expected Loss (CVaR):  ${multi_factor_results['portfolio_CVaR_95']:.2f}M")
    print("   * Component Tail Risk Contributions:")
    sectors = ["Tech Sector ($50M)", "Energy Sector ($30M)", "Financials Sector ($20M)"]
    for sector, contrib in zip(sectors, multi_factor_results["component_risk_contributions"]):
        print(f"     - {sector}: ${contrib:.2f}M")
    print("=" * 80 + "\n")

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined_output, f, indent=2)
        logger.info(f"Detailed simulation results saved to {out_path}")


if __name__ == "__main__":
    main()
