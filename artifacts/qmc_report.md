# Scrambled Sobol Quasi-Monte Carlo (QMC) Stress Simulation Report

## Summary of Results

```json
{
  "benchmark": {
    "empirical_mean_MC": 1.0327244434417224,
    "empirical_mean_unscrambled_QMC": 5856.26616004448,
    "empirical_mean_scrambled_QMC": 1.005823248488061,
    "empirical_variance_MC": 0.11713857825952698,
    "empirical_variance_unscrambled_QMC": 8.707164342663449e-25,
    "empirical_variance_scrambled_QMC": 0.03021183406249972,
    "variance_reduction_vs_MC": 3.877241547706123,
    "rmse_MC": 0.33519030198548566,
    "rmse_scrambled_QMC": 0.16951446127787576,
    "n_simulations": 1024,
    "dimension": 4,
    "n_runs": 20
  },
  "merton_credit_stress": {
    "probability_of_default": 0.27734375,
    "expected_asset_value": 104.59214713481545,
    "qmc_standard_error_pd": 0.0,
    "standard_mc_equivalent_se": 0.0044241016294971,
    "variance_reduction_factor": 1.9572675228118893e+19,
    "VaR_95": 44.67891780505958,
    "VaR_99": 56.40096506785849,
    "Expected_Shortfall_95": 51.814461161365145,
    "simulation_parameters": {
      "n_simulations_per_batch": 1024,
      "n_scramblings": 10,
      "total_paths": 10240,
      "scramble_enabled": true
    }
  },
  "multi_factor_stress": {
    "portfolio_mean_loss": 11.094096308460285,
    "portfolio_VaR_95": 49.6030405150904,
    "portfolio_VaR_99": 66.01432657644123,
    "portfolio_CVaR_95": 59.72192906827398,
    "component_risk_contributions": [
      31.042511257519713,
      21.362521383119613,
      7.316896427634644
    ],
    "n_simulations": 1024,
    "dimension": 3,
    "scramble": true
  },
  "sobol_sensitivity": {
    "first_order_indices": {
      "EBITDA Growth": 0.046437569988801775,
      "Asset Volatility": 0.10827267637178051,
      "Funded Debt Level": 0.6257348824188129,
      "Interest Rate": 0.03667413213885773
    },
    "total_order_indices": {
      "EBITDA Growth": 0.046437569988801775,
      "Asset Volatility": 0.354738241881299,
      "Funded Debt Level": 0.917763157894737,
      "Interest Rate": 0.0748530235162374
    },
    "parameter_rankings": [
      "Funded Debt Level",
      "Asset Volatility",
      "Interest Rate",
      "EBITDA Growth"
    ],
    "model_variance": 0.15003360215053763,
    "sample_size": 1024,
    "scramble_enabled": true
  },
  "covenant_headroom_stress": {
    "baseline_metrics": {
      "baseline_leverage": 4.0,
      "baseline_interest_coverage": 3.125,
      "baseline_dscr": 1.3667
    },
    "covenant_thresholds": {
      "max_leverage": 4.5,
      "min_interest_coverage": 3.0,
      "min_dscr": 1.25
    },
    "breach_probabilities": {
      "leverage_breach_probability": 0.3125,
      "interest_coverage_breach_probability": 0.458984375,
      "dscr_breach_probability": 0.3984375,
      "joint_covenant_breach_probability": 0.458984375
    },
    "mean_headroom": {
      "mean_leverage_headroom": 0.3371188200558599,
      "mean_interest_coverage_headroom": 0.12557710269300706,
      "mean_dscr_headroom": 0.11669108866867994
    },
    "tail_risk": {
      "p95_stressed_leverage": 5.665328751168956,
      "p99_stressed_leverage": 6.473103893462756,
      "CVaR_95_leverage": 6.165545044170094
    },
    "simulation_status": "DETERMINISTIC_QMC_CONVERGED",
    "n_simulations": 1024
  },
  "multi_horizon_credit_stress": {
    "horizons": [
      1.0,
      3.0,
      5.0
    ],
    "cumulative_pd_term_structure": {
      "1Y": 0.27734375,
      "3Y": 0.3916015625,
      "5Y": 0.4287109375
    },
    "marginal_pd_term_structure": {
      "1Y": 0.27734375,
      "3Y": 0.1142578125,
      "5Y": 0.037109375
    },
    "expected_asset_value_term_structure": {
      "1Y": 104.62794765887729,
      "3Y": 114.42711798648011,
      "5Y": 125.09642639960114
    },
    "VaR_95_term_structure": {
      "1Y": 44.54206121944088,
      "3Y": 65.31839537836574,
      "5Y": 73.89472267088705
    },
    "CVaR_95_term_structure": {
      "1Y": 51.76239206912464,
      "3Y": 72.16433048730975,
      "5Y": 80.66008114279786
    },
    "simulation_parameters": {
      "n_simulations": 1024,
      "dimension": 3,
      "scramble_enabled": true
    }
  }
}
```