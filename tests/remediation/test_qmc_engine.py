from core.remediation.qmc_engine import QMCStatus, ValidatedQMCEngine

def test_qmc_simulation_execution() -> None:
    engine = ValidatedQMCEngine()
    res = engine.run_simulation(num_samples=1024, seed=42)
    assert res["status"] == QMCStatus.SIMULATION_COMPLETED.value
    assert res["num_samples"] == 1024
    assert res["mean_stressed_leverage"] > 0
    assert "NOT regulatory capital or ECL" in res["tail_statistic_note"]

def test_qmc_reproducibility() -> None:
    engine = ValidatedQMCEngine()
    assert engine.verify_reproducibility(num_samples=2048, seed=123) is True

def test_qmc_convergence_validation() -> None:
    engine = ValidatedQMCEngine()
    conv = engine.validate_convergence(sample_sizes=[1024, 2048, 4096, 8192], seed=42, max_relative_error=0.05)
    assert conv["converged"] is True
    assert conv["status"] == QMCStatus.CONVERGENCE_VALIDATED.value

def test_qmc_convergence_failure_handling() -> None:
    # Artificially set very tight threshold to test failure handling
    engine = ValidatedQMCEngine()
    conv = engine.validate_convergence(sample_sizes=[64, 128], seed=42, max_relative_error=1e-8)
    assert conv["converged"] is False
    assert conv["status"] == QMCStatus.CONVERGENCE_FAILED.value
