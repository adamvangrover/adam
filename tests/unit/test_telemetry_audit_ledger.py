import pytest
from scripts.telemetry_audit_ledger import TelemetryAuditLedger

def test_telemetry_audit_ledger():
    ledger = TelemetryAuditLedger("afos-run-20260923-9941a")
    ledger.record_c0_observation(
        entity="urn:sec:edgar:financial_statements:10K:Q3_2026",
        attributed_to="agent:xbrl_edgar_ingest_daemon",
        generated_at="2026-09-23T20:50:02.118Z",
        hash_lineage="sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        status="ATTESTED_VALID"
    )
    report = ledger.generate_report("2026-09-23T20:53:51.412Z")

    assert "ADAM OS RUNTIME EXECUTION TELEMETRY // W3C PROV-O AUDIT TRACE" in report
    assert "SESSION ID: afos-run-20260923-9941a" in report
    assert "[C0_CANONICAL_OBSERVATION]" in report
    assert "prov:entity          = urn:sec:edgar:financial_statements:10K:Q3_2026" in report
    assert "prov:wasAttributedTo = agent:xbrl_edgar_ingest_daemon" in report
    assert "status               = ATTESTED_VALID" in report

def test_telemetry_audit_ledger_qmc():
    ledger = TelemetryAuditLedger("test-run")
    ledger.record_qmc_simulation(
        backend="default.qubit.pennylane_v0.38", shots=65536, topology="StronglyEntanglingLayers",
        num_qubits=8, layers=4, state_prep="AmplitudeEmbedding(Obligor_Credit_Features, normalize=True)",
        pd_alpha=2.14, pd_beta=48.21, mean_pd=4.25, var_99=11.82, epistemic_var=0.0142, tau_ood=0.350,
        status="DISTRIBUTION_STABLE", audit_hash="sha256:7f01a9b244ce9102834fde01992aa91bc0293847551029348123485769102934"
    )
    report = ledger.generate_report()
    assert "[QUANTUM_MONTE_CARLO_SIMULATION_NODE (PennyLane/Qiskit Hybrid Core)]" in report
    assert "sigma^2_epistemic = 0.0142 <= tau_OOD (0.350)" in report
    assert "Beta(alpha=2.14, beta=48.21) -> Mean PD: 4.25%" in report
