import json
import logging
from datetime import datetime, timezone, timedelta
from src.core.epistemic_calculus import EpistemicCalculus
from src.pdil.gate_12 import Gate12
from src.schemas.epistemic_types import AuthorityBoundaryRecord, TemporalBounds, EpistemicStatus

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_eval():
    logger.info("Starting ADAM Epistemic State Hierarchy Evaluation")

    calc = EpistemicCalculus(e_min=0.5, tau_ood=0.8, tau_w=0.3)
    gate_12 = Gate12()

    base_time = datetime(2026, 9, 22, 20, 0, tzinfo=timezone.utc)

    eval_cases = [
        {
            "name": "Case 1: Valid Supportive Hypothesis",
            "evidence_score": 0.85,
            "ood_metric": 0.1,
            "wasserstein_distance": 0.1,
            "resolution_available": True,
            "critical_path": False,
            "authority": "NONE",
            "temporal": TemporalBounds(
                event_time_te=base_time,
                knowledge_time_tk=base_time + timedelta(minutes=1),
                decision_time_td=base_time + timedelta(minutes=5),
            ),
            "expected_state": EpistemicStatus.SUPPORTED,
            "expected_g12": True
        },
        {
            "name": "Case 2: Authority Violation",
            "evidence_score": 0.90,
            "ood_metric": 0.1,
            "wasserstein_distance": 0.1,
            "resolution_available": True,
            "critical_path": False,
            "authority": "COMMIT", # Violation
            "temporal": TemporalBounds(
                event_time_te=base_time,
                knowledge_time_tk=base_time + timedelta(minutes=1),
                decision_time_td=base_time + timedelta(minutes=5),
            ),
            "expected_state": EpistemicStatus.SUPPORTED,
            "expected_g12": False
        },
        {
            "name": "Case 3: Temporal Integrity Violation (tk < te)",
            "evidence_score": 0.85,
            "ood_metric": 0.1,
            "wasserstein_distance": 0.1,
            "resolution_available": True,
            "critical_path": False,
            "authority": "NONE",
            "temporal": TemporalBounds(
                event_time_te=base_time + timedelta(minutes=10),
                knowledge_time_tk=base_time,
            ),
            "expected_state": EpistemicStatus.SUPPORTED,
            "expected_g12": False
        },
        {
            "name": "Case 4: Out of Distribution Data",
            "evidence_score": 0.85,
            "ood_metric": 0.95, # > tau_ood (0.8)
            "wasserstein_distance": 0.1,
            "resolution_available": True,
            "critical_path": False,
            "authority": "NONE",
            "temporal": TemporalBounds(
                event_time_te=base_time,
                knowledge_time_tk=base_time + timedelta(minutes=1),
            ),
            "expected_state": EpistemicStatus.OOD,
            "expected_g12": True # Note: G12 handles admission bounds, not state logic, but downstream C5 would reject.
        },
        {
            "name": "Case 5: Conflicted but Unresolved",
            "evidence_score": 0.85,
            "ood_metric": 0.1,
            "wasserstein_distance": 0.45, # > tau_w (0.3)
            "resolution_available": False,
            "critical_path": False,
            "authority": "NONE",
            "temporal": TemporalBounds(
                event_time_te=base_time,
                knowledge_time_tk=base_time + timedelta(minutes=1),
            ),
            "expected_state": EpistemicStatus.UNKNOWN,
            "expected_g12": True
        }
    ]

    passed_cases = 0

    for idx, case in enumerate(eval_cases):
        logger.info(f"--- Evaluating {case['name']} ---")

        # Step 1: Evaluate Epistemic Status
        status = calc.evaluate_state(
            evidence_score=case["evidence_score"],
            ood_metric=case["ood_metric"],
            wasserstein_distance=case["wasserstein_distance"],
            resolution_policy_available=case["resolution_available"],
            critical_path=case["critical_path"]
        )
        logger.info(f"Computed Status: {status.value} (Expected: {case['expected_state'].value})")

        # Step 2: Evaluate Gate 12
        record = AuthorityBoundaryRecord(
            authority=case["authority"],
            subject={"eval_case": idx},
            source_lineage={},
            epistemic_input={"status": status.value},
            model_context={},
            temporal_bounds=case["temporal"],
            provenance={},
            admission_criteria={"g12_h_schema_valid": True}
        )
        g12_admission = gate_12.evaluate_admission(record)
        logger.info(f"Gate 12 Admission: {g12_admission} (Expected: {case['expected_g12']})")

        if status == case["expected_state"] and g12_admission == case["expected_g12"]:
            logger.info("Result: PASS")
            passed_cases += 1
        else:
            logger.error("Result: FAIL")

    logger.info(f"--- Evaluation Complete: {passed_cases}/{len(eval_cases)} Passed ---")

    if passed_cases != len(eval_cases):
        raise RuntimeError("Evaluation suite failed one or more cases.")

if __name__ == "__main__":
    run_eval()
