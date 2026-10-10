import pytest
from evals.graders.multi_state_reasoning_evaluator import (
    AggregatorStateMachineEvaluator,
    BaseStateMachineEvaluator,
    ConfidenceCalibrationEvaluator,
    MultiStateReasoningEvaluator,
    ResearchReasoningRubrics,
)


def test_base_state_trace_fidelity():
    evaluator = BaseStateMachineEvaluator()

    # Valid complete execution log
    valid_logs = [
        {"step": "init"},
        {"step": "retrieve"},
        {"step": "reason"},
        {"step": "validate"},
        {"step": "synthesize"},
        {"step": "complete"},
    ]
    res_valid = evaluator.eval_state_trace_fidelity(valid_logs)
    assert res_valid["fidelity_score"] > 0.8
    assert len(res_valid["skipped_steps"]) == 0
    assert not res_valid["loop_detected"]

    # Incomplete log with skipped steps and loop
    flawed_logs = [
        {"step": "init"},
        {"step": "retrieve"},
        {"step": "retrieve"},
        {"step": "retrieve"},
        {"step": "retrieve"},
        {"step": "complete"},
    ]
    res_flawed = evaluator.eval_state_trace_fidelity(flawed_logs)
    assert res_flawed["fidelity_score"] < res_valid["fidelity_score"]
    assert "reason" in res_flawed["skipped_steps"]
    assert res_flawed["loop_detected"]


def test_adversarial_resiliency():
    evaluator = BaseStateMachineEvaluator()

    baseline_trace = [
        {"step": "init"},
        {"step": "retrieve"},
        {"step": "reason"},
        {"step": "validate"},
        {"step": "complete"},
    ]

    # Adversarial trace with infinite loop
    looping_trace = [{"step": "retrieve"} for _ in range(20)]
    res_loop = evaluator.eval_adversarial_resiliency(baseline_trace, looping_trace)
    assert res_loop["infinite_loop_detected"]
    assert res_loop["resiliency_score"] <= 0.5

    # Adversarial trace with premature exit
    premature_trace = [{"step": "init"}, {"step": "error", "status": "failed"}]
    res_exit = evaluator.eval_adversarial_resiliency(baseline_trace, premature_trace)
    assert res_exit["premature_exit_detected"]
    assert res_exit["resiliency_score"] <= 0.6


def test_fact_insertion_rate():
    evaluator = AggregatorStateMachineEvaluator()

    base_outputs = [
        "Company Alpha reported revenue of $500 million in Q3 2025.",
        "Operating margin reached 22% due to cost cutting initiatives.",
    ]

    # Grounded summary
    grounded_summary = (
        "Company Alpha reported $500 million in revenue for Q3 2025. "
        "Operating margins improved to 22% supported by cost reduction."
    )
    res_grounded = evaluator.eval_fact_insertion_rate(grounded_summary, base_outputs)
    assert res_grounded["insertion_rate"] < 0.5
    assert res_grounded["fact_fidelity_score"] > 0.5

    # Summary with injected hallucinated facts
    hallucinated_summary = (
        "Company Alpha acquired Beta Corp for $2 billion in equity. "
        "The CEO resigned amid SEC investigation and stock crashed."
    )
    res_injected = evaluator.eval_fact_insertion_rate(hallucinated_summary, base_outputs)
    assert res_injected["insertion_rate"] > 0.5
    assert len(res_injected["injected_facts"]) > 0


def test_conflict_resolution():
    evaluator = AggregatorStateMachineEvaluator()

    conflicting_inputs = [
        {"topic": "Debt Level", "fact_a": "$100M total debt", "fact_b": "$400M total debt"}
    ]

    # Summary acknowledging conflict
    summary_with_handling = (
        "There is a discrepancy regarding debt level: model A estimates $100M total debt "
        "whereas auditor filings show $400M total debt, which requires reconciliation."
    )
    res_handled = evaluator.eval_conflict_resolution(summary_with_handling, conflicting_inputs)
    assert res_handled["conflict_resolution_score"] == 1.0
    assert res_handled["conflicts_handled"] == 1

    # Summary ignoring conflict
    summary_ignored = "The company is performing well with strong cash reserves."
    res_ignored = evaluator.eval_conflict_resolution(summary_ignored, conflicting_inputs)
    assert res_ignored["conflict_resolution_score"] == 0.0
    assert res_ignored["conflicts_ignored"] == 1


def test_premise_monotonicity():
    rubrics = ResearchReasoningRubrics()

    # Monotonic reasoning steps
    monotonic_steps = [
        "Revenue increased 15% year over year.",
        "Margin expanded by 200 basis points due to operational efficiency.",
        "Free cash flow growth remains strong and bullish.",
    ]
    conclusion = "The company presents a bullish investment opportunity."
    res_mono = rubrics.eval_premise_monotonicity(monotonic_steps, conclusion)
    assert res_mono["monotonicity_score"] == 1.0
    assert len(res_mono["contradictions_found"]) == 0

    # Contradictory steps
    contradictory_steps = [
        "Free cash flow growth remains strong and bullish.",
        "However, debt default risks mean the company is insolvent and bearish.",
    ]
    res_contra = rubrics.eval_premise_monotonicity(contradictory_steps, "The overall outlook is bullish.")
    assert res_contra["monotonicity_score"] < 1.0
    assert len(res_contra["contradictions_found"]) > 0


def test_evidence_grounding():
    rubrics = ResearchReasoningRubrics()

    cited_output = (
        "According to SEC filing 10-K, revenue reached $1.2B [Source: 10-K Exhibit 99.1]. "
        "Debt ratio is 1.5x as reported in audited financial statements [1]."
    )
    res_cited = rubrics.eval_evidence_grounding(cited_output)
    assert res_cited["evidence_score"] > 0.8
    assert res_cited["citation_count"] >= 3

    uncited_output = "Company is doing great and profits will double next quarter."
    res_uncited = rubrics.eval_evidence_grounding(uncited_output)
    assert res_uncited["evidence_score"] == 0.0


def test_gap_identification():
    rubrics = ResearchReasoningRubrics()

    text_with_gaps = (
        "The current leverage profile is stable. However, we were unable to verify off-balance sheet liabilities. "
        "Key knowledge gap identified regarding international subsidiary tax exposure."
    )
    res_gaps = rubrics.eval_gap_identification(text_with_gaps)
    assert res_gaps["gap_identification_score"] >= 0.7
    assert len(res_gaps["gaps_found"]) > 0

    text_no_gaps = "Everything is completely known and 100% verified with zero doubt."
    res_no_gaps = rubrics.eval_gap_identification(text_no_gaps)
    assert res_no_gaps["gap_identification_score"] == 0.0


def test_confidence_calibration_ece():
    evaluator = ConfidenceCalibrationEvaluator()

    # Well-calibrated predictions
    accuracy_scores = [0.90, 0.85, 0.70, 0.60, 0.20]
    confidence_scores = [0.92, 0.80, 0.75, 0.55, 0.25]
    res_calibrated = evaluator.eval_confidence_calibration(accuracy_scores, confidence_scores)
    assert res_calibrated["ece"] < 0.15
    assert res_calibrated["adversarial_trap_count"] == 0
    assert res_calibrated["calibration_score"] > 0.8

    # Adversarial trap scenario: High confidence on wrong answers
    adv_accuracy = [0.10, 0.20, 0.15, 0.10]
    adv_confidence = [0.95, 0.90, 0.88, 0.92]
    res_trap = evaluator.eval_confidence_calibration(adv_accuracy, adv_confidence, is_adversarial=True)
    assert res_trap["adversarial_trap_count"] == 4
    assert res_trap["calibration_score"] < 0.3


def test_multi_state_reasoning_evaluator_composite():
    evaluator = MultiStateReasoningEvaluator()

    base_logs = [{"step": s} for s in ["init", "retrieve", "reason", "validate", "synthesize", "complete"]]
    adv_logs = [{"step": s} for s in ["init", "retrieve", "reason", "validate", "complete"]]

    summary = (
        "According to 10-K filing, Q3 revenue was $500M [Source: SEC 10-K]. "
        "There is a discrepancy in reported cash reserves, which remains uncertain."
    )
    base_outputs = ["Q3 revenue was $500M."]
    conflicting_inputs = [{"topic": "cash reserves", "fact_a": "$50M", "fact_b": "$100M"}]
    reasoning_steps = ["Q3 revenue grew 10%.", "Operational margins expanded."]
    conclusion = "Financial position is stable."

    accuracy_scores = [0.90, 0.85]
    confidence_scores = [0.90, 0.80]

    report = evaluator.evaluate_multi_state_system(
        base_state_logs=base_logs,
        adversarial_state_logs=adv_logs,
        aggregator_summary=summary,
        base_state_outputs=base_outputs,
        conflicting_inputs=conflicting_inputs,
        reasoning_steps=reasoning_steps,
        final_conclusion=conclusion,
        accuracy_scores=accuracy_scores,
        confidence_scores=confidence_scores,
    )

    assert "composite_score" in report
    assert report["composite_score"] > 0.7
    assert "axis_1_base_state_machine" in report
    assert "axis_2_aggregator_state_machine" in report
    assert "axis_3_research_reasoning" in report
    assert "axis_4_confidence_calibration" in report
