import pytest
from scripts.epistemic_evaluator import evaluate_entity, EntityEvaluation, EpistemicState

def test_successful_evaluation():
    entity = EntityEvaluation(
        entity_id="req-1",
        t_e=10, t_k=20, t_d=30, t_x=40,
        authority="NONE",
        divergence=0.1, tau_w=0.5,
        provo_traces_intact=True
    )
    res = evaluate_entity(entity)
    assert res.passed
    assert res.state == EpistemicState.SUPPORTED

def test_authority_violation():
    entity = EntityEvaluation(
        entity_id="req-2",
        t_e=10, t_k=20, t_d=30, t_x=40,
        authority="ALLOCATE",
        divergence=0.1, tau_w=0.5,
        provo_traces_intact=True
    )
    res = evaluate_entity(entity)
    assert not res.passed
    assert "Authority Violation" in res.reason

def test_temporal_violation():
    entity = EntityEvaluation(
        entity_id="req-3",
        t_e=50, t_k=20, t_d=30, t_x=40,
        authority="NONE",
        divergence=0.1, tau_w=0.5,
        provo_traces_intact=True
    )
    res = evaluate_entity(entity)
    assert not res.passed
    assert "Temporal Check" in res.reason

def test_divergence_exceeded():
    entity = EntityEvaluation(
        entity_id="req-4",
        t_e=10, t_k=20, t_d=30, t_x=40,
        authority="NONE",
        divergence=0.8, tau_w=0.5,
        provo_traces_intact=True
    )
    res = evaluate_entity(entity)
    assert not res.passed
    assert res.state == EpistemicState.CONFLICTED

def test_provo_traces_missing():
    entity = EntityEvaluation(
        entity_id="req-5",
        t_e=10, t_k=20, t_d=30, t_x=40,
        authority="NONE",
        divergence=0.1, tau_w=0.5,
        provo_traces_intact=False
    )
    res = evaluate_entity(entity)
    assert not res.passed
    assert "Provenance" in res.reason
