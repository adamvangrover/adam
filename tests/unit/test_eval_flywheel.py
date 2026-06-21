import pytest
from adam_credit_eval_flywheel import Flywheel

def test_evaluate_state_transition():
    f = Flywheel()
    assert f.evaluate_state_transition(0, 1) == 1.0
    assert f.evaluate_state_transition(1, 0) == 0.0

def test_evaluate_information_gain():
    f = Flywheel()
    assert f.evaluate_information_gain({"new_facts_extracted": 5, "tool_calls_used": 2}) == 2.5
    assert f.evaluate_information_gain({"new_facts_extracted": 0, "tool_calls_used": 0}) == 0.0
