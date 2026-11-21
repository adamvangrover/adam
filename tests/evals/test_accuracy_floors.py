import pytest

from tests.evals.run_accuracy import evaluate


@pytest.mark.parametrize("dataset", ["query_route", "query_entity"])
def test_eval_accuracy_meets_floor(dataset):
    acc, floor, misses = evaluate(dataset)
    assert acc >= floor, misses
