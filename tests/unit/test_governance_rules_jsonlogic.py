"""adam_os governance: rules.jsonLogic loads and evaluates deterministically at the edges."""
import json
from pathlib import Path

import pytest

pytest.importorskip("structlog")
pytest.importorskip("json_logic")
from pydantic import ValidationError  # noqa: E402

from adam_os.contexts.governance.engine import DeterministicPolicyEngine, FinancialContext  # noqa: E402

RULES = json.loads((Path(__file__).resolve().parents[2] / "adam_os/contexts/governance/rules.jsonLogic").read_text())


@pytest.fixture(scope="module")
def engine():
    return DeterministicPolicyEngine(RULES)


def test_rules_file_shape():
    assert RULES
    for name, rule in RULES.items():
        assert {"logic", "threshold"} <= rule.keys(), name


@pytest.mark.parametrize("rule", sorted(RULES))
def test_pass_fail_and_exact_boundary(engine, rule):
    t = RULES[rule]["threshold"]
    assert engine.evaluate(rule, {"ltv": t - 0.01}).passed
    assert engine.evaluate(rule, {"ltv": t}).passed, "boundary equality must pass (<=)"
    res = engine.evaluate(rule, {"ltv": t + 0.01})
    assert res.is_breached and res.alert_triggered and res.threshold_value == t


@pytest.mark.parametrize("ctx", [{}, {"ltv": None}])
def test_missing_variable_fails_closed(engine, ctx):
    assert engine.evaluate("default_credit_risk", ctx).is_breached


def test_unknown_rule_raises(engine):
    with pytest.raises(ValueError):
        engine.evaluate("no_such_rule", {"ltv": 0.1})


def test_divide_by_zero_guards():
    with pytest.raises(ValidationError):
        FinancialContext(entity_id="x", asset_value_usd=0, total_debt_usd=1)
    ctx = FinancialContext(entity_id="x", asset_value_usd=100, total_debt_usd=35, net_operating_income=10, debt_service=0)
    assert ctx.dscr is None
    assert DeterministicPolicyEngine().evaluate("dscr_125", ctx).is_breached


def test_financial_context_ltv(engine):
    ctx = FinancialContext(entity_id="x", asset_value_usd=200, total_debt_usd=70)
    assert ctx.ltv == pytest.approx(0.35)
    assert engine.evaluate("softbank_arm_margin_loop", ctx).passed
