import datetime
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.config.models import SR11_7_BLOCKED_MODEL_ERROR
from core.engine import meta_orchestrator as mo
from core.engine.meta_orchestrator import MetaOrchestrator, OrchestratorConfig, Route


@pytest.fixture
def orchestrator():
    with patch.object(mo, "LLMPlugin"):
        yield MetaOrchestrator(legacy_orchestrator=MagicMock())


@pytest.mark.parametrize(
    "query, forbidden",
    [
        ("monitor the situation", Route.SURVEILLANCE),
        ("watch for an alert in the report", Route.SURVEILLANCE),
        ("run a simulation", Route.RED_TEAM),
        ("build a scenario plan", Route.RED_TEAM),
        ("check multiple goals in parallel", Route.SWARM),
    ],
)
def test_ambiguous_words_do_not_trigger_specialized_routes(orchestrator, query, forbidden):
    route = orchestrator._assess_complexity(query, {})
    assert route != forbidden
    assert route is Route.LOW


@pytest.mark.parametrize(
    "query, expected",
    [
        ("run a red team exercise on Acme", Route.RED_TEAM),
        ("perform a deep dive on Apple", Route.DEEP_DIVE),
        ("surveillance of distressed credits", Route.SURVEILLANCE),
        ("stress test the book", Route.CRISIS),
        ("buy 10 shares of AAPL", Route.FO_EXECUTION),
    ],
)
def test_specialized_routes_still_match(orchestrator, query, expected):
    assert orchestrator._assess_complexity(query, {}) is expected


def test_word_boundaries_reject_substrings(orchestrator):
    # "attacker"/"shocking"/"buyer" must not match "attack"/"shock"/"buy".
    assert orchestrator._assess_complexity("the attacker was shocking to the buyer", {}) is Route.LOW


def test_force_route_override(orchestrator):
    assert orchestrator._assess_complexity("run a red team exercise", {"force_route": "esg"}) is Route.ESG
    assert orchestrator._assess_complexity("run a red team exercise", {"force_route": "bogus"}) is Route.RED_TEAM


def test_resolve_ticker_returns_none_when_unresolved(orchestrator):
    assert orchestrator.resolve_ticker("what is the price today", {}) is None
    assert orchestrator.resolve_entity("analyze the situation", {}) is None


def test_resolve_ticker_precedence(orchestrator):
    assert orchestrator.resolve_ticker("price of $msft", {"ticker": "nvda"}) == "NVDA"
    assert orchestrator.resolve_ticker("price of Apple stock", {}) == "AAPL"
    assert orchestrator.resolve_ticker("price of $msft", {}) == "MSFT"
    assert orchestrator.resolve_ticker("ESG view on TSLA", {}) == "TSLA"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler",
    [
        "_run_deep_dive_flow",
        "_run_fo_market",
        "_run_fo_execution",
        "_run_adaptive_flow",
        "_run_red_team_flow",
        "_run_esg_flow",
        "_run_compliance_flow",
    ],
)
async def test_entity_flows_request_clarification(orchestrator, handler):
    result = await getattr(orchestrator, handler)("what is going on", {})
    assert result == {"error": "no_entity_resolved", "status": "clarification_needed"}


@pytest.mark.asyncio
async def test_fo_execution_requires_quantity(orchestrator):
    orchestrator.mcp_registry = MagicMock()
    result = await orchestrator._run_fo_execution("buy $AAPL", {})
    assert result["status"] == "clarification_needed"
    orchestrator.mcp_registry.invoke.assert_not_called()


def test_all_handlers_share_uniform_signature(orchestrator):
    assert set(orchestrator._route_registry) == set(Route)
    for route, handler in orchestrator._route_registry.items():
        assert inspect.iscoroutinefunction(handler), route
        assert list(inspect.signature(handler).parameters) == ["query", "context"], route


@pytest.mark.asyncio
@pytest.mark.parametrize("route", list(Route))
async def test_route_request_dispatches_every_route_through_registry(orchestrator, route):
    handler = AsyncMock(return_value={"status": "ok"})
    orchestrator._route_registry[route] = handler
    orchestrator._reflect_on_result = AsyncMock(side_effect=lambda result, query: result)

    result = await orchestrator.route_request("anything", {"force_route": route.value, "system_prompt": "x"})

    assert result == {"status": "ok"}
    handler.assert_awaited_once()
    assert handler.await_args.args[0] == "anything"
    assert handler.await_args.args[1]["force_route"] == route.value


@pytest.mark.asyncio
async def test_handler_type_error_is_not_retried(orchestrator):
    handler = AsyncMock(side_effect=TypeError("boom"))
    orchestrator._route_registry[Route.ESG] = handler
    result = await orchestrator.route_request("x", {"force_route": "ESG", "system_prompt": "x"})
    assert result["status"] == "Routing Failed"
    assert handler.await_count == 1


def test_routing_yaml_overrides_default_model():
    with patch.object(mo, "LLMPlugin") as plugin:
        orch = MetaOrchestrator(legacy_orchestrator=MagicMock())
    assert orch.config.llm_model != "gemini-4-neon"
    assert plugin.call_args.kwargs["config"]["gemini_model_name"] == orch.config.llm_model


def test_missing_routing_yaml_keeps_config_model(tmp_path):
    cfg = OrchestratorConfig(routing_config_path=str(tmp_path / "missing.yaml"))
    with patch.object(mo, "LLMPlugin"):
        orch = MetaOrchestrator(legacy_orchestrator=MagicMock(), config=cfg)
    assert orch.config.llm_model == "gemini-4-neon"


@pytest.mark.parametrize("model", ["gpt-6.1-astra", "GPT-6.1-ASTRA-UNALIGNED", "some-astra-variant"])
def test_blocked_model_rejected_before_llm_plugin(tmp_path, model):
    cfg = OrchestratorConfig(llm_model=model, routing_config_path=str(tmp_path / "missing.yaml"))
    with patch.object(mo, "LLMPlugin") as plugin:
        with pytest.raises(SR11_7_BLOCKED_MODEL_ERROR):
            MetaOrchestrator(legacy_orchestrator=MagicMock(), config=cfg)
    plugin.assert_not_called()


def _price_rows(symbol, start_date, end_date):
    rng = np.random.default_rng(abs(hash(symbol)) % 2**32)
    start = datetime.date.fromisoformat(start_date)
    price, rows = 100.0, []
    for i in range(120):
        price *= 1 + rng.normal(0.0005, 0.01)
        rows.append({"date": (start + datetime.timedelta(days=i)).isoformat(), "close": price})
    return rows


@pytest.mark.asyncio
async def test_construct_default_portfolio_runs_optimizer(orchestrator):
    with patch.object(mo.MarketDataService, "get_historical_data", side_effect=_price_rows):
        portfolio = await orchestrator._construct_default_portfolio({})

    universe = orchestrator.config.default_universe
    assert portfolio["_constructed"] is True
    assert "_estimated" not in portfolio
    assert portfolio["method"] == "Mean-Variance (Max Sharpe)"
    assert portfolio["aum"] == orchestrator.config.default_portfolio_aum
    assert set(portfolio["allocation"]) == set(universe)
    assert sum(portfolio["allocation"].values()) == pytest.approx(1.0, abs=1e-3)
    assert {"expected_return", "volatility", "sharpe_ratio"} <= set(portfolio["metrics"])


@pytest.mark.asyncio
async def test_construct_default_portfolio_equal_weight_fallback(orchestrator):
    with patch.object(mo.MarketDataService, "get_historical_data", return_value=[]):
        portfolio = await orchestrator._construct_default_portfolio({})

    assert portfolio["_estimated"] is True
    assert portfolio["method"] == "equal_weight"
    assert set(portfolio["allocation"].values()) == {0.2}


@pytest.mark.asyncio
async def test_crisis_flow_uses_context_portfolio(orchestrator):
    orchestrator._construct_default_portfolio = AsyncMock()
    orchestrator._generic_graph_run = AsyncMock(return_value={"status": "ok"})
    portfolio = {"id": "Mine", "aum": 5}

    await orchestrator._run_crisis_flow("crisis", {"portfolio": portfolio})

    orchestrator._construct_default_portfolio.assert_not_awaited()
    assert orchestrator._generic_graph_run.await_args.args[1]["portfolio_data"] == portfolio
