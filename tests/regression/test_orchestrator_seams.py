"""Seam tests for MetaOrchestrator bugs fixed in the routing / positions-bridge rewrite."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.models import SR11_7_BLOCKED_MODEL_ERROR
from core.engine import meta_orchestrator as mo
from core.engine.crisis_simulation_graph import simulate_direct_node
from core.engine.meta_orchestrator import MetaOrchestrator, OrchestratorConfig, Route
from core.engine.states import init_crisis_state


@pytest.fixture
def orch():
    with patch.object(mo, "LLMPlugin"):
        yield MetaOrchestrator(legacy_orchestrator=MagicMock())


@pytest.mark.parametrize("query, expected", [
    ("implement the importer", Route.CODE_GEN),
    ("buy 10 shares of AAPL", Route.FO_EXECUTION),
    ("buyer sentiment is weak", Route.LOW),        # \b boundary: 'buyer' is not 'buy'
    ("the pricing desk is closed", Route.LOW),      # 'pricing' is not 'price'
    ("greenhouse emissions", Route.LOW),            # 'greenhouse' is not 'green'
])
def test_routing_word_boundaries(orch, query, expected):
    assert orch._assess_complexity(query, {}) is expected


@pytest.mark.parametrize("query", ["what is going on today", "tell me about the US economy", ""])
def test_resolve_ticker_returns_none_when_unresolved(orch, query):
    assert orch.resolve_ticker(query, {}) is None


@pytest.mark.asyncio
async def test_unresolved_ticker_asks_for_clarification(orch):
    out = await orch._run_fo_market("show the price", {})
    assert out["status"] == "clarification_needed"


def test_positions_sum_to_one_and_aum(orch):
    allocation = {"SPY": 0.4, "TLT": 0.3, "GLD": 0.1, "HYG": 0.1, "LQD": 0.1}
    out = orch._positions_from_allocation({"aum": 1_000_000, "allocation": allocation})
    assert sum(p["weight"] for p in out["positions"]) == pytest.approx(1.0)
    assert sum(p["exposure"] for p in out["positions"]) == pytest.approx(1_000_000)
    assert all(p.get("_estimated") or p.get("source") for p in out["positions"])


def test_missing_metadata_symbol_is_skipped_not_defaulted(orch):
    with patch.object(mo.logger, "warning") as warn:
        out = orch._positions_from_allocation({"aum": 100, "allocation": {"SPY": 0.5, "ZZZZ": 0.5}})
    assert [p["name"] for p in out["positions"]] == ["SPY"]
    assert out["_skipped_symbols"] == ["ZZZZ"]
    assert "ZZZZ" in str(warn.call_args_list)


def test_crisis_engine_gets_per_asset_results(orch):
    universe = orch.config.default_universe
    portfolio = orch._positions_from_allocation({"aum": 1e8, "allocation": {s: 1 / len(universe) for s in universe}})
    state = init_crisis_state("market crash", portfolio)
    state["sector_shocks"] = {"Technology": -0.8}
    with patch("core.engine.crisis_simulation_graph.sector_engine.analyze_portfolio",
               wraps=__import__("core.engine.crisis_simulation_graph", fromlist=["x"]).sector_engine.analyze_portfolio) as spy:
        simulate_direct_node(state)
    assert [a["name"] for a in spy.call_args.args[0]] == universe


@pytest.mark.asyncio
async def test_crisis_flow_bridges_user_supplied_allocation(orch):
    orch._generic_graph_run = AsyncMock(return_value={"status": "ok"})
    await orch._run_crisis_flow("crisis", {"portfolio": {"aum": 10, "allocation": {"SPY": 1.0}}})
    sent = orch._generic_graph_run.await_args.args[1]["portfolio_data"]
    assert [p["name"] for p in sent["positions"]] == ["SPY"]


@pytest.mark.parametrize("model", ["gpt-6.1-astra", "GPT-6.1-ASTRA-unaligned"])
def test_blocked_model_rejected(model):
    with patch.object(mo, "LLMPlugin"), patch.object(MetaOrchestrator, "_resolve_config", staticmethod(lambda c: c)):
        with pytest.raises(SR11_7_BLOCKED_MODEL_ERROR):
            MetaOrchestrator(legacy_orchestrator=MagicMock(), config=OrchestratorConfig(llm_model=model))


def test_default_model_is_gemini_neon_without_routing_file():
    assert OrchestratorConfig().llm_model == "gemini-4-neon"
    cfg = MetaOrchestrator._resolve_config(OrchestratorConfig(routing_config_path="/nonexistent.yaml"))
    assert cfg.llm_model == "gemini-4-neon"


@pytest.mark.parametrize("query, expected", [
    ("Deep dive on Ford Motor", "Ford Motor"),       # sentence-initial capital is not the entity
    ("Covenant review for Boeing", "Boeing"),
    ("What is the US GDP outlook?", None),           # acronyms are not entities
    ("Exxon credit outlook", "Exxon"),               # sentence-initial fallback still works
    ("Exxon credit outlook in Europe", "Exxon"),     # trailing qualifier must not displace the company
    ("Summarize compliance risks for Boeing", "Boeing"),  # sentence-initial verb is not the target
    ("Analyze Covenant Logistics", "Covenant Logistics"),  # leading name words are kept
    ("Exxon credit outlook for Europe", "Exxon"),    # regions are qualifiers, never targets
    ("Analyze Boeing exposure to sanctions on Russia", "Boeing"),
    ("Assess the Middle East", None),
])
def test_resolve_entity_prefers_non_initial_proper_nouns(orch, query, expected):
    assert orch.resolve_entity(query, {}) == expected
