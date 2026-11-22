#!/usr/bin/env python3
"""Golden-path runtime smoke: System 1, System 2, Nexus and JSONLogic governance, offline.

Writes a W3C PROV-O (JSON-LD) audit document linking every step and validates it.
Usage: python scripts/golden_path_smoke.py [--out runs/golden_path_prov.jsonld]
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def system1() -> Dict[str, Any]:
    from core.system.v22_async.async_agent_base import AsyncAgentBase
    from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent

    class Echo(AsyncAgentBase):
        async def execute(self, *args: Any, **kwargs: Any) -> Any:
            return {"echo": kwargs.get("task")}

    class Probe(BaseAgent):
        async def run(self, input_data: dict) -> None:
            await self.emit("THOUGHT", {"status": "ok", **input_data})

    v22 = await Echo({"name": "golden_echo"}).execute(task="ping")
    assert v22 == {"echo": "ping"}, v22
    await Probe("golden_probe", "smoke").run({"step": "system1"})
    return {"id": "adam:system1", "label": "System 1 (AsyncAgentBase + v30 BaseAgent.emit)"}


async def system2() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from core.agents.templates.v30_template_agent import TemplateAgentV30
    from core.schemas.agent_schema import AgentInput, AgentOutput

    class Golden(TemplateAgentV30):
        async def analyze(self, input_data: AgentInput) -> Tuple[str, List[str], float, Dict[str, Any]]:
            return "LTV within covenant", ["golden_path_fixture"], 0.9, {"ltv": input_data.context["ltv"]}

    out = await Golden({"name": "golden_system2"}).execute(AgentInput(query="check ltv", context={"ltv": 0.30}))
    assert isinstance(out, AgentOutput), type(out)
    assert out.provenance_trace, "TemplateAgentV30 must attach provenance_trace"
    assert out.metadata.get("confidence_tier") == "autonomous", out.metadata
    trace = [p.model_dump(mode="json") if hasattr(p, "model_dump") else dict(p) for p in out.provenance_trace]
    return {"id": "adam:system2", "label": "System 2 (TemplateAgentV30)"}, trace


def nexus() -> Dict[str, Any]:
    from core.engine import meta_orchestrator as mo
    from core.engine.meta_orchestrator import MetaOrchestrator, Route

    with patch.object(mo, "LLMPlugin"):
        orch = MetaOrchestrator(legacy_orchestrator=MagicMock())
    assert orch._assess_complexity("stress test the book", {}) is Route.CRISIS
    assert orch._assess_complexity("the attacker was shocking to the buyer", {}) is Route.LOW
    assert orch.resolve_ticker("what is the price today", {}) is None  # -> clarification_needed, not a guess
    return {"id": "adam:nexus", "label": "Nexus (MetaOrchestrator routing)", "model": orch.config.llm_model}


def governance() -> Dict[str, Any]:
    from adam_os.contexts.governance.engine import DeterministicPolicyEngine

    rules = json.loads((REPO / "adam_os/contexts/governance/rules.jsonLogic").read_text())
    engine = DeterministicPolicyEngine(rules)
    ok = engine.evaluate("default_credit_risk", {"entity_id": "golden", "ltv": rules["default_credit_risk"]["threshold"]})
    bad = engine.evaluate("default_credit_risk", {"entity_id": "golden", "ltv": 0.99})
    assert ok.passed and bad.is_breached
    return {"id": "adam:governance", "label": "JSONLogic governance (rules.jsonLogic)",
            "results": {"boundary": ok.passed, "breach": bad.is_breached}}


def prov_document(steps: List[Dict[str, Any]], trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    t = _now()
    graph: List[Dict[str, Any]] = [{"@id": "adam:golden_path", "@type": "prov:Agent", "prov:label": "verify_repo_health"}]
    prev = None
    for s in steps:
        act = {"@id": s["id"] + ":run", "@type": "prov:Activity", "prov:startedAtTime": t, "prov:endedAtTime": t,
               "prov:wasAssociatedWith": {"@id": "adam:golden_path"}, "prov:label": s["label"]}
        if prev:
            act["prov:used"] = {"@id": prev}
        ent = {"@id": s["id"] + ":result", "@type": "prov:Entity", "prov:wasGeneratedBy": {"@id": act["@id"]},
               "adam:detail": {k: v for k, v in s.items() if k not in ("id", "label")}}
        graph += [act, ent]
        prev = ent["@id"]
    graph.append({"@id": "adam:system2:provenance_trace", "@type": "prov:Bundle", "adam:trace": trace,
                  "prov:wasDerivedFrom": {"@id": "adam:system2:result"}})
    return {"@context": {"prov": "http://www.w3.org/ns/prov#", "adam": "https://github.com/adamvangrover/adam#"},
            "@graph": graph}


def validate_prov(doc: Dict[str, Any]) -> None:
    ids = {n["@id"] for n in doc["@graph"]}
    for n in doc["@graph"]:
        for key in ("prov:wasGeneratedBy", "prov:used", "prov:wasAssociatedWith", "prov:wasDerivedFrom"):
            if key in n:
                assert n[key]["@id"] in ids, f"dangling {key} -> {n[key]['@id']}"
    for n in doc["@graph"]:
        if n["@type"] == "prov:Entity":
            assert "prov:wasGeneratedBy" in n, n["@id"]
    assert doc["@context"]["prov"] == "http://www.w3.org/ns/prov#"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "runs" / "golden_path_prov.jsonld"))
    args = ap.parse_args()
    s1 = await system1()
    s2, trace = await system2()
    steps = [s1, s2, nexus(), governance()]
    doc = prov_document(steps, trace)
    validate_prov(doc)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, default=str))
    for s in steps:
        print(f"  ok  {s['label']}")
    print(f"  ok  W3C PROV-O audit trail -> {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
