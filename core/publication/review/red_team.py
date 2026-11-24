"""Adversarial red-team pass over an edition, run through ``core.system.red_teaming_framework``.

Offline default is ``PublicationRedTeam``: a deterministic agent with the ``execute(AgentInput) ->
AgentOutput`` contract that flips the scenario weighting, challenges fragile targets and surfaces the
unsupported / stale findings. With ``llm.enabled`` the LLM-backed ``core.agents.red_team_agent.RedTeamAgent``
can be passed instead (prompt: ``prompt_library/publication_review/red_team.md``).
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from core.publication.review.critics import fragile_targets
from core.schemas.agent_schema import AgentInput, AgentOutput
from core.schemas.publication import Edition
from core.system.red_teaming_framework import RedTeamingFramework

_CHALLENGE_CODES = {"unsupported_call", "stale_data", "contradicts_prior", "target_out_of_range", "target_jump"}


def flipped_scenario(edition: Edition) -> Dict[str, Any]:
    tails = [s for s in edition.scenarios if s.name in ("bull", "bear")]
    if not tails:
        return {}
    flip = min(tails, key=lambda s: (s.probability, s.name))
    base = next((s for s in edition.scenarios if s.name == "base"), None)
    deltas = {k: round(v / base.targets[k] - 1, 4) for k, v in flip.targets.items() if base and base.targets.get(k)}
    return {"name": flip.name, "probability": flip.probability, "narrative": flip.narrative, "vs_base": deltas}


class PublicationRedTeam:
    name = "PublicationRedTeam"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        edition: Edition = input_data.context["edition"]
        findings: List[Dict[str, str]] = input_data.context.get("findings", [])
        challenges = [f for f in findings if f["code"] in _CHALLENGE_CODES] + fragile_targets(edition, self.cfg)
        flip = flipped_scenario(edition)
        impact = round(min(10.0, 2.0 * len(challenges) + (abs(min(flip.get("vs_base", {}).values(), default=0)) * 20)), 2)
        feedback = (f"Argue {flip['name']} as base: {flip['narrative']}" if flip
                    else "No scenario set; challenge the calls and targets directly.")
        return AgentOutput(
            answer=feedback, sources=[s.ref for s in edition.sources], confidence=0.9,
            metadata={"critique": {"feedback": feedback, "impact_score": impact, "meets_standards": not challenges,
                                   "challenges": challenges, "flipped_scenario": flip},
                      "human_readable_status": f"{len(challenges)} challenge(s) raised"},
        )


def red_team_review(edition: Edition, findings: List[Dict[str, str]], cfg: Dict[str, Any], agent=None) -> Dict[str, Any]:
    agent = agent or PublicationRedTeam(cfg)
    framework = RedTeamingFramework(agent, system=edition.edition_id)
    query = AgentInput(query=edition.title, context={"edition": edition, "findings": findings})
    report = asyncio.run(framework.run(query))
    critique = framework.logs[-1]["output"].metadata["critique"]
    return {"agent": getattr(agent, "name", type(agent).__name__), "report": report,
            "impact_score": critique["impact_score"], "challenges": critique.get("challenges", []),
            "flipped_scenario": critique.get("flipped_scenario", {})}
