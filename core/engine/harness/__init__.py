"""Agent lifecycle hooks (thresholds in ``config/agent_schema.yaml`` → ``harness``).

| Hook | Stage | Does |
|---|---|---|
| ``h_ctx.pre_turn`` | pre-turn | masks stale observations, enforces the context token budget |
| ``h_ctrl.pre_turn`` / ``record_tool_call`` | pre-turn | turn cap, duplicate tool-call breaker |
| ``h_act.post_tool`` | post-tool | stdout truncation, error normalization |
| ``h_ver.pre_stop`` | pre-stop | hard abort without provenance / ``jsonLogic_version``; HITL on high-impact actions |

Wired into ``TemplateAgentV30.execute_pydantic`` and ``MetaOrchestrator.route_request``.
"""
from core.engine.harness import h_act, h_ctrl, h_ctx, h_ver
from core.engine.harness.policy import HarnessHalt, HarnessPolicy, load_harness_policy

__all__ = ["h_act", "h_ctrl", "h_ctx", "h_ver", "HarnessHalt", "HarnessPolicy", "load_harness_policy"]
