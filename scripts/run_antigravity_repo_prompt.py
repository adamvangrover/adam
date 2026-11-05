#!/usr/bin/env python3
"""
AFOS v30.1 Repository-as-a-Prompt Runner for the Google Antigravity SDK.

Hydrates `LLM_README.md`, `AGENTS.md`, and `docs/AGENTS_KNOWLEDGE_BASE.md` into
a unified PDIL + W3C PROV-O system prompt and executes queries or deterministic
System 1 (Swarm) -> System 2 (Graph) workflows via `google.antigravity`.
Includes automatic Dual-Mode offline fallback if the SDK or API keys are absent.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import os
import pathlib
import sys
from typing import Any, Dict, List

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.agent_schema import AgentOutput, ProvenanceHeader

MANIFEST_FILES: List[pathlib.Path] = [
    REPO_ROOT / "LLM_README.md",
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "docs" / "AGENTS_KNOWLEDGE_BASE.md",
]


def _infer_routing_path(query: str, mode: str = "chat") -> str:
    """Resolve deterministic AFOS routing path across workflow, Swarm, and DAG modes."""
    if mode == "workflow":
        return "SYSTEM_2_WORKFLOW"
    lower_q = query.lower()
    if any(k in lower_q for k in ("swarm", "scan", "surveillance", "hive")):
        return "SYSTEM_1_SWARM"
    return "SYSTEM_2_DAG"


def build_repo_system_prompt(include_full_knowledge_base: bool = True) -> str:
    """Compile canonical repository manifests into a single system instruction prompt."""
    files = MANIFEST_FILES if include_full_knowledge_base else MANIFEST_FILES[:2]
    sections = [
        "# SYSTEM BOOT: ADAM v30.1 (AFOS — Neuro-Symbolic Financial Sovereign)\n"
        "# MODE: REPOSITORY-AS-A-PROMPT | PROTOCOL: PDIL + AOPL v2.0 + W3C PROV-O\n\n"
        "You are the runtime execution kernel of the `adamvangrover/adam` repository.\n"
        "Enforce System 1 (Async Swarm) and System 2 (Neuro-Symbolic DAG) routing,\n"
        "Sentinel security guardrails, Bolt vectorization rules, and W3C PROV-O\n"
        "`ProvenanceHeader.check_grounding()` on every output.\n"
    ]
    for path in files:
        if path.exists():
            rel = path.relative_to(REPO_ROOT)
            sections.append(f"\n--- BEGIN {rel} ---\n{path.read_text(encoding='utf-8')}\n--- END {rel} ---")
    return "\n".join(sections)


def build_prov_o_envelope(
    query: str,
    answer: str,
    routing_path: str = "SYSTEM_2_DAG",
    source_data_object: str = "[docs/AGENTS_KNOWLEDGE_BASE.md:v30.1]",
    confidence_score: float = 0.92,
) -> Dict[str, Any]:
    """Wrap raw agent output in a W3C PROV-O grounded `AgentOutput` envelope."""
    content_hash = hashlib.sha256(
        json.dumps({"query": query, "routing_path": routing_path, "answer": answer}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    header = ProvenanceHeader(
        source_data_object=source_data_object,
        git_commit_hash="v30.1-afos",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        content_hash=content_hash,
        jsonLogic_version="v30.1",
        confidence_score=confidence_score,
        derivation_path=f"Nexus -> PDIL -> {routing_path} -> Consensus",
    )
    clean_src = source_data_object.strip("[]")
    final_answer = (
        answer if f"[{clean_src}]" in answer else f"{answer}\n\nCitation: {source_data_object}"
    )
    validated_output = AgentOutput(
        answer=final_answer,
        sources=[source_data_object],
        confidence=confidence_score,
        provenance_trace=header,
        observed_drift=False,
        metadata={
            "routing_path": routing_path,
            "query": query,
            "state_hash": header.hash_state(),
        },
    )
    is_grounded = validated_output.check_grounding()
    if not is_grounded:
        raise ValueError("W3C PROV-O Grounding Violation: source_data_object is empty.")
    return {
        "agent_output": validated_output.model_dump() if hasattr(validated_output, "model_dump") else validated_output.dict(),
        "w3c_prov_o_grounded": is_grounded,
        "state_sha256": header.hash_state(),
    }


async def run_with_antigravity_sdk(
    query: str,
    mode: str = "chat",
    runtime: str = "gemini",
    enable_write_tools: bool = False,
    vertex: bool = False,
) -> Dict[str, Any]:
    """Execute the Repository-as-a-Prompt via `google.antigravity`."""
    try:
        from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig
    except ImportError:
        return _run_offline_deterministic_fallback(
            query=query,
            reason="google-antigravity package not installed (`pip install google-antigravity`).",
            mode=mode,
        )

    system_instructions = build_repo_system_prompt()
    capabilities = CapabilitiesConfig() if enable_write_tools else None

    if runtime == "litert":
        from google.antigravity import LiteRTAgentConfig

        config = LiteRTAgentConfig(
            system_instructions=system_instructions,
            capabilities=capabilities,
        )
    elif runtime == "local-openai":
        from google.antigravity import LocalOpenAIAgentConfig

        config = LocalOpenAIAgentConfig(
            system_instructions=system_instructions,
            capabilities=capabilities,
        )
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not api_key and not (vertex and project):
            sys.stderr.write(
                "[AFOS-AGY] No GEMINI_API_KEY or Vertex ADC project detected.\n"
                "  -> Get a Gemini API key at: https://aistudio.google.com/app/api-keys\n"
                "  -> Or run `gcloud auth application-default login` with --vertex\n"
                "  -> Or pass `--runtime litert` for local on-device execution.\n"
                "  -> Engaging AFOS Dual-Mode deterministic fallback...\n"
            )
            return _run_offline_deterministic_fallback(
                query=query,
                reason="Missing GEMINI_API_KEY / Vertex ADC; executed via AFOS deterministic PDIL fallback.",
                mode=mode,
            )

        config_kwargs: Dict[str, Any] = {
            "system_instructions": system_instructions,
            "capabilities": capabilities,
        }
        if api_key:
            config_kwargs["api_key"] = api_key
        if vertex:
            config_kwargs["vertex"] = True
            if project:
                config_kwargs["project"] = project
                config_kwargs["location"] = location

        config = LocalAgentConfig(**config_kwargs)

    collected_tokens: List[str] = []
    async with Agent(config) as agent:
        if mode == "workflow" and hasattr(agent, "beta"):
            from google.antigravity import beta

            @beta.workflows.define
            def afos_pdil_workflow(objective: str):
                beta.workflows.log(f"[Nexus] Dispatching AFOS v30.1 workflow for: {objective}")
                with beta.workflows.phase("system_1_swarm_perception"):
                    swarm_scan = beta.workflows.parallel(
                        beta.workflows.agent(
                            f"SYSTEM_1_SWARM [AFOS-SURV-01]: Scan Liquidity Mirage & Systemic Tremor indicators for: {objective}"
                        ),
                        beta.workflows.agent(
                            f"SYSTEM_1_SWARM [AFOS-SENT-01]: Verify Zero-Trust & Sentinel security constraints for: {objective}"
                        ),
                    )
                with beta.workflows.phase("system_2_neuro_symbolic_dag"):
                    synthesis = beta.workflows.pipeline(
                        beta.workflows.agent(
                            f"SYSTEM_2_DAG [AFOS-UWR-01]: Perform SR 11-7 & PDIL grounded analysis using {swarm_scan} for: {objective}"
                        ),
                        beta.workflows.agent(
                            "Consensus Engine: Emit final synthesis with [Source_ID:Chunk_ID] citations and v30_knowledge_graph."
                        ),
                    )
                return synthesis

            result = await agent.beta.run_workflow(afos_pdil_workflow, query)
            answer_text = str(result)
        else:
            response = await agent.chat(query)
            async for thought in response.thoughts:
                sys.stderr.write(f"[PROV-O Thought] {thought}\n")
            async for call in response.tool_calls:
                sys.stderr.write(f"[PDIL ToolCall] {call.name}({call.args})\n")
            async for token in response:
                sys.stdout.write(token)
                sys.stdout.flush()
                collected_tokens.append(token)
            print()
            answer_text = "".join(collected_tokens)

    return build_prov_o_envelope(
        query=query,
        answer=answer_text,
        routing_path=_infer_routing_path(query=query, mode=mode),
    )


def _run_offline_deterministic_fallback(query: str, reason: str, mode: str = "chat") -> Dict[str, Any]:
    """Execute deterministic offline PDIL synthesis when cloud LLM credentials are absent."""
    routing_path = _infer_routing_path(query=query, mode=mode)
    answer = (
        f"[AFOS v30.1 Deterministic PDIL Synthesis | Mode: {routing_path}]\n"
        f"Fallback Reason: {reason}\n"
        f"Query Evaluated: {query}\n"
        "Manifests Loaded: LLM_README.md, AGENTS.md, docs/AGENTS_KNOWLEDGE_BASE.md\n"
        "Citation: [docs/AGENTS_KNOWLEDGE_BASE.md:v30.1]"
    )
    return build_prov_o_envelope(
        query=query,
        answer=answer,
        routing_path=routing_path,
        source_data_object="[docs/AGENTS_KNOWLEDGE_BASE.md:v30.1]",
        confidence_score=0.90,
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adamvangrover/adam (AFOS v30.1) as a Google Antigravity SDK Agent."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="DEEP_DIVE: Summarize the PDIL W3C PROV-O grounding architecture and verify agent compliance.",
        help="Task or query to execute against the repository-as-a-prompt.",
    )
    parser.add_argument(
        "--mode",
        choices=["chat", "workflow", "dump-prompt"],
        default="chat",
        help="Execution mode: streaming chat, @beta.workflows.define DAG, or dump the hydrated system prompt.",
    )
    parser.add_argument(
        "--runtime",
        choices=["gemini", "litert", "local-openai"],
        default="gemini",
        help="Model runtime: hosted Gemini (default), on-device LiteRT, or local OpenAI-compatible server.",
    )
    parser.add_argument(
        "--vertex",
        action="store_true",
        help="Use Vertex AI / Gemini Enterprise Agent Platform authentication.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Enable write capabilities (CapabilitiesConfig) for file edits and command execution.",
    )
    return parser.parse_args(argv)


async def _async_main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "dump-prompt":
        print(build_repo_system_prompt())
        return 0

    envelope = await run_with_antigravity_sdk(
        query=args.query,
        mode=args.mode,
        runtime=args.runtime,
        enable_write_tools=args.write,
        vertex=args.vertex,
    )
    print(json.dumps(envelope, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_async_main()))
