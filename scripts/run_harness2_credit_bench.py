#!/usr/bin/env python3
"""
CLI Runner for Harness² (arXiv:2610.14224) Credit Benchmark in ADAM OS.

Executes Level 1 Task-Level Recursion (Propose-Probe-Compose) and Level 2
Harnessing-Level Recursion across open-world corporate credit tasks, scoring
trajectories on 4 no-ground-truth contrastive behavioral signals.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

from adam_harness2 import (
    ExecutionTrace,
    ExperienceLibrary,
    Harness2Controller,
    HarnessConfig,
)
from benchmarks.credit_bench import CreditBenchAdapter
from prompts.harness2_stage_prompts import (
    format_composer_prompt,
    format_editor_prompt,
    format_reflector_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("harness2_credit_bench")

DEFAULT_CREDIT_DATASET = [
    {
        "task_id": "tmt_leverage_001",
        "domain": "credit_tmt",
        "prompt": (
            "Evaluate net leverage, debt service coverage ratio (DSCR), and capitalized software "
            "R&D add-backs for Acme Tech Corp under 10-K exhibit 99.1."
        ),
    },
    {
        "task_id": "bsl_covenant_002",
        "domain": "credit_bsl",
        "prompt": (
            "Inspect Section 7 negative covenant headroom and EBITDA add-back schedules in the "
            "Broadly Syndicated Loan agreement for Nexus Holdings LLC."
        ),
    },
    {
        "task_id": "lme_jcrew_003",
        "domain": "credit_lme",
        "prompt": (
            "Analyze Liability Management Exercise (LME) risks, unrestricted subsidiary transfer "
            "provisions, and J.Crew blocker clauses for Apex Retail Inc."
        ),
    },
]


def create_default_base_harness() -> HarnessConfig:
    """Creates canonical 2nd-Line Credit Risk Officer base harness H_0."""
    return HarnessConfig(
        system_prompt=(
            "You are a 2nd-Line Credit Risk Officer operating under SR 11-7 Model Risk Management rules. "
            "Your objective is to conduct rigorous, source-grounded credit analysis and covenant verification."
        ),
        guardrails=(
            "LOCKED (SR 11-7 Invariant): Net leverage ceiling = 6.5x. DSCR floor = 1.25x. "
            "Unsubsidized EBITDA add-backs exceeding 15% of TTM EBITDA strictly prohibited without 2LoD approval. "
            "All claims must cite primary 10-K filings or Credit Agreement sections."
        ),
        memory="Canonical deal-room state initialized.",
        tool_guidance=(
            "Invoke SEC EDGAR table parsers and deterministic python financial scripts in scripts/ "
            "before stating conclusions."
        ),
        subagents="Covenant_Stress_Tester, SEC_Filing_Parser, EBITDA_Addback_Triage_Agent",
        skills="TMT_Software_Capitalization_Triage, BSL_Covenant_Headroom_Check, LME_Blocker_Audit",
        scripts={
            "leverage_calc.py": (
                "def calc_net_leverage(total_debt: float, cash: float, adj_ebitda: float) -> float:\n"
                "    net_debt = total_debt - cash\n"
                "    return round(net_debt / adj_ebitda, 4) if adj_ebitda > 0 else float('inf')\n"
            ),
            "dscr_calc.py": (
                "def calc_dscr(ebitda: float, capex: float, tax: float, debt_service: float) -> float:\n"
                "    fcf = ebitda - capex - tax\n"
                "    return round(fcf / debt_service, 4) if debt_service > 0 else float('inf')\n"
            ),
        },
        plugins=["schema_validator", "financial_identity_checker", "citation_verifier"],
    )


def build_mock_execution_components() -> tuple[
    Callable[[HarnessConfig, str], ExecutionTrace],
    Callable[[str, ExecutionTrace, str, int], tuple[list[dict[str, str]], list[dict[str, str]]]],
    Callable[[str, ExecutionTrace, list[ExecutionTrace], str], dict[str, str]],
    Callable[[str, ExecutionTrace, ExecutionTrace, dict[str, str]], str],
]:
    """Builds standard executor, editor, composer, and reflector functions for bench runner."""

    def mock_executor(harness: HarnessConfig, task_prompt: str) -> ExecutionTrace:
        has_scripts = bool(harness.scripts)
        tool_calls = [
            {
                "name": "leverage_calc.py",
                "args": {"total_debt": 450.0, "cash": 50.0, "adj_ebitda": 80.0},
                "status": "success",
            }
        ] if has_scripts else []

        output = (
            f"Underwriting Analysis for prompt: '{task_prompt}'.\n"
            f"Grounding Evidence: Inspected 10-K Section 7 covenant exhibits and TTM EBITDA add-back schedules.\n"
            f"Calculated Net Leverage: 5.0x (within 6.5x ceiling). DSCR: 1.85x.\n"
            f"Plugins Validation: Passed all financial identity checks."
        )

        return ExecutionTrace(
            harness_id=harness.fingerprint(),
            candidate_type="base",
            output=output,
            tool_calls=tool_calls,
            tool_errors=0,
            verification_passed=True,
            latency_ms=120.0,
        )

    def mock_editor(
        task_prompt: str, trace: ExecutionTrace, prior_lessons: str, k: int
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        _ = format_editor_prompt(
            task_prompt=task_prompt, ref_trace=trace, prior_lessons=prior_lessons, k=k
        )
        inv_patches = [{"tool_guidance": "Enforce mandatory python script usage before declaring leverage."}] * k
        dom_patches = [{"skills": "Enhanced SEC 10-K footnote & covenant table extraction playbook."}] * k
        return inv_patches, dom_patches

    def mock_composer(
        task_prompt: str,
        ref_trace: ExecutionTrace,
        probe_traces: list[ExecutionTrace],
        prior_lessons: str,
    ) -> dict[str, str]:
        _ = format_composer_prompt(
            task_prompt=task_prompt,
            ref_trace=ref_trace,
            probe_traces=probe_traces,
            prior_lessons=prior_lessons,
        )
        return {
            "tool_guidance": "Enforce mandatory python script usage before declaring leverage.",
            "skills": "Enhanced SEC 10-K footnote & covenant table extraction playbook.",
        }

    def mock_reflector(
        task_prompt: str,
        ref_trace: ExecutionTrace,
        final_trace: ExecutionTrace,
        adopted_patch: dict[str, str],
    ) -> str:
        return format_reflector_prompt(
            domain="credit_risk",
            task_id="task_ref",
            task_prompt=task_prompt,
            ref_trace=ref_trace,
            final_trace=final_trace,
            adopted_patch=adopted_patch,
        )

    return mock_executor, mock_editor, mock_composer, mock_reflector


def run_benchmark(
    dataset_path: str | None = None,
    output_dir: str = "runs/harness2_credit_bench",
    k: int = 1,
    mode: str = "parallel",
    domain: str = "credit_bsl",
) -> dict[str, object]:
    """Runs the Harness² Credit Benchmark and returns results dictionary."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if dataset_path and Path(dataset_path).exists():
        logger.info(f"Loading custom dataset from {dataset_path}")
        dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    else:
        logger.info("Using default corporate credit benchmark dataset.")
        dataset = DEFAULT_CREDIT_DATASET

    base_harness = create_default_base_harness()
    library = ExperienceLibrary(output_path / "experience_library")

    executor_fn, editor_fn, composer_fn, reflector_fn = build_mock_execution_components()

    controller = Harness2Controller(
        base_harness=base_harness,
        library=library,
        executor_fn=executor_fn,
        editor_fn=editor_fn,
        composer_fn=composer_fn,
        reflector_fn=reflector_fn,
    )

    adapter = CreditBenchAdapter()
    benchmark_results = adapter.evaluate_dataset(
        controller=controller,
        dataset=dataset,
        domain=domain,
        k=k,
        mode=mode,
    )

    summary = {
        "dataset_size": len(dataset),
        "k": k,
        "mode": mode,
        "domain": domain,
        "base_harness_fingerprint": base_harness.fingerprint(),
        "results": benchmark_results,
    }

    summary_file = output_path / "benchmark_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Saved benchmark summary to {summary_file}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Harness² Credit Benchmark Runner")
    parser.add_argument("--k", type=int, default=1, help="Number of candidate pairs per step (default: 1)")
    parser.add_argument(
        "--mode",
        choices=["parallel", "sequential"],
        default="parallel",
        help="Propose-Probe-Compose mode (default: parallel)",
    )
    parser.add_argument("--domain", type=str, default="credit_bsl", help="Domain key for Experience Library")
    parser.add_argument("--dataset", type=str, default="", help="Path to JSON dataset file (optional)")
    parser.add_argument(
        "--output", type=str, default="runs/harness2_credit_bench", help="Output directory path"
    )

    args = parser.parse_args()

    logger.info(f"Starting Harness² Credit Benchmark [k={args.k}, mode={args.mode}, domain={args.domain}]")
    summary = run_benchmark(
        dataset_path=args.dataset if args.dataset else None,
        output_dir=args.output,
        k=args.k,
        mode=args.mode,
        domain=args.domain,
    )

    print("\n" + "=" * 80)
    print(" HARNESS² CREDIT BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    results = summary.get("results", [])
    if isinstance(results, list):
        for res in results:
            if isinstance(res, dict):
                scores = res.get("scores", {})
                if isinstance(scores, dict):
                    print(f"Task ID      : {res.get('task_id')}")
                    print(f"Candidate    : {res.get('candidate_type')}")
                    print(
                        f"Scores       : Determinism={scores.get('tool_script_determinism')} | "
                        f"Grounding={scores.get('source_grounding_coverage')} | "
                        f"Middleware={scores.get('middleware_plugin_pass_rate')} | "
                        f"Efficiency={scores.get('trajectory_efficiency')}"
                    )
                    print(f"Overall Score: {scores.get('overall_score')}")
                    print(f"PROV Digest  : {res.get('provenance_digest')}")
                    print("-" * 80)


if __name__ == "__main__":
    main()
