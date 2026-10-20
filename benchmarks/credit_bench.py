"""
Custom Credit Benchmark Adapter for Harness² (arXiv:2610.14224) in ADAM OS.
Evaluates agent execution trajectories on open-world corporate credit tasks using 4 behavioral contrastive signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from adam_harness2 import ExecutionTrace, Harness2Controller, HarnessConfig


@dataclass
class SignalScores:
    tool_script_determinism: float
    source_grounding_coverage: float
    middleware_plugin_pass_rate: float
    trajectory_efficiency: float
    overall_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "tool_script_determinism": round(self.tool_script_determinism, 4),
            "source_grounding_coverage": round(self.source_grounding_coverage, 4),
            "middleware_plugin_pass_rate": round(
                self.middleware_plugin_pass_rate, 4
            ),
            "trajectory_efficiency": round(self.trajectory_efficiency, 4),
            "overall_score": round(self.overall_score, 4),
        }


class CreditBenchAdapter:
    """
    Evaluates credit risk execution traces across 4 verifiable behavioral signals
    when no ground-truth answers exist.
    """

    PRIMARY_EXHIBIT_KEYWORDS = [
        r"\b10-[kk]\b",
        r"\bcredit agreement\b",
        r"\bsection 7\b",
        r"\bcovenant\b",
        r"\bebitda\b",
        r"\bfootnote\b",
        r"\bexhibit\b",
        r"\bleverage\b",
        r"\bschedules?\b",
    ]

    def evaluate_trace(
        self, trace: ExecutionTrace, harness: HarnessConfig | None = None
    ) -> SignalScores:
        """Evaluates a single ExecutionTrace across the 4 signals."""
        # Signal 1: Tool & Script Determinism
        has_scripts = bool(harness and harness.scripts)
        tool_call_count = len(trace.tool_calls)
        if has_scripts and tool_call_count > 0:
            det_score = 1.0
        elif tool_call_count > 0:
            det_score = 0.8
        else:
            det_score = 0.2 if trace.verification_passed else 0.0

        # Signal 2: Source Grounding Coverage
        text_to_check = (trace.output + " " + str(trace.tool_calls)).lower()
        matched_keywords = sum(
            1
            for kw in self.PRIMARY_EXHIBIT_KEYWORDS
            if re.search(kw, text_to_check)
        )
        grounding_score = min(1.0, matched_keywords / 4.0)

        # Signal 3: Middleware Plugin Pass Rate
        if trace.verification_passed and trace.tool_errors == 0:
            plugin_score = 1.0
        elif trace.verification_passed:
            plugin_score = max(0.0, 1.0 - (0.2 * trace.tool_errors))
        else:
            plugin_score = 0.0

        # Signal 4: Trajectory Efficiency
        if tool_call_count == 0:
            eff_score = 0.5
        else:
            error_ratio = trace.tool_errors / tool_call_count
            eff_score = max(0.0, 1.0 - error_ratio)

        overall = (
            0.30 * det_score
            + 0.30 * grounding_score
            + 0.25 * plugin_score
            + 0.15 * eff_score
        )

        return SignalScores(
            tool_script_determinism=det_score,
            source_grounding_coverage=grounding_score,
            middleware_plugin_pass_rate=plugin_score,
            trajectory_efficiency=eff_score,
            overall_score=overall,
        )

    def evaluate_dataset(
        self,
        controller: Harness2Controller,
        dataset: list[dict[str, Any]],
        domain: str = "credit",
        k: int = 1,
        mode: str = "parallel",
    ) -> list[dict[str, Any]]:
        """Executes Harness2Controller over a benchmark dataset and returns score reports."""
        results = []
        for idx, item in enumerate(dataset):
            task_id = item.get("task_id", f"task_{idx+1}")
            task_prompt = item.get("prompt", "")

            trace = controller.run_task(
                task_id=task_id,
                domain=domain,
                task_prompt=task_prompt,
                k=k,
                mode=mode,
            )
            scores = self.evaluate_trace(trace, controller.base_harness)

            results.append(
                {
                    "task_id": task_id,
                    "candidate_type": trace.candidate_type,
                    "output": trace.output,
                    "scores": scores.to_dict(),
                    "provenance_digest": trace.provenance_digest,
                }
            )
        return results
