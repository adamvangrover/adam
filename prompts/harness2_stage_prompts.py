"""
Stage Prompts for Harness² (Editor, Composer, Reflector)
Tailored for 2nd-Line Corporate Credit, SEC Filing, and Covenant Risk Oversight.
"""

from __future__ import annotations

import json
from typing import Any

EDITOR_STAGE_PROMPT = """
You are the Harness Editor operating in a Harness² runtime architecture for 2nd-Line Credit Risk Oversight.
Your role is to analyze a reference execution trajectory (τ_0) on a credit analysis task and propose orthogonal harness patches to resolve failure modes.

GOVERNANCE INVARIANTS:
1. 'guardrails' IS IMMUTABLE. You MUST NOT modify guardrails under SR 11-7 Model Risk Management rules. Any patch attempting to alter guardrails will be rejected with a PermissionError.
2. 'plugins' IS APPEND-ONLY. You may add stricter validation middleware, but you cannot remove existing checks.

INPUT DATA:
- Task Prompt: {task_prompt}
- Reference Trajectory (τ_0): {ref_trace}
- Prior Domain Lessons (EXPERIENCE.md): {prior_lessons}
- Candidate Pairs Count (k): {k}

OUTPUT REQUIREMENTS:
Generate {k} candidate patch pairs consisting of:
1. Invariant Candidate (H_inv): Fixes general agentic pathologies:
   - Tool-loop thrashing or repeated failing calls
   - Premature summarization before full document extraction
   - LLM mental arithmetic instead of deterministic code execution
   - Missing verification steps

2. Domain Candidate (H_dom): Injects domain-specific procedures for credit risk:
   - Sector-specific EBITDA add-back triage playbooks
   - SEC EDGAR table extraction heuristics
   - Covenant breach stress testing scripts (Net Leverage, FCCR, DSCR)
   - LME / Liability Management Exercise / J.Crew blocker checklists

Respond in valid JSON format containing:
{{
  "invariant_patches": [ {{ ...patch dict... }} ],
  "domain_patches": [ {{ ...patch dict... }} ]
}}
"""

COMPOSER_STAGE_PROMPT = """
You are the Harness Composer operating in a Harness² runtime architecture for 2nd-Line Credit Risk Oversight.
Your role is to contrast probe trajectories (τ_inv, τ_dom) against the baseline reference trajectory (τ_0) and synthesize an optimal composed harness patch (H^*).

NO-GROUND-TRUTH BEHAVIORAL SIGNALS TO EVALUATE:
1. Tool & Script Determinism: Did the candidate write/invoke deterministic Python scripts in `scripts/` (e.g. Net Leverage = Total Debt / Adjusted EBITDA) rather than mental LLM arithmetic?
2. Source Grounding Coverage: Did the candidate inspect primary exhibits (Credit Agreement Section 7, 10-K Footnotes, EBITDA add-back schedules) before synthesizing credit conclusions?
3. Middleware Plugin Pass Rate: Did the trajectory pass all financial identity and schema validator plugins on the first attempt without errors?
4. Trajectory Efficiency: Did the edit eliminate redundant search loops, tool thrashing, or unnecessary steps observed in τ_0?

INPUT DATA:
- Task Prompt: {task_prompt}
- Baseline Trace (τ_0): {ref_trace}
- Candidate Probe Traces: {probe_traces}
- Prior Domain Lessons: {prior_lessons}

OUTPUT REQUIREMENTS:
Select and compose the most effective non-conflicting edits from τ_inv and τ_dom that demonstrated empirical trajectory improvements.
Ensure that no edits weaken SR 11-7 regulatory guardrails.

Respond in valid JSON format containing the composed patch dict (H^*).
"""

REFLECTOR_STAGE_PROMPT = """
You are the Experience Reflector operating in a Harness² Level-2 Harnessing Recursion loop.
Your role is to summarize what worked, what failed, and why into an EXPERIENCE.md note that will condition future credit underwriting tasks in this domain.

INPUT DATA:
- Task Prompt: {task_prompt}
- Baseline Trace (τ_0): {ref_trace}
- Composed Final Trace (τ_*): {final_trace}
- Adopted Patch: {adopted_patch}

OUTPUT REQUIREMENTS:
Write a clean, structured Markdown summary suitable for persistent inclusion in EXPERIENCE.md.
Structure:
# Experience Reflection: {domain} / Task {task_id}

## 1. Baseline Pathology
- Summary of trajectory errors or inefficiencies in τ_0.

## 2. Adopted Harness Innovations
- Specific modular edits (skills, tool guidance, scripts, subagents, plugins) that produced verifiable improvements.

## 3. Reusable Domain Rule / Directive
- Clear, actionable guidance for future credit risk analyses in this sector/domain.
"""


def format_editor_prompt(
    task_prompt: str,
    ref_trace: Any,
    prior_lessons: str,
    k: int = 1,
) -> str:
    """Formats the Editor Stage Prompt with task inputs."""
    return EDITOR_STAGE_PROMPT.format(
        task_prompt=task_prompt,
        ref_trace=str(ref_trace),
        prior_lessons=prior_lessons or "No prior domain lessons recorded.",
        k=k,
    )


def format_composer_prompt(
    task_prompt: str,
    ref_trace: Any,
    probe_traces: list[Any],
    prior_lessons: str,
) -> str:
    """Formats the Composer Stage Prompt with baseline and probe trajectories."""
    return COMPOSER_STAGE_PROMPT.format(
        task_prompt=task_prompt,
        ref_trace=str(ref_trace),
        probe_traces=json.dumps([str(p) for p in probe_traces], indent=2),
        prior_lessons=prior_lessons or "No prior domain lessons recorded.",
    )


def format_reflector_prompt(
    domain: str,
    task_id: str,
    task_prompt: str,
    ref_trace: Any,
    final_trace: Any,
    adopted_patch: dict,
) -> str:
    """Formats the Experience Reflector Stage Prompt."""
    return REFLECTOR_STAGE_PROMPT.format(
        domain=domain,
        task_id=task_id,
        task_prompt=task_prompt,
        ref_trace=str(ref_trace),
        final_trace=str(final_trace),
        adopted_patch=json.dumps(adopted_patch, indent=2),
    )


def parse_editor_response(response_text: str) -> tuple[list[dict], list[dict]]:
    """Parses and validates JSON output from the Editor LLM response."""
    try:
        data = json.loads(response_text)
        inv = data.get("invariant_patches", [])
        dom = data.get("domain_patches", [])
        return inv, dom
    except Exception:
        return [], []


def parse_composer_response(response_text: str) -> dict:
    """Parses and validates JSON output from the Composer LLM response."""
    try:
        data = json.loads(response_text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
