"""recursive_engine.py - Autonomous Orchestrator, Ledger, and In-Context Self-Improvement."""

import hashlib
import json
import os
import subprocess
from typing import Any, Callable, Dict, List
from scripts.rubric_schema import EvaluationRubricResult, MacroForecastSubmission


class EpistemicMemoryLedger:
  """Persistent calibration memory.

  Ingests post-settlement diagnostics to adjust base rates and inject
  contrastive few-shot post-mortems into future inference runs.
  """

  def __init__(self, memory_file_path: str = "data/calibration_memory.json"):
    self.memory_path = memory_file_path
    self.history: List[Dict[str, Any]] = self._load()

  def _load(self) -> List[Dict[str, Any]]:
    if os.path.exists(self.memory_path):
      with open(self.memory_path, "r", encoding="utf-8") as f:
        return json.load(f)
    return []

  def record_settlement(
      self,
      challenge_id: str,
      asset: str,
      predicted_dir: str,
      confidence: float,
      actual_outcome: str,
      actual_settlement_price: float,
      point_forecast: float,
      judge_score: float,
      post_mortem_insight: str,
  ):
    is_correct = predicted_dir.lower() == actual_outcome.lower()
    directional_score = (
        (50.0 + 50.0 * confidence) if is_correct else (50.0 - 50.0 * confidence)
    )
    brier_loss = ((1.0 if is_correct else 0.0) - confidence) ** 2

    entry = {
        "challenge_id": challenge_id,
        "asset": asset,
        "predicted_direction": predicted_dir,
        "confidence": confidence,
        "actual_outcome": actual_outcome,
        "actual_settlement_price": actual_settlement_price,
        "point_forecast": point_forecast,
        "point_delta": abs(point_forecast - actual_settlement_price)
        if point_forecast
        else None,
        "correct": is_correct,
        "arena_score": directional_score,
        "brier_loss": round(brier_loss, 4),
        "judge_score": judge_score,
        "post_mortem_insight": post_mortem_insight,
    }
    self.history.append(entry)
    os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
    with open(self.memory_path, "w", encoding="utf-8") as f:
      json.dump(self.history, f, indent=2)

  def generate_calibration_context(self, top_k: int = 5) -> str:
    if not self.history:
      return (
          "NO PRIOR SETTLEMENT DRIFT RECORDED. Baseline neutral calibration"
          " enforced."
      )

    total = len(self.history)
    correct_count = sum(1 for x in self.history if x["correct"])
    base_rate = (correct_count / total) * 100.0
    mean_brier = sum(x["brier_loss"] for x in self.history) / total
    recent_samples = self.history[-top_k:]

    lines = [
        f"HISTORICAL BASE RATE: {base_rate:.1f}% ({correct_count}/{total} calls"
        f" correct) | Mean Brier Score: {mean_brier:.4f}",
        "RECURSIVE EPITEMIC MEMORY (ACTIVE SYSTEM-2 ERROR CORRECTION):",
    ]
    for s in recent_samples:
      verdict = "ACCURATE" if s["correct"] else "MISCALIBRATED"
      lines.append(
          f"- [{verdict}] Asset: {s['asset']} | Call: {s['predicted_direction']}"
          f" (c={s['confidence']:.2f}) | Outcome: {s['actual_outcome']} |"
          f" Rationale Score: {s['judge_score']:.1f}/100\n  Epistemic"
          f" Post-Mortem: {s['post_mortem_insight']}"
      )
    return "\n".join(lines)


class AppendOnlyAuditTrail:
  """Cryptographically seals forecasts into an immutable Git ledger before market settlement."""

  def __init__(self, ledger_dir: str = "forecast_ledger"):
    self.ledger_dir = ledger_dir
    os.makedirs(self.ledger_dir, exist_ok=True)

  def seal_and_commit(
      self, forecast: MacroForecastSubmission, timestamp_str: str
  ) -> str:
    filename = (
        f"{forecast.challenge_id}_{forecast.target_asset}_{timestamp_str}.json"
    )
    filepath = os.path.join(self.ledger_dir, filename)

    payload = forecast.model_dump()
    payload["sealed_at_utc"] = timestamp_str

    serialized_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    sha256_hash = hashlib.sha256(serialized_bytes).hexdigest()
    payload["sha256_seal"] = sha256_hash

    with open(filepath, "w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2)

    try:
      subprocess.run(["git", "add", filepath], check=True, capture_output=True)
      commit_msg = (
          f"forecast({forecast.target_asset}): forward prediction"
          f" [{forecast.challenge_id}] - SHA256:{sha256_hash[:8]}"
      )
      subprocess.run(
          ["git", "commit", "-m", commit_msg], check=True, capture_output=True
      )
    except subprocess.SubprocessError as e:
      print(f"[AUDIT WARN] Git commit execution bypassed: {e}")

    return filepath


class MacroAgentHarness:
  """Production coordinator managing generation, adversarial arbitration, sealing, and API transport."""

  def __init__(
      self,
      memory: EpistemicMemoryLedger,
      audit: AppendOnlyAuditTrail,
      generator_fn: Callable[[str], str],
      judge_fn: Callable[[MacroForecastSubmission], EvaluationRubricResult],
      api_dispatch_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
  ):
    self.memory = memory
    self.audit = audit
    self.generator = generator_fn
    self.judge = judge_fn
    self.dispatch_api = api_dispatch_fn

  def execute_pipeline(
      self,
      base_directive: str,
      market_challenge: Dict[str, Any],
      timestamp_str: str,
      max_refinements: int = 2,
  ) -> Dict[str, Any]:
    # 1. Inject recursive calibration priors
    calibration_context = self.memory.generate_calibration_context()
    runtime_prompt = (
        f"{base_directive}\n\n"
        "=== RECURSIVE CALIBRATION MEMORY (HISTORICAL PERFORMANCE) ===\n"
        f"{calibration_context}\n\n"
        "=== ACTIVE CHALLENGE INGESTION ===\n"
        f"{json.dumps(market_challenge, indent=2)}\n"
    )

    candidate_forecast = None
    judge_result = None

    # 2. Generation & Adversarial Refinement Loop
    for attempt in range(max_refinements + 1):
      raw_output = self.generator(runtime_prompt)
      candidate_forecast = MacroForecastSubmission.model_validate_json(
          raw_output
      )
      judge_result = self.judge(candidate_forecast)

      if judge_result.audit_verdict == "APPROVE":
        break

      # Refinement prompt injection if rejected by pre-submission gate
      runtime_prompt += (
          f"\n\n[GATE REJECTION - ATTEMPT {attempt + 1}]\n"
          f"Your submission was REJECTED with Score"
          f" {judge_result.aggregate_score:.1f}%.\nDeficiencies:"
          f" {judge_result.rejection_reasons}\nAddress every deficiency"
          " precisely and regenerate the complete JSON response."
      )

    if not judge_result or judge_result.audit_verdict != "APPROVE":
      raise RuntimeError(
          f"Pipeline aborted: Rationale failed to clear audit gate. Score:"
          f" {judge_result.aggregate_score if judge_result else 'N/A'}"
      )

    # 3. Cryptographic Append-Only Sealing
    ledger_path = self.audit.seal_and_commit(candidate_forecast, timestamp_str)

    # 4. Headline Arena Transmission
    api_receipt = self.dispatch_api(candidate_forecast.model_dump())

    return {
        "status": "DISPATCHED_AND_SEALED",
        "challenge_id": candidate_forecast.challenge_id,
        "ledger_file": ledger_path,
        "judge_audit": judge_result.model_dump(),
        "arena_receipt": api_receipt,
    }
