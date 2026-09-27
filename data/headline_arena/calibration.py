"""
ADAM-Macro-Sentinel Calibration Module
=======================================
Dynamic confidence calibration using:

1. Platt Scaling — logistic recalibration of raw confidence scores
2. Epistemic Memory Ledger — persistent settlement history for parameter updates
3. Rolling Brier diagnostics — multi-day calibration assessment
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import (
    Direction,
    EpistemicMemoryEntry,
    ForecastSubmission,
    SettlementResult,
)
from .scoring import compute_brier_score, compute_s_dir


# ─── Platt Scaling ────────────────────────────────────────────────────────────

class PlattCalibrator:
    """
    Dynamic Temperature Scaling on Confidence (c).

    Rather than treating confidence as an unconstrained generative scalar,
    calibrate it using empirical Platt scaling over historical Brier scores:

        c_calibrated = 1 / (1 + exp(-(a * c_raw + b)))

    Parameters a and b update dynamically through the EpistemicMemoryLedger
    following each settlement window.

    Initial parameters:
        a = 1.0 (identity scaling)
        b = 0.0 (no bias)
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = a
        self.b = b

    def calibrate(self, raw_confidence: float) -> float:
        """
        Apply Platt scaling to raw confidence.

        Returns calibrated confidence clamped to [0.50, 1.00].
        """
        logit = self.a * raw_confidence + self.b
        # Guard against overflow
        if logit > 20:
            calibrated = 1.0
        elif logit < -20:
            calibrated = 0.0
        else:
            calibrated = 1.0 / (1.0 + math.exp(-logit))

        # Clamp to valid confidence range
        return max(0.50, min(1.00, calibrated))

    def update_from_settlements(
        self,
        entries: list[EpistemicMemoryEntry],
        learning_rate: float = 0.01,
        n_iterations: int = 100,
    ) -> None:
        """
        Update Platt parameters (a, b) via gradient descent on the
        log-likelihood of observed settlement outcomes.

        Uses the completed entries (where correct is not None) to
        fit the logistic calibration curve.
        """
        # Filter to entries with settlement results
        settled = [e for e in entries if e.correct is not None]
        if len(settled) < 5:
            # Insufficient data — keep current parameters
            return

        for _ in range(n_iterations):
            grad_a = 0.0
            grad_b = 0.0

            for entry in settled:
                c_raw = entry.raw_confidence
                y = 1.0 if entry.correct else 0.0

                logit = self.a * c_raw + self.b
                if logit > 20:
                    p = 1.0 - 1e-10
                elif logit < -20:
                    p = 1e-10
                else:
                    p = 1.0 / (1.0 + math.exp(-logit))

                error = p - y
                grad_a += error * c_raw
                grad_b += error

            n = len(settled)
            self.a -= learning_rate * (grad_a / n)
            self.b -= learning_rate * (grad_b / n)

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, data: dict) -> "PlattCalibrator":
        return cls(a=data.get("a", 1.0), b=data.get("b", 0.0))


# ─── Epistemic Memory Ledger ─────────────────────────────────────────────────

class EpistemicMemoryLedger:
    """
    Persistent ledger tracking every forecast-settlement pair.
    Provides the data substrate for:
    - Platt scaling parameter updates
    - Rolling Brier score diagnostics
    - Regime-specific calibration analysis

    Storage: JSONL append-only log with SHA-256 chain integrity.
    """

    def __init__(self, ledger_path: str | Path):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[EpistemicMemoryEntry] = []
        self._calibrator = PlattCalibrator()
        self._load()

    def _load(self) -> None:
        """Load existing ledger entries from JSONL file."""
        if not self.ledger_path.exists():
            return

        for line in self.ledger_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                entry = EpistemicMemoryEntry(**data)
                self._entries.append(entry)
            except (json.JSONDecodeError, Exception) as e:
                # Corrupted line — log and skip
                print(f"  ⚠ Ledger corruption on line: {e}")

        # Restore calibrator from last entry's parameters
        if self._entries:
            last = self._entries[-1]
            self._calibrator = PlattCalibrator(a=last.platt_a, b=last.platt_b)

    def _append(self, entry: EpistemicMemoryEntry) -> None:
        """Append a single entry to the JSONL ledger."""
        with open(self.ledger_path, "a") as f:
            f.write(entry.model_dump_json() + "\n")
        self._entries.append(entry)

    def record_forecast(
        self,
        forecast: ForecastSubmission,
        raw_confidence: float,
        arbitration_winner: str = "champion",
    ) -> EpistemicMemoryEntry:
        """
        Record a submitted forecast in the ledger.
        Called at submission time (before settlement).
        """
        calibrated = self._calibrator.calibrate(raw_confidence)

        entry = EpistemicMemoryEntry(
            entry_id=str(uuid.uuid4()),
            challenge_id=forecast.challenge_id,
            target_asset=forecast.target_asset,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            predicted_direction=forecast.direction,
            platt_a=self._calibrator.a,
            platt_b=self._calibrator.b,
            arbitration_winner=arbitration_winner,
        )
        self._append(entry)
        return entry

    def record_settlement(
        self,
        settlement: SettlementResult,
    ) -> Optional[EpistemicMemoryEntry]:
        """
        Update a ledger entry with settlement results.
        Searches for the matching forecast entry and appends an updated record.
        """
        # Find the most recent unsettled entry for this challenge
        match = None
        for entry in reversed(self._entries):
            if (
                entry.challenge_id == settlement.challenge_id
                and entry.correct is None
            ):
                match = entry
                break

        if match is None:
            return None

        # Compute scores
        correct = settlement.predicted_direction == settlement.actual_direction
        s_dir = compute_s_dir(
            match.calibrated_confidence,
            settlement.predicted_direction,
            settlement.actual_direction,
        )
        brier = compute_brier_score(
            match.calibrated_confidence,
            settlement.predicted_direction,
            settlement.actual_direction,
        )

        # Create settlement entry
        settled_entry = EpistemicMemoryEntry(
            entry_id=match.entry_id,
            challenge_id=match.challenge_id,
            target_asset=match.target_asset,
            timestamp=datetime.now(timezone.utc),
            raw_confidence=match.raw_confidence,
            calibrated_confidence=match.calibrated_confidence,
            predicted_direction=match.predicted_direction,
            actual_direction=settlement.actual_direction,
            correct=correct,
            s_dir_score=s_dir,
            brier_residual=brier,
            platt_a=self._calibrator.a,
            platt_b=self._calibrator.b,
            arbitration_winner=match.arbitration_winner,
            notes=f"Settled: {'correct' if correct else 'incorrect'}",
        )
        self._append(settled_entry)

        # Update Platt calibrator after each settlement
        self._calibrator.update_from_settlements(self._entries)

        return settled_entry

    @property
    def calibrator(self) -> PlattCalibrator:
        return self._calibrator

    def calibrate_confidence(self, raw_confidence: float) -> float:
        """Apply current Platt scaling to a raw confidence value."""
        return self._calibrator.calibrate(raw_confidence)

    def get_rolling_accuracy(self, window: int = 20) -> float:
        """Compute rolling hit rate over last N settled entries."""
        settled = [e for e in self._entries if e.correct is not None]
        if not settled:
            return 0.5  # Prior

        recent = settled[-window:]
        hits = sum(1 for e in recent if e.correct)
        return hits / len(recent)

    def get_rolling_brier(self, window: int = 20) -> float:
        """Compute rolling average Brier score over last N settled entries."""
        settled = [e for e in self._entries if e.brier_residual is not None]
        if not settled:
            return 0.25  # Prior (random baseline)

        recent = settled[-window:]
        return sum(e.brier_residual for e in recent) / len(recent)

    def get_asset_calibration(self, target_asset: str) -> dict:
        """Get calibration diagnostics for a specific asset."""
        asset_entries = [
            e for e in self._entries
            if e.target_asset == target_asset and e.correct is not None
        ]
        if not asset_entries:
            return {
                "asset": target_asset,
                "n_forecasts": 0,
                "hit_rate": 0.5,
                "avg_brier": 0.25,
                "avg_confidence": 0.65,
            }

        hits = sum(1 for e in asset_entries if e.correct)
        return {
            "asset": target_asset,
            "n_forecasts": len(asset_entries),
            "hit_rate": hits / len(asset_entries),
            "avg_brier": (
                sum(e.brier_residual for e in asset_entries if e.brier_residual)
                / max(1, sum(1 for e in asset_entries if e.brier_residual))
            ),
            "avg_confidence": (
                sum(e.calibrated_confidence for e in asset_entries)
                / len(asset_entries)
            ),
        }

    def get_ledger_diagnostics(self) -> dict:
        """Full ledger diagnostics summary."""
        total = len(self._entries)
        settled = [e for e in self._entries if e.correct is not None]
        unsettled = total - len(settled)

        return {
            "total_entries": total,
            "settled": len(settled),
            "unsettled": unsettled,
            "hit_rate": (
                sum(1 for e in settled if e.correct) / len(settled)
                if settled else 0.5
            ),
            "avg_brier": (
                sum(e.brier_residual for e in settled if e.brier_residual)
                / max(1, sum(1 for e in settled if e.brier_residual))
                if settled else 0.25
            ),
            "platt_params": self._calibrator.to_dict(),
            "last_updated": (
                self._entries[-1].timestamp.isoformat() if self._entries else None
            ),
        }

    def export_chain_hash(self) -> str:
        """
        Compute SHA-256 chain hash over all entries for cryptographic audit.
        """
        chain = ""
        for entry in self._entries:
            chain += entry.model_dump_json()
        return hashlib.sha256(chain.encode()).hexdigest()
