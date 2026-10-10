from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True)
class InvariantResult:
    name: str
    passed: bool
    reason: str = ""

@dataclass(frozen=True)
class GateResult:
    passed: bool
    invariants: tuple[InvariantResult, ...]

    @property
    def failures(self) -> tuple[InvariantResult, ...]:
        return tuple(item for item in self.invariants if not item.passed)

def enforce_invariants(
    results: Mapping[str, bool],
    reasons: Mapping[str, str] | None = None,
) -> GateResult:
    reasons = reasons or {}

    normalized = tuple(
        InvariantResult(
            name=name,
            passed=bool(passed),
            reason=reasons.get(name, ""),
        )
        for name, passed in sorted(results.items())
    )

    # Fail closed. There is intentionally no "score" argument here.
    return GateResult(
        passed=all(item.passed for item in normalized),
        invariants=normalized,
    )
