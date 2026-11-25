"""Accuracy runner for routing/entity and publication-review eval datasets.

    python tests/evals/run_accuracy.py            # prints per-dataset accuracy + misses
"""
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(__file__).parent / "datasets"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.engine.meta_orchestrator import MetaOrchestrator, OrchestratorConfig  # noqa: E402  (bind before test stubs)


def _orchestrator():
    orch = MetaOrchestrator.__new__(MetaOrchestrator)  # routing/resolution only need config
    orch.config = OrchestratorConfig()
    return orch


def _publication_flags(case: Dict[str, Any]) -> List[str]:
    from core.publication.review import review_flags

    return sorted(f["code"] for f in review_flags(case["edition"], case.get("prior")))


def predictors() -> Dict[str, Callable[[Any], Any]]:
    orch = _orchestrator()
    return {
        "query_route": lambda q: orch._assess_complexity(q, {}).value,
        "query_entity": lambda q: orch.resolve_entity(q, {}),
        "publication_review": _publication_flags,
    }


def evaluate(name: str) -> Tuple[float, float, List[Dict[str, Any]]]:
    data = json.loads((DATASETS / f"{name}.json").read_text())
    predict = predictors()[name]
    misses = [
        {**c, "got": got} for c in data["cases"] if (got := predict(c["query"])) != c["expected"]
    ]
    return 1 - len(misses) / len(data["cases"]), data["min_accuracy"], misses


def main() -> int:
    failed = False
    for name in predictors():
        acc, floor, misses = evaluate(name)
        failed |= acc < floor
        print(f"{name}: {acc:.1%} (floor {floor:.0%})")
        for m in misses:
            print(f"  MISS {m.get('name', m['query'])!r}: expected {m['expected']!r}, got {m['got']!r}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
