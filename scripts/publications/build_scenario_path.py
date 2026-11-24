"""Build the committed synthetic market path ``data/scenarios/2026H2_path.json``.

Seeded, deterministic Brownian-bridge walk per series (config/scenarios/2026H2_path.yaml): between
consecutive anchors the walk is pinned to the anchor level, so the path agrees with the levels quoted
in the existing Market Mayhem narratives. Output is ``_estimated: true`` end to end (docs/PROVENANCE.md).

    python scripts/publications/build_scenario_path.py [--config ...] [--out ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "scenarios" / "2026H2_path.yaml"
DEFAULT_OUT = ROOT / "data" / "scenarios" / "2026H2_path.json"
VERSION = "1.0.0"


def business_days(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _bridge(rng: random.Random, n: int, vol: float, delta: float | None) -> list[float]:
    """n cumulative increments; when ``delta`` is given the walk is pinned to end exactly at it."""
    walk, acc = [], 0.0
    for _ in range(n):
        acc += rng.gauss(0.0, vol)
        walk.append(acc)
    if delta is None or n == 0:
        return walk
    return [w - (i + 1) / n * (walk[-1] - delta) for i, w in enumerate(walk)]


def build_series(rng: random.Random, days: list[date], spec: dict) -> list[float]:
    log = spec["mode"] == "log"
    to_x = math.log if log else float
    anchors = {date.fromisoformat(k): float(v) for k, v in (spec.get("anchors") or {}).items()}
    idx = {d: i for i, d in enumerate(days)}
    pins = sorted((idx[d], to_x(v)) for d, v in anchors.items())
    xs = [to_x(spec["start"])]
    last_i = 0
    for pin_i, pin_x in [*pins, (len(days) - 1, None)]:
        if pin_i <= last_i:
            continue
        steps = _bridge(rng, pin_i - last_i, spec["daily_vol"], None if pin_x is None else pin_x - xs[-1])
        base = xs[-1]
        xs.extend(base + s for s in steps)
        last_i = pin_i
    values = [math.exp(x) if log else x for x in xs]
    return [round(v, spec["decimals"]) for v in values]


def build(config: dict, config_path: str) -> dict:
    rng = random.Random(config["seed"])
    days = business_days(date.fromisoformat(config["start"]), date.fromisoformat(config["end"]))
    series = {name: {"label": s["label"], "unit": s["unit"], "values": build_series(rng, days, s),
                     "anchors": s.get("anchors") or {}} for name, s in config["series"].items()}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return {
        "_estimated": True,
        "label": "Illustrative scenario — not investment advice. Synthetic path, not market data.",
        "seed": config["seed"],
        "dates": [d.isoformat() for d in days],
        "series": series,
        "provenance_trace": [{"generator": "scripts/publications/build_scenario_path.py", "inputs": [config_path],
                              "timestamp": f"{config['end']}T00:00:00Z", "version": VERSION, "content_hash": digest}],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    cfg_path = args.config.resolve()
    rel = cfg_path.relative_to(ROOT).as_posix() if cfg_path.is_relative_to(ROOT) else str(cfg_path)
    path = build(yaml.safe_load(cfg_path.read_text()), rel)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(path, indent=1) + "\n")
    print(f"wrote {args.out} ({len(path['dates'])} business days, {len(path['series'])} series)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
