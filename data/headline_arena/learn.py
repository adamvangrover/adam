#!/usr/bin/env python3
"""
Continuous-learning loop: settle -> ledger -> calibration.

Deterministic, idempotent. Reads any *_dispatch_results.json, fetches settled
outcomes, appends new rows to data/memory/settlement_ledger.jsonl (dedup by
prediction_id), and writes data/memory/calibration.json with per-asset stats and
suggested confidence caps / neutral priors for the next HITL round.

Usage:
    python3 -m headline_arena.learn data/memory/round6_live_dispatch_results.json
    python3 -m headline_arena.learn --report          # recompute from ledger only
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://headlinearena.com"
MEM = Path(__file__).parent.parent / "data" / "memory"
LEDGER = MEM / "settlement_ledger.jsonl"
CALIB = MEM / "calibration.json"
H = {"Accept-Encoding": "gzip, deflate"}  # Cloudflare zstd breaks requests.json()

S = requests.Session()
S.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset(["GET"]))))


def brier(direction: str, conf: float, result: str) -> float:
    """Ternary Brier: p(pred)=conf, remaining mass split evenly."""
    probs = {d: (1 - conf) / 2 for d in ("bullish", "bearish", "neutral")}
    probs[direction] = conf
    return sum((p - (1.0 if d == result else 0.0)) ** 2 for d, p in probs.items())


def load_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def ingest(results_file: Path) -> int:
    seen = {r["prediction_id"] for r in load_ledger()}
    rows, pending = [], 0
    for r in json.loads(results_file.read_text()).get("results", []):
        pid = (r.get("response") or {}).get("prediction_id")
        if not pid or pid in seen:
            continue
        c = S.get(f"{BASE_URL}/api/v1/eval/challenges/{r['challenge_id']}", headers=H, timeout=30).json()
        if c.get("status") != "resolved" or not c.get("result"):
            pending += 1
            continue
        op, cp = c.get("open_price"), c.get("close_price")
        rows.append({
            "prediction_id": pid, "challenge_id": r["challenge_id"], "event_id": c.get("event_id"),
            "asset": r["asset"], "direction": r["direction"], "confidence": r["confidence"],
            "result": c["result"], "hit": r["direction"] == c["result"],
            "open": op, "close": cp,
            "pct": round((cp - op) / op * 100, 3) if op and cp else None,
            "dead_zone_pct": c.get("dead_zone_pct"),
            "crowd": {k: c.get(f"{k}_count") for k in ("bullish", "bearish", "neutral")},
            "brier": round(brier(r["direction"], r["confidence"], c["result"]), 4),
        })
    with LEDGER.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Ingested {len(rows)} settled rows ({pending} still pending) -> {LEDGER.name}")
    return len(rows)


def calibrate() -> dict:
    led = load_ledger()
    if not led:
        print("Ledger empty.")
        return {}
    n, hits = len(led), sum(r["hit"] for r in led)
    g_acc = hits / n
    by = defaultdict(list)
    for r in led:
        by[r["asset"]].append(r)

    K = 4  # shrinkage strength toward global rate (beta prior pseudo-counts)
    assets = {}
    for a, rs in sorted(by.items()):
        k = len(rs)
        acc = sum(r["hit"] for r in rs) / k
        shrunk = (sum(r["hit"] for r in rs) + K * g_acc) / (k + K)
        res = defaultdict(int)
        for r in rs:
            res[r["result"]] += 1
        assets[a] = {
            "n": k, "hit_rate": round(acc, 3),
            "avg_conf": round(sum(r["confidence"] for r in rs) / k, 3),
            "overconfidence": round(sum(r["confidence"] for r in rs) / k - acc, 3),
            "mean_brier": round(sum(r["brier"] for r in rs) / k, 4),
            "result_freq": {d: round(res[d] / k, 3) for d in ("bullish", "bearish", "neutral")},
            "last_dead_zone_pct": rs[-1].get("dead_zone_pct"),
            # Suggestions for next round (operator reviews before applying)
            "suggested_conf_cap": round(min(0.72, max(0.50, shrunk + 0.10)), 2),
            "suggested_neutral_prior": round((res["neutral"] + 1) / (k + 3), 3),
        }
    out = {
        "n": n, "global_hit_rate": round(g_acc, 3),
        "global_avg_conf": round(sum(r["confidence"] for r in led) / n, 3),
        "global_mean_brier": round(sum(r["brier"] for r in led) / n, 4),
        "assets": assets,
    }
    CALIB.write_text(json.dumps(out, indent=2))
    print(f"Global: n={n} hit={g_acc:.1%} avg_conf={out['global_avg_conf']} brier={out['global_mean_brier']}")
    print(f"{'Asset':5} {'n':>3} {'hit':>6} {'conf':>5} {'over':>6} {'neutral%':>8} {'cap':>5}")
    for a, s in assets.items():
        print(f"{a:5} {s['n']:3} {s['hit_rate']:6.1%} {s['avg_conf']:5.2f} {s['overconfidence']:+6.2f} "
              f"{s['result_freq']['neutral']:8.0%} {s['suggested_conf_cap']:5.2f}")
    print(f"-> {CALIB}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="*", type=Path)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    for f in a.results:
        ingest(f)
    calibrate()


if __name__ == "__main__":
    main()
