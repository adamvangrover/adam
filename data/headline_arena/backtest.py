#!/usr/bin/env python3
"""
Walk-forward backtest + shadow runs for flow / band-rule / track-record variants.

Uses the production probability core (MultiAgentDeliberationEngine.simulate_probabilities)
and confidence mapper (confidence.map_confidence) with variant params.
Flow is derived deterministically from the prior 3 settlements (no hand-set inputs).
Track record is walk-forward: a challenge on date d only sees predictions dated < d.

Usage:
    python3 -m headline_arena.backtest [--refresh]       # backtest all variants
    python3 -m headline_arena.backtest --shadow          # log variants on open challenges
    python3 -m headline_arena.backtest --score-shadow    # score logged shadows vs settlements
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .analytics import ASSET_UNIVERSE, AnalyticalUpgradePipeline
from .confidence import choose_direction, map_confidence
from .deliberation import MultiAgentDeliberationEngine
from .learn import brier

BASE = "https://headlinearena.com/api/v1/eval"
MEM = Path(__file__).parent.parent / "data" / "memory"
CACHE, RESULTS, SHADOW = MEM / "resolved_cache.json", MEM / "backtest_results.json", MEM / "shadow_runs.jsonl"
H = {"Accept-Encoding": "gzip, deflate"}
S = requests.Session()
S.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset(["GET"]))))

LOOKBACK = 3
FLOWS = [("zero", 0.0), ("mom", 0.4), ("rev", 0.4), ("rev", 0.8)]
BANDS = ["absolute"]                       # v1: abs vs rel showed no material difference
TRACKS = {"cal_K10_r1.0": (10, 1.0)}       # v1 winner
NPRIORS = ["static", "dyn20"]               # ticker table vs empirical neutral rate (last 20)
NMARGINS = [0.0, 0.05, 0.10]                # call neutral if within margin of top outcome
VARIANTS = [dict(flow=f, band=b, track=t, nprior=n, nmargin=m)
            for f, b, t, n, m in itertools.product(FLOWS, BANDS, TRACKS, NPRIORS, NMARGINS)]


def vname(v):
    return (f"{v['flow'][0]}{v['flow'][1]:g}|{v['band'][:3]}|{v['track']}"
            f"|{v.get('nprior', 'static')}|nm{v.get('nmargin', 0.0):g}")


def neutral_prior_for(prior: list[dict], n: int = 20):
    """Smoothed empirical neutral rate over the last n settlements (walk-forward safe)."""
    rs = prior[-n:]
    if len(rs) < 5:
        return None
    return round((sum(c["result"] == "neutral" for c in rs) + 1) / (len(rs) + 3), 3)


# ── data ────────────────────────────────────────────────────────────────
def fetch_history(refresh: bool = False) -> list[dict]:
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    out, off = {}, 0
    while True:
        j = S.get(f"{BASE}/challenges?status=resolved&limit=100&offset={off}", headers=H, timeout=60).json()
        items = [i.get("challenge", i) for i in j.get("items", [])]
        for c in items:
            if c.get("challenge_type") == "daily" and c.get("result"):
                out[c["id"]] = {k: c.get(k) for k in ("id", "event_id", "asset", "open_price",
                                "close_price", "result", "dead_zone_pct")}
        if len(items) < 100:
            break
        off += 100
    rows = sorted(out.values(), key=lambda c: (c["event_id"][-8:], c["asset"]))
    CACHE.write_text(json.dumps(rows))
    print(f"Cached {len(rows)} resolved daily challenges")
    return rows


def pct(c):
    o, cl = c.get("open_price"), c.get("close_price")
    return (cl - o) / o * 100 if o and cl else 0.0


def series(rows):
    """Per-asset chronological series. Skips non-dated event_ids (legacy UUID-format
    challenges sort incorrectly and carry stale prices)."""
    s = defaultdict(list)
    for c in rows:
        if (c.get("event_id") or "")[-8:].isdigit():
            s[c["asset"]].append(c)
    return s


def flow_for(prior: list[dict], cfg) -> float:
    mode, scale = cfg
    if mode == "zero" or not prior:
        return 0.0
    sig = math.tanh(sum(pct(c) for c in prior[-LOOKBACK:]) / len(prior[-LOOKBACK:]))
    return round((1 if mode == "mom" else -1) * scale * sig, 4)


# ── core ────────────────────────────────────────────────────────────────
ENGINE = MultiAgentDeliberationEngine(AnalyticalUpgradePipeline())


def probs_for(cid, asset, open_px, dz, flow, npi=None):
    base = ASSET_UNIVERSE[asset]
    prof = dataclasses.replace(base, current_indicated_price=open_px or base.current_indicated_price,
                               last_settlement_price=open_px or base.last_settlement_price,
                               dead_zone_pct=dz, order_flow_imbalance=flow, neutral_prior=npi)
    core = ENGINE.simulate_probabilities(cid, prof)
    p = {k: core[k] for k in ("bull", "neutral", "bear")}
    return p, core["median_move"]


class Track:
    """Walk-forward per-asset stats mirroring learn.calibrate()."""
    def __init__(self, K):
        self.K, self.rows = K, defaultdict(list)

    def calib(self, asset):
        rs = self.rows[asset]
        allr = [r for v in self.rows.values() for r in v]
        if len(rs) < 2 or not allr:
            return None
        g = sum(h for h, _ in allr) / len(allr)
        hits = sum(h for h, _ in rs)
        acc, avgc = hits / len(rs), sum(c for _, c in rs) / len(rs)
        shrunk = (hits + self.K * g) / (len(rs) + self.K)
        return {"n": len(rs), "hit_rate": acc, "overconfidence": avgc - acc,
                "suggested_conf_cap": min(0.72, max(0.50, shrunk + 0.10))}

    def add(self, asset, hit, conf):
        self.rows[asset].append((hit, conf))


def predict(v, c, prior, track, pcache, use_file=False):
    flow = flow_for(prior, v["flow"])
    npi = neutral_prior_for(prior) if v.get("nprior") == "dyn20" else None
    key = (c["id"], flow, npi)
    if key not in pcache:
        pcache[key] = probs_for(c["id"], c["asset"], c.get("open_price"), c["dead_zone_pct"] or 0.3, flow, npi)
    p, mm = pcache[key]
    d = choose_direction(p, v.get("nmargin", 0.0))
    K = TRACKS[v["track"]]
    if K is None:
        mode, override = "fixed", None
    elif use_file:
        mode, override = "auto", None               # live calibration.json (shadow)
    else:
        mode, override = "auto", (track.calib(c["asset"]) or {})  # walk-forward only
    dec = map_confidence(c["asset"], d, p, mm, c["dead_zone_pct"] or 0.3, mode=mode,
                         params={"band_rule": v["band"], "rel_max": K[1] if K else 1.2},
                         calib_override=override)
    return d, dec.confidence, dec.tier, flow


# ── backtest ────────────────────────────────────────────────────────────
def backtest(rows):
    ser = series(rows)
    idx = {c["id"]: i for a in ser.values() for i, c in enumerate(ser[a[0]["asset"]])}
    dates = sorted({c["event_id"][-8:] for c in rows})
    pcache, res = {}, {}
    for v in VARIANTS:
        K = TRACKS[v["track"]]
        track = Track(K[0]) if K else None
        preds = []
        for d8 in dates:
            day = []
            for c in (x for x in rows if x["event_id"][-8:] == d8 and x["asset"] in ASSET_UNIVERSE):
                i = idx.get(c["id"])
                if i is None or i < LOOKBACK:
                    continue
                dirn, conf, tier, flow = predict(v, c, ser[c["asset"]][:i], track, pcache)
                day.append((c, dirn, conf, tier))
            for c, dirn, conf, tier in day:  # update only after the whole day (no same-day leakage)
                hit = dirn == c["result"]
                preds.append((hit, conf, tier, dirn, brier(dirn, conf, c["result"])))
                if track:
                    track.add(c["asset"], hit, conf)
        res[vname(v)] = metrics(preds)
    # naive baselines for context
    base = {"always_neutral": [], "repeat_last": []}
    for a, s in ser.items():
        for i in range(LOOKBACK, len(s)):
            base["always_neutral"].append(s[i]["result"] == "neutral")
            base["repeat_last"].append(s[i]["result"] == s[i - 1]["result"])
    return res, {k: round(sum(v) / len(v), 3) for k, v in base.items() if v}


def metrics(preds):
    n = len(preds)
    if not n:
        return {}
    hits = [p for p in preds if p[0]]
    miss = [p for p in preds if not p[0]]
    sel = [p for p in preds if p[1] >= 0.53]
    tiers = defaultdict(list)
    for p in preds:
        tiers[p[2]].append(p[0])
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {
        "n": n, "hit": round(len(hits) / n, 3), "brier": round(mean([p[4] for p in preds]), 4),
        "avg_conf": round(mean([p[1] for p in preds]), 3),
        "discrim": round(mean([p[1] for p in hits]) - mean([p[1] for p in miss]), 4),
        "selective_share": round(len(sel) / n, 3),
        "hit_selective": round(mean([p[0] for p in sel]), 3) if sel else None,
        "neutral_calls": round(sum(p[3] == "neutral" for p in preds) / n, 3),
        "tier_hit": {t: [len(v), round(mean(v), 3)] for t, v in sorted(tiers.items())},
    }


# ── shadow ──────────────────────────────────────────────────────────────
def shadow(rows):
    ser = series(rows)
    act = S.get(f"{BASE}/challenges/active", headers=H, timeout=60).json().get("challenges", [])
    tag, n = datetime.now(timezone.utc).isoformat(timespec="seconds"), 0
    with SHADOW.open("a") as f:
        for it in act:
            c = it.get("challenge", it)
            if c.get("asset") not in ASSET_UNIVERSE:
                continue
            prior = ser.get(c["asset"], [])
            for v in VARIANTS:
                dirn, conf, tier, flow = predict(v, c, prior, None, {}, use_file=True)
                f.write(json.dumps({"tag": tag, "variant": vname(v), "challenge_id": c["id"],
                                    "asset": c["asset"], "direction": dirn, "confidence": conf,
                                    "tier": tier, "flow": flow}) + "\n")
                n += 1
    print(f"Logged {n} shadow predictions ({len(act)} challenges x {len(VARIANTS)} variants) -> {SHADOW.name}")


def score_shadow(rows):
    res = {c["id"]: c["result"] for c in rows}
    agg = defaultdict(list)
    for l in SHADOW.read_text().splitlines():
        r = json.loads(l)
        if r["challenge_id"] in res:
            out = res[r["challenge_id"]]
            agg[r["variant"]].append((r["direction"] == out, r["confidence"], r["tier"], r["direction"],
                                      brier(r["direction"], r["confidence"], out)))
    table({k: metrics(v) for k, v in agg.items()}, "SHADOW SCORE")


def table(res, title, top=None):
    print(f"\n{title}  (sorted by Brier, lower=better)")
    print(f"{'variant':28} {'n':>4} {'hit':>6} {'brier':>7} {'conf':>5} {'discrim':>8} {'sel%':>5} {'hit_sel':>7} {'neu%':>5}")
    for k, m in sorted(res.items(), key=lambda kv: kv[1].get("brier", 9))[:top]:
        hs = f"{m['hit_selective']:.1%}" if m.get("hit_selective") is not None else "-"
        print(f"{k:28} {m['n']:4} {m['hit']:6.1%} {m['brier']:7.4f} {m['avg_conf']:5.2f} "
              f"{m['discrim']:+8.4f} {m['selective_share']:5.0%} {hs:>7} {m['neutral_calls']:5.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--shadow", action="store_true")
    ap.add_argument("--score-shadow", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    rows = fetch_history(a.refresh or a.score_shadow)
    if a.shadow:
        return shadow(rows)
    if a.score_shadow:
        return score_shadow(rows)
    res, base = backtest(rows)
    RESULTS.write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(),
                                   "n_challenges": len(rows), "baselines": base, "variants": res}, indent=2))
    print(f"History: {len(rows)} resolved daily challenges | baselines: {base}")
    table(res, "BACKTEST", a.top)
    print(f"-> {RESULTS}")


if __name__ == "__main__":
    main()
