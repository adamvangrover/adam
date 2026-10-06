#!/usr/bin/env python3
"""
Submit operator-approved staged forecasts verbatim from hitl_staged_forecasts.json.

Avoids re-running the Monte Carlo (seeded via Python's randomized str hash),
so the dispatched payloads are byte-identical to what the operator reviewed.

Usage:
    python3 -m headline_arena.submit_staged --confirm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .dispatch import BASE_URL, STAGED_DIR, get_fresh_token, load_creds

HEADERS_BASE = {"Accept-Encoding": "gzip, deflate"}

# GETs are idempotent -> retry with backoff. POSTs are NOT retried automatically.
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="Operator DISPATCH_CONFIRMED")
    ap.add_argument("--out", default="round6_live_dispatch_results.json")
    args = ap.parse_args()

    staged = json.loads((STAGED_DIR / "hitl_staged_forecasts.json").read_text())
    print(f"Loaded {len(staged)} staged forecasts.")
    if not args.confirm:
        print("Dry run (no --confirm). Nothing submitted.")
        return 0

    creds = load_creds()
    token = get_fresh_token(creds)
    if not token:
        print("No access token available.", file=sys.stderr)
        return 1
    headers = {**HEADERS_BASE, "Authorization": f"Bearer {token}"}

    results, ok = [], 0
    now = datetime.now(timezone.utc)
    for fc in staged:
        cid, asset = fc["challenge_id"], fc["asset"]
        payload = dict(fc["raw_api_payload"])

        # Pre-flight: challenge still open and before deadline
        try:
            ch = SESSION.get(f"{BASE_URL}/api/v1/eval/challenges/{cid}", headers=HEADERS_BASE, timeout=30)
            cj = ch.json() if ch.status_code == 200 else {}
        except requests.RequestException as e:
            print(f"  ✗ {asset:4} [{cid[:8]}…] pre-flight failed: {e.__class__.__name__}")
            results.append({"challenge_id": cid, "asset": asset, "skipped": True, "status": "preflight_error"})
            continue
        status, deadline = cj.get("status"), cj.get("deadline")
        if status != "open" or (deadline and datetime.fromisoformat(deadline.replace("Z", "+00:00")) <= now):
            print(f"  ✗ {asset:4} [{cid[:8]}…] SKIPPED (status={status}, deadline={deadline})")
            results.append({"challenge_id": cid, "asset": asset, "skipped": True, "status": status})
            continue

        try:
            res = SESSION.post(f"{BASE_URL}/api/v1/eval/challenges/{cid}/predict",
                               json=payload, headers=headers, timeout=45)
        except requests.RequestException as e:
            print(f"  ? {asset:4} [{cid[:8]}…] POST {e.__class__.__name__} — outcome UNKNOWN, not retried")
            results.append({"challenge_id": cid, "asset": asset, "status_code": None,
                            "unknown": True, "error": e.__class__.__name__})
            continue
        data = res.json() if res.status_code in (200, 201) else {"error": res.text[:300]}
        if res.status_code in (200, 201):
            ok += 1
            print(f"  ✓ {asset:4} [{cid[:8]}…] → {payload['direction']} ({payload['confidence']:.2f}) "
                  f"scored={data.get('counts_for_score')} pred_id={str(data.get('prediction_id', ''))[:8]}…")
        else:
            print(f"  ✗ {asset:4} [{cid[:8]}…] HTTP {res.status_code}: {str(data)[:120]}")
        results.append({
            "challenge_id": cid, "asset": asset,
            "direction": payload["direction"], "confidence": payload["confidence"],
            "prov_o_hash": fc["validation"]["prov_o_hash"],
            "status_code": res.status_code,
            "scored": data.get("counts_for_score", False),
            "response": data,
        })
        time.sleep(0.3)

    record = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "total_dispatched": len(results), "success_count": ok, "results": results}
    out = STAGED_DIR / args.out
    out.write_text(json.dumps(record, indent=2))
    print(f"\nDispatched {ok}/{len(staged)}. Audit log: {out}")
    return 0 if ok == len(staged) else 2


if __name__ == "__main__":
    sys.exit(main())
