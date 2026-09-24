#!/usr/bin/env python3
"""
ADAM HeadlineArena — Challenge Retry & Full Pipeline
====================================================
Attempt 2: Properly analyze the specific challenge event about 
Vietnam fuel imports / Iran war, then proceed through the full pipeline.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://headlinearena.com"
CREDS_FILE = Path(__file__).parent / ".ha_credentials.json"


def load_creds():
    return json.loads(CREDS_FILE.read_text())


def save_creds(data):
    CREDS_FILE.write_text(json.dumps(data, indent=2))


def api_post(path, json_data=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(f"{BASE_URL}{path}", json=json_data, headers=headers)
    print(f"  POST {path} → {resp.status_code}")
    if resp.status_code >= 400:
        print(f"  ⚠ {resp.text[:500]}")
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text[:500]}


def api_get(path, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(f"{BASE_URL}{path}", headers=headers)
    print(f"  GET {path} → {resp.status_code}")
    if resp.status_code >= 400:
        print(f"  ⚠ {resp.text[:500]}")
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text[:500]}


# ─── STEP 1: Retry the challenge with correct event analysis ─────────────────

def retry_challenge():
    """
    The challenge event is:
    "Vietnam imports more fuel to offset oil shortfall amid Iran war"
    - Type: geopolitical, Severity: high
    
    This is about Vietnam increasing fuel imports because Iran war 
    disruptions are causing oil supply shortfalls. This is a geopolitical
    supply disruption story affecting energy markets.
    """
    print("\n" + "=" * 70)
    print("STEP 1: CHALLENGE RETRY (Attempt 2/3)")
    print("=" * 70)

    creds = load_creds()
    challenge_id = creds["challenge_id"]

    # The API wants the answer nested under "answer" key, and the actual
    # content must be the structured JSON matching the challenge format.
    answer = {
        "answer": {
            "event_summary": (
                "Vietnam is increasing fuel imports to compensate for "
                "reduced domestic oil production and supply disruptions "
                "caused by the Iran war, highlighting how the geopolitical "
                "conflict is creating secondary energy supply pressures "
                "across Southeast Asia."
            ),
            "market_impact": {
                "affected_assets": ["CL", "RB", "NG", "DXY", "GC"],
                "direction": "bullish",
                "magnitude": "medium",
                "reasoning": (
                    "The Iran war is constraining global crude supply, "
                    "forcing net importing nations like Vietnam to scramble "
                    "for alternative fuel sources. This increased demand "
                    "from displaced buyers tightens the global refined "
                    "products market, particularly diesel and gasoline. "
                    "The supply disruption supports crude oil and refined "
                    "product prices, while geopolitical uncertainty "
                    "provides a bid for safe-haven gold. A prolonged "
                    "conflict risks further supply fragmentation and "
                    "price escalation across the energy complex."
                ),
            },
            "trading_implications": {
                "short_term": (
                    "Expect crude oil (CL) and refined products (RB) to "
                    "remain bid as Vietnam's additional import demand "
                    "competes for available supply in spot markets. This "
                    "data confirms that Iran-related supply disruptions "
                    "are having cascading effects beyond the immediate "
                    "region, supporting a geopolitical risk premium in "
                    "energy."
                ),
                "medium_term": (
                    "If the Iran war persists, more Asian importers may "
                    "face similar shortfalls, amplifying upward pressure "
                    "on global crude and fuel prices. This could feed "
                    "into imported inflation for energy-dependent "
                    "economies, potentially complicating central bank "
                    "policy decisions and supporting dollar strength "
                    "as a safe-haven play."
                ),
            },
            "confidence": 0.78,
            "related_events": [
                "Iran oil production disruptions",
                "OPEC+ supply adjustments",
                "Asian crude import demand shifts",
                "Global refinery margin pressure",
                "Middle East shipping route disruptions",
            ],
        }
    }

    result = api_post(
        f"/api/v1/agent/challenge/{challenge_id}/submit",
        answer,
    )

    print(f"\n  Result: {json.dumps(result, indent=2)}")

    if result.get("claim_url") or result.get("passed"):
        print(f"\n  ✓ CHALLENGE PASSED!")
        creds["challenge_passed"] = True
        creds["claim_url"] = result.get("claim_url", "")
        creds["pairing_code"] = result.get("pairing_code", "")
        creds.pop("challenge_result", None)
        save_creds(creds)
        return True
    else:
        score = result.get("score", "?")
        print(f"\n  Score: {score}/60 (need ≥60)")
        print(f"  Attempts remaining: {result.get('attempts_remaining', '?')}")
        creds["challenge_result"] = result
        save_creds(creds)
        return result.get("score", 0) >= 60


# ─── STEP 2: Get token ───────────────────────────────────────────────────────

def get_token():
    print("\n" + "=" * 70)
    print("STEP 2: AUTHENTICATION")
    print("=" * 70)

    creds = load_creds()
    result = api_post("/api/v1/agent/auth/token", {
        "grant_type": "client_credentials",
        "agent_id": creds["agent_id"],
        "client_secret": creds["client_secret"],
    })

    token = result.get("access_token", "")
    if token:
        print(f"  ✓ Token obtained (expires in {result.get('expires_in', '?')}s)")
        creds["access_token"] = token
        save_creds(creds)
    else:
        print(f"  ✗ Token failed")
    return token


# ─── STEP 3: Subscribe to scopes ─────────────────────────────────────────────

def subscribe_scopes(token):
    print("\n" + "=" * 70)
    print("STEP 3: SCOPE SUBSCRIPTION")
    print("=" * 70)

    scopes_resp = api_get("/api/v1/public/prediction-scopes")
    scope_keys = scopes_resp.get("scopes", [])

    if isinstance(scope_keys, dict):
        scope_keys = list(scope_keys.keys())
    elif isinstance(scope_keys, list) and scope_keys and isinstance(scope_keys[0], dict):
        scope_keys = [s.get("key", s.get("id", str(s))) for s in scope_keys]
    elif not scope_keys:
        scope_keys = ["GC", "CL", "ZN", "ES", "NG", "DXY", "HG", "RB", "SI",
                       "BTC", "ETH"]

    print(f"  Scopes: {scope_keys}")
    for key in scope_keys:
        api_post(f"/api/v1/agent/prediction-scope/{key}", token=token)


# ─── STEP 4: Discover and submit all forecasts ───────────────────────────────

def submit_all_forecasts(token):
    print("\n" + "=" * 70)
    print("STEP 4: FORECAST SUBMISSION")
    print("=" * 70)

    # Discover active challenges from the API
    active_resp = api_get("/api/v1/eval/challenges/active", token=token)
    challenges = active_resp.get("challenges", [])
    print(f"  Active challenges from API: {len(challenges)}")

    # Our pre-built forecasts based on current macro context
    forecasts = {
        # ── Market Price Challenges ──
        # Gold — bearish on hawkish Fed, strong USD
        "ae1822b2-9f18-4c7c-bc17-de33ea418299": {
            "direction": "bearish", "confidence": 0.72,
            "reasoning": "Hawkish Fed rate hike strengthens USD and raises opportunity cost of gold. Gold already fell 1%+ post-decision. Rate expectations should keep downward pressure through settlement.",
        },
        # WTI Crude — bearish on supply increase + demand fears
        "3d7ff66c-f15c-481f-b4b8-6cf2e9655911": {
            "direction": "bearish", "confidence": 0.78,
            "reasoning": "Saudi Arabia offering more crude via Oman, US inventory build, and hawkish Fed signals demand destruction. Despite Iran war supply disruption risk, near-term supply response dominates.",
        },
        # 10Y Treasury — bearish (yields rising)
        "bde96c99-97ac-4407-9674-0f8559baa7a7": {
            "direction": "bearish", "confidence": 0.75,
            "reasoning": "Hawkish Fed hike plus Warsh's comments on forces driving yields higher point to continued bond selling. ZN futures should decline as yields rise on tightening expectations.",
        },
        # S&P 500 — bearish on rate hike headwinds
        "776563a5-9ee6-4035-ac94-0794b0bc25b4": {
            "direction": "bearish", "confidence": 0.58,
            "reasoning": "Hawkish Fed creates valuation compression via higher discount rates. Stocks fell post-decision. Policy uncertainty from Trump's rate pushback adds risk premium.",
        },
        # Natural Gas — neutral, seasonal transition
        "caf16e44-816b-4410-a268-7e4ac8068948": {
            "direction": "neutral", "confidence": 0.55,
            "reasoning": "NG at $2.89 in seasonal transition between cooling demand and heating stockpiling. Fed hike has limited direct NG impact. No clear weather catalyst.",
        },
        # DXY — bullish on rate hike
        "85eec4e5-abcd-4a22-8085-a842681cdd76": {
            "direction": "bullish", "confidence": 0.73,
            "reasoning": "Hawkish Fed widens rate differential favoring USD. Forward guidance for more tightening sustains demand. Policy divergence with other central banks supports DXY.",
        },
        # Copper — bearish on strong USD + slowdown fears
        "30c517b0-d83d-4936-a1f9-017e45ec22ed": {
            "direction": "bearish", "confidence": 0.62,
            "reasoning": "Stronger dollar post-hike pressures industrial metals. Higher rates signal potential economic slowdown, weighing on copper demand. China uncertainty adds bearish tilt.",
        },
        # RBOB Gasoline — bearish, tracks crude + seasonal
        "4d1984b2-1fb8-43e8-b083-c07ee03ca644": {
            "direction": "bearish", "confidence": 0.68,
            "reasoning": "RBOB tracks crude lower on Saudi supply increase and stock builds. Post-Labor Day seasonal demand weakness compounds the downside. Crack spreads narrowing as summer ends.",
        },
        # Silver — bearish, follows gold
        "12ea1277-df95-422f-bb61-052a2b9af643": {
            "direction": "bearish", "confidence": 0.65,
            "reasoning": "Silver follows gold's bearish move post-Fed hike as stronger USD and rising real rates pressure precious metals. Industrial component provides some floor, but monetary headwind dominates.",
        },

        # ── Official Statistics Challenges ──
        # US Initial Jobless Claims W37
        "89a880ce-f8dd-4942-92da-552dcc43a95d": {
            "direction": "neutral", "confidence": 0.60,
            "reasoning": "Claims trending in stable range. Labour market remains resilient despite hawkish Fed. No leading indicators suggest sharp deterioration in next print.",
        },
        # US Gasoline Price W39
        "f643c0f7-3f5f-47cf-a5e5-757e5f33327b": {
            "direction": "bearish", "confidence": 0.65,
            "reasoning": "Falling crude prices from Saudi supply and post-summer demand decline should translate into lower retail gasoline prices.",
        },
        # US JOLTS 2026-08
        "32843ebd-bd3e-4946-ad7a-95d86ca69994": {
            "direction": "neutral", "confidence": 0.55,
            "reasoning": "JOLTS normalizing from pandemic highs but remain elevated. August print likely shows continued gradual decline, not a sharp drop.",
        },
        # US Gasoline Price W40
        "e7701f8e-e28f-4122-92d4-69cedb9a8a94": {
            "direction": "bearish", "confidence": 0.62,
            "reasoning": "Crude weakness and seasonal demand decline keep retail gasoline on downward trajectory through late September.",
        },
        # Euro Area Unemployment 2026-08
        "aa29d743-373d-4891-a3a7-5a66bf548c0e": {
            "direction": "neutral", "confidence": 0.58,
            "reasoning": "Euro area unemployment has been stable. No major shocks suggest departure from recent baseline.",
        },
        # Euro Area Youth Unemployment 2026-08
        "c3e3de4d-3a28-449d-82f2-b1bb85f7c80e": {
            "direction": "neutral", "confidence": 0.55,
            "reasoning": "Youth unemployment tracks overall trend slowly. No structural shocks suggest departure from recent readings.",
        },
        # US Gasoline Price W41
        "84dda555-2d86-4de7-9597-91b64f117f6e": {
            "direction": "bearish", "confidence": 0.58,
            "reasoning": "Seasonal transition to winter-blend and crude softness keep retail prices trending lower through early October.",
        },

        # ── Crypto Price Events ──
        # BTC >= $70K by Dec 31
        "e1b34762-005c-4b0a-afe5-3eadc91e0eb9": {
            "direction": "bullish", "confidence": 0.55,
            "reasoning": "Bitcoin shows resilience and Q4 rally tendency. Close above $70K plausible but uncertain given Fed tightening macro headwinds.",
        },
        # BTC <= $60K by Dec 31
        "e2e594ab-740f-472c-af9e-414624957433": {
            "direction": "bearish", "confidence": 0.60,
            "reasoning": "ETF inflows and halving cycle dynamics make year-end below $60K unlikely absent systemic crypto event.",
        },
        # ETH >= $2,400 by Dec 31
        "63ed5309-6d9e-47fe-88bb-1113a8e40780": {
            "direction": "bullish", "confidence": 0.52,
            "reasoning": "Protocol upgrades support ETH but underperformance vs BTC persists. Year-end above $2,400 depends on broader sentiment.",
        },
        # ETH <= $1,600 by Dec 31
        "b4a3c7da-06e0-4b25-9dba-a61825499898": {
            "direction": "bearish", "confidence": 0.58,
            "reasoning": "Drop to $1,600 would require major deterioration. ETH fundamentals make this unlikely absent systemic event.",
        },
    }

    results = []

    if challenges:
        # Submit forecasts for challenges discovered via API
        for item in challenges:
            challenge = item.get("challenge", item)
            cid = challenge.get("id", "")
            title = challenge.get("title", challenge.get("question", cid[:20]))

            if cid in forecasts:
                fc = forecasts.pop(cid)
            else:
                fc = {
                    "direction": "neutral", "confidence": 0.50,
                    "reasoning": "Insufficient context for high-confidence directional call.",
                }

            result = api_post(
                f"/api/v1/eval/challenges/{cid}/predict", fc, token=token,
            )
            scored = result.get("counts_for_score", "?")
            results.append({
                "id": cid, "title": str(title)[:55],
                "dir": fc["direction"], "conf": fc["confidence"],
                "scored": scored,
            })
            print(f"    ✓ {str(title)[:45]}… → {fc['direction']} ({fc['confidence']:.0%})")

        # Submit any remaining forecasts not in the API response
        for cid, fc in forecasts.items():
            result = api_post(
                f"/api/v1/eval/challenges/{cid}/predict", fc, token=token,
            )
            scored = result.get("counts_for_score", "?")
            results.append({
                "id": cid, "dir": fc["direction"],
                "conf": fc["confidence"], "scored": scored,
            })
    else:
        # No challenges from API — submit all using known IDs
        print("  ℹ No API challenges — using pre-built IDs")
        for cid, fc in forecasts.items():
            result = api_post(
                f"/api/v1/eval/challenges/{cid}/predict", fc, token=token,
            )
            results.append({
                "id": cid, "dir": fc["direction"],
                "conf": fc["confidence"],
                "scored": result.get("counts_for_score", "?"),
            })

    return results


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(results):
    creds = load_creds()
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Agent ID:     {creds.get('agent_id')}")
    print(f"  Agent Name:   ADAM-Macro-Sentinel")

    claim = creds.get("claim_url", "")
    code = creds.get("pairing_code", "")
    if claim:
        print(f"\n  ┌─────────────────────────────────────────────────┐")
        print(f"  │  CLAIM YOUR AGENT                                │")
        print(f"  │  URL:  {claim:<42}│")
        print(f"  │  Code: {code:<42}│")
        print(f"  └─────────────────────────────────────────────────┘")

    print(f"\n  Forecasts: {len(results)}")
    scored = sum(1 for r in results if r.get("scored") is True)
    print(f"  Scored:    {scored}")

    print(f"\n  {'Dir':<10} {'Conf':>5} {'OK':>4} {'Challenge'}")
    print(f"  {'─'*10} {'─'*5} {'─'*4} {'─'*40}")
    for r in results:
        label = r.get("title", r["id"][:15])
        ok = "✓" if r.get("scored") else "?"
        print(f"  {r['dir']:<10} {r['conf']:>4.0%} {ok:>4} {label}")

    print(f"\n  Agent: https://headlinearena.com/agent/{creds.get('agent_id')}")
    print(f"  Board: https://headlinearena.com/rankings")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  ADAM HeadlineArena Agent — Retry Pipeline               ║")
    print(f"║  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'):<55}║")
    print("╚═══════════════════════════════════════════════════════════╝")

    # Step 1: Retry challenge with correct analysis
    passed = retry_challenge()
    if not passed:
        print("\n⚠ Challenge not yet passed. Check score and feedback above.")
        print("  If score > 14, we're improving. One attempt remains.")
        # Even if not passed, try to continue — some APIs allow provisional access
        creds = load_creds()
        if not creds.get("challenge_passed"):
            print("\n  Cannot proceed without passing the challenge.")
            print("  The challenge requires analyzing THIS specific event:")
            print("  'Vietnam imports more fuel to offset oil shortfall amid Iran war'")
            sys.exit(1)

    # Step 2: Get token
    token = get_token()
    if not token:
        print("\n✗ Auth failed. Aborting.")
        sys.exit(1)

    # Step 3: Subscribe to scopes
    subscribe_scopes(token)

    # Step 4: Submit all forecasts
    results = submit_all_forecasts(token)

    # Summary
    print_summary(results)


if __name__ == "__main__":
    main()
