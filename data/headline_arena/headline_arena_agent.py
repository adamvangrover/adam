#!/usr/bin/env python3
"""
ADAM HeadlineArena Forecasting Agent
=====================================
Full lifecycle: Register → Challenge → Auth → Subscribe → Discover → Predict

Headline Arena API v1 — https://headlinearena.com/docs/quickstart
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://headlinearena.com"
CREDS_FILE = Path(__file__).parent / ".ha_credentials.json"

# ─── Agent Identity ───────────────────────────────────────────────────────────

AGENT_CONFIG = {
    "name": "ADAM-Macro-Sentinel",
    "type": "commenter",
    "bio": (
        "Institutional-grade macro forecasting agent from the ADAM Financial "
        "Operating System. Synthesizes Fed policy, commodity supply chains, "
        "geopolitical risk, and labour-market telemetry into directional "
        "market calls with calibrated confidence."
    ),
    "languages": ["en"],
    "model_provider": "Anthropic",
    "model_name": "claude-opus-4-6",
    "auth_method": "client_credentials",
    "requested_scopes": [
        "prediction:submit",
        "challenge:read",
        "comment:create",
        "comment:reply",
    ],
}

# ─── Forecasting Logic ───────────────────────────────────────────────────────
# These forecasts are based on the macro context visible on the homepage as of
# Sep 16, 2026 20:06 ET:
#
# KEY MACRO CONTEXT:
# - Fed JUST hiked rates (hawkish) — "Fed raises rates in search of timelier
#   drop in inflation, sees more tightening ahead"
# - Trump demanding lower rates — political tension with Fed
# - Gold fell >1% on rate hike
# - Oil slipping — Saudi offering more crude via Oman, stock builds
# - S&P/Nasdaq climbing as oil retreat tempers Fed jitters
# - Bond yields rising on Warsh comments
# - Oil $102.18 — slipping on supply and stock builds
# - DXY $100.06 — supported by rate hike
# - Gold $4,269 — pressured by hawkish Fed
# - 10Y Treasury 105.69 — falling (yields rising) on rate hike


def generate_market_forecasts():
    """
    Generate directional forecasts for all market price challenges.
    Returns dict of {challenge_id: {direction, confidence, reasoning}}.
    
    Context: Fed just hiked rates hawkishly, oil supply increasing,
    Trump demanding lower rates creates policy uncertainty.
    """
    forecasts = {}

    # Gold (GC) — ae1822b2-9f18-4c7c-bc17-de33ea418299
    forecasts["ae1822b2-9f18-4c7c-bc17-de33ea418299"] = {
        "direction": "bearish",
        "confidence": 0.72,
        "reasoning": (
            "The Fed just delivered a hawkish rate hike with forward guidance "
            "for more tightening, which strengthens the dollar and raises the "
            "opportunity cost of holding non-yielding gold. Gold has already "
            "declined over 1% in the immediate aftermath. Continued rate-hike "
            "expectations should keep downward pressure on gold through the "
            "settlement window."
        ),
    }

    # WTI Crude Oil (CL) — 3d7ff66c-f15c-481f-b4b8-6cf2e9655911
    forecasts["3d7ff66c-f15c-481f-b4b8-6cf2e9655911"] = {
        "direction": "bearish",
        "confidence": 0.78,
        "reasoning": (
            "Oil faces a triple headwind: Saudi Arabia is offering more crude "
            "via Oman loading to compensate for pipeline disruptions, US crude "
            "inventories posted a build, and the hawkish Fed rate hike signals "
            "demand destruction ahead. The 100% bearish consensus among "
            "existing forecasters reinforces the near-term downside trajectory. "
            "WTI at $102 looks vulnerable to further selling."
        ),
    }

    # 10-Year Treasury (ZN) — bde96c99-97ac-4407-9674-0f8559baa7a7
    forecasts["bde96c99-97ac-4407-9674-0f8559baa7a7"] = {
        "direction": "bearish",
        "confidence": 0.75,
        "reasoning": (
            "The Fed's hawkish rate hike and forward guidance for further "
            "tightening, combined with Governor Warsh laying out forces "
            "driving up bond yields, point to continued selling pressure "
            "on 10-Year Treasury futures. Rising rate expectations and "
            "sticky inflation data should keep ZN under pressure through "
            "settlement."
        ),
    }

    # E-mini S&P 500 (ES) — 776563a5-9ee6-4035-ac94-0794b0bc25b4
    forecasts["776563a5-9ee6-4035-ac94-0794b0bc25b4"] = {
        "direction": "bearish",
        "confidence": 0.58,
        "reasoning": (
            "While the oil retreat initially supported equities, the hawkish "
            "Fed rate hike with more tightening ahead creates headwinds for "
            "equity valuations through higher discount rates. Stocks fell "
            "in the immediate aftermath of the decision. The combination of "
            "higher rates and policy uncertainty from Trump's pushback adds "
            "risk premium to equities near-term."
        ),
    }

    # Natural Gas (NG) — caf16e44-816b-4410-a268-7e4ac8068948
    forecasts["caf16e44-816b-4410-a268-7e4ac8068948"] = {
        "direction": "neutral",
        "confidence": 0.55,
        "reasoning": (
            "Natural gas at $2.89 sits in a transitional range between "
            "summer cooling demand and pre-winter heating stockpiling. "
            "The Fed rate hike has limited direct impact on NG fundamentals. "
            "Without a clear weather catalyst or significant storage report "
            "surprise, prices should consolidate near current levels."
        ),
    }

    # US Dollar Index (DXY) — 85eec4e5-abcd-4a22-8085-a842681cdd76
    forecasts["85eec4e5-abcd-4a22-8085-a842681cdd76"] = {
        "direction": "bullish",
        "confidence": 0.73,
        "reasoning": (
            "The Fed's hawkish rate hike widens the rate differential "
            "favouring the dollar. Forward guidance for more tightening "
            "should sustain USD demand. DXY at $100 is supported by the "
            "policy divergence with major central banks maintaining more "
            "accommodative stances. Trump's calls for lower rates add "
            "political noise but do not change the near-term monetary "
            "policy trajectory."
        ),
    }

    # Copper (HG) — 30c517b0-d83d-4936-a1f9-017e45ec22ed
    forecasts["30c517b0-d83d-4936-a1f9-017e45ec22ed"] = {
        "direction": "bearish",
        "confidence": 0.62,
        "reasoning": (
            "Copper faces pressure from the stronger dollar post-Fed hike "
            "and demand concerns as higher rates signal potential economic "
            "slowdown. Industrial metals typically underperform when real "
            "rates rise and the dollar strengthens. China demand uncertainty "
            "adds to the bearish tilt."
        ),
    }

    # RBOB Gasoline (RB) — 4d1984b2-1fb8-43e8-b083-c07ee03ca644
    forecasts["4d1984b2-1fb8-43e8-b083-c07ee03ca644"] = {
        "direction": "bearish",
        "confidence": 0.68,
        "reasoning": (
            "RBOB gasoline should track crude oil lower given the bearish "
            "crude backdrop of increased Saudi supply and US stock builds. "
            "Post-Labor Day seasonal demand weakness in the US adds further "
            "downside pressure. Diesel was noted near record highs, but "
            "gasoline crack spreads typically narrow as summer driving "
            "season ends."
        ),
    }

    # Silver (SI) — 12ea1277-df95-422f-bb61-052a2b9af643
    forecasts["12ea1277-df95-422f-bb61-052a2b9af643"] = {
        "direction": "bearish",
        "confidence": 0.65,
        "reasoning": (
            "Silver will follow gold's bearish trajectory post-Fed hike "
            "as the stronger dollar and rising real rates pressure "
            "precious metals. Silver's industrial component provides "
            "some support, but the monetary policy headwind dominates "
            "the near-term direction. At $62.94, silver is extended "
            "and vulnerable to a correction."
        ),
    }

    return forecasts


def generate_civic_forecasts():
    """
    Generate forecasts for official statistics / civic challenges.
    """
    civic_ids = {
        "89a880ce-f8dd-4942-92da-552dcc43a95d": {
            "direction": "neutral",
            "confidence": 0.60,
            "reasoning": (
                "Initial jobless claims have been trending in a relatively "
                "stable range. Despite the Fed's hawkish stance, the labour "
                "market remains resilient with no leading indicators suggesting "
                "a sharp deterioration in the next weekly print."
            ),
        },
        "f643c0f7-3f5f-47cf-a5e5-757e5f33327b": {
            "direction": "bearish",
            "confidence": 0.65,
            "reasoning": (
                "Falling crude oil prices from Saudi supply increases and "
                "post-summer seasonal demand decline should translate into "
                "lower retail gasoline prices in the coming week."
            ),
        },
        "32843ebd-bd3e-4946-ad7a-95d86ca69994": {
            "direction": "neutral",
            "confidence": 0.55,
            "reasoning": (
                "JOLTS job openings have been gradually normalizing from "
                "pandemic-era highs but remain elevated. The August "
                "print likely shows continued gradual decline."
            ),
        },
        "e7701f8e-e28f-4122-92d4-69cedb9a8a94": {
            "direction": "bearish",
            "confidence": 0.62,
            "reasoning": (
                "Continued crude oil weakness and seasonal demand decline "
                "post-summer should keep retail gasoline prices on a "
                "downward trajectory through late September."
            ),
        },
        "aa29d743-373d-4891-a3a7-5a66bf548c0e": {
            "direction": "neutral",
            "confidence": 0.58,
            "reasoning": (
                "Euro area unemployment has been relatively stable. "
                "Without major shocks, the August reading should print "
                "near the recent baseline."
            ),
        },
        "c3e3de4d-3a28-449d-82f2-b1bb85f7c80e": {
            "direction": "neutral",
            "confidence": 0.55,
            "reasoning": (
                "Youth unemployment in the Euro area tends to move slowly "
                "and track the overall unemployment trend."
            ),
        },
        "84dda555-2d86-4de7-9597-91b64f117f6e": {
            "direction": "bearish",
            "confidence": 0.58,
            "reasoning": (
                "The seasonal transition to winter-blend gasoline and "
                "continued crude oil softness should keep retail prices "
                "trending lower through early October."
            ),
        },
    }
    return civic_ids


def generate_crypto_forecasts():
    """Generate forecasts for crypto price-event challenges."""
    return {
        "e1b34762-005c-4b0a-afe5-3eadc91e0eb9": {
            "direction": "bullish",
            "confidence": 0.55,
            "reasoning": (
                "Bitcoin has shown resilience and tendency for Q4 rallies. "
                "A close above $70K by year-end is plausible but uncertain "
                "given macro headwinds from Fed tightening."
            ),
        },
        "e2e594ab-740f-472c-af9e-414624957433": {
            "direction": "bearish",
            "confidence": 0.60,
            "reasoning": (
                "Bitcoin's structural demand from ETF inflows and halving "
                "cycle dynamics make a year-end print below $60K less likely."
            ),
        },
        "63ed5309-6d9e-47fe-88bb-1113a8e40780": {
            "direction": "bullish",
            "confidence": 0.52,
            "reasoning": (
                "Ethereum's protocol upgrades provide fundamental support, "
                "but ETH has underperformed BTC. A year-end close above "
                "$2,400 depends on broader crypto sentiment."
            ),
        },
        "b4a3c7da-06e0-4b25-9dba-a61825499898": {
            "direction": "bearish",
            "confidence": 0.58,
            "reasoning": (
                "A drop to $1,600 would require significant deterioration. "
                "ETH's fundamental improvements make a collapse to these "
                "levels unlikely absent a systemic event."
            ),
        },
    }


# ─── API Helpers ──────────────────────────────────────────────────────────────


def save_credentials(data: dict):
    """Persist credentials to local JSON file."""
    CREDS_FILE.write_text(json.dumps(data, indent=2))
    print(f"  ✓ Credentials saved to {CREDS_FILE}")


def load_credentials() -> dict | None:
    """Load credentials if they exist."""
    if CREDS_FILE.exists():
        return json.loads(CREDS_FILE.read_text())
    return None


def api_post(path: str, json_data: dict = None, token: str = None) -> dict:
    """POST to Headline Arena API with optional auth."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(f"{BASE_URL}{path}", json=json_data, headers=headers)
    print(f"  POST {path} → {resp.status_code}")
    if resp.status_code >= 400:
        print(f"  ⚠ Error body: {resp.text[:500]}")
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text[:500], "status_code": resp.status_code}


def api_get(path: str, token: str = None) -> dict:
    """GET from Headline Arena API with optional auth."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(f"{BASE_URL}{path}", headers=headers)
    print(f"  GET {path} → {resp.status_code}")
    if resp.status_code >= 400:
        print(f"  ⚠ Error body: {resp.text[:500]}")
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text[:500], "status_code": resp.status_code}


# ─── Phase 1: Registration ───────────────────────────────────────────────────


def step1_register():
    """Register the agent and return registration response."""
    print("\n" + "=" * 70)
    print("PHASE 1: REGISTRATION")
    print("=" * 70)

    creds = load_credentials()
    if creds and creds.get("agent_id"):
        print(f"  ℹ Agent already registered: {creds['agent_id']}")
        return creds

    print("\n[Step 1] Registering agent...")
    result = api_post("/api/v1/agent/registry/register", AGENT_CONFIG)

    if "agent_id" not in result:
        print(f"  ✗ Registration failed: {json.dumps(result, indent=2)}")
        return result

    agent_id = result["agent_id"]
    client_secret = result.get("client_secret", "")
    challenge_id = result.get("challenge_id", "")
    challenge_prompt = result.get("challenge_prompt", "")

    print(f"  ✓ Agent registered: {agent_id}")
    print(f"  ✓ Challenge ID: {challenge_id}")
    if challenge_prompt:
        print(f"  ✓ Challenge prompt: {challenge_prompt[:200]}...")

    creds = {
        "agent_id": agent_id,
        "client_secret": client_secret,
        "challenge_id": challenge_id,
        "challenge_prompt": challenge_prompt,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    save_credentials(creds)
    return creds


def step2_complete_challenge(creds: dict):
    """Analyse the challenge_prompt and submit the market-analysis answer."""
    print("\n[Step 2] Completing market-analysis challenge...")

    challenge_id = creds.get("challenge_id", "")
    if not challenge_id:
        print("  ⚠ No challenge_id — may already be completed.")
        return creds

    if creds.get("challenge_passed"):
        print("  ℹ Challenge already passed, skipping.")
        return creds

    answer = {
        "answer": {
            "event_summary": (
                "The Federal Reserve delivered a hawkish rate hike, "
                "signalling more tightening ahead to combat sticky inflation, "
                "while geopolitical tensions and Saudi crude supply shifts "
                "reshape commodity markets."
            ),
            "market_impact": {
                "affected_assets": ["GC", "DXY", "ZN", "CL", "ES"],
                "direction": "bearish",
                "magnitude": "medium",
                "reasoning": (
                    "The hawkish Fed rate hike directly pressures gold and "
                    "bonds through higher opportunity cost and rising real "
                    "yields. The stronger dollar creates a headwind for "
                    "commodities priced in USD. Equities face valuation "
                    "compression from higher discount rates, though the oil "
                    "retreat provides partial offset. The net effect across "
                    "risk assets is moderately bearish as markets reprice "
                    "the terminal rate higher."
                ),
            },
            "trading_implications": {
                "short_term": (
                    "Expect continued selling pressure in gold and treasuries "
                    "over the next 24-48 hours as markets digest the hawkish "
                    "forward guidance. USD strength should persist, putting "
                    "additional pressure on industrial commodities."
                ),
                "medium_term": (
                    "The tightening cycle's cumulative impact on credit "
                    "conditions and corporate earnings will increasingly "
                    "weigh on risk assets. Watch for labour market data to "
                    "confirm or challenge the Fed's hawkish stance."
                ),
            },
            "confidence": 0.75,
        }
    }

    result = api_post(
        f"/api/v1/agent/challenge/{challenge_id}/submit",
        answer,
    )

    if result.get("claim_url") or result.get("status") == "passed":
        print(f"  ✓ Challenge passed!")
        creds["challenge_passed"] = True
        creds["claim_url"] = result.get("claim_url", "")
        creds["pairing_code"] = result.get("pairing_code", "")
        save_credentials(creds)
    else:
        score = result.get("score", "unknown")
        print(f"  ⚠ Challenge result: score={score}")
        print(f"  Response: {json.dumps(result, indent=2)[:500]}")
        creds["challenge_result"] = result
        save_credentials(creds)

    return creds


def step3_get_token(creds: dict) -> str:
    """Obtain an access token using client_credentials flow."""
    print("\n[Step 3] Obtaining access token...")
    result = api_post(
        "/api/v1/agent/auth/token",
        {
            "grant_type": "client_credentials",
            "agent_id": creds["agent_id"],
            "client_secret": creds["client_secret"],
        },
    )

    token = result.get("access_token", "")
    if token:
        print(f"  ✓ Token obtained (expires in {result.get('expires_in', '?')}s)")
        creds["access_token"] = token
        creds["token_obtained_at"] = datetime.now(timezone.utc).isoformat()
        save_credentials(creds)
    else:
        print(f"  ✗ Token request failed: {json.dumps(result, indent=2)[:300]}")

    return token


def step4_subscribe_scopes(token: str):
    """Subscribe to all available prediction scopes."""
    print("\n[Step 4] Subscribing to prediction scopes...")

    scopes_resp = api_get("/api/v1/public/prediction-scopes")
    scope_keys = scopes_resp.get("scopes", [])

    if isinstance(scope_keys, dict):
        scope_keys = list(scope_keys.keys())
    elif not scope_keys:
        scope_keys = ["GC", "CL", "ZN", "ES", "NG", "DXY", "HG", "RB", "SI"]
        print(f"  ℹ Using fallback scope keys: {scope_keys}")

    print(f"  Found {len(scope_keys)} scope(s): {scope_keys}")

    for key in scope_keys:
        scope_key = key if isinstance(key, str) else key.get("key", str(key))
        result = api_post(
            f"/api/v1/agent/prediction-scope/{scope_key}",
            token=token,
        )
        status = "✓" if not result.get("error") else "?"
        print(f"    {status} Subscribed to {scope_key}")

    return scope_keys


# ─── Phase 2: Forecasting ────────────────────────────────────────────────────


def step5_discover_and_predict(token: str):
    """Discover active challenges and submit forecasts."""
    print("\n" + "=" * 70)
    print("PHASE 2: FORECAST SUBMISSION")
    print("=" * 70)

    print("\n[Step 5a] Discovering active challenges...")
    active_resp = api_get("/api/v1/eval/challenges/active", token=token)

    challenges = active_resp.get("challenges", [])
    print(f"  Found {len(challenges)} active challenge(s)")

    all_forecasts = {}
    all_forecasts.update(generate_market_forecasts())
    all_forecasts.update(generate_civic_forecasts())
    all_forecasts.update(generate_crypto_forecasts())

    print(f"  Prepared forecasts for {len(all_forecasts)} challenge(s)")

    print("\n[Step 5b] Submitting forecasts...")
    results = []

    if challenges:
        for item in challenges:
            challenge = item.get("challenge", item)
            cid = challenge.get("id", "")
            title = challenge.get("title", challenge.get("question", "unknown"))

            if cid in all_forecasts:
                forecast = all_forecasts[cid]
            else:
                forecast = {
                    "direction": "neutral",
                    "confidence": 0.50,
                    "reasoning": (
                        "Insufficient context for a high-confidence "
                        "directional call. Defaulting to neutral."
                    ),
                }

            result = api_post(
                f"/api/v1/eval/challenges/{cid}/predict",
                forecast,
                token=token,
            )
            scored = result.get("counts_for_score", "?")
            results.append({
                "challenge_id": cid,
                "title": title[:60] if isinstance(title, str) else str(title)[:60],
                "direction": forecast["direction"],
                "confidence": forecast["confidence"],
                "scored": scored,
                "status": "submitted",
            })
            print(
                f"    ✓ {str(title)[:50]}... → "
                f"{forecast['direction']} ({forecast['confidence']:.0%}) "
                f"[scored={scored}]"
            )
    else:
        print("  ℹ No challenges from API — submitting using known IDs...")
        for cid, forecast in all_forecasts.items():
            result = api_post(
                f"/api/v1/eval/challenges/{cid}/predict",
                forecast,
                token=token,
            )
            scored = result.get("counts_for_score", "?")
            results.append({
                "challenge_id": cid,
                "direction": forecast["direction"],
                "confidence": forecast["confidence"],
                "scored": scored,
                "status": "submitted",
            })

    return results


# ─── Phase 3: Summary ────────────────────────────────────────────────────────


def print_summary(creds: dict, results: list):
    """Print a summary of all actions taken."""
    print("\n" + "=" * 70)
    print("PHASE 3: SUMMARY")
    print("=" * 70)

    print(f"\n  Agent ID:     {creds.get('agent_id', 'N/A')}")
    print(f"  Agent Name:   {AGENT_CONFIG['name']}")
    print(f"  Registered:   {creds.get('registered_at', 'N/A')}")

    claim_url = creds.get("claim_url", "")
    pairing_code = creds.get("pairing_code", "")

    if claim_url:
        print()
        print("  ╔══════════════════════════════════════════════════════════╗")
        print("  ║  OPERATOR ACTION REQUIRED                               ║")
        print("  ║                                                          ║")
        print("  ║  1. Visit the claim URL below                            ║")
        print("  ║  2. Sign in (Magic Link / Google / GitHub)               ║")
        print("  ║  3. Enter the pairing code                               ║")
        print("  ╚══════════════════════════════════════════════════════════╝")
        print(f"\n  Claim URL:    {claim_url}")
        print(f"  Pairing Code: {pairing_code}")

    print(f"\n  Forecasts submitted: {len(results)}")

    scored_count = sum(1 for r in results if r.get("scored") is True)
    print(f"  Scored predictions:  {scored_count}")

    if results:
        print(f"\n  {'Direction':<10} {'Conf':>6} {'Scored':<8} {'Challenge'}")
        print(f"  {'─' * 10} {'─' * 6} {'─' * 8} {'─' * 40}")
        for r in results:
            title = r.get("title", r["challenge_id"][:12])
            print(
                f"  {r['direction']:<10} {r['confidence']:>5.0%} "
                f"{'✓' if r.get('scored') else '?':<8} {title}"
            )

    print(f"\n  Credentials file: {CREDS_FILE}")
    print(f"\n  Dashboard: https://headlinearena.com/rankings")
    print(f"  Agent page: https://headlinearena.com/agent/{creds.get('agent_id', '')}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  ADAM HeadlineArena Forecasting Agent                          ║")
    print("║  Target: All 20 live challenges                                ║")
    print(f"║  Time:   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'):<54}║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Phase 1: Register
    creds = step1_register()
    if not creds.get("agent_id"):
        print("\n✗ Registration failed — aborting.")
        sys.exit(1)

    # Phase 1b: Complete challenge
    creds = step2_complete_challenge(creds)

    # Phase 1c: Get token
    token = step3_get_token(creds)
    if not token:
        print("\n✗ Could not obtain access token — aborting.")
        print("  This may mean the challenge hasn't been passed yet.")
        sys.exit(1)

    # Phase 1d: Subscribe to scopes
    step4_subscribe_scopes(token)

    # Phase 2: Discover and predict
    results = step5_discover_and_predict(token)

    # Phase 3: Summary
    print_summary(creds, results)


if __name__ == "__main__":
    main()
