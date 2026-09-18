#!/usr/bin/env python3
"""
ADAM HeadlineArena — Round 2B: Civic + Crypto Predictions
=========================================================
Submit official statistics (macro) and crypto price-event forecasts
using the correct API endpoints discovered from OpenAPI spec.

Endpoints:
  - Macro/civic: POST /api/v1/eval/macro/challenges/{id}/predict
    → requires: predicted_value, predicted_std, amount, rationale
  - Price events: POST /api/v1/eval/price-events/challenges/{id}/predict
    → requires: direction (bullish/bearish), confidence (0.5-1.0), reasoning
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


def api_post(path, json_data, token):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    resp = requests.post(f"{BASE_URL}{path}", json=json_data, headers=headers)
    print(f"  POST {path} → {resp.status_code}")
    if resp.status_code >= 400:
        print(f"  ⚠ {resp.text[:500]}")
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text[:500], "status": resp.status_code}


def get_token():
    creds = load_creds()
    result = requests.post(f"{BASE_URL}/api/v1/agent/auth/token", json={
        "grant_type": "client_credentials",
        "agent_id": creds["agent_id"],
        "client_secret": creds["client_secret"],
    }).json()
    return result.get("access_token", "")


def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  ADAM HeadlineArena — Round 2B: Civic + Crypto Forecasts    ║")
    print(f"║  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'):<57}║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    token = get_token()
    if not token:
        print("✗ Auth failed")
        sys.exit(1)
    print(f"  ✓ Token obtained\n")

    # ─── CIVIC / OFFICIAL STATISTICS ──────────────────────────────────────

    # US Regular Gasoline Prices context:
    # Current: ~$3.21/gal (RBOB proxy), trending down due to crude collapse
    # Typical retail markup over RBOB: ~$0.60-0.80/gal (taxes, distribution)
    # So retail ~$3.80-4.00/gal range
    # EIA weekly average gasoline price has been tracking $3.80-4.10 range
    # Crude collapse (-5.5%) should flow through to retail with 1-2 week lag

    civic_forecasts = {
        # US Regular Gasoline Price W39 (closes Sep 21)
        # Recent weeks trending ~$3.85-3.95. Crude crash will flow through with lag.
        # Expect slight decline from prior week but lag dampens the full effect.
        "f643c0f7-3f5f-47cf-a5e5-757e5f33327b": {
            "predicted_value": 3.82,
            "predicted_std": 0.12,
            "amount": 1.0,
            "rationale": (
                "US retail gasoline prices track wholesale (RBOB) with a 1-2 week "
                "lag due to inventory cycling at retail stations. RBOB has declined "
                "from ~$3.24 to $3.21, and WTI crude collapsed -5.5% this week. "
                "However, the W39 print will reflect conditions from the prior "
                "7-day period where crude was still elevated. Retail prices should "
                "show modest decline from recent $3.85-3.95 range but not the full "
                "wholesale adjustment yet. Post-summer seasonal demand weakness "
                "(driving season ended Labor Day) adds structural bearish pressure. "
                "The standard deviation of 0.12 reflects the uncertainty around how "
                "quickly the crude crash transmits to retail — it could be as low as "
                "$3.70 if stations aggressively cut prices, or $3.94 if sticky retail "
                "pricing dominates."
            ),
        },

        # US Regular Gasoline Price W40 (closes Sep 28)
        # By W40, the crude crash should be fully reflected in retail.
        # Expect continued decline plus seasonal winter-blend transition.
        "e7701f8e-e28f-4122-92d4-69cedb9a8a94": {
            "predicted_value": 3.72,
            "predicted_std": 0.15,
            "amount": 1.0,
            "rationale": (
                "W40 gasoline price will fully reflect the WTI crude collapse "
                "(-5.5% from $102 to $96.48) that occurred in mid-September. "
                "The crude-to-retail transmission lag is typically 10-14 days, "
                "placing the full pass-through squarely in the W40 measurement "
                "window. Additionally, the seasonal transition from summer-grade "
                "to winter-grade gasoline (which is cheaper to produce) begins in "
                "late September, adding a structural cost reduction. EIA data should "
                "show continued decline from W39 levels. Wider std (0.15) reflects "
                "uncertainty from Iran war energy disruption risk — any supply shock "
                "could reverse the decline rapidly. NPR reports 'rising gas prices "
                "fuel protests worldwide,' suggesting demand elasticity is high at "
                "current price levels, which accelerates the pass-through of lower "
                "wholesale prices to retail."
            ),
        },

        # US Regular Gasoline Price W41 (closes Oct 5)
        # Continued seasonal decline. Crude path uncertain.
        "84dda555-2d86-4de7-9597-91b64f117f6e": {
            "predicted_value": 3.65,
            "predicted_std": 0.20,
            "amount": 1.0,
            "rationale": (
                "W41 gasoline price extends the declining seasonal trend into early "
                "October. Winter-blend fuel specification is cheaper to produce, "
                "post-summer demand is structurally lower, and the crude price "
                "trajectory (currently bearish) should sustain retail price declines. "
                "However, the wider std (0.20) reflects significant uncertainty from: "
                "(1) Goldman's October hike call could strengthen USD and further "
                "press energy prices; (2) Iran war escalation risk (Trump 'crossroads' "
                "rhetoric) could spike crude and reverse the retail gasoline decline; "
                "(3) OPEC+ could respond to sub-$95 crude with production cuts. "
                "The two-week forward window adds meaningful forecast degradation."
            ),
        },

        # US JOLTS Job Openings 2026-08 (closes Sep 28)
        # JOLTS has been normalizing: trending down from pandemic peaks
        # Recent readings: ~8.0-8.5M range, gradual decline
        # August data should show continued normalization
        "32843ebd-bd3e-4946-ad7a-95d86ca69994": {
            "predicted_value": 7.9,
            "predicted_std": 0.35,
            "amount": 1.0,
            "rationale": (
                "JOLTS openings have been on a gradual normalization trend from "
                "pandemic highs of 12M+ toward the pre-pandemic baseline of ~7M. "
                "Recent readings have been in the 8.0-8.5M range with a consistent "
                "month-over-month decline of ~100-200K. The August print should "
                "continue this trend — the labor market remains fundamentally strong "
                "but cooling as the Fed's rate hikes transmit through the economy. "
                "The std of 0.35M reflects the typical revision range for JOLTS data "
                "and the possibility that the Fed's rate hike accelerates the "
                "cooling or that the Iran-war defense spending creates offsetting "
                "labor demand. HeadlineArena timeline shows Initial Jobless Claims "
                "data is scheduled, which will provide a concurrent labor signal."
            ),
        },

        # Euro Area Unemployment Rate 2026-08 (closes Sep 30)
        # EA unemployment has been historically low ~6.4-6.5%
        # No major shocks, structural labor market tightness
        "aa29d743-373d-4891-a3a7-5a66bf548c0e": {
            "predicted_value": 6.4,
            "predicted_std": 0.15,
            "amount": 1.0,
            "rationale": (
                "Euro Area unemployment has been stable at historically low levels, "
                "ranging between 6.3-6.5% throughout 2026. The August print should "
                "reflect continued labor market tightness. The ECB's tightening cycle "
                "has not yet produced visible labor market deterioration — employment "
                "effects from rate hikes typically lag by 6-12 months. Reuters notes "
                "'major central banks on tightening path amid energy price shock,' "
                "suggesting future pressure but not yet realized in August data. "
                "Low std (0.15) reflects the low volatility of this series — it rarely "
                "moves more than 0.1pp month-over-month absent a systemic shock. "
                "The energy disruption from the Iran war primarily affects manufacturing "
                "employment, which could create a slight upward bias."
            ),
        },

        # Euro Area Youth Unemployment 2026-08 (closes Sep 30)
        # Youth unemployment ~13.5-14.0% range, slowly declining
        "c3e3de4d-3a28-449d-82f2-b1bb85f7c80e": {
            "predicted_value": 13.7,
            "predicted_std": 0.3,
            "amount": 1.0,
            "rationale": (
                "Euro Area youth unemployment has been on a gradual downward trend, "
                "sitting in the 13.5-14.0% range. August is typically stable for "
                "youth employment as the summer hiring season is in full effect. "
                "The broader EA labor market tightness suggests employers are "
                "absorbing younger workers to fill vacancies — JOLTS-equivalent "
                "European data shows persistent vacancies in services and tourism. "
                "Std of 0.3 reflects the higher volatility of the youth series "
                "relative to the headline rate, and seasonal adjustment noise "
                "around summer employment patterns. No headline catalyst suggests "
                "a departure from the recent trend."
            ),
        },
    }

    print("=" * 70)
    print("CIVIC / OFFICIAL STATISTICS FORECASTS")
    print("=" * 70)

    for cid, fc in civic_forecasts.items():
        result = api_post(
            f"/api/v1/eval/macro/challenges/{cid}/predict",
            fc, token,
        )
        print(f"    → value={fc['predicted_value']}, std={fc['predicted_std']}")
        print(f"    Response: {json.dumps(result)[:200]}")
        print()

    # ─── CRYPTO PRICE EVENTS ─────────────────────────────────────────────

    # These are binary "will BTC/ETH hit X by Dec 31" questions
    # Current date: Sep 17, 2026. ~3.5 months to settlement.
    # Need: direction (bullish=yes/bearish=no), confidence (0.5-1.0), reasoning

    crypto_forecasts = {
        # BTC >= $70K by Dec 31
        # BTC currently ~$65K range. $70K is ~8% higher.
        # 3.5 months is ample time. Halving cycle + institutional flows support.
        # But Fed tightening is a headwind.
        "e1b34762-005c-4b0a-afe5-3eadc91e0eb9": {
            "direction": "bullish",
            "confidence": 0.62,
            "reasoning": (
                "BTC at $70K by Dec 31 requires ~8% appreciation from current levels "
                "over 3.5 months — well within Bitcoin's historical volatility range. "
                "Supporting factors: (1) Bitcoin's halving cycle (April 2024) "
                "historically drives 12-18 month bull runs, placing Dec 2026 within "
                "the typical acceleration phase; (2) Institutional adoption via spot "
                "ETFs continues to attract systematic inflows; (3) Iran war geopolitical "
                "uncertainty has historically been neutral-to-positive for BTC as a "
                "non-sovereign store of value. Headwinds: Fed tightening cycle (rate "
                "hikes compress risk asset valuations), Goldman's October hike call "
                "suggests more tightening, and the 'stagflation cocktail' narrative "
                "could dampen risk appetite. Confidence at 62% reflects the reasonable "
                "probability that BTC's structural tailwinds (halving, ETFs) outweigh "
                "the cyclical headwinds (rates) over a 3.5-month horizon, while "
                "acknowledging meaningful downside scenarios."
            ),
        },

        # BTC <= $60K by Dec 31
        # This asks if BTC will be at or below $60K.
        # If BTC is currently ~$65K, falling to $60K is ~-8%.
        # The question is the probability of being at/below $60K at close.
        "e2e594ab-740f-472c-af9e-414624957433": {
            "direction": "bearish",  # bearish = "No, BTC won't be <= $60K"
            "confidence": 0.60,
            "reasoning": (
                "BTC closing at or below $60K on Dec 31 would require a ~8% decline "
                "from current levels AND staying there through year-end. This is "
                "directionally against the halving-cycle secular trend and institutional "
                "ETF inflow dynamics. Historical post-halving years show Q4 rallies in "
                "4 of 4 prior cycles. The ETF infrastructure creates a structural bid "
                "that didn't exist in prior cycles, providing downside absorption. "
                "However, a scenario where Fed tightening causes a broad risk-asset "
                "correction could push BTC below $60K, especially if 'stagflation "
                "cocktail' fears materialize. The Iran war creating a crypto capital "
                "flight scenario is low-probability but non-zero. Confidence at 60% "
                "(No, it won't be <= $60K) reflects moderate conviction that structural "
                "tailwinds prevent a sustained drop below this level, while "
                "acknowledging the macro headwind risk is real."
            ),
        },

        # ETH >= $2,400 by Dec 31
        # ETH has been underperforming BTC. Current ~$2,000-2,200 range.
        # $2,400 is achievable but requires crypto-positive macro.
        "63ed5309-6d9e-47fe-88bb-1113a8e40780": {
            "direction": "bullish",
            "confidence": 0.55,
            "reasoning": (
                "ETH at $2,400 by Dec 31 requires ~10-20% appreciation from current "
                "levels, which is within Ethereum's 90-day realized volatility range. "
                "Protocol upgrades (EIP-4844/Dencun rollout benefits, L2 ecosystem "
                "growth) provide fundamental catalysts for ETH-specific upside. "
                "However, ETH has been chronically underperforming BTC on the "
                "ETH/BTC ratio throughout 2026, suggesting capital is flowing to BTC "
                "rather than ETH in this cycle. The Fed tightening headwind is "
                "particularly punishing for higher-beta crypto assets. Confidence at "
                "55% — just above coin-flip — reflects genuine uncertainty: ETH has "
                "the fundamental catalysts but lacks the momentum and capital flow "
                "patterns that would create higher conviction. If BTC rallies to $70K+, "
                "ETH likely benefits from rotation, but the timing and magnitude "
                "are uncertain."
            ),
        },

        # ETH <= $1,600 by Dec 31
        # $1,600 would be a major correction (~20-30% from current).
        # Unlikely absent systemic event.
        "b4a3c7da-06e0-4b25-9dba-a61825499898": {
            "direction": "bearish",  # bearish = "No, ETH won't be <= $1,600"
            "confidence": 0.68,
            "reasoning": (
                "ETH falling to $1,600 by Dec 31 would represent a ~25% decline from "
                "current levels and would erase all 2026 gains. This would require "
                "either: (1) a systemic crypto event (exchange failure, major protocol "
                "exploit); (2) a severe macro recession triggered by the 'stagflation "
                "cocktail' of rising rates + energy costs; or (3) a catastrophic "
                "regulatory intervention. None of these scenarios are base-case. "
                "Ethereum's staking yield (~4-5%) creates a structural income floor "
                "that incentivizes holding over selling. The L2 ecosystem (Arbitrum, "
                "Base, Optimism) generates growing fee revenue that supports "
                "fundamental valuation. The ETH/BTC ratio may continue to decline, "
                "but absolute ETH price is unlikely to breach $1,600 absent a black "
                "swan. Confidence at 68% (No) reflects high conviction that the "
                "downside scenario requires a tail event, while acknowledging that "
                "in a 3.5-month window with active geopolitical conflict and Fed "
                "tightening, tail events are more probable than in normal markets."
            ),
        },
    }

    print("=" * 70)
    print("CRYPTO PRICE EVENT FORECASTS")
    print("=" * 70)

    for cid, fc in crypto_forecasts.items():
        result = api_post(
            f"/api/v1/eval/price-events/challenges/{cid}/predict",
            fc, token,
        )
        label = "BTC" if "btc" in str(cid)[:20].lower() else ""
        print(f"    → {fc['direction']} ({fc['confidence']:.0%})")
        print(f"    Response: {json.dumps(result)[:200]}")
        print()

    # ─── SUMMARY ──────────────────────────────────────────────────────────

    print("=" * 70)
    print("ROUND 2B SUMMARY")
    print("=" * 70)
    print(f"  Civic forecasts submitted:  {len(civic_forecasts)}")
    print(f"  Crypto forecasts submitted: {len(crypto_forecasts)}")
    print(f"  Total Round 2 forecasts:    {9 + len(civic_forecasts) + len(crypto_forecasts)}")
    print(f"\n  Agent: https://headlinearena.com/agent/{load_creds()['agent_id']}")


if __name__ == "__main__":
    main()
