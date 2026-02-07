import sys
import os
import logging

# Ensure we can import from core
sys.path.append(os.getcwd())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ZombieCheck")

def run_zombie_check():
    logger.info("🟢 SYSTEM AUDIT: ZOMBIE CHECK INITIATED")
    logger.info("Target Sector: Legacy Industrials")
    logger.info("Scenario: 'The Stellantis Reset' (-25% Demand Shock, +200bps Cost of Capital)")
    logger.info("-" * 60)

    # Mock Data for Industrial Giants (Approximate values for simulation)
    # Figures in Millions USD
    targets = {
        "STLA (Patient Zero)": {"ebitda": 25000, "interest_expense": 2000, "debt": 30000},
        "GM (General Motors)": {"ebitda": 18000, "interest_expense": 1500, "debt": 40000},
        "F (Ford Motor Co)":   {"ebitda": 12000, "interest_expense": 1800, "debt": 100000}, # Ford Credit debt is huge
        "BA (Boeing)":         {"ebitda": 4000,  "interest_expense": 2500, "debt": 50000},
        "GE (General Electric)": {"ebitda": 7000, "interest_expense": 800, "debt": 20000},
        "CAT (Caterpillar)":   {"ebitda": 13000, "interest_expense": 600, "debt": 35000},
    }

    # Simulation Parameters
    demand_shock = -0.25  # 25% drop in revenue -> assuming 1.5x operating leverage -> 37.5% drop in EBITDA
    operating_leverage = 1.5
    rate_shock_bps = 200

    # Base Rate Assumption (e.g., effective rate on debt)
    base_rate = 0.05

    print(f"{'TICKER':<25} | {'ICR (Pre)':<10} | {'ICR (Post)':<10} | {'STATUS'}")
    print("-" * 60)

    zombie_count = 0

    for ticker, data in targets.items():
        # Pre-Shock Metrics
        pre_ebitda = data['ebitda']
        pre_interest = data['interest_expense']
        pre_icr = pre_ebitda / pre_interest if pre_interest else 99.9

        # Apply Shock
        # EBITDA Hit: Revenue drop * Operating Leverage
        ebitda_shock_pct = demand_shock * operating_leverage
        post_ebitda = pre_ebitda * (1 + ebitda_shock_pct)

        # Interest Hit: Refinancing risk?
        # Let's assume 20% of debt floats or needs refi at higher rates immediately
        floating_portion = 0.20
        debt = data['debt']
        # Old Interest + (Floating Debt * Rate Increase)
        added_interest = (debt * floating_portion) * (rate_shock_bps / 10000.0)
        post_interest = pre_interest + added_interest

        post_icr = post_ebitda / post_interest if post_interest else 0.0

        # Status
        status = "HEALTHY"
        if post_icr < 1.0:
            status = "🧟 ZOMBIE"
            zombie_count += 1
        elif post_icr < 2.0:
            status = "⚠️ AT RISK"
        elif post_icr < pre_icr * 0.7:
             status = "📉 STRESSED"

        print(f"{ticker:<25} | {pre_icr:>9.2f}x | {post_icr:>9.2f}x | {status}")

    print("-" * 60)
    print(f"CONCLUSION: {zombie_count} / {len(targets)} firms failed the stress test.")
    if zombie_count > 0:
        print("ALERT: Contagion risk is NON-ZERO. The 'Stellantis Reset' is spreading.")
    else:
        print("RESULT: Sector appears resilient, but margins are compressing.")

if __name__ == "__main__":
    run_zombie_check()
