"""
Script: Live Market Updater
===========================
Runs a continuous loop to simulate 'Live' market activity.
Updates `showcase/js/market_snapshot.js` in real-time.

Usage:
  python scripts/live_market_updater.py --interval 2 --continuous --scenario BULL_RALLY
  python scripts/live_market_updater.py --dynamic-scenarios
"""

import time
import argparse
import sys
import os
import random

# Add core to path
sys.path.append(os.getcwd())

from core.market_data.manager import MarketDataManager
from core.market_data.scenarios import SCENARIOS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=2.0, help="Update interval in seconds")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run (if not continuous)")
    parser.add_argument("--continuous", action="store_true", help="Run indefinitely")
    parser.add_argument("--scenario", type=str, help="Initial scenario to activate")
    parser.add_argument("--dynamic-scenarios", action="store_true", help="Randomly switch scenarios over time")
    args = parser.parse_args()

    manager = MarketDataManager()

    if args.scenario:
        print(manager.set_scenario(args.scenario))

    print(f"Starting Live Market Updater (Interval: {args.interval}s)...")

    step = 0
    scenario_timer = 0

    try:
        while args.continuous or step < args.steps:
            manager.simulate_step()

            scenario_name = manager.active_scenario.name
            msg = f"\rStep {step+1}: Market updated [{scenario_name}]"
            sys.stdout.write(msg.ljust(60)) # Pad to clear line
            sys.stdout.flush()

            # Dynamic Switching Logic
            if args.dynamic_scenarios:
                scenario_timer += 1
                # Switch every ~20 steps (probabilistic)
                if scenario_timer > 20 and random.random() < 0.1:
                    new_scenario = random.choice(list(SCENARIOS.keys()))
                    print(f"\n[EVENT] Switching Regime -> {new_scenario}")
                    manager.set_scenario(new_scenario)
                    scenario_timer = 0

            time.sleep(args.interval)
            step += 1
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print("\nLive update session ended.")

if __name__ == "__main__":
    main()
