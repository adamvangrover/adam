import sys
import os
import time
import random
import json
from uuid import uuid4
from datetime import datetime

# Ensure core modules are in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.unified_ledger.schema import ParentOrder, ChildOrder, OrderSide, OrderType, ExecutionVenue, TimeInForce
from core.quantitative.pricing import AvellanedaStoikovModel
from core.quantitative.matching_engine import MatchingEngine
from core.simulation.scenarios.ccar_2025 import CCAR_2025

OUTPUT_FILE = "showcase/data/unified_banking_log.json"

def run_simulation():
    print("="*60)
    print("Initializing Unified Banking World Model Simulation")
    print(f"Scenario: {CCAR_2025.name}")
    print("="*60)

    # 1. Initialize Infrastructure
    matching_engine = MatchingEngine()
    symbol = "JPM_HY_CREDIT"

    # 2. Initialize Agents
    market_maker_algo = AvellanedaStoikovModel(
        gamma=0.5,
        sigma=CCAR_2025.vix_peak / 100.0,
        T=1/252,
        k=1.5
    )

    mm_inventory = 0.0
    mid_price = 100.00

    # Data Log
    simulation_log = {
        "metadata": {
            "scenario": CCAR_2025.name,
            "description": CCAR_2025.description,
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "mm_parameters": {
                "gamma": market_maker_algo.gamma,
                "sigma": market_maker_algo.sigma
            }
        },
        "ticks": [],
        "orders": []
    }

    # 3. Simulate Wealth Management Client Order
    parent_order = ParentOrder(
        client_id="FAMILY_OFFICE_A",
        strategy_tag="LIQUIDITY_PRESERVATION",
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=5000
    )

    chunks = [500, 1000, 1500, 1000, 1000] # More granular ticks
    prev_mm_bid_id = None
    prev_mm_ask_id = None

    for i, qty in enumerate(chunks):
        tick_data = {
            "tick": i + 1,
            "timestamp": datetime.utcnow().isoformat(),
            "mid_price_start": mid_price,
            "mm_inventory_start": mm_inventory
        }

        # A. Cancel Previous Quotes
        if prev_mm_bid_id:
            matching_engine.cancel_order(symbol, prev_mm_bid_id)
        if prev_mm_ask_id:
            matching_engine.cancel_order(symbol, prev_mm_ask_id)

        # B. Market Maker Updates Quotes
        bid, ask = market_maker_algo.get_quotes(mid_price, mm_inventory, time_remaining=(len(chunks)-i)/252)
        spread = ask - bid

        tick_data["mm_quote"] = {"bid": bid, "ask": ask, "spread": spread}

        # Submit MM orders
        mm_bid_order = ChildOrder(
            parent_id=uuid4(),
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=10000,
            price=round(bid, 2),
            order_type=OrderType.LIMIT,
            venue=ExecutionVenue.INTERNAL,
            desk_id="IB_MARKET_MAKER"
        )
        mm_ask_order = ChildOrder(
            parent_id=uuid4(),
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=10000,
            price=round(ask, 2),
            order_type=OrderType.LIMIT,
            venue=ExecutionVenue.INTERNAL,
            desk_id="IB_MARKET_MAKER"
        )
        matching_engine.process_order(mm_bid_order)
        matching_engine.process_order(mm_ask_order)

        prev_mm_bid_id = mm_bid_order.order_id
        prev_mm_ask_id = mm_ask_order.order_id

        # C. Wealth Management Execution
        child_order = ChildOrder(
            parent_id=parent_order.order_id,
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=qty,
            price=round(bid, 2),
            order_type=OrderType.LIMIT,
            venue=ExecutionVenue.INTERNAL,
            desk_id="WM_EXECUTION"
        )

        result = matching_engine.process_order(child_order)

        # Log Orders
        simulation_log["orders"].append({
            "tick": i + 1,
            "side": "SELL",
            "quantity": qty,
            "execution_price": round(bid, 2), # Assuming full fill at bid
            "filled": result['filled_quantity']
        })

        for fill in result['fills']:
            mm_inventory += fill['quantity']
            # Price Impact: Impact scales with Square Root of size (Almgren-Chriss)
            impact = 0.05 * (fill['quantity'] / 500)**0.5
            mid_price -= impact

        tick_data["mid_price_end"] = mid_price
        tick_data["mm_inventory_end"] = mm_inventory
        simulation_log["ticks"].append(tick_data)

        print(f"Tick {i+1}: Price {tick_data['mid_price_start']:.2f} -> {mid_price:.2f} | Inv {mm_inventory}")

    # Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(simulation_log, f, indent=2)

    print(f"\nSimulation Complete. Log saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_simulation()
