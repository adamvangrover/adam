import asyncio
import logging
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.system1.event_bus import EventBus
from core.system1.pheromone_engine import PheromoneEngine
from core.system1.workers.market_stream_worker import MarketStreamWorker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run_system1_test():
    """
    Spawns the System 1 Event Loop.
    Deploys the Event Bus, Pheromone Engine, and 5 simultaneous Market Stream Workers.
    """
    print("\n" + "="*80)
    print("INITIALIZING PRIORITY 3: SYSTEM 1 ASYNCHRONOUS SWARM")
    print("="*80 + "\n")
    
    bus = EventBus()
    engine = PheromoneEngine(event_bus=bus)
    
    # We create a special queue to listen for the System 2 Escalation event
    escalation_queue = bus.subscribe("ESCALATION")

    # Deploy 5 Micro-Workers simultaneously observing the mock market
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "JPM"]
    workers = []
    for ticker in tickers:
        base_px = random.uniform(50, 500)
        worker = MarketStreamWorker(ticker=ticker, event_bus=bus, base_price=base_px)
        workers.append(worker)

    # Boot up the ecosystem concurrently
    print("--- DEPLOYING SWARM TASKS TO ASYNCIO EVENT LOOP ---")
    
    engine_task = asyncio.create_task(engine.start())
    worker_tasks = [asyncio.create_task(w.start()) for w in workers]
    
    all_swarm_tasks = [engine_task] + worker_tasks

    try:
        # Wait up to 10 seconds for a Systemic Crash to organically trigger an Escalation
        print("\nWaiting for localized micro-anomalies to aggregate into a systemic escalation...\n")
        escalation_event = await asyncio.wait_for(escalation_queue.get(), timeout=15.0)
        
        print("\n" + "="*80)
        print("TEST SUCCESSFUL: ESCALATION EVENT RECIEVED BY MASTER SCRIPT")
        print(f"Entities requiring immediate Cognitive Deep Dive: {escalation_event['effected_entities']}")
        print("="*80 + "\n")
        
    except asyncio.TimeoutError:
        print("\nTest timed out. The market remained stable. No escalation required.")
        
    finally:
        # Graceful shutdown of the Swarm
        print("--- INITIATING SWARM RECALL (SHUTDOWN) ---")
        engine_task.cancel()
        for t in worker_tasks:
            t.cancel()
            
        # Await cancellations
        await asyncio.gather(*all_swarm_tasks, return_exceptions=True)
        print("System 1 Swarm offline.")

if __name__ == "__main__":
    asyncio.run(run_system1_test())
