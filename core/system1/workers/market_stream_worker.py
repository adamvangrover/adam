import asyncio
import random
import logging
from typing import Dict, Any

class MarketStreamWorker:
    """
    An ultra-lightweight async micro-worker constantly polling simulated WebSocket feeds.
    Runs concurrently within the System 1 Event Loop.
    Does NOT do heavy computing. It observes baseline variance and drops Pheromones if breached.
    """
    
    def __init__(self, ticker: str, event_bus, base_price: float = 100.0):
        self.ticker = ticker
        self.bus = event_bus
        
        self.current_price = base_price
        self.baseline_volatility = 0.05  # Normal tick variance is ~0-5%
        self.anomaly_threshold = 0.15    # Drop a Pheromone if variance jumps >15%
        
        self.is_running = False
        logging.info(f"System 1 Worker Deployed for [{self.ticker}]")

    async def start(self):
        """
        The continuous observation loop.
        """
        self.is_running = True
        try:
            while self.is_running:
                # Simulate waiting for the next data tick from the hypothetical Exchange WebSocket
                await asyncio.sleep(random.uniform(0.1, 1.0))
                
                # Ingest the mock tick
                tick = self._simulate_tick()
                await self._process_tick(tick)
                
        except asyncio.CancelledError:
            logging.info(f"Micro-Worker [{self.ticker}] Terminated.")
            self.is_running = False

    def _simulate_tick(self) -> Dict[str, Any]:
        """
        Mocks a real-time price/sentiment tick.
        Occasionally throws intentional massive anomalies to test the Pheromone Engine.
        """
        # 25% chance of a massive market shock to this specific ticker to force the test
        if random.random() < 0.25:
            variance = random.uniform(0.20, 0.40) # 20-40% crash
            direction = -1
        else:
            variance = random.uniform(0, self.baseline_volatility)
            direction = random.choice([1, -1])
            
        new_price = self.current_price * (1 + (variance * direction))
        self.current_price = new_price
        
        return {
            "price": new_price,
            "variance": variance,
            "sentiment": "PANIC" if variance > self.anomaly_threshold else "NEUTRAL"
        }

    async def _process_tick(self, tick: Dict[str, Any]):
        """
        Reflexive reaction to a new data point.
        """
        variance = tick["variance"]
        
        if variance > self.anomaly_threshold:
            # We detected something bizarre. Drop a PHEROMONE flag onto the Bus for the Engine.
            # Then immediately abandon the tick and await the next one.
            message = {
                "source_worker": "MarketStreamWorker",
                "ticker": self.ticker,
                "severity": "RED_ALERT" if variance > 0.25 else "YELLOW_ALERT",
                "reason": f"Volatile shock detected: {variance*100:.2f}% shift.",
                "context": tick
            }
            logging.warning(f"[{self.ticker}] Anomaly Detected! Dropping PHEROMONE onto Event Bus.")
            await self.bus.publish("PHEROMONE", message)
            
        else:
            # Normal market activity. Discard and move on.
            pass
