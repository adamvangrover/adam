import asyncio
import logging
import time
from typing import Dict, Any

class PheromoneEngine:
    """
    Listens to the 'PHEROMONE' topic on the Event Bus.
    Tracks anomalous signals dropped by System 1 micro-workers.
    If conditions breach a predefined threshold, it escalates to System 2.
    """
    
    def __init__(self, event_bus):
        self.bus = event_bus
        self.queue = self.bus.subscribe("PHEROMONE")
        
        # State tracking: {ticker: {"count": int, "last_seen": timestamp}}
        self.alert_state: Dict[str, Dict[str, Any]] = {}
        
        # Escalation Thresholds
        self.escalation_window = 5.0  # seconds
        self.escalation_count = 3     # alerts required across different tickers
        
        self.is_running = False
        logging.info("Pheromone Engine Initialized and Listening.")

    async def start(self):
        """
        Starts the continuous listening loop.
        """
        self.is_running = True
        try:
            while self.is_running:
                # Wait for a new Pheromone message
                message = await self.queue.get()
                await self._process_pheromone(message)
                self.queue.task_done()
        except asyncio.CancelledError:
            logging.info("Pheromone Engine Shutdown Sequence Initiated.")
            self.is_running = False

    async def _process_pheromone(self, message: Dict[str, Any]):
        """
        Analyzes the incoming pheromone to decide if a systemic event is occurring.
        """
        ticker = message.get("ticker", "UNKNOWN")
        severity = message.get("severity", "YELLOW")
        reason = message.get("reason", "Anomaly Detected")
        
        logging.warning(f"[PHEROMONE DROP] Worker flagged [{ticker}]: {severity} - {reason}")
        
        # Record the State
        current_time = time.time()
        self.alert_state[ticker] = {
            "count": self.alert_state.get(ticker, {}).get("count", 0) + 1,
            "last_seen": current_time
        }
        
        # Clean stale pheromones (older than the window) before checking thresholds
        self._prune_stale_pheromones(current_time)
        
        # Check Systemic Threshold (Are multiple different things breaking at once?)
        active_tickers = list(self.alert_state.keys())
        if len(active_tickers) >= self.escalation_count:
            await self._trigger_system2_escalation(active_tickers)
            # Reset state after escalation to avoid spamming
            self.alert_state.clear()
            
    def _prune_stale_pheromones(self, current_time: float):
        """
        Removes pheromones that have expired past the temporal window.
        """
        stale_keys = []
        for ticker, data in self.alert_state.items():
            if current_time - data["last_seen"] > self.escalation_window:
                stale_keys.append(ticker)
        
        for k in stale_keys:
            del self.alert_state[k]

    async def _trigger_system2_escalation(self, active_tickers: list):
        """
        The critical bridge: System 1 demands Attention from System 2.
        """
        logging.critical("="*60)
        logging.critical(f"SYSTEM 2 ESCALATION TRIGGERED!")
        logging.critical(f"Aggregated Pheromones detected critical correlation across: {', '.join(active_tickers)}")
        logging.critical("Initiating Cognitive Orchestrator for Deep Dive Analysis...")
        logging.critical("="*60)
        
        # In a full deployment, this would invoke the LangGraph from Priority 1.
        # For this test, we publish an ESCALATION event back to the bus for the test harness to catch.
        await self.bus.publish("ESCALATION", {"status": "triggered", "effected_entities": active_tickers})
