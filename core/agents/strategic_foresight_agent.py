from typing import Dict, Any, List
import numpy as np
import logging
from core.research.oswm.inference import OSWMInference

# Configure logging
logger = logging.getLogger(__name__)

class StrategicForesightAgent:
    """
    The Strategic Foresight Agent (SFA) acts as the system's "Pre-Crime" division.
    It uses the One-Shot World Model (OSWM) to simulate future scenarios and
    identify regime shifts before they happen.
    """

    def __init__(self):
        self.oswm = OSWMInference()
        self.oswm.pretrain_on_synthetic_prior(steps=10) # Quick init for demo
        logger.info("Strategic Foresight Agent initialized with OSWM capability.")

    def generate_briefing(self, ticker: str = "SPY", horizon: int = 10) -> Dict[str, Any]:
        """
        Generates a strategic briefing based on OSWM predictions.
        """
        logger.info(f"Generating strategic briefing for {ticker} (Horizon: {horizon})...")

        # 1. Load Context
        # Using DataFetcher implicitly via OSWMInference
        context, stats = self.oswm.load_market_context(ticker, period="1mo")

        if len(context) < 10:
            logger.warning("Insufficient data for robust prediction.")
            return {"status": "INSUFFICIENT_DATA"}

        # 2. Run Simulations
        # Run multiple trajectories to estimate uncertainty
        num_sims = 5
        predictions = []
        for _ in range(num_sims):
            # We use the deterministic transformer for now, but in a full version
            # we might add dropout or temperature to the model to get variance.
            # Here we simulate 'uncertainty' by perturbing the input slightly
            perturbed_context = [c + np.random.normal(0, 0.01) for c in context[-10:]]
            pred = self.oswm.generate_scenario(perturbed_context, steps=horizon)
            predictions.append(pred)

        # 3. Analyze Results
        predictions = np.array(predictions)
        avg_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        # Denormalize
        mean, std = stats["mean"], stats["std"]
        avg_price = avg_pred * std + mean

        # Detect Regime Shift
        # If the last predicted price is significantly different (> 2 sigma) from current price
        current_price = context[-1] * std + mean
        final_price = avg_price[-1]

        drift = (final_price - current_price) / current_price
        volatility_forecast = np.mean(std_pred)

        status = "STABLE"
        if abs(drift) > 0.05:
            status = "REGIME_SHIFT_IMMINENT"
        elif volatility_forecast > 0.5:
            status = "HIGH_UNCERTAINTY"

        briefing = {
            "target": ticker,
            "status": status,
            "forecast_drift_pct": float(drift * 100),
            "volatility_index": float(volatility_forecast),
            "projected_path": avg_price.tolist(),
            "narrative": self._construct_narrative(ticker, status, drift)
        }

        logger.info(f"Briefing complete: {status}")
        return briefing

    def _construct_narrative(self, ticker, status, drift):
        if status == "REGIME_SHIFT_IMMINENT":
            direction = "CRASH" if drift < 0 else "melt-up"
            return f"CRITICAL ALERT: OSWM detects a {abs(drift*100):.1f}% {direction} trajectory for {ticker}. Hedging recommended."
        elif status == "HIGH_UNCERTAINTY":
            return f"WARNING: Market physics are breaking down for {ticker}. Predictive confidence is low."
        else:
            return f"Conditions for {ticker} appear nominal. Standard risk parameters apply."
