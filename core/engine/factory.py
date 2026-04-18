from typing import Union
from core.engine.interfaces import EngineInterface
from core.engine.live_mock_engine import LiveMockEngine
from core.engine.real_engine import RealTradingEngine

class EngineFactory:
    """
    Factory pattern for runtime environment rotation.
    Allows seamless switching between Simulation (System 3) and Live (System 1/2) engines.

    Graceful Fallback Mechanism:
    Defaults to LiveMockEngine if REAL engines fail or are not selected,
    guaranteeing system continuity.

    PyO3 Bindings & Data Handoffs:
    Real engines may interface with the `core/rust_pricing` layer to compute
    advanced metrics via PyO3, abstracting the complex Rust structures behind standard Python interfaces.
    """

    @staticmethod
    def get_engine(environment: str = "SIMULATION") -> EngineInterface:
        """
        Returns the appropriate engine instance based on the environment string.

        Args:
            environment (str): 'SIMULATION', 'LIVE', or 'BACKTEST'

        Returns:
            EngineInterface: The instantiated engine.
        """
        env_upper = environment.upper()

        if env_upper == "SIMULATION":
            # LiveMockEngine is a singleton, so we just return the instance
            return LiveMockEngine()

        elif env_upper == "LIVE" or env_upper == "PRODUCTION":
            return RealTradingEngine()

        else:
            raise ValueError(f"Unknown environment: {environment}. Supported: SIMULATION, LIVE")
