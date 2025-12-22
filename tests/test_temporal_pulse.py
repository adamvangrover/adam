import asyncio
import logging
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Ensure repo root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.system.temporal_engine import TemporalEngine, PulseTask
from core.procedures.autonomous_update import RoutineMaintenance

# Mock AgentOrchestrator to avoid heavy initialization during unit test
# But we can try to import it to ensure no syntax errors
try:
    from core.system.agent_orchestrator import AgentOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestTemporalPulse")

class TestTemporalEngine(unittest.TestCase):

    def test_pulse_task_initialization(self):
        async def dummy_coro():
            pass
        task = PulseTask("TestTask", dummy_coro, interval_seconds=10)
        self.assertEqual(task.name, "TestTask")
        self.assertEqual(task.interval, 10)
        self.assertFalse(task.is_running)

    def test_engine_registration(self):
        engine = TemporalEngine()
        async def dummy_coro():
            pass
        engine.register_task("TestTask", dummy_coro, 60)
        self.assertIn("TestTask", engine.tasks)

class TestAsyncTemporalEngine(unittest.IsolatedAsyncioTestCase):

    async def test_engine_execution(self):
        engine = TemporalEngine()
        counter = {"count": 0}

        async def increment_counter():
            counter["count"] += 1
            logger.info(f"Counter incremented to {counter['count']}")

        # Register a fast task
        engine.register_task("FastTask", increment_counter, interval_seconds=1, run_immediately=True)

        # Start engine in background
        task = asyncio.create_task(engine.start())

        # Let it run for 2.5 seconds (should run immediately + at 1s + at 2s = 3 times ideally,
        # or immediately + 1s marker + 2s marker.
        # run_immediately runs separate from the loop check.
        # Loop checks every 1s.
        # 0s: run_immediately fires. last_run updated.
        # 0s: loop start.
        # 1s: loop wake. 1s >= 1s? Yes. Fire. last_run updated.
        # 2s: loop wake. 1s >= 1s? Yes. Fire. last_run updated.
        await asyncio.sleep(2.5)

        engine.stop()
        await task # Wait for it to finish gracefully

        logger.info(f"Final count: {counter['count']}")
        self.assertGreaterEqual(counter["count"], 2, "Task should have run at least twice (immediate + interval)")

    async def test_routine_maintenance_init(self):
        # We verify we can instantiate RoutineMaintenance
        # This will create DataIngestionAgent which creates IngestionEngine
        # We might need to mock things if they require actual file structure or heavy deps

        # Mock DataIngestionAgent to avoid side effects
        with patch('core.procedures.autonomous_update.DataIngestionAgent') as MockAgent:
            routine = RoutineMaintenance(data_dir="tests/test_data")
            self.assertIsNotNone(routine)
            MockAgent.assert_called_once()

            # Verify methods exist
            self.assertTrue(hasattr(routine, 'run_market_data_refresh'))
            self.assertTrue(hasattr(routine, 'run_deep_discovery'))

if __name__ == '__main__':
    unittest.main()
