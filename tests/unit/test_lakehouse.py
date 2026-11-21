import os
import unittest
import tempfile
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, Any, List

# Create local mock schemas so the tests don't break if core/schemas/hnasp is not available
class HNASPMeta(BaseModel):
    agent_id: str
    version: str
    session_id: str
    timestamp: str

class LogicLayer(BaseModel):
    business_rules: Dict[str, Any]
    margin_requirements: Dict[str, float]
    regulatory_boundaries: List[str]

class PersonaState(BaseModel):
    evaluation: float
    potency: float
    activity: float
    focus: str

class HNASPState(BaseModel):
    meta: HNASPMeta
    logic: LogicLayer
    persona: PersonaState
    context: Dict[str, Any]

# Since we define mock schemas for testing, we must monkeypatch the lakehouse to use them
import core.hnasp.lakehouse as lakehouse_module
lakehouse_module.HNASP = HNASPState

class TestObservationLakehouse(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.lakehouse = lakehouse_module.ObservationLakehouse(storage_path=self.test_dir)

        # Helper to create valid HNASP state
        meta = HNASPMeta(
            agent_id="test_agent",
            version="1.0",
            session_id="session123",
            timestamp="2026-03-31T00:00:00Z"
        )
        logic = LogicLayer(
            business_rules={"rule1": "val"},
            margin_requirements={"margin": 5.0},
            regulatory_boundaries=["limit1"]
        )
        persona = PersonaState(
            evaluation=1.0,
            potency=1.0,
            activity=1.0,
            focus="testing"
        )
        self.hnasp = HNASPState(
            meta=meta,
            logic=logic,
            persona=persona,
            context={"data": "none"}
        )

    def tearDown(self):
        self.lakehouse.close()
        # Clean up DuckDB and Parquet files
        for f in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, f))
        os.rmdir(self.test_dir)

    def test_save_and_load_latest_state(self):
        self.lakehouse.save_trace(self.hnasp)
        loaded = self.lakehouse.load_latest_state("test_agent")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.meta.agent_id, "test_agent")
        self.assertEqual(loaded.persona.focus, "testing")



    def test_query_traces(self):
        self.lakehouse.save_trace(self.hnasp)
        self.lakehouse.save_trace(self.hnasp)

        traces = self.lakehouse.query_traces("test_agent", limit=2)
        self.assertEqual(len(traces), 2)

        traces = self.lakehouse.query_traces("test_agent", limit=1)
        self.assertEqual(len(traces), 1)

    def test_query_analytical(self):
        self.lakehouse.save_trace(self.hnasp)
        df = self.lakehouse.query_analytical("SELECT COUNT(*) as count FROM traces")
        self.assertEqual(df.iloc[0]['count'], 1)

if __name__ == '__main__':
    unittest.main()
