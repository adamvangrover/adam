import asyncio
import os
import sys
import unittest

# Ensure core is in path
sys.path.append(os.getcwd())

from core.system.nexus_zero_orchestrator import NexusZeroOrchestrator
from core.v23_graph_engine.odyssey_knowledge_graph import OdysseyKnowledgeGraph


class TestOdysseyFlow(unittest.TestCase):
    def test_graph_ingestion(self):
        kg = OdysseyKnowledgeGraph()
        entity = {
            "@id": "urn:fibo:be-le-cb:Corporation:US-Test",
            "@type": "fibo-be-le-cb:Corporation",
            "legalName": "Test Entity",
            "hasCreditFacility": []
        }
        kg.ingest_odyssey_entity(entity)
        self.assertIn("urn:fibo:be-le-cb:Corporation:US-Test", kg.graph.nodes)

    def test_j_crew_detection(self):
        kg = OdysseyKnowledgeGraph()
        # Setup risky scenario
        entity_id = "urn:fibo:be-le-cb:Corporation:US-Risky"
        kg.graph.add_node(entity_id, type="fibo-be-le-cb:Corporation")

        # Unrestricted Sub
        sub_id = "urn:sub:Unrestricted"
        kg.graph.add_node(sub_id, type="lending:UnrestrictedSubsidiary")
        kg.graph.add_edge(entity_id, sub_id, relation="OWNS")

        # Facility without blocker
        fac_id = "urn:fac:TermLoan"
        kg.graph.add_node(fac_id, type="CreditFacility", has_jcrew_blocker=False)
        kg.graph.add_edge(entity_id, fac_id, relation="BORROWS")

        risk = kg.detect_j_crew_maneuver(entity_id)
        self.assertTrue(risk["detected"])

    def test_nexus_zero_orchestration(self):
        orchestrator = NexusZeroOrchestrator()
        payload = {
            "@id": "urn:fibo:be-le-cb:Corporation:US-Flow",
            "@type": "fibo-be-le-cb:Corporation",
            "legalName": "Flow Corp",
            "hasCreditFacility": []
        }

        async def run():
            return await orchestrator.run_analysis("Check risk", payload)

        result = asyncio.run(run())
        self.assertIn("final_decision", result)
        self.assertIn("decision_xml", result["final_decision"])

if __name__ == "__main__":
    unittest.main()
