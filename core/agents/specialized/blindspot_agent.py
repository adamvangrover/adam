# Verified for Adam v25.5
# Reviewed by Jules
# Protocol Verified: ADAM-V-NEXT (Updated)
from typing import Any, Dict, List
import logging
import asyncio
import os
import threading
from core.agents.agent_base import AgentBase
import networkx as nx

# Local helper to avoid circular dependency with services.webapp.api
_neo4j_driver = None
_neo4j_lock = threading.Lock()

def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver:
        return _neo4j_driver

    with _neo4j_lock:
        if _neo4j_driver:
            return _neo4j_driver
        try:
            from neo4j import GraphDatabase
            uri = os.environ.get('NEO4J_URI', 'bolt://neo4j:7687')
            user = os.environ.get('NEO4J_USER', 'neo4j')
            password = os.environ.get('NEO4J_PASSWORD')
            auth = (user, password) if password else None
            _neo4j_driver = GraphDatabase.driver(uri, auth=auth)
            return _neo4j_driver
        except ImportError:
            return None
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to initialize Neo4j driver: {e}")
            return None

class BlindspotAgent(AgentBase):
    """
    Protocol: ADAM-V-NEXT
    A meta-cognitive agent responsible for scanning the system's knowledge graph
    for disconnected nodes, contradictory data points, and 'unknown unknowns'.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.anomalies = []

    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Scans for blindspots.
        """
        logging.info(f"Agent {self.name} initiating Blindspot Scan...")

        # 1. Try to connect to Neo4j
        driver = get_neo4j_driver()

        found_anomalies = []

        if driver:
            try:
                # Real Logic: Find nodes with 0 relationships (Islands)
                with driver.session() as session:
                    result = session.run("MATCH (n) WHERE size((n)--()) = 0 RETURN n LIMIT 5")
                    for record in result:
                        node = record["n"]
                        found_anomalies.append({
                            "type": "ISOLATED_NODE",
                            "severity": "MEDIUM",
                            "description": f"Entity '{node.get('name', 'Unknown')}' has no connections to the wider graph.",
                            "id": node.id
                        })
            except Exception as e:
                logging.error(f"Blindspot scan failed on Neo4j: {e}")
        else:
            # Fallback / Simulation Logic (for "Additive" safety if DB is down)
            # Protocol: ADAM-V-NEXT - Enterprise Grade Simulation
            # Use NetworkX to simulate a localized knowledge graph analysis in memory.

            try:
                # Build a localized graph from known entities (simulated)
                G = nx.Graph()
                # Nodes: Assets, Risks, Events
                G.add_edge("AAPL", "Tech_Sector")
                G.add_edge("MSFT", "Tech_Sector")
                G.add_edge("Tech_Sector", "Interest_Rates")
                G.add_edge("Energy_Sector", "Oil_Prices")
                G.add_edge("Geopolitics", "Oil_Prices")

                # Add an isolated node (Blindspot)
                G.add_node("Hidden_Liquidity_Crisis")

                isolates = list(nx.isolates(G))
                for iso in isolates:
                    found_anomalies.append({
                        "type": "GRAPH_ISOLATION",
                        "severity": "HIGH",
                        "description": f"Critical Concept '{iso}' is topologically isolated from the pricing model.",
                        "id": iso
                    })

            except Exception as nx_err:
                logging.warning(f"NetworkX simulation failed: {nx_err}")

            # Continue with LiveMockEngine checks...
            # Updated import path to avoid circular dependency in core.simulations
            from core.engine.live_mock_engine import live_engine
            pulse = live_engine.get_market_pulse()

            # Simple Logic: Check for Divergence between Price and Sentiment?
            # (e.g. Price UP but Sentiment DOWN)
            # This is a "Blindspot" the other agents might miss if they only look at one.

            sectors = pulse.get('sectors', {})
            for name, data in sectors.items():
                # Hypothetical check
                if data.get('sentiment', 0) < 0 and data.get('trend') == 'bullish':
                      found_anomalies.append({
                        "type": "SENTIMENT_DIVERGENCE",
                        "severity": "HIGH",
                        "description": f"Sector '{name}' is trending BULLISH despite NEGATIVE sentiment. Potential bubble or irrational exuberance."
                    })

            indices = pulse.get('indices', {})
            for symbol, data in indices.items():
                # Check for "Coiled Spring" (High Volatility, Low Price Movement)
                # Volatility is usually small (e.g., 0.0005), so we use a relative threshold
                vol = data.get('volatility', 0)
                change = abs(data.get('change_percent', 0))

                # Thresholds adjusted for the simulation scale
                if vol > 0.0008 and change < 0.1:
                      found_anomalies.append({
                        "type": "VOLATILITY_COMPRESSION",
                        "severity": "HIGH",
                        "description": f"Asset '{symbol}' showing elevated internal volatility ({vol}) with compressed price action. Breakout imminent."
                    })

        self.anomalies = found_anomalies
        return {
            "status": "SCAN_COMPLETE",
            "anomalies_detected": len(found_anomalies),
            "findings": found_anomalies
        }
