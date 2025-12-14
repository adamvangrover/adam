# core/engine/unified_knowledge_graph.py

"""
Manages the integration of the FIBO domain ontology and the W3C PROV-O provenance ontology.

This module provides the core interface for the Neuro-Symbolic Planner to query
the two-layer knowledge graph. It abstracts the underlying graph database (e.g., Neo4j,
TerminusDB) and provides methods for complex, cross-ontology queries.

Key Components:
- GraphDB Connector: Handles the connection to the underlying graph database.
- FIBO Query Interface: Provides methods to query financial concepts and
  relationships based on the Financial Industry Business Ontology.
- PROV-O Query Interface: Provides methods to query data lineage and provenance
  based on the W3C Provenance Ontology.
- Unified Query Engine: Allows the planner to run queries that traverse both
  FIBO and PROV-O simultaneously, creating a fully verifiable reasoning chain.
"""

import networkx as nx
import logging
import json
import os
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared instance for Singleton pattern
_SHARED_GRAPH_INSTANCE = None

class UnifiedKnowledgeGraph:
    def __init__(self):
        """
        Initializes the in-memory Knowledge Graph.
        Uses a Singleton pattern to avoid re-parsing the seed file on every instantiation.
        """
        global _SHARED_GRAPH_INSTANCE
        if _SHARED_GRAPH_INSTANCE is None:
            logger.info("Initializing Shared Knowledge Graph...")
            self.graph = nx.DiGraph()
            self._ingest_fibo_ontology()
            self._ingest_provenance_data()
            self._ingest_seed_data()
            _SHARED_GRAPH_INSTANCE = self.graph
        else:
            # Reuse the shared graph instance
            self.graph = _SHARED_GRAPH_INSTANCE
        
    def _ingest_fibo_ontology(self):
        """
        Mocks the ingestion of FIBO ontology nodes and edges.
        """
        logger.info("Ingesting FIBO Ontology...")
        # Define some core financial concepts and relationships
        triples = [
            # Core Financials
            ("Company", "has_risk_profile", "RiskProfile"),
            ("Company", "issues", "FinancialReport"),
            ("FinancialReport", "contains", "FinancialData"),
            ("RiskProfile", "depends_on", "FinancialData"),
            ("RiskProfile", "depends_on", "MarketData"),
            ("MarketData", "affects", "Volatility"),
            ("Volatility", "affects", "RiskScore"),
            ("RiskScore", "determines", "CreditRating"),

            # ESG & Compliance
            ("Company", "has_esg_score", "ESGScore"),
            ("ESGScore", "influenced_by", "CarbonFootprint"),
            ("ESGScore", "influenced_by", "GovernanceStructure"),
            ("Company", "subject_to", "Regulation"),
            ("Regulation", "enforced_by", "RegulatoryBody"),
            ("RegulatoryBody", "issues", "RegulatoryFine"),
            ("RegulatoryFine", "impacts", "FinancialData"),

            # Macroeconomics
            ("MacroEvent", "impacts", "MarketSector"),
            ("MarketSector", "contains", "Company"),
            ("InterestRate", "affects", "CostOfCapital"),
            ("InterestRate", "affects", "ConsumerSpend"),
            ("CostOfCapital", "affects", "Valuation"),
            ("Inflation", "affects", "ConsumerSpend"),
            ("ConsumerSpend", "affects", "Revenue"),
            ("Revenue", "is_part_of", "FinancialData"),

            # Red Teaming / Adversarial
            ("CyberAttack", "targets", "Company"),
            ("CyberAttack", "impacts", "OperationalStability"),
            ("OperationalStability", "affects", "RiskScore"),
            ("ShortSellerReport", "challenges", "Valuation"),

            # Market Sentiment & News
            ("NewsArticle", "mentions", "Company"),
            ("NewsArticle", "expresses", "Sentiment"),
            ("Sentiment", "influences", "StockPrice"),
            ("SocialMediaPost", "amplifies", "NewsArticle"),
            ("SocialMediaPost", "indicates", "RetailSentiment"),
            ("RetailSentiment", "drives", "Volatility"),
            ("AnalystUpgrade", "boosts", "Sentiment"),
            ("AnalystDowngrade", "dampens", "Sentiment"),

            # Domain specific instances (legacy fallback)
            ("Apple Inc.", "is_a", "Company"),
            ("Apple Inc.", "belongs_to", "TechnologySector"),
            ("TechnologySector", "is_a", "MarketSector"),
            ("Apple 10-K", "is_a", "FinancialReport"),
            ("AAPL Stock", "is_a", "MarketData"),
            ("SEC", "is_a", "RegulatoryBody"),
            ("GDPR", "is_a", "Regulation"),
            ("Fed Rate Hike", "is_a", "InterestRate")
        ]
        for u, r, v in triples:
            self.graph.add_edge(u, v, relation=r, type="fibo")

    def _ingest_provenance_data(self):
        """
        Mocks W3C PROV-O metadata.
        """
        logger.info("Ingesting PROV-O Metadata...")
        # Link data sources to agents/processes
        self.graph.add_node("Apple 10-K", prov_source="SEC EDGAR", prov_time="2023-09-30")
        self.graph.add_node("AAPL Stock", prov_source="Bloomberg", prov_time="2023-10-27")
        self.graph.add_node("Fed Rate Hike", prov_source="Federal Reserve API", prov_time="2023-11-01")
        self.graph.add_node("GDPR", prov_source="EU Law Database", prov_time="2018-05-25")

    def _ingest_seed_data(self):
        """
        Ingests data from data/v23_ukg_seed.json to populate the graph dynamically.
        """
        filepath = "data/v23_ukg_seed.json"
        if not os.path.exists(filepath):
            logger.warning(f"Seed data not found at {filepath}")
            return

        try:
            logger.info(f"Ingesting seed data from {filepath}...")
            with open(filepath, 'r') as f:
                data = json.load(f)

            ukg_root = data.get("v23_unified_knowledge_graph", {})
            nodes_data = ukg_root.get("nodes", {})

            lei_to_name = {}
            node_id_to_name = {}

            # 1. Legal Entities
            for entity in nodes_data.get("legal_entities", []):
                name = entity.get("legal_name")
                lei = entity.get("lei_code")
                nid = entity.get("node_id")

                if name:
                    if lei:
                        lei_to_name[lei] = name
                    if nid:
                        node_id_to_name[nid] = name

                    # Add Node with all metadata
                    self.graph.add_node(name, **entity, type="LegalEntity")
                    self.graph.add_edge(name, "Company", relation="is_a", type="fibo")

                    # Link to Sector
                    sector = entity.get("sector")
                    if sector:
                        self.graph.add_edge(name, sector, relation="belongs_to", type="fibo")
                        if sector not in self.graph:
                            self.graph.add_edge(sector, "MarketSector", relation="is_a", type="fibo")

            # 2. Supply Chain Relations
            for rel in nodes_data.get("supply_chain_relations", []):
                src = lei_to_name.get(rel.get("source_lei"))
                tgt = lei_to_name.get(rel.get("target_lei"))
                if src and tgt:
                    self.graph.add_edge(src, tgt,
                                      relation=rel.get("relationship_type", "connected_to"),
                                      type="supply_chain",
                                      criticality=rel.get("criticality_score"),
                                      desc=rel.get("dependency_description"))

            # 3. Macro Indicators
            for macro in nodes_data.get("macro_indicators", []):
                name = macro.get("name")
                if name:
                    self.graph.add_node(name, **macro, type="MacroIndicator")
                    impacts = macro.get("impact_map", {})
                    for sector, impact_type in impacts.items():
                        if sector not in self.graph:
                             self.graph.add_edge(sector, "MarketSector", relation="is_a", type="fibo")

                        self.graph.add_edge(name, sector,
                                          relation="impacts",
                                          impact_type=impact_type,
                                          type="macro_impact")

            # 4. Crisis Scenarios
            sim_params = ukg_root.get("simulation_parameters", {})
            for scenario in sim_params.get("crisis_scenarios", []):
                name = scenario.get("name")
                if name:
                    # Avoid conflict with 'type' kwarg
                    scenario_attrs = scenario.copy()
                    if 'type' in scenario_attrs:
                        scenario_attrs['scenario_type'] = scenario_attrs.pop('type')

                    self.graph.add_node(name, **scenario_attrs, type="CrisisScenario")
                    # Link to affected nodes if explicitly listed
                    for affected_id in scenario.get("affected_nodes", []):
                        target_name = node_id_to_name.get(affected_id)
                        if target_name:
                             self.graph.add_edge(name, target_name, relation="affects", type="scenario_impact")

            # 5. Regulatory Rules
            reg_rules = ukg_root.get("regulatory_rules", {})
            for framework, rules in reg_rules.items():
                framework_node = f"Framework: {framework.upper()}"
                self.graph.add_node(framework_node, type="RegulatoryFramework")

                for rule in rules:
                    rule_name = rule.get("name")
                    if rule_name:
                        self.graph.add_node(rule_name, **rule, type="Regulation")
                        self.graph.add_edge(framework_node, rule_name, relation="contains", type="regulatory_structure")
                        self.graph.add_edge(rule_name, "Company", relation="applies_to", type="regulatory_scope")

            logger.info("Seed data ingestion complete.")

        except Exception as e:
            logger.error(f"Failed to ingest seed data: {e}", exc_info=True)

    def find_symbolic_path(self, start_concept: str, end_concept: str) -> Optional[List[Dict[str, str]]]:
        """
        Finds a reasoning path between two concepts.
        """
        try:
            path = nx.shortest_path(self.graph, source=start_concept, target=end_concept)
            # Convert nodes to a list of steps with relationships
            symbolic_plan = []
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i+1]
                edge_data = self.graph.get_edge_data(u, v)
                relation = edge_data.get("relation", "connected_to")
                symbolic_plan.append({
                    "source": u,
                    "relation": relation,
                    "target": v,
                    "provenance": self.graph.nodes[u].get("prov_source", "Inferred")
                })
            return symbolic_plan
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {start_concept} and {end_concept}")
            return None
        except nx.NodeNotFound as e:
            logger.warning(f"Node not found in KG: {e}")
            return None

    def query_node_metadata(self, node_name: str) -> Dict[str, Any]:
        if node_name in self.graph.nodes:
            return self.graph.nodes[node_name]
        return {}
