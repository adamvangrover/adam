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

    def ingest_repo_graph(self, repo_graph_data: Dict[str, Any]):
        """Ingests the Repo Graph (Self-Awareness)."""
        logger.info("Ingesting Repo Graph...")
        subgraph = nx.node_link_graph(repo_graph_data)
        self.graph = nx.compose(self.graph, subgraph)
        logger.info(f"Graph now has {self.graph.number_of_nodes()} nodes.")

    def ingest_financial_data(self, companies: List[Dict[str, Any]]):
        """Ingests financial entities."""
        for company in companies:
            node_id = company.get("symbol") or company.get("company_id")
            if not node_id:
                continue

            self.graph.add_node(
                node_id,
                type="Company",
                sector=company.get("sector"),
                description=company.get("description")
            )
            # Add sector node
            if company.get("sector"):
                self.graph.add_node(company["sector"], type="Sector")
                self.graph.add_edge(node_id, company["sector"], relation="belongs_to")

    def ingest_memory_vectors(self, memory_entries: List[Dict[str, Any]]):
        """Ingests past analysis as Memory Nodes."""
        for entry in memory_entries:
            node_id = f"Memory::{entry['company_id']}::{entry['timestamp']}"
            self.graph.add_node(
                node_id,
                type="Memory",
                summary=entry.get("analysis_summary"),
                timestamp=entry.get("timestamp")
            )
            # Link to Company
            if entry.get("company_id"):
                self.graph.add_edge(node_id, entry["company_id"], relation="analyzes")

    def ingest_risk_state(self, risk_state: Dict[str, Any]):
        """
        Ingests the state from the Vertical Risk Agent (Odyssey) into the FIBO Graph.
        Maps Tickers -> LegalEntity, BalanceSheet -> FinancialReport, etc.
        """
        logger.info("Ingesting Odyssey Risk State...")

        ticker = risk_state.get("ticker")
        if not ticker:
            logger.warning("No ticker found in risk state. Skipping.")
            return

        # 1. Legal Entity
        entity_id = f"LegalEntity::{ticker}"
        self.graph.add_node(
            entity_id,
            type="LegalEntity",
            ticker=ticker,
            label=ticker  # For display
        )

        # 2. Financial Report (Snapshot)
        bs = risk_state.get("balance_sheet")
        is_stmt = risk_state.get("income_statement")

        if bs:
            # Create a simplified report ID based on fiscal year if available, else generic
            fy = bs.get("fiscal_year", "Current")
            report_id = f"FinancialReport::{ticker}::{fy}"

            # Calculate metrics if possible
            ebitda = is_stmt.get("consolidated_ebitda") if is_stmt else None
            debt = bs.get("total_debt")
            leverage = (debt / ebitda) if (debt is not None and ebitda is not None and ebitda != 0) else None

            self.graph.add_node(
                report_id,
                type="FinancialReport",
                fiscal_year=fy,
                total_debt=debt,
                ebitda=ebitda,
                leverage_ratio=leverage
            )
            self.graph.add_edge(entity_id, report_id, relation="REPORTED")

        # 3. Covenants
        covenants = risk_state.get("covenants", [])

        # Bolt Optimization: Hoist facility creation out of loop to avoid N redundant checks
        if covenants:
            facility_id = f"CreditFacility::{ticker}::General"
            if not self.graph.has_node(facility_id):
                self.graph.add_node(facility_id, type="CreditFacility", name="General Facility")
                self.graph.add_edge(entity_id, facility_id, relation="BORROWS")

            # Bolt Optimization: Batch node/edge creation for O(1) graph update overhead
            covenant_nodes = []
            covenant_edges = []

            for cov in covenants:
                cov_name = cov.get("name", "Unknown Covenant")
                cov_id = f"Covenant::{ticker}::{cov_name}"

                covenant_nodes.append((cov_id, {
                    "type": "Covenant",
                    "name": cov_name,
                    "threshold": cov.get("threshold"),
                    "operator": cov.get("operator")
                }))
                covenant_edges.append((facility_id, cov_id, {"relation": "GOVERNED_BY"}))

            self.graph.add_nodes_from(covenant_nodes)
            self.graph.add_edges_from(covenant_edges)

        # 4. Risk Model Output
        if risk_state.get("draft_memo"):
            memo = risk_state["draft_memo"]
            model_id = f"RiskModel::{ticker}::{self.graph.number_of_nodes()}"  # Unique ID
            self.graph.add_node(
                model_id,
                type="RiskModel",
                recommendation=memo.get("recommendation"),
                confidence=memo.get("confidence_score")
            )
            self.graph.add_edge(entity_id, model_id, relation="HAS_RISK_MODEL")

        logger.info(f"Ingested Risk State for {ticker}. Graph nodes: {self.graph.number_of_nodes()}")

    def ingest_regulatory_updates(self, regulations: List[Dict[str, Any]]):
        """Ingests regulatory updates as Regulation nodes."""
        for reg in regulations:
            # unique ID based on source and title
            node_id = f"Regulation::{reg.get('source')}::{reg.get('title')}"
            self.graph.add_node(
                node_id,
                type="Regulation",
                source=reg.get("source"),
                title=reg.get("title"),
                summary=reg.get("summary")
            )
            logger.info(f"Ingested Regulation: {node_id}")

    def ingest_compliance_event(self, event: Dict[str, Any]):
        """Ingests a compliance event (e.g. violation) linked to an entity."""
        # event: {transaction_id, compliance_status, violated_rules, risk_score, entity_id/ticker}
        t_id = event.get('transaction_id') or "Unknown"
        event_id = f"ComplianceEvent::{t_id}"

        self.graph.add_node(
            event_id,
            type="ComplianceEvent",
            status=event.get("compliance_status"),
            risk_score=event.get("risk_score"),
            violated_rules=event.get("violated_rules")
        )

        # Link to Entity if provided
        entity_id = event.get("entity_id") or event.get("ticker")
        if entity_id:
            # Auto-create entity node if missing to ensure linkage
            if not self.graph.has_node(entity_id):
                self.graph.add_node(entity_id, type="LegalEntity")

            self.graph.add_edge(entity_id, event_id, relation="HAS_COMPLIANCE_EVENT")

        logger.info(f"Ingested Compliance Event: {event_id}")

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
