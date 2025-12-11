import networkx as nx
import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class UnifiedKnowledgeGraph:
    """
    The central brain of the system.
    Merges:
    1. Financial Entities (Companies, Sectors)
    2. System Entities (Agents, Tools - via RepoGraph)
    3. Memory Artifacts (Past Analysis)
    """
    def __init__(self):
        self.graph = nx.DiGraph()

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
            if not node_id: continue

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
            label=ticker # For display
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
        for cov in covenants:
            cov_name = cov.get("name", "Unknown Covenant")
            cov_id = f"Covenant::{ticker}::{cov_name}"

            self.graph.add_node(
                cov_id,
                type="Covenant",
                name=cov_name,
                threshold=cov.get("threshold"),
                operator=cov.get("operator")
            )
            # In a real graph, we'd link to the Facility. Here we link to Entity via a "governed_by" proxy or direct
            # Using the schema: LegalEntity -> BORROWS -> Facility -> GOVERNED_BY -> Covenant
            # We'll create a synthetic facility for now
            facility_id = f"CreditFacility::{ticker}::General"
            if not self.graph.has_node(facility_id):
                self.graph.add_node(facility_id, type="CreditFacility", name="General Facility")
                self.graph.add_edge(entity_id, facility_id, relation="BORROWS")

            self.graph.add_edge(facility_id, cov_id, relation="GOVERNED_BY")

        # 4. Risk Model Output
        if risk_state.get("draft_memo"):
            memo = risk_state["draft_memo"]
            model_id = f"RiskModel::{ticker}::{self.graph.number_of_nodes()}" # Unique ID
            self.graph.add_node(
                model_id,
                type="RiskModel",
                recommendation=memo.get("recommendation"),
                confidence=memo.get("confidence_score")
            )
            self.graph.add_edge(entity_id, model_id, relation="HAS_RISK_MODEL")

        logger.info(f"Ingested Risk State for {ticker}. Graph nodes: {self.graph.number_of_nodes()}")

    def query_graph(self, query: str) -> List[Dict[str, Any]]:
        # Placeholder for graph traversal/search
        # For now, return neighbors of a node if query matches a node ID
        if query in self.graph:
            return [
                {"node": n, "relation": self.graph[query][n].get("relation")}
                for n in self.graph.neighbors(query)
            ]
        return []

    def save_snapshot(self, filepath: str = "data/knowledge_graph_snapshot.json"):
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
