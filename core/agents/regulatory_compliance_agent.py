#core/agents/regulatory_compliance_agent.py

from __future__ import annotations
import re
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import asyncio

# Import libraries for NLP, knowledge graph interaction, and regulatory data access
import nltk
import requests
from bs4 import BeautifulSoup
from neo4j import GraphDatabase

from core.llm_plugin import LLMPlugin
from core.data_sources.political_landscape import PoliticalLandscapeLoader
from core.agents.agent_base import AgentBase

# Configure logger
logger = logging.getLogger(__name__)

class RegulatoryComplianceAgent(AgentBase):
    """
    Ensures adherence to all applicable financial regulations and compliance standards.

    Core Capabilities:
    - Monitors regulatory changes and trends across relevant jurisdictions.
    - Analyzes financial transactions and activities for compliance.
    - Identifies potential regulatory risks and provides mitigation strategies.
    - Generates compliance reports and audit trails.
    - Collaborates with other agents to incorporate compliance considerations.
    - Provides guidance on interacting with regulatory bodies.
    - Adapts to changing political landscapes and regulatory priorities.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the RegulatoryComplianceAgent with configuration parameters.

        Args:
            config: A dictionary containing configuration parameters.
        """
        super().__init__(config, **kwargs)
        self.regulatory_knowledge = self._load_regulatory_knowledge()
        self.nlp_toolkit = self._initialize_nlp_toolkit()
        self.political_landscape = self._load_political_landscape()
        # Initialize LLM with config from agent config
        self.llm = LLMPlugin(config=self.config.get("llm_config"))

        # Load learned parameters for continuous learning
        self.learned_params = self._load_learned_profile()

    def _load_learned_profile(self) -> Dict:
        """
        Loads the learned compliance profile (weights, risk modifiers) from disk.
        """
        profile_path = "data/compliance_learned_profile.json"
        default_profile = {
            "rule_weights": {
                "threshold_violation": 0.5,
                "geopolitical_risk": 0.3
            },
            "entity_risk_modifiers": {},
            "learning_rate": 0.05
        }
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load learned profile: {e}")
        return default_profile

    def _save_learned_profile(self):
        """
        Saves the learned compliance profile to disk.
        """
        profile_path = "data/compliance_learned_profile.json"
        try:
            os.makedirs(os.path.dirname(profile_path), exist_ok=True)
            with open(profile_path, 'w') as f:
                json.dump(self.learned_params, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learned profile: {e}")

    def _load_regulatory_knowledge(self) -> Dict:
        """
        Loads regulatory knowledge from various sources.

        Returns:
            A dictionary containing regulatory knowledge.
        """
        knowledge = {}

        # 1. Load from Knowledge Graph (Neo4j)
        knowledge_graph_uri = self.config.get("knowledge_graph_uri")
        if knowledge_graph_uri:
            try:
                graph_driver = GraphDatabase.driver(knowledge_graph_uri)
                with graph_driver.session() as session:
                    # Example query to retrieve KYC regulations
                    result = session.run(
                        "MATCH (r:Regulation {name: 'KYC'}) "
                        "RETURN r.description AS description, r.rules AS rules, r.updates AS updates, r.history AS history"
                    )
                    for record in result:
                        knowledge["KYC"] = {
                            "description": record["description"],
                            "rules": record["rules"],
                            "updates": record["updates"],
                            "history": record["history"]
                        }
                    # ... add queries for other regulations
            except Exception as e:
                logger.error(f"Failed to load from Neo4j Knowledge Graph: {e}")

        # 2. Load from Regulatory APIs
        api_key = self.config.get("regulatory_api_key")
        if api_key:
            # Example API call to retrieve AML updates
            aml_updates = self._get_aml_updates(api_key)
            knowledge["AML"] = {"updates": aml_updates}
            # ... add calls for other regulations

        return knowledge

    def _initialize_nlp_toolkit(self) -> nltk.stem.WordNetLemmatizer:
        """
        Initializes the NLP toolkit with necessary resources.

        Returns:
            An instance of the NLP toolkit.
        """
        try:
            # Download required NLTK resources
            # Note: In production, these should be pre-downloaded
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('vader_lexicon', quiet=True)  # For sentiment analysis
        except Exception as e:
            logger.warning(f"NLTK download failed: {e}")

        return nltk.stem.WordNetLemmatizer()

    def _get_aml_updates(self, api_key: str) -> List[str]:
        """
        Retrieves AML updates from a regulatory API.

        Args:
            api_key: The API key for authentication.

        Returns:
            A list of AML updates.
        """
        # Example API call (replace with actual API endpoint and parameters)
        url = f"https://regulatory-api.com/aml/updates?api_key={api_key}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                updates = response.json().get("updates", [])
                return updates
            else:
                logger.warning(f"Error retrieving AML updates: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Exception retrieving AML updates: {e}")
            return []

    def _load_political_landscape(self) -> Dict:
        """
        Loads data on the current political landscape.

        Returns:
            A dictionary containing information on political leaders,
            party affiliations, key policies, and recent developments.
        """
        try:
            loader = PoliticalLandscapeLoader()
            landscape = loader.load_landscape()
            return landscape
        except Exception as e:
            logger.error(f"Error loading political landscape: {e}")
            # Return a minimal valid structure to prevent crashes
            return {
                "US": {
                    "president": "Unknown",
                    "recent_developments": ["Data unavailable"]
                }
            }

    def _analyze_transaction(self, transaction: Dict) -> Dict:
        """
        Analyzes a financial transaction for compliance with relevant regulations.

        Args:
            transaction: A dictionary representing the financial transaction.

        Returns:
            A dictionary containing compliance analysis results.
        """
        # 1. Preprocess transaction data
        transaction_text = f"{transaction.get('customer', '')} {transaction.get('amount', '')}"
        try:
            tokens = nltk.word_tokenize(transaction_text)
            lemmas = [self.nlp_toolkit.lemmatize(token) for token in tokens]
        except Exception as e:
            # logger.warning(f"NLP processing failed: {e}")
            lemmas = []

        # 2. Apply Rule-based analysis
        violated_rules = []
        amount = transaction.get('amount', 0)
        try:
            amount = float(amount)
        except (ValueError, TypeError):
            amount = 0

        if amount > 10000:
            violated_rules.append("Transaction amount exceeds threshold")

        # Load weights
        weights = self.learned_params.get("rule_weights", {})

        # 3. Political Context Analysis (Expanded)
        country = transaction.get("country", "US")
        country_data = self.political_landscape.get(country, {})
        developments = country_data.get("recent_developments", [])

        geo_risk = 0.0
        geo_risk_weight = weights.get("geopolitical_risk", 0.3)

        if isinstance(developments, list):
            for dev in developments:
                 if isinstance(dev, str) and any(kw in dev.lower() for kw in ["instability", "sanction", "crisis", "war", "conflict"]):
                     geo_risk += geo_risk_weight
                     violated_rules.append(f"Geopolitical Risk Warning: {dev}")

        # 4. Calculate risk score
        risk_score = 0.0

        # Add scores from specific rules
        threshold_weight = weights.get("threshold_violation", 0.5)
        if any("exceeds threshold" in r for r in violated_rules):
            risk_score += threshold_weight

        risk_score += geo_risk

        # Add entity-specific risk modifier
        entity_id = transaction.get("customer") or transaction.get("ticker")
        if entity_id:
            modifier = self.learned_params.get("entity_risk_modifiers", {}).get(entity_id, 0.0)
            risk_score += modifier

        risk_score = min(risk_score, 1.0) # Cap at 1.0
        risk_score = max(risk_score, 0.0) # Ensure non-negative

        analysis = {
            "transaction_id": transaction.get("id"),
            "compliance_status": "compliant" if not violated_rules else "non-compliant",
            "violated_rules": violated_rules,
            "risk_score": risk_score,
            "entity_id": entity_id
        }
        return analysis

    def _generate_compliance_report(self, transactions: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Generates a compliance report for a list of transactions.

        Args:
            transactions: A list of dictionaries representing financial transactions.

        Returns:
            A tuple containing the compliance report string and the structured results list.
        """
        report = "Compliance Report:\n"
        results = []
        for transaction in transactions:
            analysis = self._analyze_transaction(transaction)
            results.append(analysis)
            report += f"  - Transaction {analysis['transaction_id']}: {analysis['compliance_status']}\n"
            if analysis['violated_rules']:
                report += f"    Violated Rules: {', '.join(analysis['violated_rules'])}\n"
            report += f"    Risk Score: {analysis['risk_score']}\n"
        return report, results

    def _get_regulatory_updates(self) -> List[Tuple[str, str, str]]:
        """
        Retrieves the latest regulatory updates from relevant sources.

        Returns:
            A list of tuples, where each tuple contains the source,
            title, and summary of a regulatory update.
        """
        updates = []

        # 1. Scrape regulatory websites
        # Example: Scraping FINRA website for KYC updates
        finra_url = "https://www.finra.org/rules-guidance/key-topics/know-your-customer"
        try:
            response = requests.get(finra_url, timeout=10)
            if response.status_code == 200:
                finra_html = response.text
                soup = BeautifulSoup(finra_html, 'html.parser')
                # ... (Extract relevant updates from the parsed HTML)
                # Example:
                updates.append(
                    ("FINRA", "New KYC Rule", "Enhanced due diligence for high-risk customers.")
                )
        except Exception as e:
            logger.error(f"Failed to scrape FINRA: {e}")

        # 2. Access regulatory RSS feeds
        # ... (Implementation for accessing RSS feeds)

        return updates

    def provide_guidance(self, user_question: str) -> str:
        """
        Provides guidance on regulatory compliance matters using LLM.

        Args:
            user_question: The user's question regarding compliance.

        Returns:
            A string containing the guidance.
        """
        prompt = (
            "You are a Regulatory Compliance Expert. Provide guidance on the following compliance matter.\n\n"
            f"User Question: {user_question}\n\n"
            "Context (Regulatory Knowledge):\n"
            f"{self.regulatory_knowledge}\n\n"
            "Context (Political Landscape):\n"
            f"{self.political_landscape}\n\n"
            "Guidance:"
        )

        try:
            guidance = self.llm.generate_text(prompt, task="regulatory_guidance")
            return guidance
        except Exception as e:
            # Fallback if LLM fails
            logger.error(f"LLM generation failed: {e}")
            return "Please consult with a compliance expert for further assistance."

    def _integrate_knowledge(self, updates: List[Tuple[str, str, str]], knowledge_graph: Any = None):
        """
        Integrates new regulatory updates into the knowledge base (Local + Graph).
        """
        if not updates:
            return

        logger.info(f"Integrating {len(updates)} regulatory updates into Knowledge Base.")

        # 1. Update local cache
        for source, title, summary in updates:
            if source not in self.regulatory_knowledge:
                self.regulatory_knowledge[source] = {"updates": []}
            self.regulatory_knowledge[source]["updates"].append({"title": title, "summary": summary})

        # 2. Update UnifiedKnowledgeGraph if available
        if knowledge_graph and hasattr(knowledge_graph, 'ingest_regulatory_data'):
            formatted_updates = [{"source": s, "title": t, "summary": sum} for s, t, sum in updates]
            try:
                knowledge_graph.ingest_regulatory_data(formatted_updates)
                logger.info("Updates ingested into UnifiedKnowledgeGraph.")
            except Exception as e:
                logger.error(f"Failed to ingest regulatory data into Graph: {e}")

    def _continuous_learning(self, results: List[Dict]):
        """
        Analyzes the generated report to identify patterns and improve future compliance checks.
        """
        if not results:
            return

        violation_counts = {}

        # 1. Update Entity Risk Modifiers based on violation frequency
        modifiers = self.learned_params.setdefault("entity_risk_modifiers", {})
        learning_rate = self.learned_params.get("learning_rate", 0.05)

        for res in results:
            for rule in res.get('violated_rules', []):
                violation_counts[rule] = violation_counts.get(rule, 0) + 1

            entity_id = res.get("entity_id")
            if not entity_id:
                continue

            if res.get("violated_rules"):
                # Increase risk for repeat offenders
                modifiers[entity_id] = modifiers.get(entity_id, 0.0) + (learning_rate * 0.1)
                logger.info(f"Continuous Learning: Escalating risk profile for entity {entity_id}")
            else:
                # Decay risk for compliant entities (forgive over time)
                current = modifiers.get(entity_id, 0.0)
                if current > 0:
                    modifiers[entity_id] = max(0.0, current - (learning_rate * 0.05))

        if violation_counts:
            logger.info(f"Continuous Learning - Observed Violation Patterns: {violation_counts}")

        self._save_learned_profile()

    def process_feedback(self, feedback_data: Dict[str, Any]):
        """
        Processes feedback from audits or human reviewers to adjust learned parameters.

        Args:
            feedback_data: Dict containing 'transaction_id', 'correct_assessment' (bool),
                           'rule_id' (optional), 'comment'.
        """
        # This allows adjusting rule weights
        if "rule_id" in feedback_data:
            rule = feedback_data["rule_id"]
            correct = feedback_data.get("correct_assessment", True)

            weights = self.learned_params.setdefault("rule_weights", {})
            current_weight = weights.get(rule, 0.5)

            learning_rate = self.learned_params.get("learning_rate", 0.05)

            if not correct:
                # We were wrong (False Positive), decrease weight
                weights[rule] = max(0.1, current_weight - learning_rate)
                logger.info(f"Feedback: Decreasing weight for rule {rule} to {weights[rule]}")
            else:
                # We were right (True Positive), potentially increase (reinforcement)
                weights[rule] = min(1.0, current_weight + (learning_rate * 0.1))
                logger.info(f"Feedback: Increasing weight for rule {rule} to {weights[rule]}")

        self._save_learned_profile()

    def _save_audit_trail(self, results: List[Dict]):
        """
        Saves analysis results to a secure audit log file (JSONL).
        """
        audit_file = "logs/compliance_audit.jsonl"
        try:
            os.makedirs(os.path.dirname(audit_file), exist_ok=True)
            with open(audit_file, "a") as f:
                for res in results:
                    entry = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "agent_id": self.config.get("agent_id", "unknown"),
                        **res
                    }
                    f.write(json.dumps(entry) + "\n")
            logger.info(f"Audit trail saved to {audit_file}")
        except Exception as e:
            logger.error(f"Failed to save audit trail: {e}")

    async def execute(self, transactions: List[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Runs the RegulatoryComplianceAgent to analyze transactions,
        monitor regulatory updates, and provide compliance guidance.

        Args:
            transactions: A list of dictionaries representing financial transactions.
            kwargs: Additional arguments (e.g., knowledge_graph).

        Returns:
            A dictionary containing the execution results.
        """
        # Handle transactions from kwargs if not positional
        if transactions is None:
            transactions = kwargs.get("transactions", [])

        # If still no transactions, check if they are in 'target_data' (common pattern)
        if not transactions and "target_data" in kwargs:
            target_data = kwargs["target_data"]
            if isinstance(target_data, list):
                transactions = target_data
            elif isinstance(target_data, dict) and "transactions" in target_data:
                transactions = target_data["transactions"]

        # Generate Report and Structured Results
        compliance_report, analysis_results = self._generate_compliance_report(transactions)
        regulatory_updates = self._get_regulatory_updates()

        # Integrate with other agents
        # Send significant findings to RiskAssessmentAgent
        if compliance_report:
            risk_data = {
                 "source": "RegulatoryComplianceAgent",
                 "report": compliance_report,
                 "risk_type": "compliance",
                 "regulatory_updates": len(regulatory_updates),
                 "violation_count": sum(1 for r in analysis_results if r['violated_rules'])
            }
            logger.info("Sending compliance data to RiskAssessmentAgent...")
            await self.send_message("RiskAssessmentAgent", risk_data)

        # Knowledge Base Integration
        knowledge_graph = kwargs.get("knowledge_graph") # Expecting UnifiedKnowledgeGraph instance
        self._integrate_knowledge(regulatory_updates, knowledge_graph)

        # Ingest Compliance Events into Graph
        if knowledge_graph and hasattr(knowledge_graph, 'ingest_compliance_event'):
            for res in analysis_results:
                if res['violated_rules']:
                    try:
                        knowledge_graph.ingest_compliance_event(res)
                    except Exception as e:
                        logger.error(f"Failed to ingest compliance event: {e}")

        # Continuous Learning & Audit Trail
        self._continuous_learning(analysis_results)
        self._save_audit_trail(analysis_results)

        logger.info("RegulatoryComplianceAgent execution completed.")
        return {
            "status": "completed",
            "compliance_report": compliance_report,
            "regulatory_updates": regulatory_updates,
            "analysis_results": analysis_results
        }
