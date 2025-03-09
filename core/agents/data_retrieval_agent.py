# core/agents/data_retrieval_agent.py
import logging
from typing import Optional, Union, List, Dict, Any
from core.agents.agent_base import AgentBase
from core.utils.config_utils import load_config
from core.utils.data_utils import load_data
from core.system.knowledge_base import KnowledgeBase  # Import KnowledgeBase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataRetrievalAgent(AgentBase):
    """
    Agent responsible for retrieving data from various sources.  Supports:
    - Local files (JSON, CSV, YAML) via data_utils.
    - A simple Knowledge Base (core.system.knowledge_base).
    - Placeholder for future Knowledge Graph integration.
    - Basic inter-agent communication (synchronous, for now).

    This agent is designed to be a central point for data access, abstracting
    away the specifics of data storage and retrieval from other agents.
    """

    def __init__(self, config_path: str = "config/agents.yaml"):
        """
        Initializes the DataRetrievalAgent.

        Args:
            config_path: Path to the agent configuration file.
        """
        super().__init__()
        self.config = load_config(config_path)
        agent_config = self.config.get('agents', {}).get('DataRetrievalAgent', {})

        if not agent_config:
            logging.error("DataRetrievalAgent configuration not found in %s", config_path)
            raise ValueError("DataRetrievalAgent configuration not found.")

        self.persona = agent_config.get('persona', "Data Retrieval Agent")
        self.description = agent_config.get('description', "Retrieves data.")
        self.expertise = agent_config.get('expertise', ["data access", "data retrieval"])

        # Load data sources configuration
        self.data_sources_config = load_config("config/data_sources.yaml")
        if not self.data_sources_config:
            logging.error("Failed to load data sources configuration.")
            raise FileNotFoundError("data_sources.yaml could not be loaded")

        # Initialize KnowledgeBase (if implemented)
        try:
            self.knowledge_base = KnowledgeBase()  # No config needed for now
        except Exception as e:
            logging.error(f"Failed to initialize KnowledgeBase: {e}")
            self.knowledge_base = None


    def get_risk_rating(self, company_id: str) -> Optional[str]:
        """Retrieves the risk rating for a given company."""
        risk_data = load_data(self.data_sources_config['risk_ratings'])
        if risk_data:
            return risk_data.get(company_id)
        return None

    def get_market_data(self) -> Optional[Dict[str, Any]]:
        """Retrieves general market data."""
        market_data = load_data(self.data_sources_config['market_baseline'])
        return market_data


    def access_knowledge_base(self, query: str) -> Optional[str]:
        """Accesses the Knowledge Base to retrieve information."""
        if self.knowledge_base:
            try:
                return self.knowledge_base.query(query)
            except Exception as e:
                logging.error(f"Error accessing Knowledge Base: {e}")
                return None
        else:
            logging.warning("KnowledgeBase is not initialized.")
            return None

    def access_knowledge_graph(self, query: str) -> str:
        """Placeholder for knowledge graph access."""
        # TODO: Implement knowledge graph interaction.  This is a placeholder.
        logging.warning("Knowledge Graph access is not yet implemented.")
        return "Knowledge Graph access is not yet implemented."


    def facilitate_communication(self, sender_agent: str, recipient_agent: str, message: str) -> str:
        """
        Facilitates *synchronous* communication between agents (acting as an intermediary).
        This is a *very* simplified example and should be replaced with a
        message queue in a production system.

        Args:
            sender_agent: The name of the sending agent.
            recipient_agent: The name of the receiving agent.
            message: The message to send.

        Returns:
            A confirmation message (or error message).
        """
        logging.info(f"Agent Communication: From {sender_agent} to {recipient_agent}: {message}")
        # In a real system with a message queue, you would:
        # 1.  Serialize the message (e.g., to JSON).
        # 2.  Push the message onto the queue.
        # 3.  The recipient agent would be listening on the queue and process the message.
        return f"Message relayed from {sender_agent} to {recipient_agent}."


    def execute(self, input_data: str) -> str:
        """
        Executes the data retrieval process based on the input query.

        Args:
            input_data: The input query (string).

        Returns:
            The retrieved data (string), or an error message.
        """
        input_lower = input_data.lower()

        if "risk rating" in input_lower:
            parts = input_data.split()
            if len(parts) > 5:
                company_id = parts[-1].replace("?", "").replace('"', '').strip()
                risk_rating = self.get_risk_rating(company_id)
                if risk_rating:
                    return f"The risk rating for {company_id} is {risk_rating}."
                else:
                    return f"Risk rating for {company_id} not found."

        elif "market data" in input_lower:
            market_data = self.get_market_data()
            if market_data:
                return str(market_data)
            else:
                return "Market Data Not Found"

        elif "knowledge base" in input_lower:
            query = input_data.replace("knowledge base", "").strip()
            result = self.access_knowledge_base(query)
            return result if result else "No information found in the knowledge base."

        elif "knowledge graph" in input_lower:
            query = input_data.replace("knowledge graph", "").strip()
            return self.access_knowledge_graph(query)

        elif "send message" in input_lower:
            try:
                parts = input_data.split(":")
                message = parts[1].strip()
                agent_info = parts[0].replace("Send message from ", "").strip().split(" to ")
                sender = agent_info[0].strip()
                recipient = agent_info[1].strip()
                return self.facilitate_communication(sender, recipient, message)
            except (IndexError, ValueError) as e:
                logging.error(f"Invalid message format: {e}")
                return "Invalid message format. Use 'Send message from [Sender] to [Recipient]: [Message]'"

        else:
            return "Data retrieval request not understood."
