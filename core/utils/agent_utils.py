# core/utils/agent_utils.py

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_MOCK_MESSAGE_BUS: List[Dict[str, Any]] = []


def communicate_between_agents(sender_agent: str, receiver_agent: str, message: Any) -> None:
    """
    Facilitates communication between agents using the message queue.

    Args:
      sender_agent: The name of the sending agent.
      receiver_agent: The name of the receiving agent.
      message: The message to be sent.
    """
    try:
        # Construct the message with sender and receiver information
        message_with_routing = {
            "sender": sender_agent,
            "receiver": receiver_agent,
            "message": message
        }
        # Store in global bus for backward compatibility tests
        _MOCK_MESSAGE_BUS.append(message_with_routing)
        logger.debug(f"Message stored in mock bus from {sender_agent} to {receiver_agent}")
    except Exception as e:
        logger.error(f"Error in agent communication: {e}", exc_info=True)


def share_knowledge_between_agents(
    sender_agent: str, receiver_agent: str, knowledge_type: str, knowledge_data: Any
) -> None:
    """
    Enables knowledge sharing between agents.

    Args:
      sender_agent: The name of the agent sharing knowledge.
      receiver_agent: The name of the agent receiving knowledge.
      knowledge_type: The type of knowledge being shared (e.g., "market_sentiment", "financial_model").
      knowledge_data: The actual knowledge data to be shared.
    """
    try:
        # Construct the knowledge sharing message
        knowledge_message = {
            "sender": sender_agent,
            "receiver": receiver_agent,
            "knowledge_type": knowledge_type,
            "knowledge_data": knowledge_data
        }
        # Store in global bus for backward compatibility tests
        _MOCK_MESSAGE_BUS.append(knowledge_message)
        logger.debug(f"Knowledge shared in mock bus from {sender_agent} to {receiver_agent} (type: {knowledge_type})")
    except Exception as e:
        logger.error(f"Error in knowledge sharing between agents: {e}", exc_info=True)


def monitor_agent_performance(agent_name: str, metric: str, value: Any) -> None:
    """
    Monitors agent performance metrics.

    Args:
      agent_name: The name of the agent being monitored.
      metric: The performance metric being tracked (e.g., "execution_time", "accuracy", "resource_usage").
      value: The value of the metric.
    """
    # Store the performance metrics (e.g., in a database or log file)
    # Analyze the metrics to identify trends or potential issues
    # ... (Implementation for monitoring agent performance)
    logger.debug(f"Monitored performance for {agent_name}: {metric} = {value}")
    pass  # Placeholder for actual implementation


def validate_agent_inputs(agent_name: str, inputs: Dict[str, Any], required_parameters: List[str]) -> None:
    """
    Validates agent inputs against a list of required parameters.

    Args:
      agent_name: The name of the agent.
      inputs: The inputs provided to the agent.
      required_parameters: A list of required parameter names.

    Raises:
      ValueError: If any required parameter is missing.
    """
    for param in required_parameters:
        if param not in inputs:
            error_msg = f"Agent {agent_name} missing required parameter: {param}"
            logger.error(error_msg)
            raise ValueError(error_msg)


def format_agent_output(agent_name: str, output_data: Any, format: str = "json") -> Any:
    """
    Formats agent output data into the specified format (default: JSON).

    Args:
      agent_name: The name of the agent.
      output_data: The data to be formatted.
      format: The desired output format (e.g., "json", "csv", "text").

    Returns:
      The formatted output data.
    """
    if format == "json":
        return json.dumps(output_data, indent=4)
    elif format == "csv":
        # Convert output_data to CSV format
        logger.warning(f"CSV formatting not implemented for agent {agent_name}")
        pass  # Placeholder for CSV conversion logic
    elif format == "text":
        # Convert output_data to plain text format
        logger.warning(f"Text formatting not implemented for agent {agent_name}")
        pass  # Placeholder for text conversion logic
    else:
        error_msg = f"Invalid output format for agent {agent_name}: {format}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def log_agent_action(agent_name: str, action: str, details: Any) -> None:
    """
    Logs agent actions and events.

    Args:
      agent_name: The name of the agent.
      action: The action performed by the agent (e.g., "analyzed_data", "generated_report", "updated_knowledge_graph").
      details: Additional details about the action (e.g., parameters used, results obtained).
    """
    # Record the log entry (e.g., in a log file or database)
    # Include timestamp, agent name, action, and details
    # ... (Implementation for logging agent actions)
    logger.info(f"Agent {agent_name} performed {action}: {details}")
    pass  # Placeholder for actual implementation
