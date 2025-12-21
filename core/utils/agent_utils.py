#core/utils/agent_utils.py

import json

from core.utils.data_utils import send_message

# RabbitMQ connection parameters (same as in data_utils.py)
RABBITMQ_HOST = 'localhost'
RABBITMQ_QUEUE = 'adam_data'


def communicate_between_agents(sender_agent, receiver_agent, message):
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
        # Use the existing send_message function to send the message
        send_message(message_with_routing)
    except Exception as e:
        print(f"Error in agent communication: {e}")


def share_knowledge_between_agents(sender_agent, receiver_agent, knowledge_type, knowledge_data):
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
        # Use the existing send_message function to share the knowledge
        send_message(knowledge_message)
    except Exception as e:
        print(f"Error in knowledge sharing between agents: {e}")


def monitor_agent_performance(agent_name, metric, value):
    """
    Monitors agent performance metrics.

    Args:
      agent_name: The name of the agent being monitored.
      metric: The performance metric being tracked (e.g., "execution_time", "accuracy", "resource_usage").
      value: The value of the metric.
    """
    # Store the performance metrics (e.g., in a database or log file)
    # Analyze the metrics to identify trends or potential issues
    #... (Implementation for monitoring agent performance)
    pass  # Placeholder for actual implementation


def validate_agent_inputs(agent_name, inputs, required_parameters):
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
            raise ValueError(f"Agent {agent_name} missing required parameter: {param}")


def format_agent_output(agent_name, output_data, format="json"):
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
        pass  # Placeholder for CSV conversion logic
    elif format == "text":
        # Convert output_data to plain text format
        pass  # Placeholder for text conversion logic
    else:
        raise ValueError(f"Invalid output format for agent {agent_name}: {format}")


def log_agent_action(agent_name, action, details):
    """
    Logs agent actions and events.

    Args:
      agent_name: The name of the agent.
      action: The action performed by the agent (e.g., "analyzed_data", "generated_report", "updated_knowledge_graph").
      details: Additional details about the action (e.g., parameters used, results obtained).
    """
    # Record the log entry (e.g., in a log file or database)
    # Include timestamp, agent name, action, and details
    #... (Implementation for logging agent actions)
    pass  # Placeholder for actual implementation

# Add more agent utility functions as needed
