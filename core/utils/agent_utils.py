# core/utils/agent_utils.py

import json

_MOCK_MESSAGE_BUS = []

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
        # Store in global bus for backward compatibility tests
        _MOCK_MESSAGE_BUS.append(message_with_routing)
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
        # Store in global bus for backward compatibility tests
        _MOCK_MESSAGE_BUS.append(knowledge_message)
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
    # ... (Implementation for monitoring agent performance)
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
    # ... (Implementation for logging agent actions)
    pass  # Placeholder for actual implementation

# Add more agent utility functions as needed

"""
Graph Utilities for Adam v23.5

This module provides a unified interface for LangGraph components,
handling fallback logic for environments where langgraph is not installed.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    logger.warning("LangGraph not installed. Graph features will be disabled or mocked.")

    class StateGraph:
        """Mock StateGraph for environments without langgraph."""
        def __init__(self, state_schema, *args, **kwargs):
            self.state_schema = state_schema

        def add_node(self, node_name, action):
            pass

        def add_edge(self, start_node, end_node):
            pass

        def set_entry_point(self, node_name):
            pass

        def add_conditional_edges(self, source, path, path_map=None):
            pass

        def compile(self, checkpointer=None):
            return CompiledGraphMock()

    class CompiledGraphMock:
        """Mock for a compiled graph."""
        def invoke(self, inputs, config=None):
            logger.info(f"Mock graph invoked with inputs: {inputs}")
            return inputs

    class MemorySaver:
        """Mock MemorySaver."""
        pass

    END = "END"
    START = "START"
import time
import logging
from functools import wraps
import random

logger = logging.getLogger(__name__)


def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"Function {func.__name__} failed after {retries} retries: {e}")
                        raise
                    else:
                        sleep = (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                        logger.warning(f"Function {func.__name__} failed: {e}. Retrying in {sleep:.2f}s...")
                        time.sleep(sleep)
                        x += 1
        return wrapper
    return decorator
