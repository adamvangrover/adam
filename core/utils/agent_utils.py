"""
Agent Utilities for Adam OS.

Provides core functionality for inter-agent communication,
performance monitoring, input validation, and formatting.
Upgraded to follow modern typed Python practices.
"""

import csv
from functools import wraps
import io
import json
import logging
import random
import re
import ast
import time
from typing import Any, Optional, TypedDict

logger = logging.getLogger(__name__)

# Global bus for backward compatibility testing
class AgentMessage(TypedDict, total=False):
    sender: str
    receiver: str
    message: Any
    knowledge_type: str
    knowledge_data: Any

_MOCK_MESSAGE_BUS: list[AgentMessage] = []


def communicate_between_agents(
    sender_agent: str, receiver_agent: str, message: Any
) -> None:
    """Facilitates communication between agents using the message queue.

    Args:
        sender_agent: The name of the sending agent.
        receiver_agent: The name of the receiving agent.
        message: The message to be sent.
    """
    try:
        message_with_routing = {
            "sender": sender_agent,
            "receiver": receiver_agent,
            "message": message,
        }
        _MOCK_MESSAGE_BUS.append(message_with_routing)
        logger.debug(f"Message queued from {sender_agent} to {receiver_agent}")
    except Exception as e:
        logger.error(f"Error in agent communication: {e}", exc_info=True)


def share_knowledge_between_agents(
    sender_agent: str, receiver_agent: str, knowledge_type: str, knowledge_data: Any
) -> None:
    """Enables knowledge sharing between agents.

    Args:
        sender_agent: The name of the agent sharing knowledge.
        receiver_agent: The name of the agent receiving knowledge.
        knowledge_type: The type of knowledge being shared.
        knowledge_data: The actual knowledge data to be shared.
    """
    try:
        knowledge_message = {
            "sender": sender_agent,
            "receiver": receiver_agent,
            "knowledge_type": knowledge_type,
            "knowledge_data": knowledge_data,
        }
        _MOCK_MESSAGE_BUS.append(knowledge_message)
        logger.debug(
            f"Knowledge ({knowledge_type}) shared from {sender_agent} to {receiver_agent}"
        )
    except Exception as e:
        logger.error(
            f"Error in knowledge sharing between agents: {e}", exc_info=True
        )


def monitor_agent_performance(agent_name: str, metric: str, value: Any) -> None:
    """Monitors agent performance metrics.

    Args:
        agent_name: The name of the agent being monitored.
        metric: The performance metric being tracked (e.g., "execution_time").
        value: The value of the metric.
    """
    # Real implementation could push to Datadog/Prometheus.
    logger.info(
        f"PERFORMANCE | Agent: {agent_name} | Metric: {metric} | Value: {value}"
    )


def validate_agent_inputs(
    agent_name: str, inputs: dict[str, Any], required_parameters: list[str]
) -> None:
    """Validates agent inputs against a list of required parameters.

    Args:
        agent_name: The name of the agent.
        inputs: The inputs provided to the agent.
        required_parameters: A list of required parameter names.

    Raises:
        ValueError: If any required parameter is missing.
    """
    missing = [param for param in required_parameters if param not in inputs]
    if missing:
        missing_str = ", ".join(missing)
        logger.error(
            f"Agent {agent_name} missing required parameters: {missing_str}"
        )
        raise ValueError(
            f"Agent {agent_name} missing required parameter: {missing[0]}"
        )


def format_agent_output(
    agent_name: str, output_data: Any, format: str = "json"
) -> str:
    """Formats agent output data into the specified format.

    Args:
        agent_name: The name of the agent.
        output_data: The data to be formatted.
        format: The desired output format ("json", "csv", "text").

    Returns:
        The formatted output data as a string.
    """
    match format.lower():
        case "json":
            try:
                return json.dumps(output_data, indent=4)
            except TypeError as e:
                logger.error(
                    f"JSON serialization failed for agent {agent_name}: {e}",
                    exc_info=True,
                )
                return json.dumps({"error": "Unserializable data"})
        case "csv":
            if not isinstance(output_data, list) or not all(
                isinstance(i, dict) for i in output_data
            ):
                raise ValueError(
                    f"Agent {agent_name} output must be a list of dicts for CSV format."
                )
            if not output_data:
                return ""
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=output_data[0].keys())
            writer.writeheader()
            writer.writerows(output_data)
            return output.getvalue()
        case "text":
            return str(output_data)
        case _:
            raise ValueError(
                f"Invalid output format for agent {agent_name}: {format}"
            )


def log_agent_action(agent_name: str, action: str, details: Any) -> None:
    """Logs agent actions and events.

    Args:
        agent_name: The name of the agent.
        action: The action performed by the agent.
        details: Additional details about the action.
    """
    logger.info(
        f"ACTION | Agent: {agent_name} | Action: {action} | Details: {details}"
    )


def parse_json_garbage(text: str) -> dict[str, Any]:
    """Safely extracts and parses JSON from a string that might contain markdown
    or conversational garbage.

    Args:
        text: The raw LLM string containing JSON somewhere inside.

    Returns:
        A dictionary representing the parsed JSON.

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    # 1. Try to parse directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Try to extract JSON blocks from markdown (using `{3}` to avoid markdown rendering issues)
    json_match = re.search(r"`{3}(?:json)?\s*(\{.*?\})\s*`{3}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Fallback to greedy curly brace extraction
    braces_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if braces_match:
        extracted = braces_match.group(1)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            # 4. Final fallback: try to parse as a Python dictionary literal
            try:
                parsed_ast = ast.literal_eval(extracted)
                if isinstance(parsed_ast, dict):
                    return parsed_ast
            except (SyntaxError, ValueError):
                pass

    raise ValueError("Failed to extract valid JSON from the provided text.")


"""
Graph Utilities for Adam v23.5

This module provides a unified interface for LangGraph components,
handling fallback logic for environments where langgraph is not installed.
"""

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    logger.warning(
        "LangGraph not installed. Graph features will be disabled or mocked."
    )

    class StateGraph:
        """Mock StateGraph for environments without langgraph."""

        def __init__(self, state_schema: Any, *args: Any, **kwargs: Any):
            self.state_schema = state_schema

        def add_node(self, node_name: str, action: Any) -> None:
            pass

        def add_edge(self, start_node: str, end_node: str) -> None:
            pass

        def set_entry_point(self, node_name: str) -> None:
            pass

        def add_conditional_edges(
            self, source: str, path: Any, path_map: Any = None
        ) -> None:
            pass

        def compile(self, checkpointer: Any = None) -> Any:
            return CompiledGraphMock()

    class CompiledGraphMock:
        """Mock for a compiled graph."""

        def invoke(self, inputs: Any, config: Any = None) -> Any:
            logger.info(f"Mock graph invoked with inputs: {inputs}")
            return inputs

    class MemorySaver:
        """Mock MemorySaver."""

        pass

    END = "END"
    START = "START"


def retry_with_backoff(retries: int = 3, backoff_in_seconds: float = 1) -> Any:
    """Legacy retry utility."""

    def decorator(func: Any) -> Any:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(
                            f"Function {func.__name__} failed after {retries} retries: {e}",
                            exc_info=True,
                        )
                        raise
                    else:
                        sleep = (
                            backoff_in_seconds * 2**x
                        ) + random.uniform(0, 1)
                        logger.warning(
                            f"Function {func.__name__} failed: {e}. Retrying in {sleep:.2f}s..."
                        )
                        time.sleep(sleep)
                        x += 1

        return wrapper

    return decorator