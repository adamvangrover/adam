#core/utils/agent_utils.py

import pika
import json
from core.utils.data_utils import send_message, receive_messages

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

# Add more agent utility functions as needed
