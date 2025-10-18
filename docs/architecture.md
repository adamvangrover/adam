# Adam v22.0 System Architecture

## Overview

Adam v22.0 introduces a message broker-based architecture to decouple agent communication and improve system scalability. This document provides an overview of the new architecture.

## Components

### Agent Orchestrator

The Agent Orchestrator is responsible for managing the lifecycle of agents and workflows. It communicates with agents via the message broker.

### Message Broker

The message broker (currently RabbitMQ) is the central communication hub for all agents. Agents publish messages to topics, and other agents subscribe to those topics to receive messages.

### Agents

Agents are independent processes that perform specific tasks. They communicate with each other and the orchestrator via the message broker.

## Communication Flow

1. The Agent Orchestrator publishes a task to an agent-specific topic on the message broker.
2. The corresponding agent, which is subscribed to that topic, receives the message.
3. The agent executes the task and publishes the result to a reply-to topic.
4. The Agent Orchestrator, which is subscribed to the reply-to topic, receives the result.

This asynchronous communication pattern allows for greater flexibility and scalability.
