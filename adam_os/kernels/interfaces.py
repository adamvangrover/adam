from abc import ABC, abstractmethod
from typing import Any, Dict, List

# Importing the primitives and DAF components from your core ontology (afos_core.py)
from afos_core import (
    Event, 
    Policy, 
    Decision, 
    Evidence, 
    Scenario,
    ImmutableDecisionBlock
)

# ==================================================================
# Base OS Kernel Interface
# ==================================================================
class IKernel(ABC):
    """
    Base interface for all OS Kernels.
    Enforces a strict asynchronous lifecycle for boot sequences and graceful teardowns.
    """
    @abstractmethod
    async def initialize(self) -> None:
        """Allocate resources, establish DB pools, or load ML models."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Safely drain queues and close connections."""
        pass

# ==================================================================
# The 7 Core Kernels
# ==================================================================

class IKnowledgeKernel(IKernel):
    """
    Knowledge Kernel - The institutional memory.
    Answers fundamentally different questions across transactional, semantic, relational, and temporal tiers.
    """
    @abstractmethod
    async def ask_truth(self, query: str) -> Any:
        """Query relational stores (What is factually true? e.g., PostgreSQL)."""
        pass

    @abstractmethod
    async def ask_similarity(self, vector: List[float]) -> Any:
        """Query vector stores (What is semantically similar? e.g., Qdrant/Pinecone)."""
        pass

    @abstractmethod
    async def ask_connections(self, entity_id: str) -> Any:
        """Query Knowledge Graph (What are the topological risks? e.g., Neo4j)."""
        pass

    @abstractmethod
    async def ask_history(self, entity_id: str) -> List[Event]:
        """Query Event Store (What happened chronologically? Event Sourcing)."""
        pass

class IPolicyKernel(IKernel):
    """
    Policy Kernel - Compiles and executes deterministic rules.
    """
    @abstractmethod
    async def compile_policy(self, dsl: str, dsl_format: str = "jsonlogic") -> Policy:
        """Compiles human-readable DSL into an executable Policy entity."""
        pass

    @abstractmethod
    async def execute_policy(self, policy: Policy, context: Dict[str, Any]) -> Any:
        """Executes the deterministic runtime for the policy."""
        pass

class IDecisionKernel(IKernel):
    """
    Decision Kernel - Produces explainable decision graphs combining deterministic 
    policies with probabilistic (AI) risk models.
    """
    @abstractmethod
    async def compute_decision(self, target_id: str, policy: Policy, evidence: List[Evidence]) -> Decision:
        """Evaluates a target against policies and evidence to produce an outcome."""
        pass

class IExecutionKernel(IKernel):
    """
    Execution Kernel - The workflow and agent runtime (e.g., LangGraph / Temporal).
    """
    @abstractmethod
    async def plan_execution(self, workflow_dsl: str) -> Any:
        """Generates a directed acyclic graph (DAG) execution plan."""
        pass

    @abstractmethod
    async def run_workflow(self, plan: Any, parameters: Dict[str, Any]) -> Any:
        """Executes the workflow, handling async checkpointing and human-in-the-loop interrupts."""
        pass

class IGovernanceKernel(IKernel):
    """
    Governance Kernel - The cryptographic source of truth. 
    Manages provenance, audits, and Merkle-tree state transitions.
    """
    @abstractmethod
    async def register_decision_block(self, block: ImmutableDecisionBlock[Decision]) -> str:
        """Records a cryptographically sealed decision block into the immutable audit ledger."""
        pass

    @abstractmethod
    async def require_approval(self, decision: Decision) -> bool:
        """Evaluates risk thresholds to determine if a human reviewer must break the automation chain."""
        pass

class ISimulationKernel(IKernel):
    """
    Simulation Kernel - Runs macroeconomic what-if scenarios.
    """
    @abstractmethod
    async def run_simulation(self, portfolio_id: str, policy: Policy, scenario: Scenario) -> Any:
        """Replays historical portfolios against hypothetical policies and shock factors."""
        pass

class IIntegrationKernel(IKernel):
    """
    Integration Kernel - The nervous system for inbound/outbound signals.
    """
    @abstractmethod
    async def ingest_signal(self, source: str, payload: Dict[str, Any]) -> Event:
        """Validates, sanitizes, and transforms external webhook data into an internal OS Event."""
        pass

    @abstractmethod
    async def publish_event(self, event_topic: str, event: Event) -> None:
        """Broadcasts a standardized event to the system event bus (e.g., Kafka/RabbitMQ)."""
        pass