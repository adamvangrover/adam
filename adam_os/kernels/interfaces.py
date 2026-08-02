from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, List
from adam_os.core.ontology import Event, Policy, Decision, Evidence, Scenario

class IKnowledgeKernel(ABC):
    """
    Knowledge Kernel - The institutional memory.
    Answers fundamentally different questions across transactional, semantic, relational, and temporal tiers.
    """
    @abstractmethod
    def ask_truth(self, query: str) -> Any:
        """Query PostgreSQL (What is true?)"""
        pass

    @abstractmethod
    def ask_similarity(self, vector: List[float]) -> Any:
        """Query Qdrant (What is similar?)"""
        pass

    @abstractmethod
    def ask_connections(self, entity_id: str) -> Any:
        """Query Knowledge Graph (What is connected?)"""
        pass

    @abstractmethod
    def ask_history(self, entity_id: str) -> List[Event]:
        """Query Event Store (What happened?)"""
        pass

class IPolicyKernel(ABC):
    """
    Policy Kernel - Compiles and executes rules (JsonLogic, DMN, SQL, YAML).
    """
    @abstractmethod
    def compile_policy(self, dsl: str, format: str = "jsonlogic") -> Policy:
        """Compiles DSL into an executable policy."""
        pass

    @abstractmethod
    def execute_policy(self, policy: Policy, context: Dict[str, Any]) -> Any:
        """Executes the deterministic runtime for the policy."""
        pass

class IDecisionKernel(ABC):
    """
    Decision Kernel - Produces explainable decision graphs.
    """
    @abstractmethod
    def evaluate(self, target_id: str, policy: Policy, evidence: List[Evidence]) -> Decision:
        """Evaluates a target against a policy to produce a decision graph and outcome."""
        pass

class IExecutionKernel(ABC):
    """
    Execution Kernel - The workflow runtime (e.g. LangGraph).
    """
    @abstractmethod
    def plan_execution(self, workflow_dsl: str) -> Any:
        """Generates an execution plan from a DSL."""
        pass

    @abstractmethod
    def run_workflow(self, plan: Any) -> Any:
        """Executes the workflow, handling checkpointing and interrupts."""
        pass

class IGovernanceKernel(ABC):
    """
    Governance Kernel - Manages provenance, audits, and human review chains.
    """
    @abstractmethod
    def register_decision(self, decision: Decision) -> str:
        """Records a decision into the immutable audit ledger."""
        pass

    @abstractmethod
    def require_approval(self, decision: Decision) -> bool:
        """Routes a decision to human review chains based on policy thresholds."""
        pass

class ISimulationKernel(ABC):
    """
    Simulation Kernel - Runs what-if scenarios and alternative policies on historical data.
    """
    @abstractmethod
    def run_simulation(self, portfolio_id: str, policy: Policy, scenario: Scenario) -> Any:
        """Replays historical portfolios against alternative policies and shocks."""
        pass

class IIntegrationKernel(ABC):
    """
    Integration Kernel - Transforms external signals into system events.
    """
    @abstractmethod
    def ingest_signal(self, source: str, payload: Dict[str, Any]) -> Event:
        """Validates and transforms external data into an Event."""
        pass

    @abstractmethod
    def publish_event(self, event: Event) -> None:
        """Broadcasts the event to the system event bus."""
        pass
