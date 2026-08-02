from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class KernelInterface(ABC):
    """Base interface for all OS Kernels."""
    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        pass

class KnowledgeKernel(KernelInterface):
    @abstractmethod
    async def query_transactional(self, query: str) -> Any:
        pass

    @abstractmethod
    async def query_semantic(self, query: str) -> Any:
        pass

    @abstractmethod
    async def query_relational(self, query: str) -> Any:
        pass

    @abstractmethod
    async def query_temporal(self, query: str) -> Any:
        pass

class PolicyKernel(KernelInterface):
    @abstractmethod
    async def evaluate_policy(self, policy_id: str, context: Dict[str, Any]) -> Any:
        pass

class DecisionKernel(KernelInterface):
    @abstractmethod
    async def compute_decision_graph(self, inputs: Dict[str, Any]) -> Any:
        pass

class ExecutionKernel(KernelInterface):
    @abstractmethod
    async def execute_workflow(self, workflow_id: str, parameters: Dict[str, Any]) -> Any:
        pass

class GovernanceKernel(KernelInterface):
    @abstractmethod
    async def record_provenance(self, decision_id: str, provenance_data: Dict[str, Any]) -> None:
        pass

class SimulationKernel(KernelInterface):
    @abstractmethod
    async def run_simulation(self, scenario_id: str, parameters: Dict[str, Any]) -> Any:
        pass

class IntegrationKernel(KernelInterface):
    @abstractmethod
    async def publish_event(self, event_topic: str, event_data: Dict[str, Any]) -> None:
        pass
