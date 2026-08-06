from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel

class ASTNode(BaseModel):
    """Abstract Syntax Tree representation of a parsed policy."""
    node_type: str
    operator: str
    operands: List[Any]
    metadata: Dict[str, Any]

class ExecutionDAGNode(BaseModel):
    """Compiled Directed Acyclic Graph node for execution."""
    step_id: str
    dependencies: List[str]
    executable_logic: Any  # E.g., compiled lambda or jsonLogic ruleset

class PolicyParser(ABC):
    """Parses a DSL (JsonLogic, DMN, SQL) into an AST."""
    @abstractmethod
    def parse(self, dsl_content: str) -> ASTNode:
        pass

class PolicyCompiler(ABC):
    """Compiles an AST into an Execution DAG."""
    @abstractmethod
    def compile(self, ast: ASTNode) -> List[ExecutionDAGNode]:
        pass

class DeterministicRuntime(ABC):
    """Executes the compiled DAG predictably and statelessly."""
    @abstractmethod
    def execute(self, dag: List[ExecutionDAGNode], context: Dict[str, Any]) -> Any:
        pass

class PolicyEngine:
    """The facade for the Policy Kernel pipeline."""
    def __init__(self, parser: PolicyParser, compiler: PolicyCompiler, runtime: DeterministicRuntime):
        self.parser = parser
        self.compiler = compiler
        self.runtime = runtime

    def evaluate(self, policy_content: str, context: Dict[str, Any]) -> Any:
        ast = self.parser.parse(policy_content)
        dag = self.compiler.compile(ast)
        return self.runtime.execute(dag, context)
