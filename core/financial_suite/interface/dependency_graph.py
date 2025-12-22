from typing import Callable, Dict, List, Set
from core.financial_suite.context_manager import ContextManager


class DependencyGraph:
    """
    Manages dependencies between financial inputs and calculated outputs.
    Ensures efficient recalculation when inputs change.
    """

    def __init__(self, manager: ContextManager):
        self.manager = manager
        # Adjacency list: Input -> Set[Downstream Nodes]
        self.dependencies: Dict[str, Set[str]] = {
            "financials.projected_revenue_growth": {"ebitda", "fcff", "enterprise_value"},
            "valuation_context.wacc_method": {"wacc", "enterprise_value"},
            "credit_challenge.pd_method": {"pd", "rating", "cost_of_debt", "wacc"},
            # ...
        }

    def update_input(self, path: str, value: any):
        """
        Updates an input and triggers downstream recalculation.
        """
        print(f"Graph: Updating {path} -> {value}")
        self.manager.context.set_override(path, value)

        affected = self.dependencies.get(path, set())
        self.recalculate(affected)

    def recalculate(self, nodes: Set[str]):
        """
        Recalculates specific nodes.
        In this implementation, we map nodes to Manager methods.
        """
        if "enterprise_value" in nodes or "wacc" in nodes or "rating" in nodes:
            print("Graph: Triggering Full Solver...")
            self.manager.run_workstream()
        elif "waterfall" in nodes:
            # Partial update logic could go here
            pass

    def get_result(self, node: str):
        # Retrieve from manager.results
        pass
