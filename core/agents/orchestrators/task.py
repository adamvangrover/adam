from typing import Any, Callable, Dict, List


class Task:
    def __init__(self, name: str, action: Callable, dependencies: List[str] = None, **kwargs):
        self.name = name
        self.action = action
        # The names of tasks that must be completed before this one can run.
        self.dependencies = dependencies or []
        # The keyword arguments required by the action function.
        self.kwargs = kwargs
        # The result of the task execution.
        self.result = None
        # The state of the task.
        self.state = "pending" # pending, running, completed, failed

    def execute(self, dependency_results: Dict[str, Any]):
        """
        Executes the task's action, passing in the results of its dependencies.
        """
        self.state = "running"
        try:
            # The WorkflowManager passes dependency results as a dictionary where keys are
            # task names and values are their results. We pass this dictionary
            # as keyword arguments to the action.
            resolved_kwargs = {**self.kwargs, **dependency_results}
            self.result = self.action(**resolved_kwargs)
            self.state = "completed"
            return self.result
        except Exception as e:
            self.state = "failed"
            print(f"Task {self.name} failed: {e}")
            raise