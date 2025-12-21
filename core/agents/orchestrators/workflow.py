from typing import Dict, List

from .task import Task


class Workflow:
    def __init__(self, name: str, tasks: List[Task]):
        self.name = name
        self.tasks = {task.name: task for task in tasks}
        self.task_dependencies = self._build_dependency_graph()

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Builds a dictionary where keys are task names and values are lists of
        tasks that depend on them. This represents the reverse of the dependency
        list in each task.
        """
        graph = {name: [] for name in self.tasks}
        for name, task in self.tasks.items():
            for dep_name in task.dependencies:
                if dep_name not in self.tasks:
                    raise ValueError(f"Task '{name}' has an undefined dependency: '{dep_name}'")
                graph[dep_name].append(name)
        return graph

    def get_initial_tasks(self) -> List[Task]:
        """Returns a list of tasks that have no dependencies."""
        return [task for task in self.tasks.values() if not task.dependencies]