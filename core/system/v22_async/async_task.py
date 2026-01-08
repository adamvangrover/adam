# core/system/v22_async/async_task.py
from typing import List, Callable, Any, Dict, Optional


class AsyncTask:
    def __init__(self, name: str, dependencies: List[str], execute_func: Callable, input_data: Optional[Dict[str, Any]] = None):
        self.name = name
        self.dependencies = dependencies
        self.execute_func = execute_func
        self.input_data = input_data or {}

    async def execute(self, dependency_results: Dict[str, Any]) -> Any:
        # If input_data is provided, we might want to pass it to execute_func or merge it.
        # Current behavior: execute_func is called with dependency results.
        # If execute_func expects input_data, it should be in dependency_results or bound.
        return await self.execute_func(**dependency_results)
