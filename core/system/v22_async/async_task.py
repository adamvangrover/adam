# core/system/v22_async/async_task.py
from typing import Any, Callable, Dict, List


class AsyncTask:
    def __init__(self, name: str, dependencies: List[str], execute_func: Callable):
        self.name = name
        self.dependencies = dependencies
        self.execute_func = execute_func

    async def execute(self, dependency_results: Dict[str, Any]) -> Any:
        return await self.execute_func(**dependency_results)
