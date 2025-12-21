# core/system/v22_async/workflow.py
from typing import Dict

from .async_task import AsyncTask


class AsyncWorkflow:
    def __init__(self, name: str):
        self.name = name
        self.tasks: Dict[str, AsyncTask] = {}

    def add_task(self, task: AsyncTask):
        self.tasks[task.name] = task
