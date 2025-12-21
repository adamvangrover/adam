import concurrent.futures
from threading import RLock

from .workflow import Workflow


class WorkflowManager:
    def __init__(self, max_workers=10):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def execute_workflow(self, workflow: Workflow):
        """
        Executes a workflow, respecting task dependencies.
        """
        lock = RLock()
        completed_tasks = set()
        results = {}
        remaining_deps = {name: set(task.dependencies) for name, task in workflow.tasks.items()}
        futures = {}

        def _get_ready_tasks():
            ready = []
            for name, deps in remaining_deps.items():
                if not deps and name not in completed_tasks and name not in futures:
                    ready.append(workflow.tasks[name])
            return ready

        def _on_task_completed(future):
            try:
                result = future.result()
                task_name = future.task_name

                with lock:
                    results[task_name] = result
                    completed_tasks.add(task_name)

                    for name, deps in remaining_deps.items():
                        if task_name in deps:
                            deps.remove(task_name)

                    _schedule_tasks()

            except Exception as e:
                print(f"Task {future.task_name} failed in callback: {e}")

        def _schedule_tasks():
            with lock:
                ready_tasks = _get_ready_tasks()
                for task in ready_tasks:
                    dependency_results = {dep: results[dep] for dep in task.dependencies}

                    future = self.executor.submit(task.execute, dependency_results)
                    future.task_name = task.name
                    future.add_done_callback(_on_task_completed)
                    futures[task.name] = future

        _schedule_tasks()

        all_futures = list(futures.values())
        if all_futures:
            concurrent.futures.wait(all_futures)

        return results