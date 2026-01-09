import concurrent.futures
from threading import RLock
from .workflow import Workflow
import asyncio
import inspect

class WorkflowManager:
    def __init__(self, max_workers=10):
        # We can keep ThreadPoolExecutor for CPU-bound tasks if needed,
        # but for async tasks, we should run them in the current event loop.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def execute_workflow(self, workflow: Workflow):
        """
        Executes a workflow, respecting task dependencies.
        Supports both sync and async tasks transparently.
        """
        lock = RLock()
        completed_tasks = set()
        results = {}
        remaining_deps = {name: set(task.dependencies) for name, task in workflow.tasks.items()}
        futures = {}

        # Capture the current event loop if available, or create one?
        # Since this method itself is synchronous (called by execute_workflow in orchestrator),
        # but we are in an async context (from the test), we might have a running loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def _get_ready_tasks():
            ready = []
            for name, deps in remaining_deps.items():
                if not deps and name not in completed_tasks and name not in futures:
                    ready.append(workflow.tasks[name])
            return ready

        def _execute_task_wrapper(task, dependency_results):
            # This wrapper runs in a thread.
            # If the task action returns a coroutine, we must await it.
            # But we are in a thread, so we need a new loop or run_coroutine_threadsafe if we had a main loop.

            # HOWEVER, `task.execute` invokes `action(**kwargs)`.
            # If `action` is an `AsyncMock` or `async def`, `task.execute` returns a coroutine.

            res = task.execute(dependency_results)

            if asyncio.iscoroutine(res):
                # We are in a thread. We can use asyncio.run(res) if no loop is in this thread.
                # Since ThreadPoolExecutor creates new threads, they don't have loops.
                return asyncio.run(res)
            return res

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

                    # Wrap execution to handle async
                    future = self.executor.submit(_execute_task_wrapper, task, dependency_results)
                    future.task_name = task.name
                    future.add_done_callback(_on_task_completed)
                    futures[task.name] = future

        _schedule_tasks()

        # Wait for all tasks to complete
        # We need a loop to wait for tasks to be added to `futures` and completed.
        # The simple `concurrent.futures.wait` only waits for currently submitted futures.
        # But `_schedule_tasks` adds new ones.
        # We need to wait until `completed_tasks` equals total tasks or error.

        total_tasks = len(workflow.tasks)
        while len(completed_tasks) < total_tasks:
            # Check for failures?
            with lock:
                current_futures = list(futures.values())

            if not current_futures:
                # Should not happen if logic is correct and graph is DAG
                break

            done, not_done = concurrent.futures.wait(current_futures, return_when=concurrent.futures.FIRST_COMPLETED)

            # If a task failed, we might hang if we don't handle it.
            # `future.result()` in callback logs exception but doesn't stop everything.
            # For this simple implementation, let's just loop.

            # Improvement: Check for exceptions in done futures
            for f in done:
                try:
                    f.result()
                except Exception as e:
                    # If a task fails, we can't complete the workflow.
                    # Raise or break?
                    raise e

        return results
