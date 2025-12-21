import time

from core.agents.agent_base import AgentBase

from .task import Task
from .workflow import Workflow
from .workflow_manager import WorkflowManager


# A simple function to simulate a long-running task
def dummy_task(duration, task_name):
    print(f"Starting task: {task_name}")
    time.sleep(duration)
    print(f"Finished task: {task_name}")
    return f"{task_name} completed in {duration} seconds"

class ParallelOrchestrator(AgentBase):
    def __init__(self, config):
        super().__init__(config)
        self.workflow_manager = WorkflowManager(max_workers=5)

    def execute(self, num_tasks=5):
        """
        Creates and executes a workflow with a number of parallel tasks.
        """
        print(f"Creating a parallel workflow with {num_tasks} tasks.")

        tasks = [
            Task(
                name=f"dummy_task_{i}",
                action=dummy_task,
                duration=i + 1,
                task_name=f"Task {i}"
            ) for i in range(num_tasks)
        ]

        workflow = Workflow(name="parallel_workflow", tasks=tasks)

        print("Executing parallel workflow...")
        start_time = time.time()
        results = self.workflow_manager.execute_workflow(workflow)
        end_time = time.time()
        print(f"Workflow executed in {end_time - start_time:.2f} seconds.")

        return self._synthesize_results(results)

    def _synthesize_results(self, results):
        return {
            "source_agent": self.__class__.__name__,
            "confidence_score": 1.0,
            "data": results,
            "summary": "Successfully executed a parallel workflow."
        }