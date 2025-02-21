# scripts/run_adam.py

import yaml
from core.system.agent_orchestrator import AgentOrchestrator
from core.system.task_scheduler import TaskScheduler
#... (import other necessary modules and classes)

def main():
    # 1. Load configuration
    with open('config/agents.yaml', 'r') as f:
        agents_config = yaml.safe_load(f)
    with open('config/system.yaml', 'r') as f:
        system_config = yaml.safe_load(f)

    # 2. Initialize components
    orchestrator = AgentOrchestrator(agents_config)
    scheduler = TaskScheduler(system_config)

    # 3. Schedule tasks
    scheduler.schedule_tasks()

    # 4. Run scheduler (or execute specific tasks manually)
    scheduler.run_scheduler()
    # Alternatively, you can execute specific tasks manually:
    # orchestrator.execute_workflow("generate_newsletter")
    # orchestrator.execute_workflow("perform_company_analysis", company_data={'name': 'Example Corp',...})

if __name__ == "__main__":
    main()
