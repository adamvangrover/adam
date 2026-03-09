import asyncio
import json
import logging
import os
from datetime import datetime

from core.agents.upgrade_swarm.upgrade_agents import (
    DocumenterAgent,
    InnovatorAgent,
    ModernizerAgent,
    OptimizerAgent,
    PruningAgent,
    RefactorAgent,
    ValidatorAgent,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UpgradeManager")

PHASE_MAP = {
    "Monday": PruningAgent,
    "Tuesday": RefactorAgent,
    "Wednesday": OptimizerAgent,
    "Thursday": ModernizerAgent,
    "Friday": InnovatorAgent,
    "Saturday": DocumenterAgent,
    "Sunday": ValidatorAgent
}

class UpgradeManager:
    """Orchestrates the 7-phase daily upgrade swarm based on the target list."""
    def __init__(self, tracker_path: str = "data/upgrade_targets.json"):
        self.tracker_path = tracker_path
        self.tracker_data = self._load_tracker()
        self.today_name = datetime.now().strftime("%A")

    def _load_tracker(self) -> dict:
        if not os.path.exists(self.tracker_path):
            logger.error(f"Tracker file not found at {self.tracker_path}")
            return {"book_of_work": []}
        with open(self.tracker_path, "r") as f:
            return json.load(f)

    def _save_tracker(self):
        with open(self.tracker_path, "w") as f:
            json.dump(self.tracker_data, f, indent=2)

    def _get_next_target(self) -> dict:
        """Finds the next pending target that aligns with today's phase."""
        for target in self.tracker_data.get("book_of_work", []):
            if target["status"] == "pending" and target["current_phase"] == self.today_name:
                return target
        return None

    def _progress_target_phase(self, target: dict):
        """Moves the target to the next day's phase, or marks it complete."""
        days = list(PHASE_MAP.keys())
        current_idx = days.index(target["current_phase"])

        if current_idx < len(days) - 1:
            target["current_phase"] = days[current_idx + 1]
        else:
            target["status"] = "completed"
            target["current_phase"] = "Done"

            # Move to completed list
            self.tracker_data["book_of_work"].remove(target)
            if "completed" not in self.tracker_data:
                self.tracker_data["completed"] = []
            self.tracker_data["completed"].append(target)

    async def execute_daily_cycle(self):
        """Runs the upgrade cycle for the current day."""
        logger.info(f"Starting Upgrade Cycle for {self.today_name}")

        target = self._get_next_target()
        if not target:
            logger.info(f"No pending targets found for {self.today_name}.")
            return

        target_path = target["target_path"]
        logger.info(f"Selected target: {target_path}")

        if not os.path.exists(target_path):
            logger.error(f"Target file does not exist: {target_path}")
            target["status"] = "error"
            self._save_tracker()
            return

        with open(target_path, "r") as f:
            content = f.read()

        agent_class = PHASE_MAP.get(self.today_name)
        if not agent_class:
            logger.error(f"No agent configured for phase: {self.today_name}")
            return

        # Instantiate agent and execute
        agent = agent_class(config={"name": f"{self.today_name}Agent"})
        result = await agent.execute(target_path, content)

        # Log result
        logger.info(f"Execution complete. Agent: {result['agent']}, Target: {result['target']}")
        logger.info(f"Tokens used: {result.get('token_usage', 0)}")

        # In a real run, you'd apply the patch/result here, or create a PR.
        # For this scaffolding, we just track it.

        # Record history and advance
        target.setdefault("history", []).append({
            "phase": self.today_name,
            "date": datetime.now().isoformat(),
            "status": "success",
            "tokens": result.get("token_usage", 0)
        })

        self._progress_target_phase(target)
        self._save_tracker()
        logger.info(f"Tracker updated. Target is now at phase: {target.get('current_phase')}")

if __name__ == "__main__":
    # Test execution
    manager = UpgradeManager()
    asyncio.run(manager.execute_daily_cycle())
