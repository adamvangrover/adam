# core/engine/swarm/worker_node.py

"""
SwarmWorker: The base unit of the Hive Mind.
Designed for massive parallelism, minimal state, and specific task execution.
"""

import asyncio
import logging
import uuid
import ast
import subprocess
import tempfile
import os
from typing import Dict, Any, Optional
from core.engine.swarm.pheromone_board import PheromoneBoard

# Try to import LLMPlugin, but don't fail if missing (graceful degradation)
try:
    from core.llm_plugin import LLMPlugin
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

logger = logging.getLogger(__name__)

class SwarmWorker:
    def __init__(self, board: PheromoneBoard, role: str = "generalist"):
        self.id = str(uuid.uuid4())[:8]
        self.board = board
        self.role = role
        self.is_active = True
        logger.info(f"SwarmWorker {self.id} ({self.role}) initialized.")

    async def run(self):
        """
        Main loop for the worker.
        """
        while self.is_active:
            # 1. Look for work (sniff pheromones)
            tasks = await self.board.sniff(signal_type=f"TASK_{self.role.upper()}")

            if tasks:
                # Pick the highest intensity task
                tasks.sort(key=lambda x: x.intensity, reverse=True)
                task_signal = tasks[0]

                # Try to claim it (consume)
                await self.board.consume(task_signal)

                # Execute
                try:
                    await self.execute_task(task_signal.data)
                except Exception as e:
                    logger.error(f"Worker {self.id} failed task: {e}")
                    await self.board.deposit("ERROR", {"worker": self.id, "error": str(e)}, intensity=5.0)
            else:
                # No work, wait a bit
                await asyncio.sleep(1)

    async def execute_task(self, data: Dict[str, Any]):
        """
        Execute the specific logic. Override this in subclasses.
        """
        logger.info(f"Worker {self.id} executing task: {data}")
        await asyncio.sleep(0.5)
        # Report success via pheromone
        result_data = {"status": "success", "original_task": data, "worker": self.id}
        await self.board.deposit("RESULT", result_data, intensity=10.0, source=self.id)

    def stop(self):
        self.is_active = False

class AnalysisWorker(SwarmWorker):
    async def execute_task(self, data: Dict[str, Any]):
        target = data.get("target")
        logger.info(f"Worker {self.id} analyzing {target}...")

        # Mock analysis
        await asyncio.sleep(1.0)
        score = len(target) * 10 if target else 0

        result = {
            "target": target,
            "risk_score": score,
            "analyst_comment": f"Analyzed by {self.id}"
        }

        await self.board.deposit("ANALYSIS_RESULT", result, intensity=10.0, source=self.id)

class CoderWorker(SwarmWorker):
    def __init__(self, board: PheromoneBoard, role: str = "coder"):
        super().__init__(board, role)
        if LLM_AVAILABLE:
            # Initialize with default settings or mock
            self.llm = LLMPlugin(config={"provider": "gemini", "gemini_model_name": "gemini-3-flash"})
        else:
            self.llm = None

    async def execute_task(self, data: Dict[str, Any]):
        prompt = data.get("prompt", "")
        task_id = data.get("id", str(uuid.uuid4()))
        logger.info(f"CoderWorker {self.id} generating code for: {prompt[:50]}...")

        code_snippet = ""
        if self.llm:
            try:
                # Use LLM to generate code
                # This is synchronous call wrapped in async in real app, here assuming LLMPlugin might block or we run in thread
                # For safety, we just simulate or do a quick generation if mock
                code_snippet = f"# Generated code for {prompt}\ndef generated_function():\n    print('Hello from Swarm')\n"
                # In production: code_snippet = self.llm.generate_code(prompt)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                code_snippet = "# Error in generation"
        else:
            code_snippet = f"# LLM Unavailable. Mock code for: {prompt}"

        result = {
            "task_id": task_id,
            "code": code_snippet,
            "worker": self.id
        }
        await self.board.deposit("CODE_RESULT", result, intensity=10.0, source=self.id)
        # Also deposit a review task
        await self.board.deposit("TASK_REVIEWER", {"code": code_snippet, "origin_task_id": task_id}, intensity=8.0, source=self.id)

class ReviewerWorker(SwarmWorker):
    async def execute_task(self, data: Dict[str, Any]):
        code = data.get("code", "")
        origin_id = data.get("origin_task_id")
        logger.info(f"ReviewerWorker {self.id} reviewing code...")

        issues = []
        try:
            # Static Analysis using AST
            tree = ast.parse(code)
            # Basic check: look for print statements (as an example of 'bad practice' in prod)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print":
                    issues.append("Found 'print' statement. Use logging instead.")
        except SyntaxError as e:
            issues.append(f"Syntax Error: {e}")
        except Exception as e:
            issues.append(f"Analysis Error: {e}")

        verdict = "APPROVED" if not issues else "NEEDS_REVISION"

        result = {
            "origin_task_id": origin_id,
            "verdict": verdict,
            "issues": issues,
            "worker": self.id
        }
        await self.board.deposit("REVIEW_RESULT", result, intensity=10.0, source=self.id)

class TesterWorker(SwarmWorker):
    async def execute_task(self, data: Dict[str, Any]):
        # In a real scenario, this would write code to a temp file and run pytest
        test_target = data.get("target_file", "test_mock.py")
        logger.info(f"TesterWorker {self.id} running tests on {test_target}...")

        # Mock test execution
        await asyncio.sleep(2.0)

        # Simulate pass/fail
        success = True
        output = "Tests Passed: 5/5"

        result = {
            "target": test_target,
            "success": success,
            "output": output,
            "worker": self.id
        }
        await self.board.deposit("TEST_RESULT", result, intensity=10.0, source=self.id)

class SentinelWorker(SwarmWorker):
    """
    Credit Sentinel (System 1 -> System 2 Bridge).
    Passively monitors swarm outputs for high-risk anomalies.
    """
    async def run(self):
        logger.info(f"SentinelWorker {self.id} starting watch...")
        while self.is_active:
            # Sniff for ANALYSIS_RESULT
            # Note: We do NOT consume, to allow other components (like HiveMind aggregator) to also see it.
            # But PheromoneBoard.sniff doesn't consume.
            results = await self.board.sniff(signal_type="ANALYSIS_RESULT")

            for signal in results:
                # We use a simple in-memory cache to avoid re-alerting on the same signal ID
                # In a real system, we'd check if we've processed this signal.id
                # Here, we assume signals are transient or we just check content.

                data = signal.data
                risk_score = data.get("risk_score", 0)
                target = data.get("target", "Unknown")

                # Check for Anomaly (e.g., Risk > 50)
                # In the mock AnalysisWorker, score = len(target) * 10
                # So "distressed_co" (13 chars) -> 130 score.

                if risk_score > 50:
                    logger.warning(f"Sentinel {self.id} detected ANOMALY: {target} (Score: {risk_score}). Triggering System 2.")

                    alert_payload = {
                        "target": target,
                        "risk_score": risk_score,
                        "anomaly_type": "High Risk Score",
                        "detected_by": self.id
                    }

                    # Deposit RISK_ALERT which MetaOrchestrator can pick up
                    await self.board.deposit("RISK_ALERT", alert_payload, intensity=20.0, source=self.id)

                    # Optional: We could also deposit TASK_PLANNER directly if we wanted to bypass orchestrator logic
                    # await self.board.deposit("TASK_PLANNER", {"intent": "DEEP_DIVE", "target": target}, intensity=50.0)

            await asyncio.sleep(0.5)
