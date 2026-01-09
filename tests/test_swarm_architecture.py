
import pytest
import asyncio
from core.engine.swarm.pheromone_board import PheromoneBoard
from core.engine.swarm.worker_node import SwarmWorker, AnalysisWorker, CoderWorker, ReviewerWorker
from core.engine.swarm.hive_mind import HiveMind

@pytest.mark.asyncio
async def test_pheromone_board_basic():
    board = PheromoneBoard()
    await board.deposit("TEST_SIGNAL", {"key": "value"}, intensity=10.0)

    signals = await board.sniff("TEST_SIGNAL")
    assert len(signals) == 1
    assert signals[0].data["key"] == "value"

    await board.consume(signals[0])
    signals_after = await board.sniff("TEST_SIGNAL")
    assert len(signals_after) == 0

@pytest.mark.asyncio
async def test_worker_analysis():
    board = PheromoneBoard()
    worker = AnalysisWorker(board, role="analyst")

    # Manually trigger execute_task
    task_data = {"target": "TSLA"}
    await worker.execute_task(task_data)

    # Check result
    signals = await board.sniff("ANALYSIS_RESULT")
    assert len(signals) >= 1
    assert signals[0].data["target"] == "TSLA"

@pytest.mark.asyncio
async def test_coder_reviewer_loop():
    board = PheromoneBoard()
    coder = CoderWorker(board)
    reviewer = ReviewerWorker(board)

    # 1. Deposit Code Task
    await board.deposit("TASK_CODER", {"prompt": "print('hello')", "id": "task-1"})

    # 2. Run Coder (simulated run step)
    # Instead of running full loop, we execute task directly or use logic
    # But CoderWorker logic is in execute_task which deposits to board

    await coder.execute_task({"prompt": "print('hello')", "id": "task-1"})

    # Check for Code Result
    code_results = await board.sniff("CODE_RESULT")
    assert len(code_results) == 1

    # Check for Review Task (Coder should trigger Reviewer)
    review_tasks = await board.sniff("TASK_REVIEWER")
    assert len(review_tasks) == 1

    # 3. Run Reviewer
    await reviewer.execute_task(review_tasks[0].data)

    # Check Review Result
    review_results = await board.sniff("REVIEW_RESULT")
    assert len(review_results) == 1
    assert review_results[0].data["origin_task_id"] == "task-1"
    # Logic should catch 'print' usage
    assert "Found 'print'" in str(review_results[0].data["issues"])

@pytest.mark.asyncio
async def test_hive_mind_initialization():
    hm = HiveMind(worker_count=10)
    await hm.initialize()

    assert len(hm.workers) == 10

    # Check role distribution
    roles = [w.role for w in hm.workers]
    assert "analyst" in roles
    assert "coder" in roles
    assert "reviewer" in roles
    assert "tester" in roles

    await hm.shutdown()
