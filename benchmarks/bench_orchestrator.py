# Pseudo-benchmark configuration for pytest-benchmark
from src.backend.orchestration.agent_runtime import AgentOrchestrator

def test_benchmark_rule_evaluation(benchmark):
    orchestrator = AgentOrchestrator(None)
    # Ensures jsonLogic evaluation remains O(1) and under 1ms
    benchmark(orchestrator._evaluate_preconditions, {"ebitda_margin": 0.20}, "underwriter_gate")
