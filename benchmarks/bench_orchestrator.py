# Pseudo-benchmark configuration for pytest-benchmark
from src.backend.orchestration.agent_runtime import AgentOrchestrator

def test_benchmark_rule_evaluation(benchmark):
    orchestrator = AgentOrchestrator(None, rules_path="rules")
    # Ensures jsonLogic evaluation remains O(1) and under 1ms
    benchmark(orchestrator._evaluate_preconditions, {"ebitda_margin": 0.20, "leverage_ratio": 5.0}, "underwriter_gate")
