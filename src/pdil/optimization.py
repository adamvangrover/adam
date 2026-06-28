from typing import Dict, Any

class TokenOptimizer:
    """Manages compute and token optimization for LLM interactions."""
    def __init__(self, token_limit: int = 4000):
        self.token_limit = token_limit
        self.usage_stats: Dict[str, Dict[str, Any]] = {}

    def optimize_context(self, context: str) -> str:
        """Truncates or summarizes context to fit within token limits."""
        if len(context) > self.token_limit * 4: # approx 4 chars per token
            return context[:self.token_limit * 2] + "... [TRUNCATED] ..." + context[-self.token_limit * 2:]
        return context

    def record_usage(self, model_name: str, tokens_used: int, cost_per_1k: float) -> Dict[str, Any]:
        """Tracks cost and compute utilization per model."""
        if model_name not in self.usage_stats:
            self.usage_stats[model_name] = {"tokens": 0, "cost": 0.0}

        self.usage_stats[model_name]["tokens"] += tokens_used
        self.usage_stats[model_name]["cost"] += (tokens_used / 1000.0) * cost_per_1k

        return self.usage_stats[model_name]

__all__ = ["TokenOptimizer"]
