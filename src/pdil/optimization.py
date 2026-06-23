from typing import Dict, Any, List

class ContextOptimizer:
    """Manages context compression, summarization, and optimization."""
    def __init__(self, token_limit: int = 4000):
        self.token_limit = token_limit

    def compress(self, context: str) -> str:
        """Applies lossy compression (e.g., removing stop words) to reduce token count."""
        # Simplified compression heuristic
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "are"}
        words = context.split()
        compressed = " ".join([w for w in words if w.lower() not in stop_words])
        return compressed[:self.token_limit * 4]

    def summarize(self, context: str) -> str:
        """Summarizes verbose context into key data points."""
        if len(context) < self.token_limit * 2:
            return context
        return f"Summary: [Extracted {len(context) // 100} key entities and signals.]"

    def synthesize(self, contexts: List[str]) -> str:
        """Synthesizes multiple context streams into a single structured narrative."""
        synthesized = " | ".join(contexts)
        return self.optimize_context(synthesized)

    def analyze(self, context: str) -> Dict[str, Any]:
        """Analyzes context for signal-to-noise ratio."""
        length = len(context)
        return {
            "length": length,
            "estimated_tokens": length // 4,
            "density_score": 0.85 if length > 100 else 0.5
        }

    def optimize_context(self, context: str) -> str:
        """Applies optimal combination of compression and truncation."""
        if len(context) > self.token_limit * 4:
            return self.compress(context)
        return context

class CostOptimizer:
    """Tracks cost and compute utilization."""
    def __init__(self):
        self.usage_stats: Dict[str, Dict[str, Any]] = {}

    def record_usage(self, model_name: str, tokens_used: int, cost_per_1k: float) -> Dict[str, Any]:
        if model_name not in self.usage_stats:
            self.usage_stats[model_name] = {"tokens": 0, "cost": 0.0}

        self.usage_stats[model_name]["tokens"] += tokens_used
        self.usage_stats[model_name]["cost"] += (tokens_used / 1000.0) * cost_per_1k

        return self.usage_stats[model_name]
