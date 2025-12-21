from typing import Callable, List


class TreeOfThoughts:
    """
    Implements the Tree of Thoughts (ToT) reasoning framework.
    Uses a Search Algorithm (BFS/DFS) to explore a space of 'Thought Steps'.

    Components:
    1. Generator: Proposes k possible next steps.
    2. Evaluator: Scores each step.
    3. Search: Manages the tree exploration.
    """

    def __init__(self, generator_func: Callable, evaluator_func: Callable, max_depth: int = 3, width: int = 3):
        self.generator = generator_func
        self.evaluator = evaluator_func
        self.max_depth = max_depth
        self.width = width # 'k' in the paper

    def solve(self, initial_state: str, method: str = "bfs"):
        """
        Solves the problem using the specified search method.
        """
        if method == "bfs":
            return self._bfs(initial_state)
        elif method == "dfs":
            return self._dfs(initial_state)
        else:
            raise ValueError("Unknown method. Use 'bfs' or 'dfs'.")

    def _bfs(self, initial_state: str) -> List[str]:
        """
        Breadth-First Search.
        Maintains a set of candidate partial solutions.
        """
        current_level = [initial_state]

        for depth in range(self.max_depth):
            print(f"--- Depth {depth} ---")
            next_level_candidates = []

            # 1. Generate
            for state in current_level:
                proposals = self.generator(state, self.width)
                for prop in proposals:
                    # 2. Evaluate
                    score = self.evaluator(prop)
                    next_level_candidates.append((score, prop))

            # 3. Prune (Keep top k)
            # Sort by score descending
            next_level_candidates.sort(key=lambda x: x[0], reverse=True)
            top_k = next_level_candidates[:self.width]

            print(f"Top {len(top_k)} thoughts: {[(s, p[-20:]) for s, p in top_k]}")

            current_level = [p for s, p in top_k]

            # Check for solution (mock condition)
            if any(self._is_solution(p) for p in current_level):
                return [p for p in current_level if self._is_solution(p)]

        return current_level

    def _dfs(self, initial_state: str):
        # Placeholder for DFS implementation
        pass

    def _is_solution(self, state: str) -> bool:
        # Mock solution check
        return "SOLVED" in state

# --- Mock LLM Functions ---

def mock_generator(state: str, k: int) -> List[str]:
    """
    Simulates an LLM generating k possible continuations.
    """
    # In reality: llm.generate(state, n=k)
    thoughts = [
        f"{state} -> Step A (Logic OK)",
        f"{state} -> Step B (Logic Flawed)",
        f"{state} -> Step C (Logic Excellent)",
        f"{state} -> Step D (Irrelevant)"
    ]
    return thoughts[:k]

def mock_evaluator(state: str) -> float:
    """
    Simulates an LLM scoring a thought.
    """
    # In reality: llm.score(state)
    import random
    if "Excellent" in state: return 0.9
    if "OK" in state: return 0.6
    if "Flawed" in state: return 0.2
    return random.random()

if __name__ == "__main__":
    print("Running Tree of Thoughts (BFS)...")
    tot = TreeOfThoughts(mock_generator, mock_evaluator, max_depth=3, width=2)
    result = tot.solve("Problem: How to allocate capital?")
    print("Final Result:", result)
