from typing import List

class ArchitectureInvariants:
    @staticmethod
    def get_invariants() -> List[str]:
        return [
            "policy_before_mutation",
            "provenance_before_commit",
            "frozen_context",
            "deterministic_numeric_execution"
        ]

    @staticmethod
    def enforce_policy_before_mutation(action: str) -> bool:
        # Placeholder for strict enforcement logic
        return True
