class DeterministicSolver:
    @staticmethod
    def calculate_expected_loss(pd: float, lgd: float, ead: float) -> float:
        return pd * lgd * ead
