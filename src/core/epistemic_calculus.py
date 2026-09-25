from src.schemas.epistemic_types import EpistemicStatus

class EpistemicCalculus:
    def __init__(self, e_min: float, tau_ood: float, tau_w: float):
        self.e_min = e_min
        self.tau_ood = tau_ood
        self.tau_w = tau_w

    def evaluate_state(
        self,
        evidence_score: float,
        ood_metric: float,
        wasserstein_distance: float,
        resolution_policy_available: bool,
        critical_path: bool
    ) -> EpistemicStatus:
        is_insufficient = evidence_score < self.e_min
        is_ood = ood_metric > self.tau_ood
        is_conflicted = wasserstein_distance > self.tau_w
        is_unresolved = is_conflicted and not resolution_policy_available

        if is_unresolved or (is_insufficient and critical_path):
            return EpistemicStatus.UNKNOWN

        if is_insufficient:
            return EpistemicStatus.INSUFFICIENT_EVIDENCE

        if is_ood:
            return EpistemicStatus.OOD

        if is_conflicted:
            return EpistemicStatus.CONFLICTED

        return EpistemicStatus.SUPPORTED
