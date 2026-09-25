from src.schemas.epistemic_types import AuthorityBoundaryRecord

class Gate12:
    def evaluate_admission(self, record: AuthorityBoundaryRecord) -> bool:
        if record.authority != "NONE":
            return False

        tb = record.temporal_bounds
        if tb.event_time_te > tb.knowledge_time_tk:
            return False

        if tb.decision_time_td and tb.knowledge_time_tk > tb.decision_time_td:
            return False

        if tb.execution_time_tx and tb.decision_time_td and tb.decision_time_td > tb.execution_time_tx:
            return False

        if not all(record.admission_criteria.values()):
            return False

        return True
