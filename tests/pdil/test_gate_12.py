import pytest
from datetime import datetime, timezone
from src.pdil.gate_12 import Gate12
from src.schemas.epistemic_types import AuthorityBoundaryRecord, TemporalBounds

def test_gate_12_admission():
    gate = Gate12()
    record = AuthorityBoundaryRecord(
        authority="NONE",
        subject={},
        source_lineage={},
        epistemic_input={},
        model_context={},
        temporal_bounds=TemporalBounds(
            event_time_te=datetime(2026, 9, 22, 20, 0, tzinfo=timezone.utc),
            knowledge_time_tk=datetime(2026, 9, 22, 20, 1, tzinfo=timezone.utc)
        ),
        provenance={},
        admission_criteria={"g12_h_schema_valid": True}
    )
    assert gate.evaluate_admission(record) is True
