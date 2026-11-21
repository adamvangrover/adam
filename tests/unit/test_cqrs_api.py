import pytest
from fastapi.testclient import TestClient

from adam_os.api.main import app
from adam_os.contexts.ledger.commands import OriginateLoanCommand, UpdateDebtCommand, RevalueAssetCommand
from adam_os.contexts.ledger.handler import CommandHandler
from adam_os.contexts.governance.engine import DeterministicPolicyEngine
from adam_os.core.events import LoanOriginated

client = TestClient(app)

def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_dscr_rule_evaluation() -> None:
    engine = DeterministicPolicyEngine()

    # Safe DSCR (> 1.25)
    safe_context = {"net_operating_income": 150.0, "debt_service": 100.0} # DSCR = 1.5
    safe_result = engine.evaluate("dscr_125", safe_context)
    assert safe_result.is_breached is False

    # Breached DSCR (< 1.25)
    breach_context = {"net_operating_income": 110.0, "debt_service": 100.0} # DSCR = 1.1
    breach_result = engine.evaluate("dscr_125", breach_context)
    assert breach_result.is_breached is True

def test_command_handler_logic() -> None:
    handler = CommandHandler()

    # Test Originate
    cmd = OriginateLoanCommand(entity_id="loan-xyz", principal_amount=5000.0, asset_value=10000.0)
    events = handler.handle(cmd)

    assert len(events) == 1
    assert isinstance(events[0], LoanOriginated)
    assert events[0].principal_amount == 5000.0

    # Test failing update on uninitialized (no previous events passed)
    update_cmd = UpdateDebtCommand(entity_id="loan-new", new_debt_amount=2000.0)
    with pytest.raises(ValueError, match="Cannot update debt for an uninitialized loan."):
        handler.handle(update_cmd)

def test_api_originate_loan() -> None:
    response = client.post(
        "/ledger/originate",
        json={"entity_id": "loan-api-1", "principal_amount": 1000.0, "asset_value": 2500.0}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Loan successfully originated."
    assert response.json()["events_generated"] == 1

def test_api_update_debt_fails_without_state() -> None:
    # Since our mocked route currently doesn't load state from a DB, updating debt straight away will fail
    # which proves the handler's business logic is throwing appropriately via the API
    response = client.post(
        "/ledger/update_debt",
        json={"entity_id": "loan-api-2", "new_debt_amount": 1500.0}
    )
    assert response.status_code == 400
    assert "uninitialized loan" in response.json()["detail"]
