import pytest
from adam_v3.kernel.jsonrpc import JsonRpcHandler

def test_jsonrpc_handler():
    handler = JsonRpcHandler()
    request = {
        "jsonrpc": "2.0",
        "method": "evaluate_conviction",
        "params": {"trace_id": "123", "agent_output": {}},
        "id": 1
    }
    response = handler.handle(request)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["result"]["executed_method"] == "evaluate_conviction"
