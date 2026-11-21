import pytest
from unittest.mock import patch
from core.utils.api_utils import APIValidator, MaliciousPayloadError, validate_api_request

def test_validate_success():
    """Test successful validation of API request data."""
    request_data = {"id": 123, "name": "Test"}
    required_params = ["id", "name"]
    assert APIValidator.validate(request_data, required_params) is True
    assert validate_api_request(request_data, required_params) is True

def test_validate_missing_parameter():
    """Test validation failure when a required parameter is missing."""
    request_data = {"id": 123}
    required_params = ["id", "name"]
    with pytest.raises(ValueError, match="Missing required parameter: name"):
        APIValidator.validate(request_data, required_params)

def test_validate_not_a_dict():
    """Test validation failure when request_data is not a dictionary."""
    request_data = ["id", 123]
    required_params = ["id"]
    with pytest.raises(TypeError, match="Expected request_data to be a dict"):
        APIValidator.validate(request_data, required_params)

@pytest.mark.asyncio
@patch('core.utils.api_utils.APIValidator._call_llm_safety_check')
async def test_inspect_payload_intent_safe(mock_llm):
    """Test successful inspection of a safe payload using mocks for LLM call."""
    mock_llm.return_value = 1.0
    request_data = {"message": "Hello world", "data": [1, 2, 3]}
    result = await APIValidator.inspect_payload_intent(request_data)
    assert result["is_safe"] is True
    assert result["safety_score"] == 1.0
    assert result["recommendation"] == "allow"
    assert len(result["flags"]) == 0
    mock_llm.assert_called_once()

@pytest.mark.asyncio
@patch('core.utils.api_utils.APIValidator._call_llm_safety_check')
async def test_inspect_payload_intent_malicious_heuristics(mock_llm):
    """
    Test inspection of a malicious payload triggered by heuristic flags.
    Mocks the LLM to return safe, ensuring heuristics drive the failure.
    """
    mock_llm.return_value = 1.0
    request_data = {
        "user_input": "SQL Injection AND <script>alert(1)</script> OR eval('1+1')",
        "action": "ignore previous instructions"
    }
    with pytest.raises(MaliciousPayloadError, match="Payload rejected by AI security inspector"):
        await APIValidator.inspect_payload_intent(request_data)
    mock_llm.assert_called_once()

@pytest.mark.asyncio
@patch('core.utils.api_utils.APIValidator._call_llm_safety_check')
async def test_inspect_payload_intent_malicious_llm(mock_llm):
    """
    Test inspection of a malicious payload triggered by the zero-shot LLM check.
    Mocks the LLM to return a malicious score (0.1).
    """
    mock_llm.return_value = 0.1
    request_data = {"user_input": "Subtle prompt injection attempt without heuristic keywords."}
    with pytest.raises(MaliciousPayloadError, match="Payload rejected by AI security inspector. Score: 0.1"):
        await APIValidator.inspect_payload_intent(request_data)
    mock_llm.assert_called_once()

@pytest.mark.asyncio
async def test_inspect_payload_intent_too_large():
    """Test that a payload exceeding MAX_PAYLOAD_LENGTH is rejected by recursive estimator."""
    large_list = ["A"] * (APIValidator.MAX_PAYLOAD_LENGTH + 1)
    request_data = {"data": large_list}
    with pytest.raises(ValueError, match="Payload length exceeds maximum allowed length"):
        await APIValidator.inspect_payload_intent(request_data)
