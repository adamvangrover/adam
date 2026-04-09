import pytest
import asyncio
from typing import Dict, Any

from core.security.red_team.quantum_scanner import QuantumScanner
from core.security.red_team.response_engine import RealTimeResponseEngine
from core.agents.red_team_agent import RedTeamAgent
from core.agents.agent_base import AgentInput

# --- Tests for QuantumScanner ---

def test_quantum_scanner_vulnerable():
    scanner = QuantumScanner()
    result = scanner.scan_encryption_algorithm("RSA-2048")
    assert result["vulnerable"] is True
    assert result["severity"] == "CRITICAL"
    assert "Shor's algorithm" in result["reason"]

def test_quantum_scanner_safe():
    scanner = QuantumScanner()
    result = scanner.scan_encryption_algorithm("AES-256")
    assert result["vulnerable"] is False
    assert result["severity"] == "INFO"

def test_quantum_scanner_system():
    scanner = QuantumScanner()
    result = scanner.scan_system(["Kyber", "RSA-4096", "AES-256"])
    assert result["system_vulnerable"] is True
    assert result["overall_severity"] == "CRITICAL"
    assert len(result["findings"]) == 3

# --- Tests for RealTimeResponseEngine ---

def test_response_engine_critical_threat():
    engine = RealTimeResponseEngine()
    threat = {
        "type": "DATA_EXFILTRATION",
        "severity_score": 8.0
    }
    analysis = engine.analyze_threat(threat)
    assert analysis["urgency"] == "IMMEDIATE"
    assert analysis["recommended_action"] == "REVOKE_ALL_CREDENTIALS"
    assert analysis["calculated_severity"] >= 9.0

def test_response_engine_mitigation():
    engine = RealTimeResponseEngine()
    analysis = {
        "threat_id": "TEST_123",
        "recommended_action": "BLOCK_IP_AND_ALERT"
    }
    mitigation = engine.execute_mitigation(analysis)
    assert mitigation["status"] == "SUCCESS"
    assert mitigation["action_executed"] == "BLOCK_IP_AND_ALERT"

# --- Tests for RedTeamAgent Integration ---

@pytest.mark.asyncio
async def test_red_team_agent_cybersecurity_integration():
    # Use a dummy config
    agent = RedTeamAgent(config={"name": "TestRedTeamAgent"})

    # Create AgentInput with cyber threats and vulnerable encryption
    input_data = AgentInput(
        query="Test Cybersecurity Target",
        context={
            "encryption_algorithms": ["RSA-1024", "AES-256"],
            "cyber_threats": [
                {"type": "DDoS", "severity_score": 7.5}
            ]
        }
    )

    # Execute the agent
    output = await agent.execute(input_data)

    # Assertions
    # We expect the total_impact to be high due to the quantum vulnerability and threat
    metadata = output.metadata
    assert "critique" in metadata

    # Check impact score was boosted
    impact_score = metadata["critique"]["impact_score"]
    assert impact_score > 0.0 # Should be boosted by at least 3 + 2
