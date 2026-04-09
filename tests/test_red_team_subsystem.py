import pytest
import asyncio
from typing import Dict, Any

from core.security.red_team.quantum_scanner import QuantumScanner
from core.security.red_team.response_engine import RealTimeResponseEngine
from core.security.red_team.sandbox_env import SandboxEnvironment
from core.agents.red_team_agent import RedTeamAgent
from core.agents.agent_base import AgentInput
from core.system.red_teaming_framework import RedTeamingFramework

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

def test_zero_day_mitigation():
    engine = RealTimeResponseEngine()
    threat = {"type": "ZERO_DAY", "severity_score": 5.0}
    analysis = engine.analyze_threat(threat)
    assert analysis["urgency"] == "IMMEDIATE"
    assert analysis["recommended_action"] == "ISOLATE_ZERO_DAY_MICRO_SEGMENTATION"
    assert analysis["calculated_severity"] == 10.0

def test_quantum_attack_mitigation():
    engine = RealTimeResponseEngine()
    threat = {"type": "QUANTUM_DECRYPTION_ATTACK", "severity_score": 2.0}
    analysis = engine.analyze_threat(threat)
    assert analysis["urgency"] == "IMMEDIATE"
    assert analysis["recommended_action"] == "CYCLE_TO_LATTICE_BASED_ENCRYPTION"
    assert analysis["calculated_severity"] == 10.0

def test_adversarial_ai_mitigations():
    engine = RealTimeResponseEngine()
    threat = {"type": "ADVERSARIAL_AI_MARKET_STRUCTURE", "severity_score": 3.0}
    analysis = engine.analyze_threat(threat)
    assert analysis["urgency"] == "IMMEDIATE"
    assert analysis["recommended_action"] == "ENGAGE_AI_COUNTERMEASURES_FOR_ADVERSARIAL_AI_MARKET_STRUCTURE"
    assert analysis["calculated_severity"] == 10.0

# --- Tests for SandboxEnvironment ---

def test_sandbox_detonation():
    sandbox = SandboxEnvironment()
    threat_1 = {"type": "SQL_INJECTION", "severity_score": 5.0}
    threat_2 = {"type": "ZERO_DAY", "severity_score": 9.5}

    sandbox.detonate(threat_1)
    sandbox.detonate(threat_2)

    blast_radius = sandbox.analyze_impact()
    assert blast_radius == 12  # 5.0 -> +2, 9.5 -> +10

    containment = sandbox.contain()
    assert containment["status"] == "SUCCESS"
    assert sandbox.blast_radius == 0
    assert len(sandbox.active_threats) == 0

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

# --- Tests for RedTeamingFramework ---

@pytest.mark.asyncio
async def test_framework_run():
    agent = RedTeamAgent(config={"name": "FrameworkTestAgent"})
    framework = RedTeamingFramework(red_team_agent=agent, system="Test Production Environment")

    input_data = AgentInput(
        query="Test Cybersecurity Target",
        context={
            "encryption_algorithms": ["RSA-1024", "AES-256"],
            "cyber_threats": [
                {"type": "ZERO_DAY", "severity_score": 9.5}
            ]
        }
    )

    report = await framework.run(input_data)

    assert "=== Red Team Exercise Report ===" in report
    assert "Target System: Test Production Environment" in report
    assert len(framework.logs) == 1
    assert framework.logs[0]["input"] == input_data
