import json
import os
import sys
import pytest

# Adjust path to find scripts/report_generation.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from report_generation import generate_integration_report

def test_integration_log_structure():
    """Test that the integration log file exists and has the expected structure."""
    log_path = os.path.join('data', 'v23_integration_log.json')
    assert os.path.exists(log_path), "Integration log file missing."

    with open(log_path, 'r') as f:
        data = json.load(f)

    assert 'v23_integration_log' in data
    log = data['v23_integration_log']

    assert 'meta' in log
    assert 'delta_analysis' in log
    assert 'revised_strategic_synthesis' in log

    assert log['meta']['system_id'] == 'Adam-v23.5-Apex'
    assert log['revised_strategic_synthesis']['status'] == 'Calibration Complete'

def test_generate_integration_report():
    """Test the report generation function with mock data."""
    mock_data = {
        'v23_integration_log': {
            'meta': {
                'system_id': 'Test-System',
                'timestamp': '2026-01-01',
                'source_model': 'Test-Source'
            },
            'delta_analysis': {
                'valuation_divergence': {
                    'variance': '-5%',
                    'v23_intrinsic': 100,
                    'v30_intrinsic': 95,
                    'driver': 'Test Driver'
                },
                'risk_vector_update': {
                    'previous_assessment': 'Low',
                    'new_intelligence': 'High',
                    'implication': 'Panic'
                }
            },
            'revised_strategic_synthesis': {
                'status': 'Testing',
                'outlook': 'Neutral',
                'adjusted_price_levels': {
                    'sovereign_floor': '10',
                    'mean_reversion_zone': '20-30',
                    'speculative_ceiling': '40'
                },
                'final_directive': {
                    'action': 'Wait',
                    'rationale': 'Testing rationale',
                    'monitor': 'Test monitor'
                }
            }
        }
    }

    report = generate_integration_report(mock_data)

    assert "Test-System" in report
    assert "Test-Source" in report
    assert "$100.00" in report
    assert "$95.00" in report
    assert "Test Driver" in report
    assert "Testing" in report
    assert "Neutral" in report
    assert "Wait" in report
    assert "Testing rationale" in report
