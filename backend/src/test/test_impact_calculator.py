import pytest
import re # For test_parse_parameter_value_distributions if needed, already in impact_calculator

# Adjust import path based on how pytest will discover your modules.

import sys
import os
# Forcefully add the module's directory to sys.path
# This assumes the test file is in backend/src/test/
# and the module to test is in backend/src/main/python/
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main', 'python'))
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from impact_calculator import ( # Now try direct import
    parse_parameter_value,
    calculate_single_impact,
    calculate_driver_impacts,
    EFFECT_FUNCTION_REGISTRY # To verify functions are registered
)

# --- Test parse_parameter_value ---

def test_parse_parameter_value_numeric():
    assert parse_parameter_value(10, {}) == 10.0
    assert parse_parameter_value(10.5, {}) == 10.5
    assert parse_parameter_value("20", {}) == 20.0
    assert parse_parameter_value("-5.5", {}) == -5.5

def test_parse_parameter_value_context_lookup():
    context = {"company.pe": 15.0, "macro.rate": 0.05, "nested": {"value": 25}}
    assert parse_parameter_value("company.pe", context) == 15.0
    assert parse_parameter_value("macro.rate", context) == 0.05
    assert parse_parameter_value("nested.value", context) == 25.0
    assert parse_parameter_value("non.existent.key", context) is None
    assert parse_parameter_value("company.name", {"company": {"name": "TestCo"}}) is None # Non-numeric

def test_parse_parameter_value_distributions():
    assert parse_parameter_value("Normal(10, 2)", {}) == 10.0  # Mean
    assert parse_parameter_value("Normal( 10.5, 2.1 )", {}) == 10.5
    assert parse_parameter_value("Uniform(5, 15)", {}) == 10.0 # Mean
    assert parse_parameter_value("Uniform( 5.0, 15.5 )", {}) == 10.25
    assert parse_parameter_value("Normal(abc, 2)", {}) is None # Invalid mean
    assert parse_parameter_value("Uniform(5, xyz)", {}) is None # Invalid max

def test_parse_parameter_value_invalid():
    assert parse_parameter_value(None, {}) is None
    assert parse_parameter_value([1, 2], {}) is None # List not supported
    assert parse_parameter_value("NotADistribution(1,2)", {}) is None # Unrecognized string


# --- Test Effect Functions (indirectly via calculate_single_impact) ---

@pytest.fixture
def sample_context():
    return {
        "company.current_revenue": 1000.0,
        "company.current_pe": 20.0,
        "company.debt_level": 500.0,
        "company.product_launch_status": "successful", # For conditions
        "company.industryId": "IND_TECH"
    }

def test_calculate_single_impact_percentage_change(sample_context):
    impact_def = {
        "target": "company.current_revenue", # This key in context holds the current value
        "effect_function": "percentage_change",
        "parameters": {"percentage": 0.10}, # +10%
        "probability_of_occurrence": 0.8,
        "time_horizon": "1Y"
    }
    result = calculate_single_impact(impact_def, "DRV_TEST", sample_context)
    assert result is not None
    assert result["calculated_impact_value"] == pytest.approx(1100.0)
    assert result["target_variable"] == "company.current_revenue"
    assert result["probability_of_occurrence"] == 0.8

def test_calculate_single_impact_additive_change(sample_context):
    impact_def = {
        "target": "company.debt_level",
        "effect_function": "additive_change",
        "parameters": {"change_amount": -50.0}, # Reduce by 50
        "probability_of_occurrence": 0.7
    }
    result = calculate_single_impact(impact_def, "DRV_TEST", sample_context)
    assert result is not None
    assert result["calculated_impact_value"] == pytest.approx(450.0)

def test_calculate_single_impact_direct_value(sample_context):
    impact_def = {
        "target": "company.new_segment_value", # Target doesn't need to exist in context for direct_value
        "effect_function": "direct_value",
        "parameters": {"value": 123.45},
        "probability_of_occurrence": 0.9
    }
    result = calculate_single_impact(impact_def, "DRV_TEST", sample_context)
    assert result is not None
    assert result["calculated_impact_value"] == 123.45

def test_calculate_single_impact_missing_current_value_for_function(sample_context):
    impact_def = {
        "target": "company.non_existent_metric", # This won't be in context
        "effect_function": "percentage_change", # Needs current_value
        "parameters": {"percentage": 0.10}
    }
    result = calculate_single_impact(impact_def, "DRV_TEST", sample_context)
    assert result is None # Because current_value for 'company.non_existent_metric' is missing

def test_calculate_single_impact_missing_parameter_for_function(sample_context):
    impact_def = {
        "target": "company.current_revenue",
        "effect_function": "percentage_change", # Needs 'percentage' parameter
        "parameters": {} # Missing 'percentage'
    }
    result = calculate_single_impact(impact_def, "DRV_TEST", sample_context)
    assert result is None

def test_calculate_single_impact_unknown_function(sample_context):
    impact_def = {
        "target": "company.current_revenue",
        "effect_function": "unknown_function_name",
        "parameters": {}
    }
    result = calculate_single_impact(impact_def, "DRV_TEST", sample_context)
    assert result is None


# --- Test calculate_driver_impacts (overall driver processing) ---

@pytest.fixture
def sample_driver_def_for_calc():
    return {
        "id": "DRV_COMPLEX",
        "name": "Complex Test Driver",
        "impact_model": {
            "first_order_impacts": [
                { # Impact 1: Will be calculated
                    "target": "company.current_revenue",
                    "effect_function": "percentage_change",
                    "parameters": {"percentage": 0.10}, # +10%
                    "probability_of_occurrence": 0.8,
                    "time_horizon": "1Y",
                    "conditions": ["company.industryId == 'IND_TECH'"]
                },
                { # Impact 2: Condition not met
                    "target": "company.current_pe",
                    "effect_function": "additive_change",
                    "parameters": {"change_amount": -2.0},
                    "probability_of_occurrence": 0.7,
                    "conditions": ["company.product_launch_status == 'failed'"] # context has 'successful'
                },
                { # Impact 3: Will be calculated (no conditions)
                    "target": "company.debt_level",
                    "effect_function": "additive_change",
                    "parameters": {"change_amount": "Normal(50,10)"}, # Becomes 50.0
                    "probability_of_occurrence": 0.9
                },
                { # Impact 4: Function needs current_value from context that doesn't exist
                    "target": "company.non_existent_metric",
                    "effect_function": "percentage_change",
                    "parameters": {"percentage": 0.05}
                }
            ]
        }
    }

def test_calculate_driver_impacts_with_conditions_and_parsing(sample_driver_def_for_calc, sample_context):
    results = calculate_driver_impacts(sample_driver_def_for_calc, sample_context)
    assert len(results) == 2 # Expecting Impact 1 and Impact 3

    # Check Impact 1 (Revenue Increase)
    impact1_found = any(r["target_variable"] == "company.current_revenue" and \
                        r["calculated_impact_value"] == pytest.approx(1100.0) for r in results)
    assert impact1_found, "Expected revenue impact not found or incorrect."

    # Check Impact 3 (Debt Level Increase)
    impact3_found = any(r["target_variable"] == "company.debt_level" and \
                        r["calculated_impact_value"] == pytest.approx(550.0) for r in results) # 500 + Normal(50,10).mean()
    assert impact3_found, "Expected debt level impact not found or incorrect."

def test_calculate_driver_impacts_no_valid_impacts(sample_context):
    driver_def_no_valid = {
        "id": "DRV_NO_IMPACTS",
        "impact_model": { "first_order_impacts": [
            {"target": "company.A", "effect_function": "non_existent_func", "parameters": {}},
            {"target": "company.B", "effect_function": "percentage_change", "parameters": {}, "conditions": ["company.product_launch_status == 'never_true'"]}
        ]}
    }
    results = calculate_driver_impacts(driver_def_no_valid, sample_context)
    assert len(results) == 0
