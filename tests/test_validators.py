"""Property-based tests for input validation utilities.

These tests validate that the validation functions correctly reject invalid inputs
and provide appropriate error messages.
"""

import pytest
from hypothesis import given, settings, strategies as st, assume
from utils.validators import (
    validate_positive_number,
    validate_percentage,
    validate_pricing_params,
    ValidationLimits
)
from services.pricing_service import MODEL_CONFIGS


# Test data generators for invalid parameters
@st.composite
def invalid_pricing_params(draw, model):
    """Generate invalid pricing parameters for a given model.
    
    This generator creates parameter sets that violate at least one validation rule.
    """
    config = MODEL_CONFIGS[model]
    required_params = config['parameters']
    
    params = {}
    
    # Strategy: pick one parameter to make invalid, keep others valid
    invalid_param = draw(st.sampled_from(required_params))
    
    for param_name in required_params:
        if param_name == 'option_type':
            if param_name == invalid_param:
                # Invalid option type
                params[param_name] = draw(st.sampled_from(['call', 'put', 'x', 'invalid', '']))
            else:
                params[param_name] = 'c'
        
        elif param_name in ['x', 'fs', 'f1', 'f2']:
            # Strike or spot price
            if param_name == invalid_param:
                # Generate out-of-bounds values - use simpler approach
                params[param_name] = draw(st.sampled_from([
                    -100.0,  # Negative
                    0.0,     # Zero
                    0.001,   # Below minimum
                    ValidationLimits.MAX_FS * 1.5  # Too large
                ]))
            else:
                params[param_name] = 100.0
        
        elif param_name == 't':
            # Time to expiration
            if param_name == invalid_param:
                params[param_name] = draw(st.sampled_from([
                    -1.0,  # Negative
                    0.0,   # Zero
                    0.0001,  # Below minimum
                    ValidationLimits.MAX_T * 1.5  # Too large
                ]))
            else:
                params[param_name] = 1.0
        
        elif param_name in ['r', 'q', 'rf']:
            # Interest rates
            if param_name == invalid_param:
                params[param_name] = draw(st.sampled_from([
                    ValidationLimits.MIN_r - 0.5,  # Too low
                    ValidationLimits.MAX_r + 0.5   # Too high
                ]))
            else:
                params[param_name] = 0.05
        
        elif param_name in ['v', 'v1', 'v2']:
            # Volatility
            if param_name == invalid_param:
                params[param_name] = draw(st.sampled_from([
                    -0.5,  # Negative
                    0.0,   # Zero
                    0.001,  # Below minimum
                    ValidationLimits.MAX_V + 0.5  # Too high
                ]))
            else:
                params[param_name] = 0.2
        
        elif param_name == 't_a':
            # Averaging time for Asian options
            # Need to ensure t is set first
            if 't' not in params:
                params['t'] = 1.0
            t_value = params['t']
            
            if param_name == invalid_param:
                params[param_name] = draw(st.sampled_from([
                    -1.0,  # Negative
                    t_value + 1.0  # Greater than t
                ]))
            else:
                # Valid t_a: between 0 and t
                params[param_name] = t_value * 0.5
        
        elif param_name == 'corr':
            # Correlation
            if param_name == invalid_param:
                params[param_name] = draw(st.sampled_from([
                    -1.5,  # Too low
                    1.5    # Too high
                ]))
            else:
                params[param_name] = 0.5
    
    return params


# Feature: quant-trading-system, Property 3: Invalid parameter rejection
@given(model=st.sampled_from(list(MODEL_CONFIGS.keys())), data=st.data())
@settings(max_examples=100)
def test_invalid_parameter_rejection(model, data):
    """Property 3: Invalid parameter rejection.
    
    For any set of invalid pricing parameters, the validation function should
    return false and provide specific error messages for each invalid field.
    
    Validates: Requirements 1.4, 9.2
    """
    # Generate invalid parameters for this model
    params = data.draw(invalid_pricing_params(model))
    
    # Validate the parameters
    is_valid, error_messages = validate_pricing_params(params, model)
    
    # The validation should fail
    assert not is_valid, \
        f"Validation should have failed for invalid params: {params}"
    
    # There should be at least one error message
    assert len(error_messages) > 0, \
        f"Expected error messages for invalid params: {params}"
    
    # Each error message should be a non-empty string
    for error_msg in error_messages:
        assert isinstance(error_msg, str), \
            f"Error message should be a string, got {type(error_msg)}"
        assert len(error_msg) > 0, \
            "Error message should not be empty"


# Additional unit tests for specific validation functions
def test_validate_positive_number_with_negative():
    """Test that validate_positive_number rejects negative values."""
    is_valid, error_msg = validate_positive_number(-5.0, "test_param")
    assert not is_valid
    assert "positive" in error_msg.lower()


def test_validate_positive_number_with_zero():
    """Test that validate_positive_number rejects zero."""
    is_valid, error_msg = validate_positive_number(0.0, "test_param")
    assert not is_valid
    assert "positive" in error_msg.lower()


def test_validate_positive_number_with_positive():
    """Test that validate_positive_number accepts positive values."""
    is_valid, error_msg = validate_positive_number(5.0, "test_param")
    assert is_valid
    assert error_msg == ""


def test_validate_percentage_below_range():
    """Test that validate_percentage rejects values below -100%."""
    is_valid, error_msg = validate_percentage(-1.5, "test_param")
    assert not is_valid
    assert "between" in error_msg.lower()


def test_validate_percentage_above_range():
    """Test that validate_percentage rejects values above 200%."""
    is_valid, error_msg = validate_percentage(2.5, "test_param")
    assert not is_valid
    assert "between" in error_msg.lower()


def test_validate_percentage_within_range():
    """Test that validate_percentage accepts values within range."""
    is_valid, error_msg = validate_percentage(0.05, "test_param")
    assert is_valid
    assert error_msg == ""


def test_validate_pricing_params_missing_parameter():
    """Test that validate_pricing_params detects missing parameters."""
    # Black-Scholes requires: option_type, fs, x, t, r, v
    params = {
        'option_type': 'c',
        'fs': 100.0,
        'x': 100.0,
        't': 1.0,
        'r': 0.05
        # Missing 'v'
    }
    
    is_valid, error_messages = validate_pricing_params(params, 'black_scholes')
    assert not is_valid
    assert any('v' in msg.lower() for msg in error_messages)


def test_validate_pricing_params_invalid_option_type():
    """Test that validate_pricing_params rejects invalid option types."""
    params = {
        'option_type': 'call',  # Should be 'c' or 'p'
        'fs': 100.0,
        'x': 100.0,
        't': 1.0,
        'r': 0.05,
        'v': 0.2
    }
    
    is_valid, error_messages = validate_pricing_params(params, 'black_scholes')
    assert not is_valid
    assert any('option_type' in msg.lower() for msg in error_messages)


def test_validate_pricing_params_all_valid():
    """Test that validate_pricing_params accepts all valid parameters."""
    params = {
        'option_type': 'c',
        'fs': 100.0,
        'x': 100.0,
        't': 1.0,
        'r': 0.05,
        'v': 0.2
    }
    
    is_valid, error_messages = validate_pricing_params(params, 'black_scholes')
    assert is_valid
    assert len(error_messages) == 0


def test_validate_pricing_params_unknown_model():
    """Test that validate_pricing_params rejects unknown models."""
    params = {
        'option_type': 'c',
        'fs': 100.0,
        'x': 100.0,
        't': 1.0,
        'r': 0.05,
        'v': 0.2
    }
    
    is_valid, error_messages = validate_pricing_params(params, 'unknown_model')
    assert not is_valid
    assert any('unknown' in msg.lower() for msg in error_messages)
