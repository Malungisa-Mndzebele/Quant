"""Property-based tests for export functionality.

These tests validate correctness properties for CSV and chart export functions
using the Hypothesis library for property-based testing.
"""

import pytest
import pandas as pd
import io
from hypothesis import given, settings, strategies as st
from services.pricing_service import PricingResult, MODEL_CONFIGS
from app import export_pricing_to_csv


# Test data generators
@st.composite
def valid_pricing_result(draw):
    """Generate a valid PricingResult object."""
    model = draw(st.sampled_from(list(MODEL_CONFIGS.keys())))
    
    # Generate realistic option pricing values
    value = draw(st.floats(min_value=0.01, max_value=1000.0))
    delta = draw(st.floats(min_value=-1.0, max_value=1.0))
    gamma = draw(st.floats(min_value=0.0, max_value=1.0))
    theta = draw(st.floats(min_value=-100.0, max_value=0.0))
    vega = draw(st.floats(min_value=0.0, max_value=100.0))
    rho = draw(st.floats(min_value=-100.0, max_value=100.0))
    
    # Generate parameters based on model
    params = {
        'option_type': draw(st.sampled_from(['c', 'p'])),
        'fs': draw(st.floats(min_value=10.0, max_value=500.0)),
        'x': draw(st.floats(min_value=10.0, max_value=500.0)),
        't': draw(st.floats(min_value=0.01, max_value=5.0)),
        'r': draw(st.floats(min_value=-0.2, max_value=0.2)),
        'v': draw(st.floats(min_value=0.05, max_value=2.0))
    }
    
    # Add model-specific parameters
    if model in ['merton', 'american']:
        params['q'] = draw(st.floats(min_value=-0.2, max_value=0.2))
    elif model == 'garman_kohlhagen':
        params['rf'] = draw(st.floats(min_value=-0.2, max_value=0.2))
    elif model == 'asian_76':
        params['t_a'] = draw(st.floats(min_value=0.0, max_value=params['t'] * 0.99))
    elif model == 'kirks_76':
        params['f1'] = params.pop('fs')
        params['f2'] = draw(st.floats(min_value=10.0, max_value=500.0))
        params['v1'] = params.pop('v')
        params['v2'] = draw(st.floats(min_value=0.05, max_value=2.0))
        params['corr'] = draw(st.floats(min_value=-0.95, max_value=0.95))
    
    result = PricingResult(
        value=value,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
        model=model,
        parameters=params
    )
    
    return result, params, model


# Feature: quant-trading-system, Property 19: CSV export completeness
@given(data=st.data())
@settings(max_examples=100, deadline=None)
def test_csv_export_completeness(data):
    """Property 19: CSV export completeness.
    
    For any pricing result exported to CSV, the file should contain all calculated
    values (price and all Greeks) with column headers.
    
    Validates: Requirements 10.2
    """
    # Generate a valid pricing result
    result, params, model = data.draw(valid_pricing_result())
    
    # Export to CSV
    csv_string = export_pricing_to_csv(result, params, model)
    
    # Parse the CSV
    df = pd.read_csv(io.StringIO(csv_string))
    
    # Verify that all required columns are present
    required_columns = [
        'Model',
        'Option_Type',
        'Option_Price',
        'Delta',
        'Gamma',
        'Theta',
        'Vega',
        'Rho'
    ]
    
    for col in required_columns:
        assert col in df.columns, \
            f"Required column '{col}' missing from CSV export. Found columns: {list(df.columns)}"
    
    # Verify that the CSV has at least one row of data
    assert len(df) > 0, "CSV export should contain at least one row of data"
    
    # Verify that all calculated values are present in the first row
    # Use approximate equality to account for CSV serialization/deserialization precision loss
    import math
    
    def approx_equal(a, b, rel_tol=1e-9, abs_tol=1e-9):
        """Check if two floats are approximately equal."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    assert approx_equal(df['Option_Price'].iloc[0], result.value), \
        f"Option price mismatch: CSV has {df['Option_Price'].iloc[0]}, expected {result.value}"
    assert approx_equal(df['Delta'].iloc[0], result.delta), \
        f"Delta mismatch: CSV has {df['Delta'].iloc[0]}, expected {result.delta}"
    assert approx_equal(df['Gamma'].iloc[0], result.gamma), \
        f"Gamma mismatch: CSV has {df['Gamma'].iloc[0]}, expected {result.gamma}"
    assert approx_equal(df['Theta'].iloc[0], result.theta), \
        f"Theta mismatch: CSV has {df['Theta'].iloc[0]}, expected {result.theta}"
    assert approx_equal(df['Vega'].iloc[0], result.vega), \
        f"Vega mismatch: CSV has {df['Vega'].iloc[0]}, expected {result.vega}"
    assert approx_equal(df['Rho'].iloc[0], result.rho), \
        f"Rho mismatch: CSV has {df['Rho'].iloc[0]}, expected {result.rho}"
    
    # Verify that the model name is present
    assert df['Model'].iloc[0] == MODEL_CONFIGS[model]['display_name'], \
        f"Model name mismatch: CSV has {df['Model'].iloc[0]}, expected {MODEL_CONFIGS[model]['display_name']}"
    
    # Verify that option type is present
    expected_option_type = 'Call' if params['option_type'] == 'c' else 'Put'
    assert df['Option_Type'].iloc[0] == expected_option_type, \
        f"Option type mismatch: CSV has {df['Option_Type'].iloc[0]}, expected {expected_option_type}"
    
    # Verify that CSV has column headers (first line should be headers)
    assert csv_string.startswith('Model,'), \
        "CSV should start with column headers"
