"""Property-based tests for formatting utilities.

These tests validate that formatting functions produce output with the correct
precision and format.
"""

import pytest
import re
from hypothesis import given, settings, strategies as st
from utils.formatters import format_price, format_greek, format_percentage


# Feature: quant-trading-system, Property 15: Number formatting precision
@given(value=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_number_formatting_precision(value):
    """Property 15: Number formatting precision.
    
    For any displayed pricing result, price values should be formatted to exactly
    4 decimal places and Greek values should be formatted to exactly 4 decimal places.
    
    Validates: Requirements 7.4
    """
    # Format the value as a price
    formatted_price = format_price(value)
    
    # Format the value as a Greek
    formatted_greek = format_greek(value)
    
    # Both should be strings
    assert isinstance(formatted_price, str), \
        f"format_price should return a string, got {type(formatted_price)}"
    assert isinstance(formatted_greek, str), \
        f"format_greek should return a string, got {type(formatted_greek)}"
    
    # Check that price has exactly 4 decimal places
    # Pattern: optional minus, digits, decimal point, exactly 4 digits
    price_pattern = r'^-?\d+\.\d{4}$'
    assert re.match(price_pattern, formatted_price), \
        f"Price '{formatted_price}' should have exactly 4 decimal places"
    
    # Check that Greek has exactly 4 decimal places
    greek_pattern = r'^-?\d+\.\d{4}$'
    assert re.match(greek_pattern, formatted_greek), \
        f"Greek '{formatted_greek}' should have exactly 4 decimal places"
    
    # Verify that parsing the formatted string gives a value close to the original
    # (within rounding tolerance - 0.0001 accounts for rounding to 4 decimal places)
    parsed_price = float(formatted_price)
    assert abs(parsed_price - value) < 0.0001, \
        f"Formatted price {formatted_price} should round to approximately {value}"
    
    parsed_greek = float(formatted_greek)
    assert abs(parsed_greek - value) < 0.0001, \
        f"Formatted Greek {formatted_greek} should round to approximately {value}"


# Additional unit tests for specific formatting scenarios
def test_format_price_positive():
    """Test format_price with a positive value."""
    result = format_price(123.456789)
    assert result == "123.4568"


def test_format_price_negative():
    """Test format_price with a negative value."""
    result = format_price(-123.456789)
    assert result == "-123.4568"


def test_format_price_small():
    """Test format_price with a small value."""
    result = format_price(0.1)
    assert result == "0.1000"


def test_format_price_zero():
    """Test format_price with zero."""
    result = format_price(0.0)
    assert result == "0.0000"


def test_format_greek_positive():
    """Test format_greek with a positive value."""
    result = format_greek(0.567891)
    assert result == "0.5679"


def test_format_greek_negative():
    """Test format_greek with a negative value."""
    result = format_greek(-0.1234567)
    assert result == "-0.1235"


def test_format_greek_zero():
    """Test format_greek with zero."""
    result = format_greek(0.0)
    assert result == "0.0000"


def test_format_percentage_positive():
    """Test format_percentage with a positive decimal value."""
    result = format_percentage(0.05)
    assert result == "5.00%"


def test_format_percentage_large():
    """Test format_percentage with a value greater than 1."""
    result = format_percentage(1.5)
    assert result == "150.00%"


def test_format_percentage_negative():
    """Test format_percentage with a negative value."""
    result = format_percentage(-0.1)
    assert result == "-10.00%"


def test_format_percentage_zero():
    """Test format_percentage with zero."""
    result = format_percentage(0.0)
    assert result == "0.00%"


def test_format_percentage_precision():
    """Test format_percentage rounds to 2 decimal places."""
    result = format_percentage(0.12345)
    assert result == "12.35%"
