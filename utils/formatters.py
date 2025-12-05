"""Formatting utilities for displaying option pricing results.

This module provides formatting functions to ensure consistent display of
prices, Greeks, and percentages throughout the application.
"""


def format_price(value: float) -> str:
    """Format price to 4 decimal places.
    
    Args:
        value: The price value to format
        
    Returns:
        Formatted string with exactly 4 decimal places
        
    Example:
        >>> format_price(123.456789)
        '123.4568'
        >>> format_price(0.1)
        '0.1000'
    """
    return f"{value:.4f}"


def format_greek(value: float) -> str:
    """Format Greek value to 4 decimal places.
    
    Greeks (Delta, Gamma, Theta, Vega, Rho) are formatted with 4 decimal
    places for consistent display.
    
    Args:
        value: The Greek value to format
        
    Returns:
        Formatted string with exactly 4 decimal places
        
    Example:
        >>> format_greek(0.567891)
        '0.5679'
        >>> format_greek(-0.1234567)
        '-0.1235'
    """
    return f"{value:.4f}"


def format_percentage(value: float) -> str:
    """Format percentage value for display.
    
    Converts decimal values to percentage format (e.g., 0.05 -> "5.00%").
    
    Args:
        value: The percentage value as a decimal (e.g., 0.05 for 5%)
        
    Returns:
        Formatted percentage string with 2 decimal places
        
    Example:
        >>> format_percentage(0.05)
        '5.00%'
        >>> format_percentage(0.12345)
        '12.35%'
        >>> format_percentage(1.5)
        '150.00%'
    """
    return f"{value * 100:.2f}%"
