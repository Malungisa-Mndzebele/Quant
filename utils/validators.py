"""Input validation utilities for option pricing parameters.

This module provides validation functions to ensure all user inputs meet the
requirements defined by the optlib.gbs._GBS_Limits class and are appropriate
for the selected pricing model.
"""

from typing import Dict, List, Tuple
from services.pricing_service import MODEL_CONFIGS


# Validation limits (from optlib.gbs._GBS_Limits)
class ValidationLimits:
    """Validation limits for option pricing parameters."""
    MAX32 = 2147483248.0
    
    MIN_T = 1.0 / 1000.0  # Minimum time to expiration
    MIN_X = 0.01          # Minimum strike price
    MIN_FS = 0.01         # Minimum forward/spot price
    MIN_V = 0.005         # Minimum volatility (0.5%)
    MIN_TA = 0            # Minimum averaging time for Asian options
    MIN_b = -1            # Minimum cost of carry
    MIN_r = -1            # Minimum interest rate
    
    MAX_T = 100           # Maximum time to expiration
    MAX_X = MAX32         # Maximum strike price
    MAX_FS = MAX32        # Maximum forward/spot price
    MAX_b = 1             # Maximum cost of carry
    MAX_r = 2             # Maximum interest rate (200%)
    MAX_V = 2             # Maximum volatility (200%)


def validate_positive_number(value: float, name: str) -> Tuple[bool, str]:
    """Validate that a number is positive.
    
    Args:
        value: The numeric value to validate
        name: The parameter name for error messages
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
        If valid, error_message is empty string
    """
    try:
        float_value = float(value)
        if float_value <= 0:
            return False, f"{name} must be positive (greater than 0)"
        return True, ""
    except (TypeError, ValueError):
        return False, f"{name} must be a valid number"


def validate_percentage(value: float, name: str) -> Tuple[bool, str]:
    """Validate that a value is a valid percentage for rates/volatility.
    
    This validates that the value is within acceptable ranges for interest rates
    and volatility parameters (-100% to 200%).
    
    Args:
        value: The percentage value to validate (as decimal, e.g., 0.05 for 5%)
        name: The parameter name for error messages
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
        If valid, error_message is empty string
    """
    try:
        float_value = float(value)
        if float_value < -1.0 or float_value > 2.0:
            return False, f"{name} must be between -100% and 200% (-1.0 to 2.0 as decimal)"
        return True, ""
    except (TypeError, ValueError):
        return False, f"{name} must be a valid number"


def validate_pricing_params(params: Dict, model: str) -> Tuple[bool, List[str]]:
    """Validate pricing parameters for a given model.
    
    This function checks that all required parameters are present and within
    acceptable ranges for the specified pricing model.
    
    Args:
        params: Dictionary of parameter names to values
        model: Name of the pricing model
        
    Returns:
        Tuple of (is_valid: bool, error_messages: List[str])
        If valid, error_messages is an empty list
    """
    errors = []
    
    # Check if model exists
    if model not in MODEL_CONFIGS:
        errors.append(f"Unknown pricing model: {model}")
        return False, errors
    
    config = MODEL_CONFIGS[model]
    required_params = config['parameters']
    
    # Check for missing parameters
    for param in required_params:
        if param not in params:
            errors.append(f"Missing required parameter: {param}")
    
    # If there are missing parameters, return early
    if errors:
        return False, errors
    
    # Validate option_type
    if 'option_type' in params:
        option_type = params['option_type']
        if option_type not in ['c', 'p']:
            errors.append("option_type must be 'c' (call) or 'p' (put)")
    
    # Validate strike price (x)
    if 'x' in params:
        try:
            x = float(params['x'])
            if x < ValidationLimits.MIN_X or x > ValidationLimits.MAX_X:
                errors.append(
                    f"Strike price (x) must be between {ValidationLimits.MIN_X} "
                    f"and {ValidationLimits.MAX_X}"
                )
        except (TypeError, ValueError):
            errors.append("Strike price (x) must be a valid number")
    
    # Validate forward/spot price (fs)
    if 'fs' in params:
        try:
            fs = float(params['fs'])
            if fs < ValidationLimits.MIN_FS or fs > ValidationLimits.MAX_FS:
                errors.append(
                    f"Underlying price (fs) must be between {ValidationLimits.MIN_FS} "
                    f"and {ValidationLimits.MAX_FS}"
                )
        except (TypeError, ValueError):
            errors.append("Underlying price (fs) must be a valid number")
    
    # Validate time to expiration (t)
    if 't' in params:
        try:
            t = float(params['t'])
            if t < ValidationLimits.MIN_T or t > ValidationLimits.MAX_T:
                errors.append(
                    f"Time to expiration (t) must be between {ValidationLimits.MIN_T} "
                    f"and {ValidationLimits.MAX_T} years"
                )
        except (TypeError, ValueError):
            errors.append("Time to expiration (t) must be a valid number")
    
    # Validate risk-free rate (r)
    if 'r' in params:
        try:
            r = float(params['r'])
            if r < ValidationLimits.MIN_r or r > ValidationLimits.MAX_r:
                errors.append(
                    f"Risk-free rate (r) must be between {ValidationLimits.MIN_r} "
                    f"and {ValidationLimits.MAX_r}"
                )
        except (TypeError, ValueError):
            errors.append("Risk-free rate (r) must be a valid number")
    
    # Validate volatility (v)
    if 'v' in params:
        try:
            v = float(params['v'])
            if v < ValidationLimits.MIN_V or v > ValidationLimits.MAX_V:
                errors.append(
                    f"Volatility (v) must be between {ValidationLimits.MIN_V} "
                    f"and {ValidationLimits.MAX_V}"
                )
        except (TypeError, ValueError):
            errors.append("Volatility (v) must be a valid number")
    
    # Validate dividend yield (q) for Merton and American models
    if 'q' in params:
        try:
            q = float(params['q'])
            if q < ValidationLimits.MIN_r or q > ValidationLimits.MAX_r:
                errors.append(
                    f"Dividend yield (q) must be between {ValidationLimits.MIN_r} "
                    f"and {ValidationLimits.MAX_r}"
                )
        except (TypeError, ValueError):
            errors.append("Dividend yield (q) must be a valid number")
    
    # Validate foreign risk-free rate (rf) for Garman-Kohlhagen
    if 'rf' in params:
        try:
            rf = float(params['rf'])
            if rf < ValidationLimits.MIN_r or rf > ValidationLimits.MAX_r:
                errors.append(
                    f"Foreign risk-free rate (rf) must be between {ValidationLimits.MIN_r} "
                    f"and {ValidationLimits.MAX_r}"
                )
        except (TypeError, ValueError):
            errors.append("Foreign risk-free rate (rf) must be a valid number")
    
    # Validate averaging time (t_a) for Asian options
    if 't_a' in params:
        try:
            t_a = float(params['t_a'])
            t = float(params.get('t', 1.0))
            if t_a < ValidationLimits.MIN_TA or t_a > t:
                errors.append(
                    f"Averaging time (t_a) must be between {ValidationLimits.MIN_TA} "
                    f"and time to expiration (t={t})"
                )
        except (TypeError, ValueError):
            errors.append("Averaging time (t_a) must be a valid number")
    
    # Validate Kirk's approximation parameters
    if 'f1' in params:
        try:
            f1 = float(params['f1'])
            if f1 < ValidationLimits.MIN_FS or f1 > ValidationLimits.MAX_FS:
                errors.append(
                    f"First futures price (f1) must be between {ValidationLimits.MIN_FS} "
                    f"and {ValidationLimits.MAX_FS}"
                )
        except (TypeError, ValueError):
            errors.append("First futures price (f1) must be a valid number")
    
    if 'f2' in params:
        try:
            f2 = float(params['f2'])
            if f2 < ValidationLimits.MIN_FS or f2 > ValidationLimits.MAX_FS:
                errors.append(
                    f"Second futures price (f2) must be between {ValidationLimits.MIN_FS} "
                    f"and {ValidationLimits.MAX_FS}"
                )
        except (TypeError, ValueError):
            errors.append("Second futures price (f2) must be a valid number")
    
    if 'v1' in params:
        try:
            v1 = float(params['v1'])
            if v1 < ValidationLimits.MIN_V or v1 > ValidationLimits.MAX_V:
                errors.append(
                    f"First volatility (v1) must be between {ValidationLimits.MIN_V} "
                    f"and {ValidationLimits.MAX_V}"
                )
        except (TypeError, ValueError):
            errors.append("First volatility (v1) must be a valid number")
    
    if 'v2' in params:
        try:
            v2 = float(params['v2'])
            if v2 < ValidationLimits.MIN_V or v2 > ValidationLimits.MAX_V:
                errors.append(
                    f"Second volatility (v2) must be between {ValidationLimits.MIN_V} "
                    f"and {ValidationLimits.MAX_V}"
                )
        except (TypeError, ValueError):
            errors.append("Second volatility (v2) must be a valid number")
    
    if 'corr' in params:
        try:
            corr = float(params['corr'])
            if corr < -1.0 or corr > 1.0:
                errors.append("Correlation (corr) must be between -1.0 and 1.0")
        except (TypeError, ValueError):
            errors.append("Correlation (corr) must be a valid number")
    
    # Return validation result
    is_valid = len(errors) == 0
    return is_valid, errors
