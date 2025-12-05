"""Pricing service module for option pricing calculations.

This module wraps the optlib.gbs pricing functions and provides a unified interface
for calculating option prices, Greeks, and implied volatility across all supported models.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import optlib.gbs as gbs


@dataclass
class PricingResult:
    """Result of an option pricing calculation."""
    value: float      # Option price
    delta: float      # First derivative w.r.t. underlying price
    gamma: float      # Second derivative w.r.t. underlying price
    theta: float      # First derivative w.r.t. time
    vega: float       # First derivative w.r.t. volatility
    rho: float        # First derivative w.r.t. interest rate
    model: str        # Model used for calculation
    parameters: dict  # Input parameters used


# Model configuration mapping
MODEL_CONFIGS = {
    'black_scholes': {
        'display_name': 'Black-Scholes',
        'parameters': ['option_type', 'fs', 'x', 't', 'r', 'v'],
        'function': gbs.black_scholes,
        'description': 'Stock options (no dividend yield)',
        'supports_implied_vol': True,
        'implied_vol_function': gbs.euro_implied_vol,
        'is_american': False
    },
    'merton': {
        'display_name': 'Merton',
        'parameters': ['option_type', 'fs', 'x', 't', 'r', 'q', 'v'],
        'function': gbs.merton,
        'description': 'Stock options with continuous dividend yield',
        'supports_implied_vol': True,
        'implied_vol_function': gbs.euro_implied_vol,
        'is_american': False
    },
    'black_76': {
        'display_name': 'Black-76',
        'parameters': ['option_type', 'fs', 'x', 't', 'r', 'v'],
        'function': gbs.black_76,
        'description': 'Commodity options',
        'supports_implied_vol': True,
        'implied_vol_function': gbs.euro_implied_vol_76,
        'is_american': False
    },
    'garman_kohlhagen': {
        'display_name': 'Garman-Kohlhagen',
        'parameters': ['option_type', 'fs', 'x', 't', 'r', 'rf', 'v'],
        'function': gbs.garman_kohlhagen,
        'description': 'Foreign exchange (FX) options',
        'supports_implied_vol': True,
        'implied_vol_function': gbs.euro_implied_vol,
        'is_american': False
    },
    'asian_76': {
        'display_name': 'Asian-76',
        'parameters': ['option_type', 'fs', 'x', 't', 't_a', 'r', 'v'],
        'function': gbs.asian_76,
        'description': 'Average price options on commodities',
        'supports_implied_vol': False,
        'implied_vol_function': None,
        'is_american': False
    },
    'kirks_76': {
        'display_name': "Kirk's Approximation",
        'parameters': ['option_type', 'f1', 'f2', 'x', 't', 'r', 'v1', 'v2', 'corr'],
        'function': gbs.kirks_76,
        'description': 'Spread options',
        'supports_implied_vol': False,
        'implied_vol_function': None,
        'is_american': False
    },
    'american': {
        'display_name': 'American',
        'parameters': ['option_type', 'fs', 'x', 't', 'r', 'q', 'v'],
        'function': gbs.american,
        'description': 'American-style options',
        'supports_implied_vol': True,
        'implied_vol_function': gbs.amer_implied_vol,
        'is_american': True
    },
    'american_76': {
        'display_name': 'American-76',
        'parameters': ['option_type', 'fs', 'x', 't', 'r', 'v'],
        'function': gbs.american_76,
        'description': 'American-style commodity options',
        'supports_implied_vol': True,
        'implied_vol_function': gbs.amer_implied_vol_76,
        'is_american': True
    }
}


def get_model_parameters(model: str) -> List[str]:
    """Return list of required parameters for a given model.
    
    Args:
        model: Name of the pricing model
        
    Returns:
        List of parameter names required by the model
        
    Raises:
        ValueError: If model is not recognized
    """
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model}. Available models: {list(MODEL_CONFIGS.keys())}")
    
    return MODEL_CONFIGS[model]['parameters']


def calculate_option_price(model: str, params: Dict) -> PricingResult:
    """Calculate option price using specified model and parameters.
    
    Args:
        model: One of the supported pricing models
        params: Dictionary containing pricing parameters
        
    Returns:
        PricingResult with value and all Greeks
        
    Raises:
        ValueError: If model is not recognized
        gbs.GBS_InputError: If parameters are invalid
    """
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model}. Available models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model]
    pricing_function = config['function']
    
    # Extract parameters in the correct order
    args = []
    for param_name in config['parameters']:
        if param_name not in params:
            raise ValueError(f"Missing required parameter: {param_name}")
        args.append(params[param_name])
    
    # Call the pricing function
    value, delta, gamma, theta, vega, rho = pricing_function(*args)
    
    return PricingResult(
        value=value,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
        model=model,
        parameters=params.copy()
    )


def calculate_implied_volatility(model: str, params: Dict, observed_price: float) -> float:
    """Calculate implied volatility from observed market price.
    
    Args:
        model: Pricing model to use
        params: Dictionary containing pricing parameters (without volatility)
        observed_price: Observed market price of the option
        
    Returns:
        Implied volatility as a float
        
    Raises:
        ValueError: If model doesn't support implied volatility or is not recognized
        gbs.GBS_CalculationError: If implied volatility calculation fails to converge
    """
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model}. Available models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model]
    
    if not config['supports_implied_vol']:
        raise ValueError(f"Model {model} does not support implied volatility calculation")
    
    implied_vol_function = config['implied_vol_function']
    
    # Extract parameters for implied vol calculation (excluding volatility)
    # The implied vol functions have different signatures than pricing functions
    # They need: option_type, fs, x, t, r, q/rf (if applicable), cp (observed price)
    
    # Build args based on the model
    if model in ['black_scholes', 'american']:
        # These use q parameter
        args = [
            params['option_type'],
            params['fs'],
            params['x'],
            params['t'],
            params['r'],
            params.get('q', 0),  # q for merton/american, 0 for black_scholes
            observed_price
        ]
    elif model in ['black_76', 'american_76']:
        # These don't have q parameter
        args = [
            params['option_type'],
            params['fs'],
            params['x'],
            params['t'],
            params['r'],
            observed_price
        ]
    elif model == 'merton':
        # Merton uses q
        args = [
            params['option_type'],
            params['fs'],
            params['x'],
            params['t'],
            params['r'],
            params['q'],
            observed_price
        ]
    elif model == 'garman_kohlhagen':
        # Garman-Kohlhagen uses rf (foreign rate) as q
        args = [
            params['option_type'],
            params['fs'],
            params['x'],
            params['t'],
            params['r'],
            params['rf'],
            observed_price
        ]
    else:
        raise ValueError(f"Implied volatility not implemented for model: {model}")
    
    return implied_vol_function(*args)
