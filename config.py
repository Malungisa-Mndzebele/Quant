"""Configuration module for the quantitative trading system."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
TDA_API_KEY = os.getenv('TDA_API_KEY', '')
USE_MOCK_DATA = os.getenv('USE_MOCK_DATA', 'True').lower() == 'true'

# Application Configuration
APP_TITLE = "Quantitative Trading System"
APP_ICON = "📈"
APP_LAYOUT = "wide"

# Pricing Models Configuration
PRICING_MODELS = {
    'black_scholes': {
        'name': 'black_scholes',
        'display_name': 'Black-Scholes',
        'description': 'Standard European option pricing model',
        'parameters': ['fs', 'x', 't', 'r', 'v'],
        'supports_implied_vol': True,
        'option_style': 'european'
    },
    'merton': {
        'name': 'merton',
        'display_name': 'Merton (Dividend Yield)',
        'description': 'Black-Scholes with continuous dividend yield',
        'parameters': ['fs', 'x', 't', 'r', 'q', 'v'],
        'supports_implied_vol': True,
        'option_style': 'european'
    },
    'black_76': {
        'name': 'black_76',
        'display_name': 'Black-76',
        'description': 'Model for options on futures',
        'parameters': ['fs', 'x', 't', 'r', 'v'],
        'supports_implied_vol': True,
        'option_style': 'european'
    },
    'garman_kohlhagen': {
        'name': 'garman_kohlhagen',
        'display_name': 'Garman-Kohlhagen',
        'description': 'Model for currency options',
        'parameters': ['fs', 'x', 't', 'r', 'rf', 'v'],
        'supports_implied_vol': True,
        'option_style': 'european'
    },
    'asian_76': {
        'name': 'asian_76',
        'display_name': 'Asian-76',
        'description': 'Model for Asian options on futures',
        'parameters': ['fs', 'x', 't', 'r', 'v', 'ta'],
        'supports_implied_vol': False,
        'option_style': 'asian'
    },
    'kirks_76': {
        'name': 'kirks_76',
        'display_name': "Kirk's Approximation",
        'description': 'Model for spread options',
        'parameters': ['f1', 'f2', 'x', 't', 'r', 'v1', 'v2', 'corr'],
        'supports_implied_vol': False,
        'option_style': 'spread'
    },
    'american': {
        'name': 'american',
        'display_name': 'American (Bjerksund-Stensland)',
        'description': 'American option pricing model',
        'parameters': ['fs', 'x', 't', 'r', 'q', 'v'],
        'supports_implied_vol': True,
        'option_style': 'american'
    },
    'american_76': {
        'name': 'american_76',
        'display_name': 'American-76',
        'description': 'American options on futures',
        'parameters': ['fs', 'x', 't', 'r', 'v'],
        'supports_implied_vol': True,
        'option_style': 'american'
    }
}

# Parameter Display Names and Descriptions
PARAMETER_INFO = {
    'fs': {
        'display_name': 'Underlying Price',
        'description': 'Current price of the underlying asset',
        'default': 100.0,
        'min': 0.01,
        'max': 10000.0
    },
    'x': {
        'display_name': 'Strike Price',
        'description': 'Exercise price of the option',
        'default': 100.0,
        'min': 0.01,
        'max': 10000.0
    },
    't': {
        'display_name': 'Time to Expiration (years)',
        'description': 'Time until option expiration in years',
        'default': 1.0,
        'min': 0.001,
        'max': 100.0
    },
    'r': {
        'display_name': 'Risk-Free Rate',
        'description': 'Annual risk-free interest rate (as decimal)',
        'default': 0.05,
        'min': -1.0,
        'max': 2.0
    },
    'q': {
        'display_name': 'Dividend Yield',
        'description': 'Continuous dividend yield (as decimal)',
        'default': 0.02,
        'min': 0.0,
        'max': 1.0
    },
    'v': {
        'display_name': 'Volatility',
        'description': 'Annual volatility (as decimal)',
        'default': 0.20,
        'min': 0.005,
        'max': 2.0
    },
    'rf': {
        'display_name': 'Foreign Risk-Free Rate',
        'description': 'Foreign currency risk-free rate (as decimal)',
        'default': 0.03,
        'min': -1.0,
        'max': 2.0
    },
    'ta': {
        'display_name': 'Time Averaged',
        'description': 'Time already averaged for Asian options (years)',
        'default': 0.5,
        'min': 0.0,
        'max': 100.0
    },
    'f1': {
        'display_name': 'First Futures Price',
        'description': 'Price of first underlying futures contract',
        'default': 100.0,
        'min': 0.01,
        'max': 10000.0
    },
    'f2': {
        'display_name': 'Second Futures Price',
        'description': 'Price of second underlying futures contract',
        'default': 95.0,
        'min': 0.01,
        'max': 10000.0
    },
    'v1': {
        'display_name': 'First Volatility',
        'description': 'Volatility of first underlying (as decimal)',
        'default': 0.20,
        'min': 0.005,
        'max': 2.0
    },
    'v2': {
        'display_name': 'Second Volatility',
        'description': 'Volatility of second underlying (as decimal)',
        'default': 0.25,
        'min': 0.005,
        'max': 2.0
    },
    'corr': {
        'display_name': 'Correlation',
        'description': 'Correlation between the two underlyings',
        'default': 0.5,
        'min': -1.0,
        'max': 1.0
    }
}

# Caching Configuration
CACHE_TTL_SECONDS = 300  # 5 minutes for API data

# Formatting Configuration
PRICE_DECIMALS = 4
GREEK_DECIMALS = 4
PERCENTAGE_DECIMALS = 2

# Chart Configuration
CHART_HEIGHT = 500
CHART_TEMPLATE = "plotly_white"

# Export Configuration
EXPORT_FORMATS = ['CSV', 'JSON', 'PDF']
CHART_EXPORT_FORMATS = ['PNG', 'SVG']
