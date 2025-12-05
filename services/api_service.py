"""API service module for TDAmeritrade integration.

This module provides wrapper functions for optlib.api and optlib.instruments,
adding error handling and a consistent interface for the web application.
"""

import logging
from typing import Optional, Dict, Any
import requests
import random
from datetime import datetime, timedelta
import config
from optlib.instruments import OptionChain, Pricehistory
from optlib.api import InputError

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API-related errors."""
    pass


class NetworkError(APIError):
    """Exception raised for network-related failures."""
    pass


class InvalidAPIKeyError(APIError):
    """Exception raised when API key is invalid or expired."""
    pass


class RateLimitError(APIError):
    """Exception raised when API rate limit is exceeded."""
    pass



class SymbolNotFoundError(APIError):
    """Exception raised when requested symbol is not found."""
    pass


def _generate_mock_option_chain(symbol: str) -> OptionChain:
    """Generate a mock OptionChain for testing/demo purposes."""
    underlying_price = 100.0 + random.uniform(-5, 5)
    
    # Generate some expiration dates
    exps = []
    for i in range(1, 4):
        date = datetime.now() + timedelta(days=30*i)
        date_str = date.strftime("%Y-%m-%d")
        exps.append(date_str)
        
    # Generate strikes
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    
    call_map = {}
    put_map = {}
    
    for exp in exps:
        call_map[exp] = {}
        put_map[exp] = {}
        
        for strike in strikes:
            # Simple mock pricing logic
            time_to_exp = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days / 365.0
            vol = 0.2
            
            # Mock option data structure matching TDA API
            option_data = [{
                "putCall": "CALL",
                "symbol": f"{symbol}_{exp}_C_{strike}",
                "description": f"{symbol} {exp} {strike} Call",
                "bid": max(0.01, (underlying_price - strike) + 2), # Very rough approx
                "ask": max(0.01, (underlying_price - strike) + 2.5),
                "last": max(0.01, (underlying_price - strike) + 2.25),
                "mark": max(0.01, (underlying_price - strike) + 2.25),
                "totalVolume": random.randint(100, 10000),
                "openInterest": random.randint(100, 5000),
                "volatility": vol,
                "delta": 0.5,
                "gamma": 0.05,
                "theta": -0.1,
                "vega": 0.2,
                "rho": 0.05,
                "strikePrice": strike,
                "expirationDate": (datetime.strptime(exp, "%Y-%m-%d") - datetime(1970, 1, 1)).total_seconds() * 1000,
                "daysToExpiration": time_to_exp * 365,
                "expirationType": "S",
                "multiplier": 100,
                "settlementType": "P",
                "deliverableNote": "",
                "isIndexOption": None,
                "percentChange": random.uniform(-5, 5),
                "markChange": random.uniform(-1, 1),
                "markPercentChange": random.uniform(-5, 5),
                "nonStandard": False,
                "inTheMoney": underlying_price > strike,
                "mini": False
            }]
            
            call_map[exp][str(strike)] = option_data
            
            # Put data
            put_option_data = [{
                "putCall": "PUT",
                "symbol": f"{symbol}_{exp}_P_{strike}",
                "description": f"{symbol} {exp} {strike} Put",
                "bid": max(0.01, (strike - underlying_price) + 2),
                "ask": max(0.01, (strike - underlying_price) + 2.5),
                "last": max(0.01, (strike - underlying_price) + 2.25),
                "mark": max(0.01, (strike - underlying_price) + 2.25),
                "totalVolume": random.randint(100, 10000),
                "openInterest": random.randint(100, 5000),
                "volatility": vol,
                "delta": -0.5,
                "gamma": 0.05,
                "theta": -0.1,
                "vega": 0.2,
                "rho": -0.05,
                "strikePrice": strike,
                "expirationDate": (datetime.strptime(exp, "%Y-%m-%d") - datetime(1970, 1, 1)).total_seconds() * 1000,
                "daysToExpiration": time_to_exp * 365,
                "expirationType": "S",
                "multiplier": 100,
                "settlementType": "P",
                "deliverableNote": "",
                "isIndexOption": None,
                "percentChange": random.uniform(-5, 5),
                "markChange": random.uniform(-1, 1),
                "markPercentChange": random.uniform(-5, 5),
                "nonStandard": False,
                "inTheMoney": strike > underlying_price,
                "mini": False
            }]
            put_map[exp][str(strike)] = put_option_data

    return OptionChain(
        symbol=symbol,
        isDelayed=True,
        isIndex=False,
        interestRate=0.05,
        underlyingPrice=underlying_price,
        volatility=0.2,
        callExpDateMap=call_map,
        putExpDateMap=put_map
    )


def _generate_mock_price_history(symbol: str) -> Pricehistory:
    """Generate mock Pricehistory for testing/demo purposes."""
    candles = []
    price = 100.0
    
    # Generate 30 days of history
    for i in range(30):
        date = datetime.now() - timedelta(days=30-i)
        change = random.uniform(-2, 2)
        price += change
        
        candles.append({
            "open": price - random.uniform(0, 1),
            "high": price + random.uniform(0, 2),
            "low": price - random.uniform(0, 2),
            "close": price,
            "volume": random.randint(100000, 1000000),
            "datetime": int(date.timestamp() * 1000)
        })
        
    return Pricehistory(
        symbol=symbol,
        empty=False,
        candles=candles
    )


def fetch_option_chain(
    symbol: str,
    apikey: Optional[str] = None,
    **kwargs
) -> OptionChain:
    """Fetch option chain from TDAmeritrade API.
    
    Wrapper for optlib.instruments.OptionChain.get() with enhanced error handling.
    
    Args:
        symbol: Stock ticker symbol
        apikey: TDAmeritrade API key (optional, can use env variable)
        **kwargs: Additional parameters for API call (contract_type, strike_count, etc.)
        
    Returns:
        OptionChain object from optlib.instruments
        
    Raises:
        NetworkError: If network connection fails
        InvalidAPIKeyError: If API key is invalid or expired
        RateLimitError: If API rate limit is exceeded
        SymbolNotFoundError: If symbol is not found
        APIError: For other API-related errors
    """
    try:
        logger.info(f"Fetching option chain for symbol: {symbol}")
        
        if config.USE_MOCK_DATA:
            logger.info("Using mock data for option chain")
            return _generate_mock_option_chain(symbol)

        # Call the optlib wrapper
        chain = OptionChain.get(
            symbol=symbol,
            apikey=apikey,
            **kwargs
        )
        
        logger.info(f"Successfully fetched option chain for {symbol}")
        return chain
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Network connection error: {e}")
        raise NetworkError(
            "Failed to connect to TDAmeritrade API. Please check your internet connection."
        ) from e
        
    except requests.exceptions.Timeout as e:
        logger.error(f"Request timeout: {e}")
        raise NetworkError(
            "Request to TDAmeritrade API timed out. Please try again."
        ) from e
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e}")
        
        if e.response.status_code == 401:
            raise InvalidAPIKeyError(
                "Invalid or expired API key. Please check your TDA_API_KEY."
            ) from e
        elif e.response.status_code == 429:
            raise RateLimitError(
                "API rate limit exceeded. Please wait a moment and try again."
            ) from e
        elif e.response.status_code == 404:
            raise SymbolNotFoundError(
                f"Symbol '{symbol}' not found. Please check the symbol and try again."
            ) from e
        else:
            raise APIError(
                f"API request failed with status {e.response.status_code}: {e.response.text}"
            ) from e
            
    except InputError as e:
        logger.error(f"Input error: {e}")
        raise InvalidAPIKeyError(
            "API key not found. Please provide a valid TDA_API_KEY."
        ) from e
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise APIError(f"Invalid API response: {e}") from e
        
    except Exception as e:
        logger.error(f"Unexpected error fetching option chain: {e}")
        raise APIError(f"Unexpected error: {e}") from e


def fetch_historical_data(
    symbol: str,
    apikey: Optional[str] = None,
    **kwargs
) -> Pricehistory:
    """Fetch historical price data from TDAmeritrade API.
    
    Wrapper for optlib.instruments.Pricehistory.get() with enhanced error handling.
    
    Args:
        symbol: Stock ticker symbol
        apikey: TDAmeritrade API key (optional, can use env variable)
        **kwargs: Additional parameters for API call (period_type, frequency_type, etc.)
        
    Returns:
        Pricehistory object from optlib.instruments
        
    Raises:
        NetworkError: If network connection fails
        InvalidAPIKeyError: If API key is invalid or expired
        RateLimitError: If API rate limit is exceeded
        SymbolNotFoundError: If symbol is not found
        APIError: For other API-related errors
    """
    try:
        logger.info(f"Fetching historical data for symbol: {symbol}")
        
        if config.USE_MOCK_DATA:
            logger.info("Using mock data for historical prices")
            return _generate_mock_price_history(symbol)

        # Call the optlib wrapper
        history = Pricehistory.get(
            symbol=symbol,
            apikey=apikey,
            **kwargs
        )
        
        logger.info(f"Successfully fetched historical data for {symbol}")
        return history
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Network connection error: {e}")
        raise NetworkError(
            "Failed to connect to TDAmeritrade API. Please check your internet connection."
        ) from e
        
    except requests.exceptions.Timeout as e:
        logger.error(f"Request timeout: {e}")
        raise NetworkError(
            "Request to TDAmeritrade API timed out. Please try again."
        ) from e
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e}")
        
        if e.response.status_code == 401:
            raise InvalidAPIKeyError(
                "Invalid or expired API key. Please check your TDA_API_KEY."
            ) from e
        elif e.response.status_code == 429:
            raise RateLimitError(
                "API rate limit exceeded. Please wait a moment and try again."
            ) from e
        elif e.response.status_code == 404:
            raise SymbolNotFoundError(
                f"Symbol '{symbol}' not found. Please check the symbol and try again."
            ) from e
        else:
            raise APIError(
                f"API request failed with status {e.response.status_code}: {e.response.text}"
            ) from e
            
    except InputError as e:
        logger.error(f"Input error: {e}")
        raise InvalidAPIKeyError(
            "API key not found. Please provide a valid TDA_API_KEY."
        ) from e
        
    except Exception as e:
        logger.error(f"Unexpected error fetching historical data: {e}")
        raise APIError(f"Unexpected error: {e}") from e


def validate_api_key(api_key: str) -> bool:
    """Validate TDAmeritrade API key by making a test request.
    
    Makes a lightweight API call to verify the key is valid and has connectivity.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        logger.info("Validating API key")
        
        if config.USE_MOCK_DATA:
            logger.info("Mock data enabled, skipping API key validation")
            return True

        # Make a simple test request - fetch option chain for a known symbol
        # with minimal data to test connectivity
        OptionChain.get(
            symbol="SPY",
            apikey=api_key,
            strike_count=1  # Minimize data transfer
        )
        
        logger.info("API key validation successful")
        return True
        
    except (InvalidAPIKeyError, InputError):
        logger.warning("API key validation failed - invalid key")
        return False
        
    except (NetworkError, RateLimitError, APIError):
        logger.warning("API key validation failed - network or API error")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error during API key validation: {e}")
        return False
