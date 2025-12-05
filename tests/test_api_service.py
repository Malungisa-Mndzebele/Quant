"""Unit tests for API service error handling."""

import pytest
from unittest.mock import patch, Mock
import requests
from services.api_service import (
    fetch_option_chain,
    fetch_historical_data,
    validate_api_key,
    NetworkError,
    InvalidAPIKeyError,
    RateLimitError,
    SymbolNotFoundError,
    APIError
)
from optlib.api import InputError


@pytest.fixture(autouse=True)
def disable_mock_data():
    """Disable mock data for all tests in this file."""
    with patch('services.api_service.config.USE_MOCK_DATA', False):
        yield


class TestFetchOptionChainErrorHandling:
    """Test error handling for fetch_option_chain function."""
    
    def test_network_connection_error(self):
        """Test handling of network connection failures."""
        with patch('services.api_service.OptionChain.get') as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            with pytest.raises(NetworkError) as exc_info:
                fetch_option_chain("AAPL", apikey="test_key")
            
            assert "Failed to connect" in str(exc_info.value)
    
    def test_network_timeout_error(self):
        """Test handling of request timeouts."""
        with patch('services.api_service.OptionChain.get') as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
            
            with pytest.raises(NetworkError) as exc_info:
                fetch_option_chain("AAPL", apikey="test_key")
            
            assert "timed out" in str(exc_info.value)
    
    def test_invalid_api_key_401(self):
        """Test handling of invalid API key (401 error)."""
        with patch('services.api_service.OptionChain.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            http_error = requests.exceptions.HTTPError()
            http_error.response = mock_response
            mock_get.side_effect = http_error
            
            with pytest.raises(InvalidAPIKeyError) as exc_info:
                fetch_option_chain("AAPL", apikey="invalid_key")
            
            assert "Invalid or expired API key" in str(exc_info.value)
    
    def test_rate_limit_error_429(self):
        """Test handling of rate limit errors (429 error)."""
        with patch('services.api_service.OptionChain.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.text = "Too Many Requests"
            http_error = requests.exceptions.HTTPError()
            http_error.response = mock_response
            mock_get.side_effect = http_error
            
            with pytest.raises(RateLimitError) as exc_info:
                fetch_option_chain("AAPL", apikey="test_key")
            
            assert "rate limit exceeded" in str(exc_info.value)
    
    def test_symbol_not_found_404(self):
        """Test handling of symbol not found (404 error)."""
        with patch('services.api_service.OptionChain.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.text = "Not Found"
            http_error = requests.exceptions.HTTPError()
            http_error.response = mock_response
            mock_get.side_effect = http_error
            
            with pytest.raises(SymbolNotFoundError) as exc_info:
                fetch_option_chain("INVALID", apikey="test_key")
            
            assert "not found" in str(exc_info.value)
    
    def test_missing_api_key_input_error(self):
        """Test handling of missing API key."""
        with patch('services.api_service.OptionChain.get') as mock_get:
            mock_get.side_effect = InputError("TDA_API_KEY not found in environment")
            
            with pytest.raises(InvalidAPIKeyError) as exc_info:
                fetch_option_chain("AAPL")
            
            assert "API key not found" in str(exc_info.value)
    
    def test_other_http_error(self):
        """Test handling of other HTTP errors."""
        with patch('services.api_service.OptionChain.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            http_error = requests.exceptions.HTTPError()
            http_error.response = mock_response
            mock_get.side_effect = http_error
            
            with pytest.raises(APIError) as exc_info:
                fetch_option_chain("AAPL", apikey="test_key")
            
            assert "500" in str(exc_info.value)


class TestFetchHistoricalDataErrorHandling:
    """Test error handling for fetch_historical_data function."""
    
    def test_network_connection_error(self):
        """Test handling of network connection failures."""
        with patch('services.api_service.Pricehistory.get') as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            with pytest.raises(NetworkError) as exc_info:
                fetch_historical_data("AAPL", apikey="test_key")
            
            assert "Failed to connect" in str(exc_info.value)
    
    def test_network_timeout_error(self):
        """Test handling of request timeouts."""
        with patch('services.api_service.Pricehistory.get') as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
            
            with pytest.raises(NetworkError) as exc_info:
                fetch_historical_data("AAPL", apikey="test_key")
            
            assert "timed out" in str(exc_info.value)
    
    def test_invalid_api_key_401(self):
        """Test handling of invalid API key (401 error)."""
        with patch('services.api_service.Pricehistory.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            http_error = requests.exceptions.HTTPError()
            http_error.response = mock_response
            mock_get.side_effect = http_error
            
            with pytest.raises(InvalidAPIKeyError) as exc_info:
                fetch_historical_data("AAPL", apikey="invalid_key")
            
            assert "Invalid or expired API key" in str(exc_info.value)
    
    def test_rate_limit_error_429(self):
        """Test handling of rate limit errors (429 error)."""
        with patch('services.api_service.Pricehistory.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.text = "Too Many Requests"
            http_error = requests.exceptions.HTTPError()
            http_error.response = mock_response
            mock_get.side_effect = http_error
            
            with pytest.raises(RateLimitError) as exc_info:
                fetch_historical_data("AAPL", apikey="test_key")
            
            assert "rate limit exceeded" in str(exc_info.value)
    
    def test_symbol_not_found_404(self):
        """Test handling of symbol not found (404 error)."""
        with patch('services.api_service.Pricehistory.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.text = "Not Found"
            http_error = requests.exceptions.HTTPError()
            http_error.response = mock_response
            mock_get.side_effect = http_error
            
            with pytest.raises(SymbolNotFoundError) as exc_info:
                fetch_historical_data("INVALID", apikey="test_key")
            
            assert "not found" in str(exc_info.value)
    
    def test_missing_api_key_input_error(self):
        """Test handling of missing API key."""
        with patch('services.api_service.Pricehistory.get') as mock_get:
            mock_get.side_effect = InputError("TDA_API_KEY not found in environment")
            
            with pytest.raises(InvalidAPIKeyError) as exc_info:
                fetch_historical_data("AAPL")
            
            assert "API key not found" in str(exc_info.value)


class TestValidateAPIKey:
    """Test API key validation function."""
    
    def test_valid_api_key(self):
        """Test validation with a valid API key."""
        with patch('services.api_service.OptionChain.get') as mock_get:
            # Mock successful response
            mock_get.return_value = Mock()
            
            result = validate_api_key("valid_key")
            
            assert result is True
            mock_get.assert_called_once()
    
    def test_invalid_api_key(self):
        """Test validation with an invalid API key."""
        with patch('services.api_service.OptionChain.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            http_error = requests.exceptions.HTTPError()
            http_error.response = mock_response
            mock_get.side_effect = http_error
            
            result = validate_api_key("invalid_key")
            
            assert result is False
    
    def test_missing_api_key(self):
        """Test validation with missing API key."""
        with patch('services.api_service.OptionChain.get') as mock_get:
            mock_get.side_effect = InputError("TDA_API_KEY not found in environment")
            
            result = validate_api_key("")
            
            assert result is False
    
    def test_network_error_during_validation(self):
        """Test validation when network error occurs."""
        with patch('services.api_service.OptionChain.get') as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            result = validate_api_key("test_key")
            
            assert result is False
    
    def test_rate_limit_during_validation(self):
        """Test validation when rate limit is hit."""
        with patch('services.api_service.OptionChain.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.text = "Too Many Requests"
            http_error = requests.exceptions.HTTPError()
            http_error.response = mock_response
            mock_get.side_effect = http_error
            
            result = validate_api_key("test_key")
            
            assert result is False
