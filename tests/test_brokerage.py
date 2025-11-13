"""Tests for brokerage connectors."""

import pytest
from datetime import datetime
import os

from src.brokers import (
    SimulatedBrokerageAdapter,
    BrokerageFactory,
    CredentialManager,
    BrokerageCredentials,
    AuthenticationError,
    OrderError,
    BrokerageError
)
from src.models.order import Order, OrderAction, OrderType, OrderStatusEnum
from src.models.config import BrokerageConfig, TradingMode


class TestSimulatedBrokerageAdapter:
    """Tests for SimulatedBrokerageAdapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = SimulatedBrokerageAdapter(initial_capital=50000.0)
        
        assert adapter.mode == "simulation"
        assert adapter.initial_capital == 50000.0
        assert adapter.cash_balance == 50000.0
        assert len(adapter.positions) == 0
        assert adapter.connection_status.is_authenticated == False
    
    def test_authentication(self):
        """Test authentication process."""
        adapter = SimulatedBrokerageAdapter()
        
        result = adapter.authenticate({})
        
        assert result == True
        assert adapter.connection_status.is_authenticated == True
        assert adapter.connection_status.is_connected == True

    def test_submit_order_without_authentication(self):
        """Test that submitting order without authentication raises error."""
        adapter = SimulatedBrokerageAdapter()
        
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        with pytest.raises(AuthenticationError):
            adapter.submit_order(order)
    
    def test_submit_market_order_buy(self):
        """Test submitting a market buy order."""
        adapter = SimulatedBrokerageAdapter(initial_capital=10000.0)
        adapter.authenticate({})
        
        # Set market price
        adapter.update_market_price("AAPL", 150.0)
        
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        order_id = adapter.submit_order(order)
        
        assert order_id == "TEST-001"
        assert order.status == OrderStatusEnum.FILLED
        assert order.filled_price == 150.0
        assert order.filled_quantity == 10
        assert adapter.cash_balance == 10000.0 - (150.0 * 10)
        assert "AAPL" in adapter.positions
        assert adapter.positions["AAPL"].quantity == 10

    def test_submit_market_order_sell(self):
        """Test submitting a market sell order."""
        adapter = SimulatedBrokerageAdapter(initial_capital=10000.0)
        adapter.authenticate({})
        
        # Set market price and create initial position
        adapter.update_market_price("AAPL", 150.0)
        buy_order = Order(
            order_id="BUY-001",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        adapter.submit_order(buy_order)
        
        # Now sell
        adapter.update_market_price("AAPL", 160.0)
        sell_order = Order(
            order_id="SELL-001",
            symbol="AAPL",
            action=OrderAction.SELL,
            quantity=5,
            order_type=OrderType.MARKET
        )
        
        order_id = adapter.submit_order(sell_order)
        
        assert order_id == "SELL-001"
        assert sell_order.status == OrderStatusEnum.FILLED
        assert sell_order.filled_price == 160.0
        assert adapter.positions["AAPL"].quantity == 5
    
    def test_insufficient_funds(self):
        """Test order rejection due to insufficient funds."""
        adapter = SimulatedBrokerageAdapter(initial_capital=1000.0)
        adapter.authenticate({})
        
        adapter.update_market_price("AAPL", 150.0)
        
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,  # Would cost $15,000
            order_type=OrderType.MARKET
        )
        
        with pytest.raises(OrderError, match="Insufficient funds"):
            adapter.submit_order(order)

    def test_insufficient_shares(self):
        """Test order rejection due to insufficient shares."""
        adapter = SimulatedBrokerageAdapter()
        adapter.authenticate({})
        
        adapter.update_market_price("AAPL", 150.0)
        
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            action=OrderAction.SELL,
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        with pytest.raises(OrderError, match="Insufficient shares"):
            adapter.submit_order(order)
    
    def test_get_order_status(self):
        """Test retrieving order status."""
        adapter = SimulatedBrokerageAdapter()
        adapter.authenticate({})
        
        adapter.update_market_price("AAPL", 150.0)
        
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        adapter.submit_order(order)
        
        status = adapter.get_order_status("TEST-001")
        
        assert status.order_id == "TEST-001"
        assert status.status == "FILLED"
        assert status.filled_quantity == 10
        assert status.average_fill_price == 150.0
    
    def test_get_account_info(self):
        """Test retrieving account information."""
        adapter = SimulatedBrokerageAdapter(initial_capital=10000.0)
        adapter.authenticate({})
        
        account_info = adapter.get_account_info()
        
        assert account_info.cash_balance == 10000.0
        assert account_info.portfolio_value == 10000.0
        assert account_info.buying_power == 10000.0

    def test_get_positions(self):
        """Test retrieving positions."""
        adapter = SimulatedBrokerageAdapter()
        adapter.authenticate({})
        
        # Create some positions
        adapter.update_market_price("AAPL", 150.0)
        adapter.update_market_price("GOOGL", 2800.0)
        
        order1 = Order(
            order_id="TEST-001",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        adapter.submit_order(order1)
        
        order2 = Order(
            order_id="TEST-002",
            symbol="GOOGL",
            action=OrderAction.BUY,
            quantity=5,
            order_type=OrderType.MARKET
        )
        adapter.submit_order(order2)
        
        positions = adapter.get_positions()
        
        assert len(positions) == 2
        symbols = [pos.symbol for pos in positions]
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
    
    def test_update_market_prices(self):
        """Test updating market prices."""
        adapter = SimulatedBrokerageAdapter()
        adapter.authenticate({})
        
        prices = {
            "AAPL": 150.0,
            "GOOGL": 2800.0,
            "MSFT": 300.0
        }
        
        adapter.update_market_prices(prices)
        
        assert adapter.current_prices["AAPL"] == 150.0
        assert adapter.current_prices["GOOGL"] == 2800.0
        assert adapter.current_prices["MSFT"] == 300.0


class TestCredentialManager:
    """Tests for CredentialManager."""
    
    def test_load_from_dict(self):
        """Test loading credentials from dictionary."""
        creds_dict = {
            'api_key': 'test_key_123',
            'api_secret': 'test_secret_456',
            'account_id': 'ACC-789'
        }
        
        credentials = CredentialManager.load_from_dict(creds_dict)
        
        assert credentials.api_key == 'test_key_123'
        assert credentials.api_secret == 'test_secret_456'
        assert credentials.account_id == 'ACC-789'
    
    def test_load_from_dict_missing_key(self):
        """Test error when api_key is missing."""
        creds_dict = {
            'api_secret': 'test_secret_456'
        }
        
        with pytest.raises(AuthenticationError, match="Missing credential: api_key"):
            CredentialManager.load_from_dict(creds_dict)
    
    def test_load_from_dict_missing_secret(self):
        """Test error when api_secret is missing."""
        creds_dict = {
            'api_key': 'test_key_123'
        }
        
        with pytest.raises(AuthenticationError, match="Missing credential: api_secret"):
            CredentialManager.load_from_dict(creds_dict)
    
    def test_mask_credential(self):
        """Test credential masking for safe logging."""
        credential = "my_secret_api_key_12345"
        masked = CredentialManager.mask_credential(credential, visible_chars=4)
        
        assert masked.endswith("2345")
        assert masked.startswith("*")
        assert len(masked) == len(credential)


class TestBrokerageFactory:
    """Tests for BrokerageFactory."""
    
    def test_create_simulated_adapter(self):
        """Test creating simulated adapter."""
        config = BrokerageConfig(provider="simulated")
        
        adapter = BrokerageFactory.create_adapter(
            config=config,
            trading_mode=TradingMode.SIMULATION,
            initial_capital=50000.0
        )
        
        assert isinstance(adapter, SimulatedBrokerageAdapter)
        assert adapter.initial_capital == 50000.0
    
    def test_create_adapter_unknown_provider(self):
        """Test error when provider is unknown."""
        config = BrokerageConfig(
            provider="unknown_broker",
            api_key="test_key",
            api_secret="test_secret"
        )
        
        with pytest.raises(BrokerageError, match="Unknown brokerage provider"):
            BrokerageFactory.create_adapter(
                config=config,
                trading_mode=TradingMode.SIMULATION
            )
    
    def test_validate_live_mode_with_simulated_broker(self):
        """Test that live mode with simulated broker is rejected."""
        config = BrokerageConfig(provider="simulated")
        
        with pytest.raises(BrokerageError, match="Cannot use simulated brokerage in LIVE trading mode"):
            BrokerageFactory.create_adapter(
                config=config,
                trading_mode=TradingMode.LIVE
            )
    
    def test_get_available_providers(self):
        """Test getting list of available providers."""
        providers = BrokerageFactory.get_available_providers()
        
        assert "simulated" in providers
        assert "simulation" in providers
        assert "paper" in providers
    
    def test_validate_mode_switch_simulation_to_live(self):
        """Test validation when switching from simulation to live."""
        config = BrokerageConfig(provider="simulated")
        
        is_valid, message = BrokerageFactory.validate_mode_switch(
            current_mode=TradingMode.SIMULATION,
            new_mode=TradingMode.LIVE,
            config=config
        )
        
        assert is_valid == False
        assert "Cannot switch to LIVE mode" in message
    
    def test_validate_mode_switch_live_to_simulation(self):
        """Test validation when switching from live to simulation."""
        config = BrokerageConfig(provider="public", api_key="key", api_secret="secret")
        
        is_valid, message = BrokerageFactory.validate_mode_switch(
            current_mode=TradingMode.LIVE,
            new_mode=TradingMode.SIMULATION,
            config=config
        )
        
        assert is_valid == True
        assert "SIMULATION mode" in message
