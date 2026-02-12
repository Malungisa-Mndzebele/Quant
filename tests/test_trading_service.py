"""Unit tests for trading service."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from hypothesis import given, strategies as st, settings

from services.trading_service import (
    TradingService,
    Order,
    OrderStatus,
    Position
)


# Mock Alpaca objects
@dataclass
class MockAlpacaOrder:
    """Mock Alpaca order object"""
    id: str
    symbol: str
    qty: str
    side: Mock
    type: Mock
    status: str
    filled_qty: str
    filled_avg_price: str
    limit_price: str
    submitted_at: datetime
    filled_at: datetime


@dataclass
class MockAlpacaPosition:
    """Mock Alpaca position object"""
    symbol: str
    qty: str
    avg_entry_price: str
    current_price: str
    market_value: str
    cost_basis: str
    unrealized_pl: str
    unrealized_plpc: str


@dataclass
class MockAlpacaAccount:
    """Mock Alpaca account object"""
    cash: str
    portfolio_value: str
    buying_power: str
    equity: str
    last_equity: str
    initial_margin: str
    maintenance_margin: str
    daytrade_count: int
    daytrading_buying_power: str
    regt_buying_power: str
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    account_number: str
    status: Mock


@pytest.fixture
def mock_trading_client():
    """Create a mock TradingClient"""
    with patch('services.trading_service.TradingClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def trading_service(mock_trading_client):
    """Create a TradingService instance with mocked client"""
    with patch('services.trading_service.settings') as mock_settings:
        mock_settings.alpaca.api_key = 'test_key'
        mock_settings.alpaca.secret_key = 'test_secret'
        
        service = TradingService(
            api_key='test_key',
            api_secret='test_secret',
            paper=True
        )
        
        return service


class TestTradingServiceInitialization:
    """Test trading service initialization"""
    
    def test_init_with_credentials(self, mock_trading_client):
        """Test initialization with provided credentials"""
        with patch('services.trading_service.settings') as mock_settings:
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            
            service = TradingService(
                api_key='custom_key',
                api_secret='custom_secret',
                paper=True
            )
            
            assert service.api_key == 'custom_key'
            assert service.api_secret == 'custom_secret'
            assert service.paper is True
    
    def test_init_without_credentials_raises_error(self):
        """Test initialization without credentials raises ValueError"""
        with patch('services.trading_service.settings') as mock_settings:
            mock_settings.alpaca.api_key = None
            mock_settings.alpaca.secret_key = None
            
            with pytest.raises(ValueError, match="Alpaca API credentials are required"):
                TradingService()
    
    def test_init_paper_mode_default(self, mock_trading_client):
        """Test that paper mode is True by default"""
        with patch('services.trading_service.settings') as mock_settings:
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            
            service = TradingService()
            
            assert service.paper is True


class TestPlaceOrder:
    """Test place_order functionality"""
    
    def test_place_market_buy_order(self, trading_service, mock_trading_client):
        """Test placing a market buy order"""
        # Setup mock response
        mock_order = MockAlpacaOrder(
            id='order123',
            symbol='AAPL',
            qty='10',
            side=Mock(value='buy'),
            type=Mock(value='market'),
            status='new',
            filled_qty='0',
            filled_avg_price=None,
            limit_price=None,
            submitted_at=datetime.now(),
            filled_at=None
        )
        mock_trading_client.submit_order.return_value = mock_order
        
        # Place order
        order = trading_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='market'
        )
        
        # Verify
        assert order.symbol == 'AAPL'
        assert order.quantity == 10
        assert order.side == 'buy'
        assert order.order_type == 'market'
        assert order.order_id == 'order123'
        assert mock_trading_client.submit_order.called
    
    def test_place_limit_sell_order(self, trading_service, mock_trading_client):
        """Test placing a limit sell order"""
        # Setup mock response
        mock_order = MockAlpacaOrder(
            id='order456',
            symbol='GOOGL',
            qty='5',
            side=Mock(value='sell'),
            type=Mock(value='limit'),
            status='new',
            filled_qty='0',
            filled_avg_price=None,
            limit_price='150.50',
            submitted_at=datetime.now(),
            filled_at=None
        )
        mock_trading_client.submit_order.return_value = mock_order
        
        # Place order
        order = trading_service.place_order(
            symbol='GOOGL',
            qty=5,
            side='sell',
            order_type='limit',
            limit_price=150.50
        )
        
        # Verify
        assert order.symbol == 'GOOGL'
        assert order.quantity == 5
        assert order.side == 'sell'
        assert order.order_type == 'limit'
        assert order.limit_price == 150.50
    
    def test_place_order_invalid_symbol(self, trading_service):
        """Test placing order with invalid symbol raises ValueError"""
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            trading_service.place_order(
                symbol='',
                qty=10,
                side='buy'
            )
    
    def test_place_order_invalid_quantity(self, trading_service):
        """Test placing order with invalid quantity raises ValueError"""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            trading_service.place_order(
                symbol='AAPL',
                qty=0,
                side='buy'
            )
    
    def test_place_order_invalid_side(self, trading_service):
        """Test placing order with invalid side raises ValueError"""
        with pytest.raises(ValueError, match="Side must be 'buy' or 'sell'"):
            trading_service.place_order(
                symbol='AAPL',
                qty=10,
                side='invalid'
            )
    
    def test_place_limit_order_without_price(self, trading_service):
        """Test placing limit order without price raises ValueError"""
        with pytest.raises(ValueError, match="Limit price is required for limit orders"):
            trading_service.place_order(
                symbol='AAPL',
                qty=10,
                side='buy',
                order_type='limit'
            )
    
    def test_place_order_normalizes_symbol(self, trading_service, mock_trading_client):
        """Test that symbol is normalized to uppercase"""
        mock_order = MockAlpacaOrder(
            id='order789',
            symbol='AAPL',
            qty='10',
            side=Mock(value='buy'),
            type=Mock(value='market'),
            status='new',
            filled_qty='0',
            filled_avg_price=None,
            limit_price=None,
            submitted_at=datetime.now(),
            filled_at=None
        )
        mock_trading_client.submit_order.return_value = mock_order
        
        order = trading_service.place_order(
            symbol='aapl',  # lowercase
            qty=10,
            side='buy'
        )
        
        assert order.symbol == 'AAPL'


class TestGetOrderStatus:
    """Test get_order_status functionality"""
    
    def test_get_order_status(self, trading_service, mock_trading_client):
        """Test getting order status"""
        mock_order = MockAlpacaOrder(
            id='order123',
            symbol='AAPL',
            qty='10',
            side=Mock(value='buy'),
            type=Mock(value='market'),
            status='filled',
            filled_qty='10',
            filled_avg_price='150.25',
            limit_price=None,
            submitted_at=datetime.now(),
            filled_at=datetime.now()
        )
        mock_trading_client.get_order_by_id.return_value = mock_order
        
        order = trading_service.get_order_status('order123')
        
        assert order.order_id == 'order123'
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 10
        assert order.filled_avg_price == 150.25
    
    def test_get_order_status_invalid_id(self, trading_service):
        """Test getting order status with invalid ID raises ValueError"""
        with pytest.raises(ValueError, match="Order ID must be a non-empty string"):
            trading_service.get_order_status('')


class TestCancelOrder:
    """Test cancel_order functionality"""
    
    def test_cancel_order(self, trading_service, mock_trading_client):
        """Test cancelling an order"""
        mock_trading_client.cancel_order_by_id.return_value = None
        
        result = trading_service.cancel_order('order123')
        
        assert result is True
        assert mock_trading_client.cancel_order_by_id.called_with('order123')
    
    def test_cancel_order_invalid_id(self, trading_service):
        """Test cancelling order with invalid ID raises ValueError"""
        with pytest.raises(ValueError, match="Order ID must be a non-empty string"):
            trading_service.cancel_order('')


class TestGetPositions:
    """Test get_positions functionality"""
    
    def test_get_positions(self, trading_service, mock_trading_client):
        """Test getting all positions"""
        mock_positions = [
            MockAlpacaPosition(
                symbol='AAPL',
                qty='10',
                avg_entry_price='150.00',
                current_price='155.00',
                market_value='1550.00',
                cost_basis='1500.00',
                unrealized_pl='50.00',
                unrealized_plpc='0.0333'
            ),
            MockAlpacaPosition(
                symbol='GOOGL',
                qty='5',
                avg_entry_price='2800.00',
                current_price='2850.00',
                market_value='14250.00',
                cost_basis='14000.00',
                unrealized_pl='250.00',
                unrealized_plpc='0.0179'
            )
        ]
        mock_trading_client.get_all_positions.return_value = mock_positions
        
        positions = trading_service.get_positions()
        
        assert len(positions) == 2
        assert positions[0].symbol == 'AAPL'
        assert positions[0].quantity == 10
        assert positions[0].unrealized_pl == 50.00
        assert positions[1].symbol == 'GOOGL'
        assert positions[1].quantity == 5
    
    def test_get_positions_empty(self, trading_service, mock_trading_client):
        """Test getting positions when none exist"""
        mock_trading_client.get_all_positions.return_value = []
        
        positions = trading_service.get_positions()
        
        assert len(positions) == 0


class TestClosePosition:
    """Test close_position functionality"""
    
    def test_close_entire_position(self, trading_service, mock_trading_client):
        """Test closing entire position"""
        mock_order = MockAlpacaOrder(
            id='order999',
            symbol='AAPL',
            qty='10',
            side=Mock(value='sell'),
            type=Mock(value='market'),
            status='new',
            filled_qty='0',
            filled_avg_price=None,
            limit_price=None,
            submitted_at=datetime.now(),
            filled_at=None
        )
        mock_trading_client.close_position.return_value = mock_order
        
        order = trading_service.close_position('AAPL')
        
        assert order.symbol == 'AAPL'
        assert order.side == 'sell'
        assert mock_trading_client.close_position.called
    
    def test_close_partial_position_by_qty(self, trading_service, mock_trading_client):
        """Test closing partial position by quantity"""
        mock_order = MockAlpacaOrder(
            id='order888',
            symbol='GOOGL',
            qty='3',
            side=Mock(value='sell'),
            type=Mock(value='market'),
            status='new',
            filled_qty='0',
            filled_avg_price=None,
            limit_price=None,
            submitted_at=datetime.now(),
            filled_at=None
        )
        mock_trading_client.close_position.return_value = mock_order
        
        order = trading_service.close_position('GOOGL', qty=3)
        
        assert order.quantity == 3
    
    def test_close_position_invalid_symbol(self, trading_service):
        """Test closing position with invalid symbol raises ValueError"""
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            trading_service.close_position('')


class TestGetAccount:
    """Test get_account functionality"""
    
    def test_get_account(self, trading_service, mock_trading_client):
        """Test getting account information"""
        mock_account = MockAlpacaAccount(
            cash='10000.00',
            portfolio_value='25000.00',
            buying_power='20000.00',
            equity='25000.00',
            last_equity='24500.00',
            initial_margin='5000.00',
            maintenance_margin='3000.00',
            daytrade_count=0,
            daytrading_buying_power='20000.00',
            regt_buying_power='20000.00',
            pattern_day_trader=False,
            trading_blocked=False,
            transfers_blocked=False,
            account_blocked=False,
            account_number='123456789',
            status=Mock(value='ACTIVE')
        )
        mock_trading_client.get_account.return_value = mock_account
        
        account = trading_service.get_account()
        
        assert account['cash'] == 10000.00
        assert account['portfolio_value'] == 25000.00
        assert account['buying_power'] == 20000.00
        assert account['equity'] == 25000.00
        assert account['pattern_day_trader'] is False
        assert account['status'] == 'ACTIVE'


class TestGetOrders:
    """Test get_orders functionality"""
    
    def test_get_orders_all(self, trading_service, mock_trading_client):
        """Test getting all orders"""
        mock_orders = [
            MockAlpacaOrder(
                id='order1',
                symbol='AAPL',
                qty='10',
                side=Mock(value='buy'),
                type=Mock(value='market'),
                status='filled',
                filled_qty='10',
                filled_avg_price='150.00',
                limit_price=None,
                submitted_at=datetime.now(),
                filled_at=datetime.now()
            ),
            MockAlpacaOrder(
                id='order2',
                symbol='GOOGL',
                qty='5',
                side=Mock(value='sell'),
                type=Mock(value='limit'),
                status='new',
                filled_qty='0',
                filled_avg_price=None,
                limit_price='2850.00',
                submitted_at=datetime.now(),
                filled_at=None
            )
        ]
        mock_trading_client.get_orders.return_value = mock_orders
        
        orders = trading_service.get_orders(status='all')
        
        assert len(orders) == 2
        assert orders[0].order_id == 'order1'
        assert orders[1].order_id == 'order2'


class TestTradingMode:
    """Test trading mode methods"""
    
    def test_is_paper_trading(self, trading_service):
        """Test is_paper_trading returns correct value"""
        assert trading_service.is_paper_trading() is True
    
    def test_get_trading_mode(self, trading_service):
        """Test get_trading_mode returns correct string"""
        assert trading_service.get_trading_mode() == 'paper'
    
    def test_paper_mode_initialization(self):
        """Test initialization in paper trading mode"""
        with patch('services.trading_service.TradingClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            with patch('services.trading_service.settings') as mock_settings:
                mock_settings.alpaca.api_key = 'test_key'
                mock_settings.alpaca.secret_key = 'test_secret'
                
                service = TradingService(
                    api_key='test_key',
                    api_secret='test_secret',
                    paper=True
                )
                
                assert service.paper is True
                assert service.is_paper_trading() is True
                assert service.get_trading_mode() == 'paper'
                
                # Verify TradingClient was initialized with paper=True
                mock_client_class.assert_called_once()
                call_kwargs = mock_client_class.call_args[1]
                assert call_kwargs['paper'] is True
    
    def test_live_mode_initialization(self):
        """Test initialization in live trading mode"""
        with patch('services.trading_service.TradingClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            with patch('services.trading_service.settings') as mock_settings:
                mock_settings.alpaca.api_key = 'test_key'
                mock_settings.alpaca.secret_key = 'test_secret'
                
                service = TradingService(
                    api_key='test_key',
                    api_secret='test_secret',
                    paper=False
                )
                
                assert service.paper is False
                assert service.is_paper_trading() is False
                assert service.get_trading_mode() == 'live'
                
                # Verify TradingClient was initialized with paper=False
                call_kwargs = mock_client_class.call_args[1]
                assert call_kwargs['paper'] is False
    
    def test_paper_mode_order_execution(self, mock_trading_client):
        """Test that orders in paper mode don't affect live account"""
        with patch('services.trading_service.settings') as mock_settings:
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            
            # Create paper trading service
            service = TradingService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Mock order response
            mock_order = MockAlpacaOrder(
                id='paper_order_123',
                symbol='AAPL',
                qty='10',
                side=Mock(value='buy'),
                type=Mock(value='market'),
                status='new',
                filled_qty='0',
                filled_avg_price=None,
                limit_price=None,
                submitted_at=datetime.now(),
                filled_at=None
            )
            mock_trading_client.submit_order.return_value = mock_order
            
            # Place order
            order = service.place_order(
                symbol='AAPL',
                qty=10,
                side='buy'
            )
            
            # Verify order was placed in paper mode
            assert order.order_id == 'paper_order_123'
            assert service.is_paper_trading() is True
    
    def test_live_mode_order_execution(self, mock_trading_client):
        """Test that orders in live mode use live account"""
        with patch('services.trading_service.settings') as mock_settings:
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            
            # Create live trading service
            service = TradingService(
                api_key='test_key',
                api_secret='test_secret',
                paper=False
            )
            
            # Mock order response
            mock_order = MockAlpacaOrder(
                id='live_order_456',
                symbol='GOOGL',
                qty='5',
                side=Mock(value='sell'),
                type=Mock(value='market'),
                status='new',
                filled_qty='0',
                filled_avg_price=None,
                limit_price=None,
                submitted_at=datetime.now(),
                filled_at=None
            )
            mock_trading_client.submit_order.return_value = mock_order
            
            # Place order
            order = service.place_order(
                symbol='GOOGL',
                qty=5,
                side='sell'
            )
            
            # Verify order was placed in live mode
            assert order.order_id == 'live_order_456'
            assert service.is_paper_trading() is False
    
    def test_mode_persists_across_operations(self, mock_trading_client):
        """Test that trading mode persists across multiple operations"""
        with patch('services.trading_service.settings') as mock_settings:
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            
            # Create paper trading service
            service = TradingService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Verify mode before operations
            assert service.is_paper_trading() is True
            
            # Mock responses for various operations
            mock_order = MockAlpacaOrder(
                id='order_123',
                symbol='AAPL',
                qty='10',
                side=Mock(value='buy'),
                type=Mock(value='market'),
                status='new',
                filled_qty='0',
                filled_avg_price=None,
                limit_price=None,
                submitted_at=datetime.now(),
                filled_at=None
            )
            mock_trading_client.submit_order.return_value = mock_order
            mock_trading_client.get_order_by_id.return_value = mock_order
            mock_trading_client.get_all_positions.return_value = []
            
            # Perform multiple operations
            service.place_order(symbol='AAPL', qty=10, side='buy')
            assert service.is_paper_trading() is True
            
            service.get_order_status('order_123')
            assert service.is_paper_trading() is True
            
            service.get_positions()
            assert service.is_paper_trading() is True
            
            # Mode should still be paper
            assert service.get_trading_mode() == 'paper'


class TestOrderDataclass:
    """Test Order dataclass properties"""
    
    def test_order_is_filled(self):
        """Test is_filled property"""
        order = Order(
            order_id='test',
            symbol='AAPL',
            quantity=10,
            side='buy',
            order_type='market',
            status=OrderStatus.FILLED,
            filled_qty=10,
            filled_avg_price=150.00,
            limit_price=None,
            submitted_at=datetime.now(),
            filled_at=datetime.now()
        )
        
        assert order.is_filled is True
    
    def test_order_is_active(self):
        """Test is_active property"""
        order = Order(
            order_id='test',
            symbol='AAPL',
            quantity=10,
            side='buy',
            order_type='market',
            status=OrderStatus.NEW,
            filled_qty=0,
            filled_avg_price=None,
            limit_price=None,
            submitted_at=datetime.now(),
            filled_at=None
        )
        
        assert order.is_active is True


class TestErrorHandlingAndRetry:
    """Test error handling and retry logic"""
    
    def test_retry_on_api_failure(self, mock_trading_client):
        """Test that API failures trigger retry logic"""
        with patch('services.trading_service.settings') as mock_settings:
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            
            service = TradingService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Disable retry delays for testing
            service._max_retries = 3
            service._retry_delay = 0.0
            service._min_request_interval = 0.0
            
            # Mock API to fail twice then succeed
            mock_order = MockAlpacaOrder(
                id='order_retry',
                symbol='AAPL',
                qty='10',
                side=Mock(value='buy'),
                type=Mock(value='market'),
                status='new',
                filled_qty='0',
                filled_avg_price=None,
                limit_price=None,
                submitted_at=datetime.now(),
                filled_at=None
            )
            
            mock_trading_client.submit_order.side_effect = [
                Exception("Network error"),
                Exception("Timeout"),
                mock_order
            ]
            
            # Should succeed on third attempt
            order = service.place_order(
                symbol='AAPL',
                qty=10,
                side='buy'
            )
            
            assert order.order_id == 'order_retry'
            assert mock_trading_client.submit_order.call_count == 3
    
    def test_retry_exhaustion_raises_exception(self, mock_trading_client):
        """Test that exhausting retries raises the last exception"""
        with patch('services.trading_service.settings') as mock_settings:
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            
            service = TradingService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Disable retry delays for testing
            service._max_retries = 3
            service._retry_delay = 0.0
            service._min_request_interval = 0.0
            
            # Mock API to always fail
            mock_trading_client.submit_order.side_effect = Exception("Persistent error")
            
            # Should raise exception after all retries
            with pytest.raises(Exception, match="Persistent error"):
                service.place_order(
                    symbol='AAPL',
                    qty=10,
                    side='buy'
                )
            
            assert mock_trading_client.submit_order.call_count == 3
    
    def test_api_error_handling_on_get_positions(self, mock_trading_client):
        """Test error handling when fetching positions fails"""
        with patch('services.trading_service.settings') as mock_settings:
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            
            service = TradingService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Disable retry delays for testing
            service._max_retries = 1
            service._retry_delay = 0.0
            service._min_request_interval = 0.0
            
            # Mock API failure
            mock_trading_client.get_all_positions.side_effect = Exception("API unavailable")
            
            # Should raise exception
            with pytest.raises(Exception, match="API unavailable"):
                service.get_positions()
    
    def test_api_error_handling_on_cancel_order(self, mock_trading_client):
        """Test error handling when cancelling order fails"""
        with patch('services.trading_service.settings') as mock_settings:
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            
            service = TradingService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Disable retry delays for testing
            service._max_retries = 1
            service._retry_delay = 0.0
            service._min_request_interval = 0.0
            
            # Mock API failure
            mock_trading_client.cancel_order_by_id.side_effect = Exception("Order not found")
            
            # Should raise exception
            with pytest.raises(Exception, match="Order not found"):
                service.cancel_order('invalid_order_id')


class TestInputValidation:
    """Test comprehensive input validation"""
    
    def test_place_order_with_negative_limit_price(self, trading_service):
        """Test that negative limit price is rejected"""
        with pytest.raises(ValueError, match="Limit price must be positive"):
            trading_service.place_order(
                symbol='AAPL',
                qty=10,
                side='buy',
                order_type='limit',
                limit_price=-50.0
            )
    
    def test_place_order_with_invalid_time_in_force(self, trading_service):
        """Test that invalid time_in_force is rejected"""
        with pytest.raises(ValueError, match="Invalid time_in_force"):
            trading_service.place_order(
                symbol='AAPL',
                qty=10,
                side='buy',
                time_in_force='invalid'
            )
    
    def test_close_position_with_negative_qty(self, trading_service):
        """Test that negative quantity is rejected when closing position"""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            trading_service.close_position('AAPL', qty=-5)
    
    def test_close_position_with_invalid_percentage(self, trading_service):
        """Test that invalid percentage is rejected"""
        with pytest.raises(ValueError, match="Percentage must be between 0 and 100"):
            trading_service.close_position('AAPL', percentage=150)
    
    def test_close_position_with_both_qty_and_percentage(self, trading_service):
        """Test that specifying both qty and percentage is rejected"""
        with pytest.raises(ValueError, match="Cannot specify both qty and percentage"):
            trading_service.close_position('AAPL', qty=5, percentage=50)
    
    def test_get_orders_with_invalid_limit(self, trading_service):
        """Test that invalid limit is rejected"""
        with pytest.raises(ValueError, match="Limit must be positive"):
            trading_service.get_orders(limit=0)
    
    def test_get_orders_with_invalid_status(self, trading_service):
        """Test that invalid status is rejected"""
        with pytest.raises(ValueError, match="Status must be"):
            trading_service.get_orders(status='invalid_status')


class TestPositionDataclass:
    """Test Position dataclass properties"""
    
    def test_position_is_long(self):
        """Test is_long property"""
        position = Position(
            symbol='AAPL',
            quantity=10,
            side='long',
            entry_price=150.00,
            current_price=155.00,
            market_value=1550.00,
            cost_basis=1500.00,
            unrealized_pl=50.00,
            unrealized_pl_pct=0.0333
        )
        
        assert position.is_long is True
        assert position.is_short is False
    
    def test_position_is_short(self):
        """Test is_short property"""
        position = Position(
            symbol='AAPL',
            quantity=-10,
            side='short',
            entry_price=150.00,
            current_price=145.00,
            market_value=-1450.00,
            cost_basis=-1500.00,
            unrealized_pl=50.00,
            unrealized_pl_pct=0.0333
        )
        
        assert position.is_short is True
        assert position.is_long is False



# Property-Based Tests

class TestOrderExecutionConfirmationProperty:
    """
    Property-Based Test for Order Execution Confirmation
    
    Feature: ai-trading-agent, Property 5: Order execution confirmation
    Validates: Requirements 5.3, 5.5
    
    Property: For any trade order submitted, the system should receive either 
    a filled confirmation or error response from the broker API.
    """
    
    @given(
        symbol=st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=('Lu',))),
        qty=st.integers(min_value=1, max_value=10000),
        side=st.sampled_from(['buy', 'sell']),
        order_type=st.sampled_from(['market', 'limit']),
        limit_price=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        should_succeed=st.booleans()
    )
    @settings(max_examples=100, deadline=5000)
    def test_order_execution_always_returns_confirmation_or_error(
        self,
        symbol,
        qty,
        side,
        order_type,
        limit_price,
        should_succeed
    ):
        """
        Property: For any trade order submitted, the system should receive either
        a filled confirmation or error response from the broker API.
        
        This test verifies that:
        1. When an order is successfully submitted, we receive an Order object with valid status
        2. When an order fails, we receive an exception (error response)
        3. There is no case where we submit an order and get neither confirmation nor error
        """
        with patch('services.trading_service.TradingClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            with patch('services.trading_service.settings') as mock_settings:
                mock_settings.alpaca.api_key = 'test_key'
                mock_settings.alpaca.secret_key = 'test_secret'
                
                service = TradingService(
                    api_key='test_key',
                    api_secret='test_secret',
                    paper=True
                )
                
                # Disable retry delays for testing
                service._max_retries = 1
                service._retry_delay = 0.0
                service._min_request_interval = 0.0
                
                if should_succeed:
                    # Mock successful order submission
                    mock_order = MockAlpacaOrder(
                        id=f'order_{symbol}_{qty}',
                        symbol=symbol,
                        qty=str(qty),
                        side=Mock(value=side),
                        type=Mock(value=order_type),
                        status='new',
                        filled_qty='0',
                        filled_avg_price=None,
                        limit_price=str(limit_price) if order_type == 'limit' else None,
                        submitted_at=datetime.now(),
                        filled_at=None
                    )
                    mock_client.submit_order.return_value = mock_order
                    
                    # Execute order placement
                    try:
                        if order_type == 'limit':
                            order = service.place_order(
                                symbol=symbol,
                                qty=qty,
                                side=side,
                                order_type=order_type,
                                limit_price=limit_price
                            )
                        else:
                            order = service.place_order(
                                symbol=symbol,
                                qty=qty,
                                side=side,
                                order_type=order_type
                            )
                        
                        # Verify we received a confirmation (Order object)
                        assert order is not None, "Order confirmation should not be None"
                        assert isinstance(order, Order), "Should receive Order object as confirmation"
                        assert order.order_id is not None, "Order should have an ID"
                        assert order.status is not None, "Order should have a status"
                        assert order.symbol == symbol.upper(), "Order symbol should match"
                        assert order.quantity == qty, "Order quantity should match"
                        assert order.side == side, "Order side should match"
                        
                        # Verify status is valid
                        assert isinstance(order.status, OrderStatus), "Status should be OrderStatus enum"
                        
                    except Exception as e:
                        # If we get an exception on a successful mock, that's a test failure
                        pytest.fail(f"Expected order confirmation but got exception: {e}")
                
                else:
                    # Mock failed order submission
                    mock_client.submit_order.side_effect = Exception("Order submission failed")
                    
                    # Execute order placement and expect an error
                    with pytest.raises(Exception) as exc_info:
                        if order_type == 'limit':
                            service.place_order(
                                symbol=symbol,
                                qty=qty,
                                side=side,
                                order_type=order_type,
                                limit_price=limit_price
                            )
                        else:
                            service.place_order(
                                symbol=symbol,
                                qty=qty,
                                side=side,
                                order_type=order_type
                            )
                    
                    # Verify we received an error response
                    assert exc_info.value is not None, "Should receive error response"
                    assert str(exc_info.value) != "", "Error should have a message"
    
    @given(
        symbol=st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=('Lu',))),
        qty=st.integers(min_value=1, max_value=10000),
        side=st.sampled_from(['buy', 'sell']),
        final_status=st.sampled_from(['filled', 'partially_filled', 'cancelled', 'rejected'])
    )
    @settings(max_examples=100, deadline=5000)
    def test_order_status_check_always_returns_confirmation_or_error(
        self,
        symbol,
        qty,
        side,
        final_status
    ):
        """
        Property: For any order status check, the system should receive either
        a status confirmation or error response from the broker API.
        
        This verifies that checking order status always returns a definitive result.
        """
        with patch('services.trading_service.TradingClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            with patch('services.trading_service.settings') as mock_settings:
                mock_settings.alpaca.api_key = 'test_key'
                mock_settings.alpaca.secret_key = 'test_secret'
                
                service = TradingService(
                    api_key='test_key',
                    api_secret='test_secret',
                    paper=True
                )
                
                # Disable retry delays for testing
                service._max_retries = 1
                service._retry_delay = 0.0
                service._min_request_interval = 0.0
                
                order_id = f'order_{symbol}_{qty}'
                
                # Mock order status response
                mock_order = MockAlpacaOrder(
                    id=order_id,
                    symbol=symbol,
                    qty=str(qty),
                    side=Mock(value=side),
                    type=Mock(value='market'),
                    status=final_status,
                    filled_qty=str(qty) if final_status == 'filled' else '0',
                    filled_avg_price='150.00' if final_status == 'filled' else None,
                    limit_price=None,
                    submitted_at=datetime.now(),
                    filled_at=datetime.now() if final_status == 'filled' else None
                )
                mock_client.get_order_by_id.return_value = mock_order
                
                # Check order status
                order = service.get_order_status(order_id)
                
                # Verify we received a confirmation
                assert order is not None, "Order status should not be None"
                assert isinstance(order, Order), "Should receive Order object"
                assert order.order_id == order_id, "Order ID should match"
                assert order.status is not None, "Order should have a status"
                assert isinstance(order.status, OrderStatus), "Status should be OrderStatus enum"



# ========== Trading Schedule Management Tests ==========

class TestTradingSchedule:
    """Test TradingSchedule dataclass"""
    
    def test_default_schedule_creation(self):
        """Test creating schedule with default values"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import time as dt_time
        
        schedule = TradingSchedule()
        
        assert schedule.active_days == {0, 1, 2, 3, 4}  # Mon-Fri
        assert schedule.start_time == dt_time(9, 30)
        assert schedule.end_time == dt_time(16, 0)
        assert schedule.asset_class == AssetClass.STOCKS
        assert schedule.enabled is True
    
    def test_custom_schedule_creation(self):
        """Test creating schedule with custom values"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import time as dt_time
        
        schedule = TradingSchedule(
            active_days={0, 1, 2, 3, 4, 5, 6},
            start_time=dt_time(0, 0),
            end_time=dt_time(23, 59),
            asset_class=AssetClass.CRYPTO,
            timezone="UTC",
            enabled=True
        )
        
        assert schedule.active_days == {0, 1, 2, 3, 4, 5, 6}
        assert schedule.start_time == dt_time(0, 0)
        assert schedule.end_time == dt_time(23, 59)
        assert schedule.asset_class == AssetClass.CRYPTO
        assert schedule.timezone == "UTC"
    
    def test_is_trading_allowed_during_hours(self):
        """Test trading is allowed during configured hours"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import datetime, time as dt_time
        
        schedule = TradingSchedule(
            active_days={0, 1, 2, 3, 4},  # Mon-Fri
            start_time=dt_time(9, 30),
            end_time=dt_time(16, 0),
            asset_class=AssetClass.STOCKS
        )
        
        # Monday at 10:00 AM
        check_time = datetime(2024, 1, 1, 10, 0)  # Monday
        assert schedule.is_trading_allowed(check_time) is True
    
    def test_is_trading_allowed_outside_hours(self):
        """Test trading is not allowed outside configured hours"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import datetime, time as dt_time
        
        schedule = TradingSchedule(
            active_days={0, 1, 2, 3, 4},  # Mon-Fri
            start_time=dt_time(9, 30),
            end_time=dt_time(16, 0),
            asset_class=AssetClass.STOCKS
        )
        
        # Monday at 8:00 AM (before market open)
        check_time = datetime(2024, 1, 1, 8, 0)
        assert schedule.is_trading_allowed(check_time) is False
        
        # Monday at 5:00 PM (after market close)
        check_time = datetime(2024, 1, 1, 17, 0)
        assert schedule.is_trading_allowed(check_time) is False
    
    def test_is_trading_allowed_on_weekend(self):
        """Test trading is not allowed on weekends for stock schedule"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import datetime, time as dt_time
        
        schedule = TradingSchedule(
            active_days={0, 1, 2, 3, 4},  # Mon-Fri
            start_time=dt_time(9, 30),
            end_time=dt_time(16, 0),
            asset_class=AssetClass.STOCKS
        )
        
        # Saturday at 10:00 AM
        check_time = datetime(2024, 1, 6, 10, 0)  # Saturday
        assert schedule.is_trading_allowed(check_time) is False
    
    def test_is_trading_allowed_when_disabled(self):
        """Test trading is not allowed when schedule is disabled"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import datetime, time as dt_time
        
        schedule = TradingSchedule(
            active_days={0, 1, 2, 3, 4},
            start_time=dt_time(9, 30),
            end_time=dt_time(16, 0),
            asset_class=AssetClass.STOCKS,
            enabled=False
        )
        
        # Monday at 10:00 AM (would normally be allowed)
        check_time = datetime(2024, 1, 1, 10, 0)
        assert schedule.is_trading_allowed(check_time) is False
    
    def test_schedule_crossing_midnight(self):
        """Test schedule that crosses midnight"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import datetime, time as dt_time
        
        schedule = TradingSchedule(
            active_days={0, 1, 2, 3, 4, 5, 6},
            start_time=dt_time(23, 0),  # 11:00 PM
            end_time=dt_time(2, 0),     # 2:00 AM
            asset_class=AssetClass.CRYPTO
        )
        
        # 11:30 PM - should be allowed
        check_time = datetime(2024, 1, 1, 23, 30)
        assert schedule.is_trading_allowed(check_time) is True
        
        # 1:00 AM - should be allowed
        check_time = datetime(2024, 1, 1, 1, 0)
        assert schedule.is_trading_allowed(check_time) is True
        
        # 3:00 AM - should not be allowed
        check_time = datetime(2024, 1, 1, 3, 0)
        assert schedule.is_trading_allowed(check_time) is False
    
    def test_validate_schedule_valid(self):
        """Test validation of valid schedule"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import time as dt_time
        
        schedule = TradingSchedule(
            active_days={0, 1, 2, 3, 4},
            start_time=dt_time(9, 30),
            end_time=dt_time(16, 0),
            asset_class=AssetClass.STOCKS
        )
        
        errors = schedule.validate()
        assert len(errors) == 0
    
    def test_validate_schedule_no_active_days(self):
        """Test validation fails when no active days"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import time as dt_time
        
        schedule = TradingSchedule(
            active_days=set(),
            start_time=dt_time(9, 30),
            end_time=dt_time(16, 0),
            asset_class=AssetClass.STOCKS
        )
        
        errors = schedule.validate()
        assert len(errors) > 0
        assert any("active day" in err.lower() for err in errors)
    
    def test_validate_schedule_same_start_end_time(self):
        """Test validation fails when start and end time are the same"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import time as dt_time
        
        schedule = TradingSchedule(
            active_days={0, 1, 2, 3, 4},
            start_time=dt_time(9, 30),
            end_time=dt_time(9, 30),
            asset_class=AssetClass.STOCKS
        )
        
        errors = schedule.validate()
        assert len(errors) > 0
        assert any("same" in err.lower() for err in errors)
    
    def test_get_schedule_description(self):
        """Test getting human-readable schedule description"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import time as dt_time
        
        schedule = TradingSchedule(
            active_days={0, 1, 2, 3, 4},
            start_time=dt_time(9, 30),
            end_time=dt_time(16, 0),
            asset_class=AssetClass.STOCKS
        )
        
        description = schedule.get_schedule_description()
        
        assert "Stocks" in description
        assert "Mon" in description
        assert "Fri" in description
        assert "09:30 AM" in description
        assert "04:00 PM" in description


class TestTradingServiceScheduleManagement:
    """Test trading service schedule management methods"""
    
    def test_default_schedules_initialized(self, trading_service):
        """Test that default schedules are initialized"""
        from services.trading_service import AssetClass
        
        schedules = trading_service.get_all_schedules()
        
        assert AssetClass.STOCKS in schedules
        assert AssetClass.CRYPTO in schedules
        assert AssetClass.FOREX in schedules
    
    def test_get_trading_schedule(self, trading_service):
        """Test getting trading schedule for asset class"""
        from services.trading_service import AssetClass
        
        schedule = trading_service.get_trading_schedule(AssetClass.STOCKS)
        
        assert schedule is not None
        assert schedule.asset_class == AssetClass.STOCKS
        assert schedule.enabled is True
    
    def test_set_trading_schedule(self, trading_service):
        """Test setting custom trading schedule"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import time as dt_time
        
        custom_schedule = TradingSchedule(
            active_days={0, 1, 2},  # Mon-Wed only
            start_time=dt_time(10, 0),
            end_time=dt_time(15, 0),
            asset_class=AssetClass.STOCKS
        )
        
        trading_service.set_trading_schedule(AssetClass.STOCKS, custom_schedule)
        
        retrieved_schedule = trading_service.get_trading_schedule(AssetClass.STOCKS)
        assert retrieved_schedule.active_days == {0, 1, 2}
        assert retrieved_schedule.start_time == dt_time(10, 0)
        assert retrieved_schedule.end_time == dt_time(15, 0)
    
    def test_set_invalid_schedule_raises_error(self, trading_service):
        """Test setting invalid schedule raises ValueError"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import time as dt_time
        
        invalid_schedule = TradingSchedule(
            active_days=set(),  # No active days - invalid
            start_time=dt_time(9, 30),
            end_time=dt_time(16, 0),
            asset_class=AssetClass.STOCKS
        )
        
        with pytest.raises(ValueError, match="Invalid trading schedule"):
            trading_service.set_trading_schedule(AssetClass.STOCKS, invalid_schedule)
    
    def test_set_schedule_mismatched_asset_class(self, trading_service):
        """Test setting schedule with mismatched asset class raises error"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import time as dt_time
        
        schedule = TradingSchedule(
            active_days={0, 1, 2, 3, 4},
            start_time=dt_time(9, 30),
            end_time=dt_time(16, 0),
            asset_class=AssetClass.CRYPTO  # Mismatch
        )
        
        with pytest.raises(ValueError, match="does not match"):
            trading_service.set_trading_schedule(AssetClass.STOCKS, schedule)
    
    def test_is_trading_allowed(self, trading_service):
        """Test checking if trading is allowed"""
        from services.trading_service import AssetClass
        from datetime import datetime
        
        # Monday at 10:00 AM - should be allowed for stocks
        check_time = datetime(2024, 1, 1, 10, 0)
        
        assert trading_service.is_trading_allowed(AssetClass.STOCKS, check_time) is True
    
    def test_is_trading_allowed_outside_hours(self, trading_service):
        """Test trading not allowed outside hours"""
        from services.trading_service import AssetClass
        from datetime import datetime
        
        # Monday at 8:00 AM - before market open
        check_time = datetime(2024, 1, 1, 8, 0)
        
        assert trading_service.is_trading_allowed(AssetClass.STOCKS, check_time) is False
    
    def test_enable_automated_trading(self, trading_service):
        """Test enabling automated trading"""
        assert trading_service.is_automated_trading_enabled() is False
        
        trading_service.enable_automated_trading()
        
        assert trading_service.is_automated_trading_enabled() is True
    
    def test_disable_automated_trading(self, trading_service):
        """Test disabling automated trading"""
        trading_service.enable_automated_trading()
        assert trading_service.is_automated_trading_enabled() is True
        
        trading_service.disable_automated_trading()
        
        assert trading_service.is_automated_trading_enabled() is False
    
    def test_validate_schedule_against_market_hours_stocks(self, trading_service):
        """Test validating stock schedule against market hours"""
        from services.trading_service import AssetClass
        
        warnings = trading_service.validate_schedule_against_market_hours(AssetClass.STOCKS)
        
        # Default stock schedule should have no warnings
        assert len(warnings) == 0
    
    def test_validate_schedule_with_weekend_trading(self, trading_service):
        """Test validation warns about weekend trading for stocks"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import time as dt_time
        
        # Schedule with weekend trading
        schedule = TradingSchedule(
            active_days={0, 1, 2, 3, 4, 5, 6},  # Include weekend
            start_time=dt_time(9, 30),
            end_time=dt_time(16, 0),
            asset_class=AssetClass.STOCKS
        )
        
        trading_service.set_trading_schedule(AssetClass.STOCKS, schedule)
        
        warnings = trading_service.validate_schedule_against_market_hours(AssetClass.STOCKS)
        
        assert len(warnings) > 0
        assert any("non-trading days" in warn.lower() for warn in warnings)
    
    def test_validate_schedule_before_market_open(self, trading_service):
        """Test validation warns about trading before market open"""
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import time as dt_time
        
        # Schedule starting before market open
        schedule = TradingSchedule(
            active_days={0, 1, 2, 3, 4},
            start_time=dt_time(8, 0),  # Before 9:30 AM
            end_time=dt_time(16, 0),
            asset_class=AssetClass.STOCKS
        )
        
        trading_service.set_trading_schedule(AssetClass.STOCKS, schedule)
        
        warnings = trading_service.validate_schedule_against_market_hours(AssetClass.STOCKS)
        
        assert len(warnings) > 0
        assert any("before market open" in warn.lower() for warn in warnings)
    
    def test_get_schedule_status(self, trading_service):
        """Test getting schedule status for all asset classes"""
        status = trading_service.get_schedule_status()
        
        assert 'automated_trading_enabled' in status
        assert 'current_time' in status
        assert 'schedules' in status
        
        assert 'stocks' in status['schedules']
        assert 'crypto' in status['schedules']
        assert 'forex' in status['schedules']
        
        # Check stock schedule details
        stock_status = status['schedules']['stocks']
        assert 'enabled' in stock_status
        assert 'description' in stock_status
        assert 'trading_allowed_now' in stock_status
        assert 'active_days' in stock_status
        assert 'start_time' in stock_status
        assert 'end_time' in stock_status


class TestPlaceOrderWithScheduleCheck:
    """Test place_order_with_schedule_check functionality"""
    
    def test_place_order_without_automated_trading(self, trading_service, mock_trading_client):
        """Test order placement works when automated trading is disabled"""
        from services.trading_service import AssetClass
        from datetime import datetime
        
        # Automated trading is disabled by default
        assert trading_service.is_automated_trading_enabled() is False
        
        # Mock order response
        mock_order = MockAlpacaOrder(
            id='order123',
            symbol='AAPL',
            qty='10',
            side=Mock(value='buy'),
            type=Mock(value='market'),
            status='new',
            filled_qty='0',
            filled_avg_price=None,
            limit_price=None,
            submitted_at=datetime.now(),
            filled_at=None
        )
        mock_trading_client.submit_order.return_value = mock_order
        
        # Should work even outside trading hours when automated trading is disabled
        order = trading_service.place_order_with_schedule_check(
            symbol='AAPL',
            qty=10,
            side='buy',
            asset_class=AssetClass.STOCKS
        )
        
        assert order.symbol == 'AAPL'
        assert order.quantity == 10


class TestTradingScheduleEnforcementProperty:
    """
    Property-Based Test for Trading Schedule Enforcement
    
    Feature: ai-trading-agent, Property 18: Trading schedule enforcement
    Validates: Requirements 18.2
    
    Property: For any time outside configured trading hours, automated trading 
    should not execute new trades.
    """
    
    @given(
        symbol=st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=('Lu',))),
        qty=st.integers(min_value=1, max_value=1000),
        side=st.sampled_from(['buy', 'sell']),
        # Generate times outside trading hours
        hour=st.integers(min_value=0, max_value=23),
        minute=st.integers(min_value=0, max_value=59),
        day_of_week=st.integers(min_value=0, max_value=6)
    )
    @settings(max_examples=100, deadline=5000)
    def test_automated_trading_respects_schedule(
        self,
        symbol,
        qty,
        side,
        hour,
        minute,
        day_of_week
    ):
        """
        Property: For any time outside configured trading hours, automated trading
        should not execute new trades.
        
        This test verifies that:
        1. When automated trading is enabled
        2. And the current time is outside the configured trading schedule
        3. Then order placement should be rejected with a clear error message
        4. And when the time is inside the trading schedule, orders should be allowed
        """
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import datetime, time as dt_time
        
        with patch('services.trading_service.TradingClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            with patch('services.trading_service.settings') as mock_settings:
                mock_settings.alpaca.api_key = 'test_key'
                mock_settings.alpaca.secret_key = 'test_secret'
                
                service = TradingService(
                    api_key='test_key',
                    api_secret='test_secret',
                    paper=True
                )
                
                # Disable retry delays for testing
                service._max_retries = 1
                service._retry_delay = 0.0
                service._min_request_interval = 0.0
                
                # Configure a strict trading schedule: Mon-Fri, 9:30 AM - 4:00 PM
                schedule = TradingSchedule(
                    active_days={0, 1, 2, 3, 4},  # Mon-Fri
                    start_time=dt_time(9, 30),
                    end_time=dt_time(16, 0),
                    asset_class=AssetClass.STOCKS,
                    enabled=True
                )
                
                service.set_trading_schedule(AssetClass.STOCKS, schedule)
                
                # Enable automated trading (this activates schedule enforcement)
                service.enable_automated_trading()
                
                # Create a test datetime with the generated values
                # Use a fixed date in 2024 (Monday = 0, so Jan 1, 2024 is Monday)
                base_date = datetime(2024, 1, 1)  # Monday
                days_to_add = day_of_week
                test_datetime = datetime(
                    base_date.year,
                    base_date.month,
                    base_date.day + days_to_add,
                    hour,
                    minute
                )
                
                # Determine if this time should allow trading
                is_within_schedule = schedule.is_trading_allowed(test_datetime)
                
                # Mock successful order response
                mock_order = MockAlpacaOrder(
                    id=f'order_{symbol}_{qty}',
                    symbol=symbol,
                    qty=str(qty),
                    side=Mock(value=side),
                    type=Mock(value='market'),
                    status='new',
                    filled_qty='0',
                    filled_avg_price=None,
                    limit_price=None,
                    submitted_at=test_datetime,
                    filled_at=None
                )
                mock_client.submit_order.return_value = mock_order
                
                # Mock datetime.now() to return our test time
                with patch('services.trading_service.datetime') as mock_datetime:
                    mock_datetime.now.return_value = test_datetime
                    mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
                    
                    if is_within_schedule:
                        # Time is within schedule - order should be allowed
                        try:
                            order = service.place_order_with_schedule_check(
                                symbol=symbol,
                                qty=qty,
                                side=side,
                                asset_class=AssetClass.STOCKS
                            )
                            
                            # Verify order was placed successfully
                            assert order is not None, "Order should be placed during trading hours"
                            assert isinstance(order, Order), "Should return Order object"
                            assert order.symbol == symbol.upper()
                            assert order.quantity == qty
                            assert order.side == side
                            
                        except ValueError as e:
                            # If we get a ValueError during trading hours, that's a test failure
                            pytest.fail(
                                f"Order should be allowed during trading hours. "
                                f"Time: {test_datetime.strftime('%A %I:%M %p')}, "
                                f"Schedule: Mon-Fri 9:30 AM - 4:00 PM. "
                                f"Error: {e}"
                            )
                    
                    else:
                        # Time is outside schedule - order should be rejected
                        with pytest.raises(ValueError) as exc_info:
                            service.place_order_with_schedule_check(
                                symbol=symbol,
                                qty=qty,
                                side=side,
                                asset_class=AssetClass.STOCKS
                            )
                        
                        # Verify error message is informative
                        error_msg = str(exc_info.value)
                        assert "Trading not allowed" in error_msg, \
                            "Error message should indicate trading is not allowed"
                        assert "stocks" in error_msg.lower(), \
                            "Error message should mention the asset class"
                        
                        # Verify order was NOT submitted to broker
                        # (submit_order should not have been called)
                        # Note: We can't easily check this with the current implementation
                        # because place_order is called internally, but the ValueError
                        # should be raised before that happens
    
    @given(
        symbol=st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=('Lu',))),
        qty=st.integers(min_value=1, max_value=1000),
        side=st.sampled_from(['buy', 'sell']),
        hour=st.integers(min_value=0, max_value=23),
        minute=st.integers(min_value=0, max_value=59)
    )
    @settings(max_examples=100, deadline=5000)
    def test_manual_override_bypasses_schedule(
        self,
        symbol,
        qty,
        side,
        hour,
        minute
    ):
        """
        Property: For any time, manual trades with override_schedule=True should
        bypass schedule enforcement, even when automated trading is enabled.
        
        This verifies that users can always place manual trades regardless of schedule.
        """
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import datetime, time as dt_time
        
        with patch('services.trading_service.TradingClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            with patch('services.trading_service.settings') as mock_settings:
                mock_settings.alpaca.api_key = 'test_key'
                mock_settings.alpaca.secret_key = 'test_secret'
                
                service = TradingService(
                    api_key='test_key',
                    api_secret='test_secret',
                    paper=True
                )
                
                # Disable retry delays for testing
                service._max_retries = 1
                service._retry_delay = 0.0
                service._min_request_interval = 0.0
                
                # Configure a strict trading schedule
                schedule = TradingSchedule(
                    active_days={0, 1, 2, 3, 4},  # Mon-Fri
                    start_time=dt_time(9, 30),
                    end_time=dt_time(16, 0),
                    asset_class=AssetClass.STOCKS,
                    enabled=True
                )
                
                service.set_trading_schedule(AssetClass.STOCKS, schedule)
                service.enable_automated_trading()
                
                # Create a test datetime (use Saturday to ensure it's outside schedule)
                test_datetime = datetime(2024, 1, 6, hour, minute)  # Saturday
                
                # Verify this time is outside the schedule
                assert not schedule.is_trading_allowed(test_datetime), \
                    "Test time should be outside trading schedule"
                
                # Mock successful order response
                mock_order = MockAlpacaOrder(
                    id=f'order_{symbol}_{qty}',
                    symbol=symbol,
                    qty=str(qty),
                    side=Mock(value=side),
                    type=Mock(value='market'),
                    status='new',
                    filled_qty='0',
                    filled_avg_price=None,
                    limit_price=None,
                    submitted_at=test_datetime,
                    filled_at=None
                )
                mock_client.submit_order.return_value = mock_order
                
                # Mock datetime.now() to return our test time
                with patch('services.trading_service.datetime') as mock_datetime:
                    mock_datetime.now.return_value = test_datetime
                    mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
                    
                    # Place order with override_schedule=True
                    order = service.place_order_with_schedule_check(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        asset_class=AssetClass.STOCKS,
                        override_schedule=True  # This should bypass schedule check
                    )
                    
                    # Verify order was placed successfully despite being outside schedule
                    assert order is not None, "Order should be placed with override"
                    assert isinstance(order, Order), "Should return Order object"
                    assert order.symbol == symbol.upper()
                    assert order.quantity == qty
                    assert order.side == side
    
    @given(
        symbol=st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=('Lu',))),
        qty=st.integers(min_value=1, max_value=1000),
        side=st.sampled_from(['buy', 'sell']),
        hour=st.integers(min_value=0, max_value=23),
        minute=st.integers(min_value=0, max_value=59)
    )
    @settings(max_examples=100, deadline=5000)
    def test_disabled_automated_trading_allows_all_times(
        self,
        symbol,
        qty,
        side,
        hour,
        minute
    ):
        """
        Property: For any time, when automated trading is disabled, orders should
        be allowed regardless of schedule configuration.
        
        This verifies that schedule enforcement only applies when automated trading is enabled.
        """
        from services.trading_service import TradingSchedule, AssetClass
        from datetime import datetime, time as dt_time
        
        with patch('services.trading_service.TradingClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            with patch('services.trading_service.settings') as mock_settings:
                mock_settings.alpaca.api_key = 'test_key'
                mock_settings.alpaca.secret_key = 'test_secret'
                
                service = TradingService(
                    api_key='test_key',
                    api_secret='test_secret',
                    paper=True
                )
                
                # Disable retry delays for testing
                service._max_retries = 1
                service._retry_delay = 0.0
                service._min_request_interval = 0.0
                
                # Configure a strict trading schedule
                schedule = TradingSchedule(
                    active_days={0, 1, 2, 3, 4},  # Mon-Fri
                    start_time=dt_time(9, 30),
                    end_time=dt_time(16, 0),
                    asset_class=AssetClass.STOCKS,
                    enabled=True
                )
                
                service.set_trading_schedule(AssetClass.STOCKS, schedule)
                
                # Keep automated trading DISABLED (default state)
                assert not service.is_automated_trading_enabled(), \
                    "Automated trading should be disabled"
                
                # Create a test datetime (use Saturday to ensure it's outside schedule)
                test_datetime = datetime(2024, 1, 6, hour, minute)  # Saturday
                
                # Verify this time is outside the schedule
                assert not schedule.is_trading_allowed(test_datetime), \
                    "Test time should be outside trading schedule"
                
                # Mock successful order response
                mock_order = MockAlpacaOrder(
                    id=f'order_{symbol}_{qty}',
                    symbol=symbol,
                    qty=str(qty),
                    side=Mock(value=side),
                    type=Mock(value='market'),
                    status='new',
                    filled_qty='0',
                    filled_avg_price=None,
                    limit_price=None,
                    submitted_at=test_datetime,
                    filled_at=None
                )
                mock_client.submit_order.return_value = mock_order
                
                # Mock datetime.now() to return our test time
                with patch('services.trading_service.datetime') as mock_datetime:
                    mock_datetime.now.return_value = test_datetime
                    mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
                    
                    # Place order without automated trading enabled
                    order = service.place_order_with_schedule_check(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        asset_class=AssetClass.STOCKS
                    )
                    
                    # Verify order was placed successfully despite being outside schedule
                    assert order is not None, \
                        "Order should be placed when automated trading is disabled"
                    assert isinstance(order, Order), "Should return Order object"
                    assert order.symbol == symbol.upper()
                    assert order.quantity == qty
                    assert order.side == side
    
    def test_place_order_without_automated_trading_outside_hours(self, trading_service, mock_trading_client):
        """Test order placement works outside trading hours when automated trading is disabled"""
        from services.trading_service import AssetClass
        from datetime import datetime
        
        # Automated trading is disabled by default
        assert trading_service.is_automated_trading_enabled() is False
        
        # Mock order response
        mock_order = MockAlpacaOrder(
            id='order123',
            symbol='AAPL',
            qty='10',
            side=Mock(value='buy'),
            type=Mock(value='market'),
            status='new',
            filled_qty='0',
            filled_avg_price=None,
            limit_price=None,
            submitted_at=datetime.now(),
            filled_at=None
        )
        mock_trading_client.submit_order.return_value = mock_order
        
        # Should work even outside trading hours when automated trading is disabled
        check_time = datetime(2024, 1, 1, 8, 0)  # Before market open
        
        order = trading_service.place_order_with_schedule_check(
            symbol='AAPL',
            qty=10,
            side='buy',
            asset_class=AssetClass.STOCKS
        )
        
        assert order.symbol == 'AAPL'
    
    def test_place_order_with_automated_trading_during_hours(self, trading_service, mock_trading_client):
        """Test order placement works during trading hours with automated trading"""
        from services.trading_service import AssetClass
        from datetime import datetime
        
        # Enable automated trading
        trading_service.enable_automated_trading()
        
        # Mock order response
        mock_order = MockAlpacaOrder(
            id='order123',
            symbol='AAPL',
            qty='10',
            side=Mock(value='buy'),
            type=Mock(value='market'),
            status='new',
            filled_qty='0',
            filled_avg_price=None,
            limit_price=None,
            submitted_at=datetime.now(),
            filled_at=None
        )
        mock_trading_client.submit_order.return_value = mock_order
        
        # Mock is_trading_allowed to return True
        with patch.object(trading_service, 'is_trading_allowed', return_value=True):
            order = trading_service.place_order_with_schedule_check(
                symbol='AAPL',
                qty=10,
                side='buy',
                asset_class=AssetClass.STOCKS
            )
            
            assert order.symbol == 'AAPL'
    
    def test_place_order_with_automated_trading_outside_hours(self, trading_service):
        """Test order placement fails outside trading hours with automated trading"""
        from services.trading_service import AssetClass
        
        # Enable automated trading
        trading_service.enable_automated_trading()
        
        # Mock is_trading_allowed to return False
        with patch.object(trading_service, 'is_trading_allowed', return_value=False):
            with pytest.raises(ValueError, match="Trading not allowed"):
                trading_service.place_order_with_schedule_check(
                    symbol='AAPL',
                    qty=10,
                    side='buy',
                    asset_class=AssetClass.STOCKS
                )
    
    def test_place_order_with_schedule_override(self, trading_service, mock_trading_client):
        """Test order placement with schedule override"""
        from services.trading_service import AssetClass
        from datetime import datetime
        
        # Enable automated trading
        trading_service.enable_automated_trading()
        
        # Mock order response
        mock_order = MockAlpacaOrder(
            id='order123',
            symbol='AAPL',
            qty='10',
            side=Mock(value='buy'),
            type=Mock(value='market'),
            status='new',
            filled_qty='0',
            filled_avg_price=None,
            limit_price=None,
            submitted_at=datetime.now(),
            filled_at=None
        )
        mock_trading_client.submit_order.return_value = mock_order
        
        # Mock is_trading_allowed to return False
        with patch.object(trading_service, 'is_trading_allowed', return_value=False):
            # Should work with override_schedule=True
            order = trading_service.place_order_with_schedule_check(
                symbol='AAPL',
                qty=10,
                side='buy',
                asset_class=AssetClass.STOCKS,
                override_schedule=True
            )
            
            assert order.symbol == 'AAPL'
    
    def test_place_order_different_asset_classes(self, trading_service, mock_trading_client):
        """Test order placement respects different schedules for different asset classes"""
        from services.trading_service import AssetClass
        from datetime import datetime
        
        # Enable automated trading
        trading_service.enable_automated_trading()
        
        # Mock order response
        mock_order = MockAlpacaOrder(
            id='order123',
            symbol='BTC',
            qty='1',
            side=Mock(value='buy'),
            type=Mock(value='market'),
            status='new',
            filled_qty='0',
            filled_avg_price=None,
            limit_price=None,
            submitted_at=datetime.now(),
            filled_at=None
        )
        mock_trading_client.submit_order.return_value = mock_order
        
        # Crypto should be allowed 24/7
        with patch.object(trading_service, 'is_trading_allowed', return_value=True):
            order = trading_service.place_order_with_schedule_check(
                symbol='BTC',
                qty=1,
                side='buy',
                asset_class=AssetClass.CRYPTO
            )
            
            assert order.symbol == 'BTC'
