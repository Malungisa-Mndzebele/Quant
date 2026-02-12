"""Tests for paper trading service."""

import pytest
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, MagicMock

from services.paper_trading_service import (
    PaperTradingService,
    PaperAccount,
    PaperPosition
)
from services.trading_service import OrderStatus


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def mock_market_data():
    """Create mock market data service"""
    mock = Mock()
    
    # Mock quote with bid/ask prices
    mock_quote = Mock()
    mock_quote.bid_price = 100.0
    mock_quote.ask_price = 100.0
    
    mock.get_latest_quote = Mock(return_value=mock_quote)
    
    return mock


@pytest.fixture
def paper_service(temp_db, mock_market_data):
    """Create paper trading service for testing"""
    return PaperTradingService(
        db_path=temp_db,
        market_data_service=mock_market_data,
        initial_capital=100000.0
    )


class TestPaperTradingService:
    """Test paper trading service"""
    
    def test_initialization(self, paper_service):
        """Test service initialization"""
        assert paper_service.account is not None
        assert paper_service.account.initial_capital == 100000.0
        assert paper_service.account.cash == 100000.0
        assert paper_service.account.portfolio_value == 100000.0
    
    def test_is_paper_trading(self, paper_service):
        """Test paper trading mode check"""
        assert paper_service.is_paper_trading() is True
        assert paper_service.get_trading_mode() == 'paper'
    
    def test_place_market_buy_order(self, paper_service, mock_market_data):
        """Test placing a market buy order"""
        # Set mock price
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        # Place order
        order = paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='market'
        )
        
        assert order.symbol == 'AAPL'
        assert order.quantity == 10
        assert order.side == 'buy'
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 10
        assert order.filled_avg_price == 150.0
        
        # Check account updated
        account = paper_service.get_account()
        assert account['cash'] == 100000.0 - (10 * 150.0)
    
    def test_place_market_sell_order(self, paper_service, mock_market_data):
        """Test placing a market sell order"""
        # First buy some shares
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='market'
        )
        
        # Now sell
        mock_quote.bid_price = 160.0
        mock_quote.ask_price = 160.0
        
        order = paper_service.place_order(
            symbol='AAPL',
            qty=5,
            side='sell',
            order_type='market'
        )
        
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 5
        
        # Check position updated
        positions = paper_service.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == 5
    
    def test_insufficient_funds(self, paper_service, mock_market_data):
        """Test order rejection due to insufficient funds"""
        # Set high price
        mock_quote = Mock()
        mock_quote.bid_price = 50000.0
        mock_quote.ask_price = 50000.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        # Try to buy more than we can afford
        with pytest.raises(ValueError, match="Insufficient funds"):
            paper_service.place_order(
                symbol='AAPL',
                qty=10,
                side='buy',
                order_type='market'
            )
    
    def test_insufficient_shares(self, paper_service):
        """Test order rejection due to insufficient shares"""
        # Try to sell shares we don't have
        with pytest.raises(ValueError, match="Insufficient shares"):
            paper_service.place_order(
                symbol='AAPL',
                qty=10,
                side='sell',
                order_type='market'
            )
    
    def test_get_positions(self, paper_service, mock_market_data):
        """Test getting positions"""
        # Initially no positions
        positions = paper_service.get_positions()
        assert len(positions) == 0
        
        # Buy some shares
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='market'
        )
        
        paper_service.place_order(
            symbol='GOOGL',
            qty=5,
            side='buy',
            order_type='market'
        )
        
        # Check positions
        positions = paper_service.get_positions()
        assert len(positions) == 2
        
        symbols = {pos.symbol for pos in positions}
        assert 'AAPL' in symbols
        assert 'GOOGL' in symbols
    
    def test_close_position(self, paper_service, mock_market_data):
        """Test closing a position"""
        # Buy shares
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='market'
        )
        
        # Close position
        order = paper_service.close_position('AAPL')
        
        assert order.side == 'sell'
        assert order.quantity == 10
        assert order.status == OrderStatus.FILLED
        
        # Check position closed
        positions = paper_service.get_positions()
        assert len(positions) == 0
    
    def test_get_account(self, paper_service):
        """Test getting account information"""
        account = paper_service.get_account()
        
        assert account['account_id'] is not None
        assert account['cash'] == 100000.0
        assert account['portfolio_value'] == 100000.0
        assert account['initial_capital'] == 100000.0
        assert account['total_return'] == 0.0
        assert account['total_return_pct'] == 0.0
        assert account['mode'] == 'PAPER TRADING'
    
    def test_get_order_status(self, paper_service, mock_market_data):
        """Test getting order status"""
        # Place order
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        order = paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='market'
        )
        
        # Get status
        status = paper_service.get_order_status(order.order_id)
        
        assert status.order_id == order.order_id
        assert status.status == OrderStatus.FILLED
    
    def test_cancel_order(self, paper_service, mock_market_data):
        """Test cancelling an order"""
        # Place limit order that won't fill
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        order = paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='limit',
            limit_price=100.0  # Below market
        )
        
        assert order.status == OrderStatus.PENDING
        
        # Cancel order
        result = paper_service.cancel_order(order.order_id)
        assert result is True
        
        # Check status
        status = paper_service.get_order_status(order.order_id)
        assert status.status == OrderStatus.CANCELLED
    
    def test_reset_account(self, paper_service, mock_market_data):
        """Test resetting account"""
        # Make some trades
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='market'
        )
        
        # Reset
        paper_service.reset_account()
        
        # Check reset
        account = paper_service.get_account()
        assert account['cash'] == 100000.0
        assert account['portfolio_value'] == 100000.0
        
        positions = paper_service.get_positions()
        assert len(positions) == 0
    
    def test_reset_account_with_new_capital(self, paper_service):
        """Test resetting account with new capital"""
        paper_service.reset_account(initial_capital=50000.0)
        
        account = paper_service.get_account()
        assert account['initial_capital'] == 50000.0
        assert account['cash'] == 50000.0
        assert account['portfolio_value'] == 50000.0
    
    def test_save_session(self, paper_service, mock_market_data):
        """Test saving session"""
        # Make some trades
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='market'
        )
        
        # Save session
        session_file = paper_service.save_session('test_session')
        
        assert os.path.exists(session_file)
        assert 'test_session' in session_file
        
        # Cleanup
        if os.path.exists(session_file):
            os.unlink(session_file)
    
    def test_get_performance_summary(self, paper_service, mock_market_data):
        """Test getting performance summary"""
        # Make some trades
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='market'
        )
        
        # Get summary
        summary = paper_service.get_performance_summary()
        
        assert summary['account_id'] is not None
        assert summary['initial_capital'] == 100000.0
        assert summary['num_positions'] == 1
        assert summary['mode'] == 'PAPER TRADING'
    
    def test_position_pnl_calculation(self, paper_service, mock_market_data):
        """Test position P&L calculation"""
        # Buy at 150
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='market'
        )
        
        # Price goes up to 160
        mock_quote.bid_price = 160.0
        mock_quote.ask_price = 160.0
        
        positions = paper_service.get_positions()
        assert len(positions) == 1
        
        position = positions[0]
        assert position.entry_price == 150.0
        assert position.current_price == 160.0
        assert position.unrealized_pl == 100.0  # 10 shares * $10 gain
        assert position.unrealized_pl_pct == pytest.approx(6.67, rel=0.1)
    
    def test_average_price_calculation(self, paper_service, mock_market_data):
        """Test average price calculation for multiple buys"""
        # Buy 10 shares at 150
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='market'
        )
        
        # Buy 10 more shares at 160
        mock_quote.bid_price = 160.0
        mock_quote.ask_price = 160.0
        
        paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='market'
        )
        
        # Check average price
        positions = paper_service.get_positions()
        assert len(positions) == 1
        
        position = positions[0]
        assert position.quantity == 20
        assert position.entry_price == 155.0  # (10*150 + 10*160) / 20
    
    def test_limit_order_below_market(self, paper_service, mock_market_data):
        """Test limit buy order below market price"""
        # Market price is 150
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        # Place limit order at 140 (below market)
        order = paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='limit',
            limit_price=140.0
        )
        
        # Order should be pending
        assert order.status == OrderStatus.PENDING
        assert order.filled_qty == 0
        
        # No position should be created
        positions = paper_service.get_positions()
        assert len(positions) == 0
    
    def test_limit_order_above_market(self, paper_service, mock_market_data):
        """Test limit buy order above market price"""
        # Market price is 150
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        # Place limit order at 160 (above market)
        order = paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='limit',
            limit_price=160.0
        )
        
        # Order should fill immediately at limit price
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 10
        assert order.filled_avg_price == 160.0
    
    def test_invalid_symbol(self, paper_service):
        """Test order with invalid symbol"""
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            paper_service.place_order(
                symbol='',
                qty=10,
                side='buy',
                order_type='market'
            )
    
    def test_invalid_quantity(self, paper_service):
        """Test order with invalid quantity"""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            paper_service.place_order(
                symbol='AAPL',
                qty=0,
                side='buy',
                order_type='market'
            )
    
    def test_invalid_side(self, paper_service):
        """Test order with invalid side"""
        with pytest.raises(ValueError, match="Side must be 'buy' or 'sell'"):
            paper_service.place_order(
                symbol='AAPL',
                qty=10,
                side='invalid',
                order_type='market'
            )
    
    def test_invalid_order_type(self, paper_service):
        """Test order with invalid order type"""
        with pytest.raises(ValueError, match="Order type must be 'market' or 'limit'"):
            paper_service.place_order(
                symbol='AAPL',
                qty=10,
                side='buy',
                order_type='invalid'
            )
    
    def test_limit_order_without_price(self, paper_service):
        """Test limit order without limit price"""
        with pytest.raises(ValueError, match="Limit price is required for limit orders"):
            paper_service.place_order(
                symbol='AAPL',
                qty=10,
                side='buy',
                order_type='limit'
            )
    
    def test_close_nonexistent_position(self, paper_service):
        """Test closing a position that doesn't exist"""
        with pytest.raises(ValueError, match="No position found"):
            paper_service.close_position('AAPL')
    
    def test_get_nonexistent_order(self, paper_service):
        """Test getting status of nonexistent order"""
        with pytest.raises(ValueError, match="Order not found"):
            paper_service.get_order_status('invalid_order_id')
    
    def test_cancel_filled_order(self, paper_service, mock_market_data):
        """Test cancelling an already filled order"""
        # Place and fill order
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote.return_value = mock_quote
        
        order = paper_service.place_order(
            symbol='AAPL',
            qty=10,
            side='buy',
            order_type='market'
        )
        
        # Try to cancel filled order
        with pytest.raises(ValueError, match="Cannot cancel order in status"):
            paper_service.cancel_order(order.order_id)


class TestPaperAccount:
    """Test PaperAccount dataclass"""
    
    def test_to_dict(self):
        """Test converting account to dictionary"""
        now = datetime.now()
        account = PaperAccount(
            account_id='test_123',
            initial_capital=100000.0,
            cash=95000.0,
            portfolio_value=105000.0,
            equity=105000.0,
            buying_power=190000.0,
            created_at=now,
            last_updated=now
        )
        
        data = account.to_dict()
        
        assert data['account_id'] == 'test_123'
        assert data['initial_capital'] == 100000.0
        assert data['cash'] == 95000.0
        assert isinstance(data['created_at'], str)


class TestPaperPosition:
    """Test PaperPosition dataclass"""
    
    def test_to_dict(self):
        """Test converting position to dictionary"""
        now = datetime.now()
        position = PaperPosition(
            symbol='AAPL',
            quantity=10,
            side='long',
            entry_price=150.0,
            current_price=160.0,
            market_value=1600.0,
            cost_basis=1500.0,
            unrealized_pl=100.0,
            unrealized_pl_pct=6.67,
            entry_time=now
        )
        
        data = position.to_dict()
        
        assert data['symbol'] == 'AAPL'
        assert data['quantity'] == 10
        assert isinstance(data['entry_time'], str)
    
    def test_is_long(self):
        """Test is_long property"""
        position = PaperPosition(
            symbol='AAPL',
            quantity=10,
            side='long',
            entry_price=150.0,
            current_price=160.0,
            market_value=1600.0,
            cost_basis=1500.0,
            unrealized_pl=100.0,
            unrealized_pl_pct=6.67,
            entry_time=datetime.now()
        )
        
        assert position.is_long is True
        assert position.is_short is False
    
    def test_is_short(self):
        """Test is_short property"""
        position = PaperPosition(
            symbol='AAPL',
            quantity=10,
            side='short',
            entry_price=150.0,
            current_price=160.0,
            market_value=1600.0,
            cost_basis=1500.0,
            unrealized_pl=-100.0,
            unrealized_pl_pct=-6.67,
            entry_time=datetime.now()
        )
        
        assert position.is_short is True
        assert position.is_long is False



# Property-Based Tests
from hypothesis import given, strategies as st, settings, HealthCheck
import sqlite3


@given(
    num_trades=st.integers(min_value=1, max_value=10),
    initial_capital=st.floats(min_value=10000.0, max_value=1000000.0)
)
@settings(
    max_examples=100, 
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_paper_trading_isolation(num_trades, initial_capital):
    """
    Property Test: Paper trading isolation
    
    Feature: ai-trading-agent, Property 9: Paper trading isolation
    
    For any trade executed in paper trading mode, it should not affect 
    real broker account positions or balances.
    
    This test verifies that:
    1. Paper trading uses a separate database
    2. Paper trading never calls the real Alpaca Trading API
    3. All operations are isolated to the paper trading service
    4. No real money or positions are affected
    5. Paper trading mode is clearly identified
    
    Validates: Requirements 9.2
    """
    # Create temporary database for this test
    fd, temp_db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    try:
        # Create mock market data service
        mock_market_data = Mock()
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.0
        mock_market_data.get_latest_quote = Mock(return_value=mock_quote)
        
        # Create paper trading service
        paper_service = PaperTradingService(
            db_path=temp_db_path,
            market_data_service=mock_market_data,
            initial_capital=initial_capital
        )
        
        # CRITICAL ISOLATION CHECKS
        
        # 1. Verify paper trading mode is clearly identified
        assert paper_service.is_paper_trading() is True, \
            "Service must report as paper trading"
        assert paper_service.get_trading_mode() == 'paper', \
            "Trading mode must be 'paper'"
        
        # 2. Verify the service does NOT have a real trading client
        # Paper trading should never instantiate TradingClient from Alpaca
        assert not hasattr(paper_service, 'client'), \
            "Paper trading service must not have a real trading client"
        
        # 3. Verify database path is separate and clearly identified
        assert temp_db_path in paper_service.db_path or 'paper' in paper_service.db_path.lower(), \
            "Paper trading must use a separate, clearly identified database"
        
        # 4. Verify account is marked as paper trading
        account = paper_service.get_account()
        assert account['mode'] == 'PAPER TRADING', \
            "Account mode must be clearly marked as PAPER TRADING"
        assert account['initial_capital'] == initial_capital, \
            "Initial capital should match what was configured"
        
        # 5. Execute some trades and verify they're isolated
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        trades_executed = 0
        
        for i in range(min(num_trades, len(symbols))):
            symbol = symbols[i]
            try:
                # Place a buy order
                order = paper_service.place_order(
                    symbol=symbol,
                    qty=10,
                    side='buy',
                    order_type='market'
                )
                
                # Verify order ID is marked as paper trading
                assert order.order_id.startswith('paper_'), \
                    f"Paper trading orders must have 'paper_' prefix, got: {order.order_id}"
                
                # Verify order was filled (we have sufficient funds)
                if order.status == OrderStatus.FILLED:
                    trades_executed += 1
                
            except ValueError:
                # Insufficient funds is acceptable
                pass
        
        # 6. Verify database tables are paper trading specific
        conn = sqlite3.connect(paper_service.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE 'paper_%'
        """)
        paper_tables = cursor.fetchall()
        assert len(paper_tables) > 0, \
            "Paper trading must use tables with 'paper_' prefix for isolation"
        
        # Verify we have the expected paper trading tables
        table_names = [row[0] for row in paper_tables]
        expected_tables = ['paper_account', 'paper_positions', 'paper_orders', 'paper_transactions']
        for expected in expected_tables:
            assert expected in table_names, \
                f"Missing expected paper trading table: {expected}"
        
        conn.close()
        
        # 7. Verify that if trades were executed, they only affected paper account
        if trades_executed > 0:
            final_account = paper_service.get_account()
            
            # Cash should have decreased (we bought stocks)
            assert final_account['cash'] <= initial_capital, \
                "Paper trading cash should not exceed initial capital after buying"
            
            # Portfolio value should still be tracked
            assert final_account['portfolio_value'] > 0, \
                "Paper trading should track portfolio value"
            
            # Mode should still be paper trading
            assert final_account['mode'] == 'PAPER TRADING', \
                "Account mode must remain PAPER TRADING after trades"
            
            # Verify positions exist
            positions = paper_service.get_positions()
            assert len(positions) > 0, \
                "Paper trading should track positions"
            
            # All positions should be in paper trading
            for pos in positions:
                assert pos.symbol in symbols, \
                    "Position symbols should match what we traded"
        
        # 8. CRITICAL: Verify no real API client was ever created
        # This is the core isolation property - paper trading must NEVER
        # interact with real broker APIs
        assert not hasattr(paper_service, 'client'), \
            "Paper trading must never create a real trading client"
        
        # 9. Verify the service always reports as paper trading
        assert paper_service.is_paper_trading() is True, \
            "Service must always report as paper trading"
        
        # 10. Verify database isolation - check that database is separate file
        assert os.path.exists(paper_service.db_path), \
            "Paper trading database must exist"
        assert paper_service.db_path == temp_db_path, \
            "Paper trading must use the specified separate database"
    
    finally:
        # Cleanup
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)
