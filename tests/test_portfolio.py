"""Unit tests for portfolio tracking system."""
import pytest
from datetime import datetime, timedelta

from src.models.portfolio import Portfolio
from src.models.position import Position
from src.models.order import Trade, OrderAction
from src.models.performance import PerformanceMetrics


class TestPortfolioInitialization:
    """Test portfolio initialization."""
    
    def test_create_portfolio_with_initial_capital(self):
        """Test creating a portfolio with initial capital."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        assert portfolio.cash_balance == 100000.0
        assert portfolio.initial_value == 100000.0
        assert len(portfolio.positions) == 0
        assert len(portfolio.trade_history) == 0
    
    def test_create_portfolio_with_zero_capital_raises_error(self):
        """Test that zero initial capital raises error."""
        with pytest.raises(ValueError) as exc_info:
            Portfolio(initial_capital=0.0)
        
        assert "must be positive" in str(exc_info.value)
    
    def test_create_portfolio_with_negative_capital_raises_error(self):
        """Test that negative initial capital raises error."""
        with pytest.raises(ValueError) as exc_info:
            Portfolio(initial_capital=-1000.0)
        
        assert "must be positive" in str(exc_info.value)


class TestPositionManagement:
    """Test position management operations."""
    
    def test_add_new_position(self):
        """Test adding a new position to portfolio."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == 100
        assert portfolio.positions["AAPL"].entry_price == 150.0
    
    def test_add_multiple_positions(self):
        """Test adding multiple different positions."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.add_position("GOOGL", 50, 2800.0)
        
        assert len(portfolio.positions) == 2
        assert "AAPL" in portfolio.positions
        assert "GOOGL" in portfolio.positions
    
    def test_add_to_existing_position_calculates_average_price(self):
        """Test that adding to existing position calculates weighted average."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.add_position("AAPL", 50, 160.0)
        
        position = portfolio.positions["AAPL"]
        assert position.quantity == 150
        # Weighted average: (100*150 + 50*160) / 150 = 153.33
        assert abs(position.entry_price - 153.33) < 0.01
    
    def test_add_negative_quantity_reduces_position(self):
        """Test that adding negative quantity reduces position."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.add_position("AAPL", -50, 160.0)
        
        position = portfolio.positions["AAPL"]
        assert position.quantity == 50
    
    def test_add_negative_quantity_closes_position(self):
        """Test that reducing position to zero removes it."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.add_position("AAPL", -100, 160.0)
        
        assert "AAPL" not in portfolio.positions
    
    def test_add_zero_quantity_raises_error(self):
        """Test that zero quantity raises error."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        with pytest.raises(ValueError) as exc_info:
            portfolio.add_position("AAPL", 0, 150.0)
        
        assert "zero quantity" in str(exc_info.value)
    
    def test_add_position_with_zero_price_raises_error(self):
        """Test that zero price raises error."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        with pytest.raises(ValueError) as exc_info:
            portfolio.add_position("AAPL", 100, 0.0)
        
        assert "must be positive" in str(exc_info.value)
    
    def test_remove_position(self):
        """Test removing a position from portfolio."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.remove_position("AAPL")
        
        assert "AAPL" not in portfolio.positions
    
    def test_remove_nonexistent_position_does_nothing(self):
        """Test that removing non-existent position doesn't raise error."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        # Should not raise error
        portfolio.remove_position("AAPL")
        
        assert len(portfolio.positions) == 0
    
    def test_get_position(self):
        """Test retrieving a specific position."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        position = portfolio.get_position("AAPL")
        
        assert position is not None
        assert position.symbol == "AAPL"
        assert position.quantity == 100
    
    def test_get_nonexistent_position_returns_none(self):
        """Test that getting non-existent position returns None."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        position = portfolio.get_position("AAPL")
        
        assert position is None


class TestPriceUpdates:
    """Test price update functionality."""
    
    def test_update_prices(self):
        """Test updating prices for positions."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.add_position("GOOGL", 50, 2800.0)
        
        portfolio.update_prices({"AAPL": 155.0, "GOOGL": 2850.0})
        
        assert portfolio.positions["AAPL"].current_price == 155.0
        assert portfolio.positions["GOOGL"].current_price == 2850.0
    
    def test_update_prices_partial(self):
        """Test updating prices for subset of positions."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.add_position("GOOGL", 50, 2800.0)
        
        portfolio.update_prices({"AAPL": 155.0})
        
        assert portfolio.positions["AAPL"].current_price == 155.0
        assert portfolio.positions["GOOGL"].current_price == 2800.0  # Unchanged
    
    def test_update_prices_for_nonexistent_symbol_ignored(self):
        """Test that updating price for non-existent symbol is ignored."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        
        # Should not raise error
        portfolio.update_prices({"GOOGL": 2850.0})
        
        assert "GOOGL" not in portfolio.positions


class TestPortfolioValueCalculations:
    """Test portfolio value calculations."""
    
    def test_get_total_value_cash_only(self):
        """Test total value with only cash."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        total_value = portfolio.get_total_value()
        
        assert total_value == 100000.0
    
    def test_get_total_value_with_positions(self):
        """Test total value with positions."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.cash_balance = 85000.0  # Simulate buying
        
        total_value = portfolio.get_total_value()
        
        # Cash: 85000, Position: 100 * 150 = 15000
        assert total_value == 100000.0
    
    def test_get_total_value_with_price_changes(self):
        """Test total value after price changes."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.cash_balance = 85000.0
        
        portfolio.update_prices({"AAPL": 160.0})
        total_value = portfolio.get_total_value()
        
        # Cash: 85000, Position: 100 * 160 = 16000
        assert total_value == 101000.0
    
    def test_get_total_value_with_current_prices_parameter(self):
        """Test total value with current_prices parameter."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.cash_balance = 85000.0
        
        total_value = portfolio.get_total_value({"AAPL": 165.0})
        
        # Cash: 85000, Position: 100 * 165 = 16500
        assert total_value == 101500.0
    
    def test_get_positions_value(self):
        """Test getting total value of positions only."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.add_position("GOOGL", 50, 2800.0)
        
        positions_value = portfolio.get_positions_value()
        
        # AAPL: 100 * 150 = 15000, GOOGL: 50 * 2800 = 140000
        assert positions_value == 155000.0
    
    def test_get_unrealized_pnl(self):
        """Test calculating unrealized profit/loss."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.update_prices({"AAPL": 160.0})
        
        unrealized_pnl = portfolio.get_unrealized_pnl()
        
        # (160 - 150) * 100 = 1000
        assert unrealized_pnl == 1000.0
    
    def test_get_unrealized_pnl_with_loss(self):
        """Test calculating unrealized loss."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.update_prices({"AAPL": 140.0})
        
        unrealized_pnl = portfolio.get_unrealized_pnl()
        
        # (140 - 150) * 100 = -1000
        assert unrealized_pnl == -1000.0


class TestReturnCalculations:
    """Test return calculations."""
    
    def test_get_total_return_positive(self):
        """Test calculating positive total return."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.cash_balance = 85000.0
        portfolio.update_prices({"AAPL": 160.0})
        
        total_return = portfolio.get_total_return()
        
        # Initial: 100000, Current: 101000, Return: 1%
        assert abs(total_return - 1.0) < 0.01
    
    def test_get_total_return_negative(self):
        """Test calculating negative total return."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.cash_balance = 85000.0
        portfolio.update_prices({"AAPL": 140.0})
        
        total_return = portfolio.get_total_return()
        
        # Initial: 100000, Current: 99000, Return: -1%
        assert abs(total_return - (-1.0)) < 0.01
    
    def test_get_total_return_with_explicit_value(self):
        """Test calculating return with explicit current value."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        total_return = portfolio.get_total_return(current_value=110000.0)
        
        # Return: 10%
        assert total_return == 10.0
    
    def test_get_daily_return_with_insufficient_data(self):
        """Test daily return with insufficient data."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        daily_return = portfolio.get_daily_return()
        
        assert daily_return == 0.0
    
    def test_get_daily_return_with_data(self):
        """Test calculating daily return."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        now = datetime.now()
        portfolio.record_daily_value(now - timedelta(days=1), 100000.0)
        portfolio.record_daily_value(now, 101000.0)
        
        daily_return = portfolio.get_daily_return()
        
        # (101000 - 100000) / 100000 * 100 = 1%
        assert abs(daily_return - 1.0) < 0.01


class TestDrawdownTracking:
    """Test drawdown tracking."""
    
    def test_update_peak_value(self):
        """Test updating peak value."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.update_peak_value(105000.0)
        
        assert portfolio._peak_value == 105000.0
    
    def test_peak_value_only_increases(self):
        """Test that peak value only increases."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.update_peak_value(105000.0)
        portfolio.update_peak_value(103000.0)
        
        assert portfolio._peak_value == 105000.0
    
    def test_get_max_drawdown_no_drawdown(self):
        """Test max drawdown when at peak."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        max_dd = portfolio.get_max_drawdown(current_value=100000.0)
        
        assert max_dd == 0.0
    
    def test_get_max_drawdown_with_loss(self):
        """Test max drawdown with loss."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.update_peak_value(110000.0)
        max_dd = portfolio.get_max_drawdown(current_value=99000.0)
        
        # (110000 - 99000) / 110000 * 100 = 10%
        assert abs(max_dd - 10.0) < 0.01


class TestTradeHistory:
    """Test trade history tracking."""
    
    def test_record_trade(self):
        """Test recording a trade."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        trade = Trade(
            trade_id="T1",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            pnl=None
        )
        
        portfolio.record_trade(trade)
        
        assert len(portfolio.trade_history) == 1
        assert portfolio.trade_history[0] == trade
    
    def test_record_multiple_trades(self):
        """Test recording multiple trades."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        trade1 = Trade(
            trade_id="T1",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            pnl=None
        )
        trade2 = Trade(
            trade_id="T2",
            symbol="GOOGL",
            action=OrderAction.BUY,
            quantity=50,
            price=2800.0,
            timestamp=datetime.now(),
            pnl=None
        )
        
        portfolio.record_trade(trade1)
        portfolio.record_trade(trade2)
        
        assert len(portfolio.trade_history) == 2
    
    def test_get_win_rate_no_trades(self):
        """Test win rate with no trades."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        win_rate = portfolio.get_win_rate()
        
        assert win_rate == 0.0
    
    def test_get_win_rate_all_winning(self):
        """Test win rate with all winning trades."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        for i in range(5):
            trade = Trade(
                trade_id=f"T{i}",
                symbol="AAPL",
                action=OrderAction.SELL,
                quantity=100,
                price=150.0,
                timestamp=datetime.now(),
                pnl=100.0
            )
            portfolio.record_trade(trade)
        
        win_rate = portfolio.get_win_rate()
        
        assert win_rate == 100.0
    
    def test_get_win_rate_mixed(self):
        """Test win rate with mixed trades."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        # 3 winning trades
        for i in range(3):
            trade = Trade(
                trade_id=f"TW{i}",
                symbol="AAPL",
                action=OrderAction.SELL,
                quantity=100,
                price=150.0,
                timestamp=datetime.now(),
                pnl=100.0
            )
            portfolio.record_trade(trade)
        
        # 2 losing trades
        for i in range(2):
            trade = Trade(
                trade_id=f"TL{i}",
                symbol="AAPL",
                action=OrderAction.SELL,
                quantity=100,
                price=150.0,
                timestamp=datetime.now(),
                pnl=-50.0
            )
            portfolio.record_trade(trade)
        
        win_rate = portfolio.get_win_rate()
        
        # 3 out of 5 = 60%
        assert win_rate == 60.0


class TestPerformanceMetrics:
    """Test performance metrics calculation."""
    
    def test_calculate_returns_basic(self):
        """Test calculating performance metrics."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.cash_balance = 85000.0
        portfolio.update_prices({"AAPL": 160.0})
        
        metrics = portfolio.calculate_returns()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert abs(metrics.total_return - 1.0) < 0.01
        assert metrics.num_trades == 0
        assert abs(metrics.total_pnl - 1000.0) < 0.01
    
    def test_calculate_returns_with_trade_history(self):
        """Test metrics with trade history."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        # Add some trades
        trade1 = Trade(
            trade_id="T1",
            symbol="AAPL",
            action=OrderAction.SELL,
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            pnl=500.0
        )
        trade2 = Trade(
            trade_id="T2",
            symbol="GOOGL",
            action=OrderAction.SELL,
            quantity=50,
            price=2800.0,
            timestamp=datetime.now(),
            pnl=-200.0
        )
        
        portfolio.record_trade(trade1)
        portfolio.record_trade(trade2)
        
        metrics = portfolio.calculate_returns()
        
        assert metrics.num_trades == 2
        assert metrics.win_rate == 50.0
    
    def test_calculate_returns_with_daily_values(self):
        """Test metrics with daily value history."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        now = datetime.now()
        # Record 5 days of values
        portfolio.record_daily_value(now - timedelta(days=4), 100000.0)
        portfolio.record_daily_value(now - timedelta(days=3), 101000.0)
        portfolio.record_daily_value(now - timedelta(days=2), 102000.0)
        portfolio.record_daily_value(now - timedelta(days=1), 101500.0)
        portfolio.record_daily_value(now, 103000.0)
        
        metrics = portfolio.calculate_returns()
        
        # Should have Sharpe ratio calculated
        assert metrics.sharpe_ratio != 0.0
        # Should have max drawdown from equity curve
        assert metrics.max_drawdown >= 0.0
    
    def test_calculate_returns_with_current_prices(self):
        """Test metrics with current prices parameter."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.cash_balance = 85000.0
        
        metrics = portfolio.calculate_returns({"AAPL": 165.0})
        
        # Total value: 85000 + 100*165 = 101500
        # Return: 1.5%
        assert abs(metrics.total_return - 1.5) < 0.01


class TestPortfolioRepresentation:
    """Test portfolio string representation."""
    
    def test_portfolio_repr(self):
        """Test portfolio string representation."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.cash_balance = 85000.0
        
        repr_str = repr(portfolio)
        
        assert "Portfolio" in repr_str
        assert "cash=85000.00" in repr_str
        assert "positions=1" in repr_str
        assert "total_value=100000.00" in repr_str
