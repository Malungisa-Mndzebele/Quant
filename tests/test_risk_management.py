"""Tests for risk management system."""

import pytest
from datetime import datetime, date
from src.risk.risk_manager import RiskManager, RiskViolation
from src.models.config import RiskConfig
from src.models.portfolio import Portfolio
from src.models.order import Order, OrderAction, OrderType, OrderStatusEnum
from src.models.position import Position


class TestRiskManager:
    """Test suite for RiskManager."""
    
    @pytest.fixture
    def risk_config(self):
        """Create a standard risk configuration."""
        return RiskConfig(
            max_position_size_pct=10.0,
            max_daily_loss_pct=5.0,
            max_portfolio_leverage=1.0,
            allowed_symbols=None,
            max_positions=5,
            max_order_value=10000.0
        )
    
    @pytest.fixture
    def risk_manager(self, risk_config):
        """Create a risk manager instance."""
        return RiskManager(risk_config)
    
    @pytest.fixture
    def portfolio(self):
        """Create a portfolio with initial capital."""
        return Portfolio(initial_capital=100000.0)
    
    def test_risk_manager_initialization(self, risk_config):
        """Test risk manager initializes correctly."""
        rm = RiskManager(risk_config)
        assert rm.config == risk_config
        assert rm._daily_start_value is None
        assert rm._daily_pnl == 0.0
        assert rm._last_reset_date is None
    
    def test_validate_order_success(self, risk_manager, portfolio):
        """Test successful order validation."""
        order = Order(
            order_id="test_001",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        is_valid, reason = risk_manager.validate_order(order, portfolio, current_price=150.0)
        assert is_valid is True
        assert reason == ""
    
    def test_validate_order_insufficient_cash(self, risk_manager, portfolio):
        """Test order validation fails with insufficient cash."""
        order = Order(
            order_id="test_002",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=1000,
            order_type=OrderType.MARKET
        )
        
        is_valid, reason = risk_manager.validate_order(order, portfolio, current_price=150.0)
        assert is_valid is False
        assert "Insufficient cash" in reason
    
    def test_validate_order_position_size_limit(self, risk_manager, portfolio):
        """Test order validation fails when exceeding position size limit."""
        # Temporarily remove max_order_value limit to test position size limit specifically
        risk_manager.config.max_order_value = None
        
        # Max position size is 10% of $100,000 = $10,000
        # Try to buy $15,000 worth
        order = Order(
            order_id="test_003",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        is_valid, reason = risk_manager.validate_order(order, portfolio, current_price=150.0)
        assert is_valid is False
        assert "position size limit" in reason
    
    def test_validate_order_allowed_within_position_limit(self, risk_manager, portfolio):
        """Test order validation succeeds within position size limit."""
        # Max position size is 10% of $100,000 = $10,000
        # Buy $9,000 worth (60 shares at $150)
        order = Order(
            order_id="test_004",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=60,
            order_type=OrderType.MARKET
        )
        
        is_valid, reason = risk_manager.validate_order(order, portfolio, current_price=150.0)
        assert is_valid is True
        assert reason == ""
    
    def test_validate_order_sell_always_allowed(self, risk_manager, portfolio):
        """Test sell orders are always allowed (reducing position)."""
        # Add a position first
        portfolio.add_position("AAPL", 100, 150.0)
        
        # Sell order should be allowed regardless of position size limit
        order = Order(
            order_id="test_005",
            symbol="AAPL",
            action=OrderAction.SELL,
            quantity=50,
            order_type=OrderType.MARKET
        )
        
        is_valid, reason = risk_manager.validate_order(order, portfolio, current_price=150.0)
        assert is_valid is True
        assert reason == ""
    
    def test_validate_order_insufficient_position(self, risk_manager, portfolio):
        """Test sell order fails with insufficient position."""
        # Add a small position
        portfolio.add_position("AAPL", 10, 150.0)
        
        # Try to sell more than we have
        order = Order(
            order_id="test_006",
            symbol="AAPL",
            action=OrderAction.SELL,
            quantity=50,
            order_type=OrderType.MARKET
        )
        
        is_valid, reason = risk_manager.validate_order(order, portfolio, current_price=150.0)
        assert is_valid is False
        assert "Insufficient position" in reason
    
    def test_validate_order_max_positions_limit(self, risk_manager, portfolio):
        """Test order validation fails when max positions limit reached."""
        # Add 5 positions (max_positions = 5)
        for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]):
            portfolio.add_position(symbol, 10, 100.0)
        
        # Try to add a 6th position
        order = Order(
            order_id="test_007",
            symbol="NVDA",
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        is_valid, reason = risk_manager.validate_order(order, portfolio, current_price=100.0)
        assert is_valid is False
        assert "Maximum number of positions" in reason
    
    def test_validate_order_max_order_value(self, risk_manager, portfolio):
        """Test order validation fails when exceeding max order value."""
        # max_order_value = $10,000
        # Try to place order worth $15,000
        order = Order(
            order_id="test_008",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        is_valid, reason = risk_manager.validate_order(order, portfolio, current_price=150.0)
        assert is_valid is False
        assert "exceeds maximum order value" in reason
    
    def test_validate_order_allowed_symbols(self, risk_manager, portfolio):
        """Test order validation with symbol whitelist."""
        # Set allowed symbols
        risk_manager.config.allowed_symbols = ["AAPL", "GOOGL", "MSFT"]
        
        # Valid symbol
        order1 = Order(
            order_id="test_009",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        is_valid, reason = risk_manager.validate_order(order1, portfolio, current_price=150.0)
        assert is_valid is True
        
        # Invalid symbol
        order2 = Order(
            order_id="test_010",
            symbol="TSLA",
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        is_valid, reason = risk_manager.validate_order(order2, portfolio, current_price=200.0)
        assert is_valid is False
        assert "not in allowed symbols" in reason
    
    def test_daily_loss_limit_enforcement(self, risk_manager, portfolio):
        """Test daily loss limit is enforced correctly."""
        # Initialize daily tracking
        risk_manager._reset_daily_tracking(portfolio)
        initial_value = portfolio.get_total_value()
        
        # Simulate a 6% loss (exceeds 5% limit)
        loss_amount = initial_value * 0.06
        portfolio.cash_balance -= loss_amount
        
        # Check should fail
        assert risk_manager.check_daily_loss_limit(portfolio) is False
    
    def test_daily_loss_limit_within_threshold(self, risk_manager, portfolio):
        """Test daily loss limit passes when within threshold."""
        # Initialize daily tracking
        risk_manager._reset_daily_tracking(portfolio)
        initial_value = portfolio.get_total_value()
        
        # Simulate a 3% loss (within 5% limit)
        loss_amount = initial_value * 0.03
        portfolio.cash_balance -= loss_amount
        
        # Check should pass
        assert risk_manager.check_daily_loss_limit(portfolio) is True
    
    def test_daily_pnl_calculation(self, risk_manager, portfolio):
        """Test daily P&L calculation."""
        risk_manager._reset_daily_tracking(portfolio)
        initial_value = portfolio.get_total_value()
        
        # Simulate profit
        profit = 5000.0
        portfolio.cash_balance += profit
        
        daily_pnl = risk_manager.get_daily_pnl(portfolio)
        assert daily_pnl == profit
        
        daily_pnl_pct = risk_manager.get_daily_pnl_pct(portfolio)
        expected_pct = (profit / initial_value) * 100
        assert abs(daily_pnl_pct - expected_pct) < 0.01
    
    def test_calculate_exposure(self, risk_manager, portfolio):
        """Test exposure calculation."""
        # Add some positions
        portfolio.add_position("AAPL", 100, 150.0)  # $15,000
        portfolio.add_position("GOOGL", 50, 200.0)  # $10,000
        # Total positions: $25,000
        # Cash: $75,000 (100,000 - 25,000)
        portfolio.cash_balance = 75000.0
        
        exposure = risk_manager.calculate_exposure(portfolio)
        
        assert exposure['total_exposure'] == 25000.0
        assert abs(exposure['exposure_pct'] - 25.0) < 0.01
        assert abs(exposure['cash_pct'] - 75.0) < 0.01
        assert abs(exposure['leverage'] - 0.25) < 0.01
    
    def test_position_concentration(self, risk_manager, portfolio):
        """Test position concentration calculation."""
        # Add positions with different sizes
        portfolio.add_position("AAPL", 100, 150.0)  # $15,000
        portfolio.add_position("GOOGL", 50, 100.0)  # $5,000
        portfolio.cash_balance = 80000.0
        # Total portfolio: $100,000
        
        concentration = risk_manager.calculate_position_concentration(portfolio)
        
        assert abs(concentration["AAPL"] - 15.0) < 0.01
        assert abs(concentration["GOOGL"] - 5.0) < 0.01
    
    def test_largest_position(self, risk_manager, portfolio):
        """Test largest position identification."""
        portfolio.add_position("AAPL", 100, 150.0)  # $15,000
        portfolio.add_position("GOOGL", 50, 100.0)  # $5,000
        portfolio.cash_balance = 80000.0
        
        symbol, pct = risk_manager.get_largest_position_pct(portfolio)
        
        assert symbol == "AAPL"
        assert abs(pct - 15.0) < 0.01
    
    def test_risk_metrics_report(self, risk_manager, portfolio):
        """Test comprehensive risk metrics report generation."""
        # Set up portfolio
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.cash_balance = 85000.0
        risk_manager._reset_daily_tracking(portfolio)
        
        metrics = risk_manager.get_risk_metrics_report(portfolio)
        
        assert 'portfolio_value' in metrics
        assert 'cash_balance' in metrics
        assert 'positions_count' in metrics
        assert 'exposure' in metrics
        assert 'position_concentration' in metrics
        assert 'largest_position' in metrics
        assert 'daily_pnl' in metrics
        assert 'risk_limits' in metrics
        
        assert metrics['positions_count'] == 1
        assert metrics['cash_balance'] == 85000.0
    
    def test_format_risk_report(self, risk_manager, portfolio):
        """Test risk report formatting."""
        portfolio.add_position("AAPL", 100, 150.0)
        portfolio.cash_balance = 85000.0
        risk_manager._reset_daily_tracking(portfolio)
        
        report = risk_manager.format_risk_report(portfolio)
        
        assert isinstance(report, str)
        assert "RISK METRICS REPORT" in report
        assert "Portfolio Value" in report
        assert "EXPOSURE" in report
        assert "DAILY P&L" in report
        assert "RISK LIMITS" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
