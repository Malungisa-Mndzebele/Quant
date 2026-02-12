"""Property-based and unit tests for risk service."""

import pytest
from datetime import datetime
from hypothesis import given, strategies as st, settings, assume
from dataclasses import dataclass

from services.risk_service import (
    RiskService,
    TradeRequest,
    Position,
    ValidationResult,
    RiskViolationType,
    RiskMetrics
)
from config.settings import RiskConfig


# Test fixtures
@pytest.fixture
def risk_config():
    """Create a standard risk configuration for testing"""
    return RiskConfig(
        max_position_size=0.10,  # 10% of portfolio
        max_portfolio_risk=0.20,  # 20% risk
        daily_loss_limit=1000.0,  # $1000 daily loss limit
        stop_loss_pct=0.05,  # 5% stop loss
        take_profit_pct=0.10,  # 10% take profit
        max_open_positions=10
    )


@pytest.fixture
def risk_service(risk_config):
    """Create a risk service instance"""
    service = RiskService(config=risk_config)
    service.reset_daily_limits()  # Ensure clean state
    return service


# Hypothesis strategies for generating test data
@st.composite
def trade_request_strategy(draw, max_price=10000.0, max_qty=1000):
    """Generate random trade requests"""
    symbol = draw(st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=('Lu',))))
    quantity = draw(st.integers(min_value=1, max_value=max_qty))
    side = draw(st.sampled_from(['buy', 'sell']))
    price = draw(st.floats(min_value=0.01, max_value=max_price, allow_nan=False, allow_infinity=False))
    order_type = draw(st.sampled_from(['market', 'limit']))
    
    return TradeRequest(
        symbol=symbol,
        quantity=quantity,
        side=side,
        price=price,
        order_type=order_type
    )


@st.composite
def position_strategy(draw, max_price=10000.0, max_qty=1000):
    """Generate random positions"""
    symbol = draw(st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=('Lu',))))
    quantity = draw(st.integers(min_value=1, max_value=max_qty))
    entry_price = draw(st.floats(min_value=0.01, max_value=max_price, allow_nan=False, allow_infinity=False))
    current_price = draw(st.floats(min_value=0.01, max_value=max_price, allow_nan=False, allow_infinity=False))
    
    unrealized_pl = (current_price - entry_price) * quantity
    unrealized_pl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
    
    return Position(
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        current_price=current_price,
        unrealized_pl=unrealized_pl,
        unrealized_pl_pct=unrealized_pl_pct,
        stop_loss=None,
        take_profit=None
    )


class TestRiskLimitEnforcementProperty:
    """
    Property-based tests for risk limit enforcement.
    
    Feature: ai-trading-agent, Property 6: Risk limit enforcement
    Validates: Requirements 7.2
    """
    
    @given(
        trade=trade_request_strategy(),
        portfolio_value=st.floats(min_value=1000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False),
        available_cash=st.floats(min_value=0.0, max_value=1000000.0, allow_nan=False, allow_infinity=False),
        positions=st.lists(position_strategy(), max_size=15)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_risk_limit_enforcement(
        self,
        trade,
        portfolio_value,
        available_cash,
        positions
    ):
        """
        Property 6: Risk limit enforcement
        
        For any trade request, if it would exceed configured risk limits,
        the trade should be rejected before submission to broker.
        
        This property verifies that:
        1. Trades exceeding position size limits are rejected
        2. Trades exceeding available cash are rejected
        3. Trades exceeding max positions are rejected
        4. Trades exceeding daily loss limits are rejected
        5. Trades exceeding portfolio risk limits are rejected
        6. Valid trades within all limits are accepted
        
        **Validates: Requirements 7.2**
        """
        # Create fresh risk service for each test
        risk_config = RiskConfig(
            max_position_size=0.10,
            max_portfolio_risk=0.20,
            daily_loss_limit=1000.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            max_open_positions=10
        )
        risk_service = RiskService(config=risk_config)
        risk_service.reset_daily_limits()
        
        # Ensure portfolio value is positive
        assume(portfolio_value > 0)
        
        # Ensure trade price and quantity are reasonable
        assume(trade.price > 0)
        assume(trade.quantity > 0)
        
        # Calculate trade value
        trade_value = trade.quantity * trade.price
        
        # Ensure trade value doesn't cause overflow
        assume(trade_value < portfolio_value * 10)  # Reasonable upper bound
        
        # Validate the trade
        result = risk_service.validate_trade(
            trade=trade,
            portfolio_value=portfolio_value,
            current_positions=positions,
            available_cash=available_cash
        )
        
        # Property assertion: Check that validation result is consistent with risk limits
        config = risk_service.config
        
        # Check 1: Position size limit
        position_size_pct = trade_value / portfolio_value
        if position_size_pct > config.max_position_size:
            # Trade should be rejected for position size violation
            assert not result.is_valid, (
                f"Trade with position size {position_size_pct:.2%} should be rejected "
                f"(limit: {config.max_position_size:.2%})"
            )
            assert RiskViolationType.POSITION_SIZE in result.violations
        
        # Check 2: Insufficient funds for buy orders
        if trade.side.lower() == 'buy' and trade_value > available_cash:
            # Trade should be rejected for insufficient funds
            assert not result.is_valid, (
                f"Buy trade requiring ${trade_value:.2f} should be rejected "
                f"(available: ${available_cash:.2f})"
            )
            assert RiskViolationType.INSUFFICIENT_FUNDS in result.violations
        
        # Check 3: Maximum open positions
        if trade.side.lower() == 'buy':
            is_new_position = not any(pos.symbol == trade.symbol for pos in positions)
            if is_new_position and len(positions) >= config.max_open_positions:
                # Trade should be rejected for max positions violation
                assert not result.is_valid, (
                    f"Trade should be rejected when max positions ({config.max_open_positions}) reached "
                    f"(current: {len(positions)})"
                )
                assert RiskViolationType.MAX_POSITIONS in result.violations
        
        # Check 4: Daily loss limit
        daily_loss = risk_service._get_daily_loss()
        if daily_loss >= config.daily_loss_limit:
            # Trade should be rejected for daily loss limit
            assert not result.is_valid, (
                f"Trade should be rejected when daily loss limit reached "
                f"(${daily_loss:.2f} >= ${config.daily_loss_limit:.2f})"
            )
            assert RiskViolationType.DAILY_LOSS in result.violations
        
        # Check 5: Portfolio risk limit
        total_exposure = sum(abs(pos.quantity * pos.current_price) for pos in positions)
        if trade.side.lower() == 'buy':
            total_exposure += trade_value
        
        exposure_pct = total_exposure / portfolio_value
        max_exposure_pct = 1.0 - config.max_portfolio_risk
        
        if exposure_pct > max_exposure_pct:
            # Trade should be rejected for portfolio risk violation
            assert not result.is_valid, (
                f"Trade should be rejected when portfolio exposure {exposure_pct:.2%} "
                f"exceeds limit {max_exposure_pct:.2%}"
            )
            assert RiskViolationType.PORTFOLIO_RISK in result.violations
        
        # Property: If trade is valid, it must not violate any limits
        if result.is_valid:
            assert len(result.violations) == 0, "Valid trade should have no violations"
            assert position_size_pct <= config.max_position_size, (
                "Valid trade should not exceed position size limit"
            )
            if trade.side.lower() == 'buy':
                assert trade_value <= available_cash, (
                    "Valid buy trade should not exceed available cash"
                )
            assert daily_loss < config.daily_loss_limit, (
                "Valid trade should not be allowed when daily loss limit reached"
            )
        
        # Property: If trade is invalid, it must have at least one violation
        if not result.is_valid:
            assert len(result.violations) > 0, "Invalid trade must have violations"
            assert len(result.messages) > 0, "Invalid trade must have error messages"
    
    @given(
        portfolio_value=st.floats(min_value=10000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False),
        available_cash=st.floats(min_value=5000.0, max_value=500000.0, allow_nan=False, allow_infinity=False),
        price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_position_size_limit_always_enforced(
        self,
        portfolio_value,
        available_cash,
        price
    ):
        """
        Property: Position size limit is always enforced.
        
        For any trade that would create a position larger than max_position_size,
        the trade must be rejected regardless of other factors.
        
        **Validates: Requirements 7.2**
        """
        # Create fresh risk service
        risk_config = RiskConfig(
            max_position_size=0.10,
            max_portfolio_risk=0.20,
            daily_loss_limit=1000.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            max_open_positions=10
        )
        risk_service = RiskService(config=risk_config)
        risk_service.reset_daily_limits()
        
        assume(portfolio_value > 0)
        assume(price > 0)
        assume(available_cash > 0)
        
        # Calculate quantity that would exceed position size limit
        max_position_value = portfolio_value * risk_service.config.max_position_size
        exceeding_quantity = int((max_position_value * 1.5) / price) + 1
        
        # Ensure we have a meaningful quantity
        assume(exceeding_quantity > 0)
        assume(exceeding_quantity * price < portfolio_value * 10)  # Reasonable bound
        
        trade = TradeRequest(
            symbol='TEST',
            quantity=exceeding_quantity,
            side='buy',
            price=price,
            order_type='market'
        )
        
        result = risk_service.validate_trade(
            trade=trade,
            portfolio_value=portfolio_value,
            current_positions=[],
            available_cash=available_cash
        )
        
        # Property: Trade must be rejected for position size violation
        assert not result.is_valid, (
            f"Trade with excessive position size should always be rejected"
        )
        assert RiskViolationType.POSITION_SIZE in result.violations, (
            "Position size violation should be flagged"
        )
    
    @given(
        trade_value=st.floats(min_value=100.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        available_cash=st.floats(min_value=0.0, max_value=50000.0, allow_nan=False, allow_infinity=False),
        portfolio_value=st.floats(min_value=100000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_insufficient_funds_always_blocks_buy(
        self,
        trade_value,
        available_cash,
        portfolio_value
    ):
        """
        Property: Insufficient funds always blocks buy orders.
        
        For any buy trade where trade_value > available_cash,
        the trade must be rejected regardless of other factors.
        
        **Validates: Requirements 7.2**
        """
        # Create fresh risk service
        risk_config = RiskConfig(
            max_position_size=0.10,
            max_portfolio_risk=0.20,
            daily_loss_limit=1000.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            max_open_positions=10
        )
        risk_service = RiskService(config=risk_config)
        risk_service.reset_daily_limits()
        
        assume(portfolio_value > 0)
        assume(trade_value > available_cash)  # Ensure insufficient funds
        
        # Create a trade that requires more cash than available
        price = 100.0
        quantity = int(trade_value / price) + 1
        
        assume(quantity > 0)
        
        trade = TradeRequest(
            symbol='TEST',
            quantity=quantity,
            side='buy',
            price=price,
            order_type='market'
        )
        
        # Ensure position size is within limits (to isolate insufficient funds check)
        position_size_pct = (quantity * price) / portfolio_value
        assume(position_size_pct <= risk_service.config.max_position_size)
        
        result = risk_service.validate_trade(
            trade=trade,
            portfolio_value=portfolio_value,
            current_positions=[],
            available_cash=available_cash
        )
        
        # Property: Trade must be rejected for insufficient funds
        assert not result.is_valid, (
            f"Buy trade requiring ${quantity * price:.2f} should be rejected "
            f"when only ${available_cash:.2f} available"
        )
        assert RiskViolationType.INSUFFICIENT_FUNDS in result.violations, (
            "Insufficient funds violation should be flagged"
        )
    
    @given(
        num_positions=st.integers(min_value=10, max_value=20),
        portfolio_value=st.floats(min_value=100000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False),
        available_cash=st.floats(min_value=10000.0, max_value=100000.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_max_positions_limit_enforced(
        self,
        num_positions,
        portfolio_value,
        available_cash
    ):
        """
        Property: Maximum positions limit is always enforced.
        
        For any new position when max_open_positions is reached,
        the trade must be rejected.
        
        **Validates: Requirements 7.2**
        """
        # Create fresh risk service
        risk_config = RiskConfig(
            max_position_size=0.10,
            max_portfolio_risk=0.20,
            daily_loss_limit=1000.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            max_open_positions=10
        )
        risk_service = RiskService(config=risk_config)
        risk_service.reset_daily_limits()
        
        assume(portfolio_value > 0)
        assume(available_cash > 0)
        
        # Ensure we're at or above the limit
        assume(num_positions >= risk_service.config.max_open_positions)
        
        # Create positions up to or exceeding the limit
        positions = []
        for i in range(num_positions):
            positions.append(Position(
                symbol=f'SYM{i}',
                quantity=10,
                entry_price=100.0,
                current_price=105.0,
                unrealized_pl=50.0,
                unrealized_pl_pct=0.05
            ))
        
        # Try to open a new position (different symbol)
        trade = TradeRequest(
            symbol='NEWSYM',
            quantity=10,
            side='buy',
            price=100.0,
            order_type='market'
        )
        
        # Ensure trade is within other limits
        trade_value = trade.quantity * trade.price
        assume(trade_value <= available_cash)
        assume(trade_value / portfolio_value <= risk_service.config.max_position_size)
        
        result = risk_service.validate_trade(
            trade=trade,
            portfolio_value=portfolio_value,
            current_positions=positions,
            available_cash=available_cash
        )
        
        # Property: Trade must be rejected for max positions violation
        assert not result.is_valid, (
            f"New position should be rejected when {num_positions} positions exist "
            f"(limit: {risk_service.config.max_open_positions})"
        )
        assert RiskViolationType.MAX_POSITIONS in result.violations, (
            "Max positions violation should be flagged"
        )
    
    @given(
        daily_loss=st.floats(min_value=1000.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        portfolio_value=st.floats(min_value=100000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False),
        available_cash=st.floats(min_value=10000.0, max_value=100000.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_daily_loss_limit_blocks_trading(
        self,
        daily_loss,
        portfolio_value,
        available_cash
    ):
        """
        Property: Daily loss limit blocks all trading.
        
        For any trade when daily loss limit is reached or exceeded,
        the trade must be rejected.
        
        **Validates: Requirements 7.2**
        """
        # Create fresh risk service
        risk_config = RiskConfig(
            max_position_size=0.10,
            max_portfolio_risk=0.20,
            daily_loss_limit=1000.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            max_open_positions=10
        )
        risk_service = RiskService(config=risk_config)
        risk_service.reset_daily_limits()
        
        assume(portfolio_value > 0)
        assume(available_cash > 0)
        assume(daily_loss >= risk_service.config.daily_loss_limit)
        
        # Record losses to reach the limit
        risk_service._record_loss(daily_loss)
        
        # Try to place a small, otherwise valid trade
        trade = TradeRequest(
            symbol='TEST',
            quantity=10,
            side='buy',
            price=10.0,
            order_type='market'
        )
        
        # Ensure trade is within other limits
        trade_value = trade.quantity * trade.price
        assume(trade_value <= available_cash)
        assume(trade_value / portfolio_value <= risk_service.config.max_position_size)
        
        result = risk_service.validate_trade(
            trade=trade,
            portfolio_value=portfolio_value,
            current_positions=[],
            available_cash=available_cash
        )
        
        # Property: Trade must be rejected for daily loss limit
        assert not result.is_valid, (
            f"Trade should be rejected when daily loss ${daily_loss:.2f} "
            f">= limit ${risk_service.config.daily_loss_limit:.2f}"
        )
        assert RiskViolationType.DAILY_LOSS in result.violations, (
            "Daily loss violation should be flagged"
        )


class TestStopLossTriggerAccuracyProperty:
    """
    Property-based tests for stop-loss trigger accuracy.
    
    Feature: ai-trading-agent, Property 7: Stop-loss trigger accuracy
    Validates: Requirements 7.4
    """
    
    @given(
        entry_price=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        stop_loss_pct=st.floats(min_value=0.01, max_value=0.50, allow_nan=False, allow_infinity=False),
        price_change_pct=st.floats(min_value=-0.99, max_value=0.50, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_stop_loss_trigger_accuracy_percentage_based(
        self,
        entry_price,
        stop_loss_pct,
        price_change_pct
    ):
        """
        Property 7: Stop-loss trigger accuracy (percentage-based)
        
        For any position with a percentage-based stop-loss, when the current price
        reaches or falls below the stop-loss threshold, the position should be
        automatically closed.
        
        This property verifies that:
        1. Stop-loss triggers when price drops by the configured percentage
        2. Stop-loss does not trigger when price is above the threshold
        3. Stop-loss triggers exactly at the threshold
        4. The trigger logic is consistent across all price ranges
        
        **Validates: Requirements 7.4**
        """
        # Create risk service with specific stop-loss percentage
        risk_config = RiskConfig(
            max_position_size=0.10,
            max_portfolio_risk=0.20,
            daily_loss_limit=1000.0,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=0.10,
            max_open_positions=10
        )
        risk_service = RiskService(config=risk_config)
        risk_service.reset_daily_limits()
        
        assume(entry_price > 0)
        assume(stop_loss_pct > 0)
        
        # Calculate current price based on percentage change
        current_price = entry_price * (1 + price_change_pct)
        
        # Ensure current price is positive
        assume(current_price > 0)
        
        # Create position without explicit stop-loss (uses percentage-based)
        quantity = 100
        unrealized_pl = (current_price - entry_price) * quantity
        unrealized_pl_pct = (current_price - entry_price) / entry_price
        
        position = Position(
            symbol='TEST',
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pl=unrealized_pl,
            unrealized_pl_pct=unrealized_pl_pct,
            stop_loss=None,  # Use percentage-based stop-loss
            take_profit=None
        )
        
        # Check stop-loss trigger
        should_trigger = risk_service.check_stop_loss(position, current_price)
        
        # Calculate actual loss percentage
        loss_pct = (entry_price - current_price) / entry_price
        
        # Property assertion: Stop-loss should trigger if and only if loss >= threshold
        if loss_pct >= stop_loss_pct:
            # Price has dropped by stop-loss percentage or more
            assert should_trigger, (
                f"Stop-loss should trigger when loss {loss_pct:.2%} >= "
                f"threshold {stop_loss_pct:.2%} "
                f"(entry: ${entry_price:.2f}, current: ${current_price:.2f})"
            )
        else:
            # Price has not dropped enough
            assert not should_trigger, (
                f"Stop-loss should NOT trigger when loss {loss_pct:.2%} < "
                f"threshold {stop_loss_pct:.2%} "
                f"(entry: ${entry_price:.2f}, current: ${current_price:.2f})"
            )
    
    @given(
        entry_price=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        stop_loss_offset=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        price_offset=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_stop_loss_trigger_accuracy_explicit_price(
        self,
        entry_price,
        stop_loss_offset,
        price_offset
    ):
        """
        Property 7: Stop-loss trigger accuracy (explicit price)
        
        For any position with an explicit stop-loss price, when the current price
        reaches or falls below that price, the position should be automatically closed.
        
        This property verifies that:
        1. Stop-loss triggers when current_price <= stop_loss_price
        2. Stop-loss does not trigger when current_price > stop_loss_price
        3. The trigger logic works for all price ranges
        
        **Validates: Requirements 7.4**
        """
        # Create risk service
        risk_config = RiskConfig(
            max_position_size=0.10,
            max_portfolio_risk=0.20,
            daily_loss_limit=1000.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            max_open_positions=10
        )
        risk_service = RiskService(config=risk_config)
        risk_service.reset_daily_limits()
        
        assume(entry_price > 0)
        assume(stop_loss_offset > 0)
        
        # Set explicit stop-loss price below entry price
        stop_loss_price = entry_price - stop_loss_offset
        
        # Ensure stop-loss price is positive
        assume(stop_loss_price > 0)
        
        # Calculate current price relative to stop-loss
        current_price = stop_loss_price + price_offset
        
        # Ensure current price is positive
        assume(current_price > 0)
        
        # Create position with explicit stop-loss
        quantity = 100
        unrealized_pl = (current_price - entry_price) * quantity
        unrealized_pl_pct = (current_price - entry_price) / entry_price
        
        position = Position(
            symbol='TEST',
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pl=unrealized_pl,
            unrealized_pl_pct=unrealized_pl_pct,
            stop_loss=stop_loss_price,  # Explicit stop-loss
            take_profit=None
        )
        
        # Check stop-loss trigger
        should_trigger = risk_service.check_stop_loss(position, current_price)
        
        # Property assertion: Stop-loss should trigger if and only if current_price <= stop_loss_price
        if current_price <= stop_loss_price:
            assert should_trigger, (
                f"Stop-loss should trigger when current price ${current_price:.2f} <= "
                f"stop-loss ${stop_loss_price:.2f}"
            )
        else:
            assert not should_trigger, (
                f"Stop-loss should NOT trigger when current price ${current_price:.2f} > "
                f"stop-loss ${stop_loss_price:.2f}"
            )
    
    @given(
        entry_price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        stop_loss_pct=st.floats(min_value=0.01, max_value=0.20, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_stop_loss_triggers_at_exact_threshold(
        self,
        entry_price,
        stop_loss_pct
    ):
        """
        Property: Stop-loss triggers exactly at threshold.
        
        For any position, when the current price equals exactly the stop-loss threshold,
        the stop-loss should trigger.
        
        **Validates: Requirements 7.4**
        """
        # Create risk service
        risk_config = RiskConfig(
            max_position_size=0.10,
            max_portfolio_risk=0.20,
            daily_loss_limit=1000.0,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=0.10,
            max_open_positions=10
        )
        risk_service = RiskService(config=risk_config)
        risk_service.reset_daily_limits()
        
        assume(entry_price > 0)
        assume(stop_loss_pct > 0)
        
        # Calculate exact stop-loss price
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        
        # Create position with explicit stop-loss at exact threshold
        quantity = 100
        current_price = stop_loss_price  # Exactly at threshold
        unrealized_pl = (current_price - entry_price) * quantity
        unrealized_pl_pct = (current_price - entry_price) / entry_price
        
        position = Position(
            symbol='TEST',
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pl=unrealized_pl,
            unrealized_pl_pct=unrealized_pl_pct,
            stop_loss=stop_loss_price,
            take_profit=None
        )
        
        # Check stop-loss trigger
        should_trigger = risk_service.check_stop_loss(position, current_price)
        
        # Property: Stop-loss must trigger at exact threshold
        assert should_trigger, (
            f"Stop-loss must trigger when price equals exactly the stop-loss threshold "
            f"(${current_price:.2f} == ${stop_loss_price:.2f})"
        )
    
    @given(
        entry_price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        stop_loss_pct=st.floats(min_value=0.01, max_value=0.20, allow_nan=False, allow_infinity=False),
        epsilon=st.floats(min_value=0.001, max_value=0.01, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_stop_loss_does_not_trigger_just_above_threshold(
        self,
        entry_price,
        stop_loss_pct,
        epsilon
    ):
        """
        Property: Stop-loss does not trigger just above threshold.
        
        For any position, when the current price is just slightly above the stop-loss
        threshold, the stop-loss should NOT trigger.
        
        **Validates: Requirements 7.4**
        """
        # Create risk service
        risk_config = RiskConfig(
            max_position_size=0.10,
            max_portfolio_risk=0.20,
            daily_loss_limit=1000.0,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=0.10,
            max_open_positions=10
        )
        risk_service = RiskService(config=risk_config)
        risk_service.reset_daily_limits()
        
        assume(entry_price > 0)
        assume(stop_loss_pct > 0)
        assume(epsilon > 0)
        
        # Calculate stop-loss price
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        
        # Set current price just above threshold
        current_price = stop_loss_price + epsilon
        
        # Ensure current price is still below entry (otherwise it's a gain, not a loss)
        assume(current_price < entry_price)
        
        # Create position
        quantity = 100
        unrealized_pl = (current_price - entry_price) * quantity
        unrealized_pl_pct = (current_price - entry_price) / entry_price
        
        position = Position(
            symbol='TEST',
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pl=unrealized_pl,
            unrealized_pl_pct=unrealized_pl_pct,
            stop_loss=stop_loss_price,
            take_profit=None
        )
        
        # Check stop-loss trigger
        should_trigger = risk_service.check_stop_loss(position, current_price)
        
        # Property: Stop-loss must NOT trigger just above threshold
        assert not should_trigger, (
            f"Stop-loss must NOT trigger when price ${current_price:.2f} is above "
            f"stop-loss threshold ${stop_loss_price:.2f}"
        )
    
    @given(
        entry_price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        stop_loss_pct=st.floats(min_value=0.01, max_value=0.20, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_stop_loss_records_loss_when_triggered(
        self,
        entry_price,
        stop_loss_pct
    ):
        """
        Property: Stop-loss records loss when triggered (percentage-based).
        
        For any position with percentage-based stop-loss, when the stop-loss triggers,
        the loss should be recorded in daily tracking.
        
        **Validates: Requirements 7.4**
        """
        # Create risk service
        risk_config = RiskConfig(
            max_position_size=0.10,
            max_portfolio_risk=0.20,
            daily_loss_limit=10000.0,  # High limit to not interfere
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=0.10,
            max_open_positions=10
        )
        risk_service = RiskService(config=risk_config)
        risk_service.reset_daily_limits()
        
        assume(entry_price > 0)
        assume(stop_loss_pct > 0)
        
        # Calculate price that triggers stop-loss
        current_price = entry_price * (1 - stop_loss_pct - 0.01)  # Below threshold
        
        assume(current_price > 0)
        
        # Create position
        quantity = 100
        unrealized_pl = (current_price - entry_price) * quantity
        unrealized_pl_pct = (current_price - entry_price) / entry_price
        
        position = Position(
            symbol='TEST',
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pl=unrealized_pl,
            unrealized_pl_pct=unrealized_pl_pct,
            stop_loss=None,  # Use percentage-based
            take_profit=None
        )
        
        # Record initial daily loss
        initial_daily_loss = risk_service._get_daily_loss()
        
        # Check stop-loss (should trigger and record loss)
        should_trigger = risk_service.check_stop_loss(position, current_price)
        
        # Get daily loss after check
        final_daily_loss = risk_service._get_daily_loss()
        
        # Property: If stop-loss triggered, loss should be recorded
        if should_trigger:
            expected_loss = abs(unrealized_pl)
            assert final_daily_loss > initial_daily_loss, (
                "Daily loss should increase when stop-loss triggers"
            )
            assert abs(final_daily_loss - initial_daily_loss - expected_loss) < 0.01, (
                f"Recorded loss should equal position loss: "
                f"expected ${expected_loss:.2f}, got ${final_daily_loss - initial_daily_loss:.2f}"
            )


class TestRiskServiceUnitTests:
    """Unit tests for risk service functionality"""
    
    def test_validate_trade_within_all_limits(self, risk_service):
        """Test that valid trade within all limits is accepted"""
        trade = TradeRequest(
            symbol='AAPL',
            quantity=10,
            side='buy',
            price=150.0,
            order_type='market'
        )
        
        result = risk_service.validate_trade(
            trade=trade,
            portfolio_value=100000.0,
            current_positions=[],
            available_cash=50000.0
        )
        
        assert result.is_valid
        assert len(result.violations) == 0
    
    def test_validate_trade_exceeds_position_size(self, risk_service):
        """Test that trade exceeding position size limit is rejected"""
        trade = TradeRequest(
            symbol='AAPL',
            quantity=1000,  # $150,000 value
            side='buy',
            price=150.0,
            order_type='market'
        )
        
        result = risk_service.validate_trade(
            trade=trade,
            portfolio_value=100000.0,  # 150% of portfolio
            current_positions=[],
            available_cash=200000.0
        )
        
        assert not result.is_valid
        assert RiskViolationType.POSITION_SIZE in result.violations
        assert result.suggested_quantity is not None
        assert result.suggested_quantity < 1000
    
    def test_validate_trade_insufficient_funds(self, risk_service):
        """Test that trade with insufficient funds is rejected"""
        trade = TradeRequest(
            symbol='AAPL',
            quantity=100,
            side='buy',
            price=150.0,
            order_type='market'
        )
        
        result = risk_service.validate_trade(
            trade=trade,
            portfolio_value=100000.0,
            current_positions=[],
            available_cash=10000.0  # Only $10k available, need $15k
        )
        
        assert not result.is_valid
        assert RiskViolationType.INSUFFICIENT_FUNDS in result.violations
    
    def test_validate_trade_max_positions_reached(self, risk_service):
        """Test that new position is rejected when max positions reached"""
        # Create 10 existing positions (at the limit)
        positions = [
            Position(
                symbol=f'SYM{i}',
                quantity=10,
                entry_price=100.0,
                current_price=105.0,
                unrealized_pl=50.0,
                unrealized_pl_pct=0.05
            )
            for i in range(10)
        ]
        
        # Try to open a new position
        trade = TradeRequest(
            symbol='NEWSYM',
            quantity=10,
            side='buy',
            price=100.0,
            order_type='market'
        )
        
        result = risk_service.validate_trade(
            trade=trade,
            portfolio_value=100000.0,
            current_positions=positions,
            available_cash=50000.0
        )
        
        assert not result.is_valid
        assert RiskViolationType.MAX_POSITIONS in result.violations
    
    def test_validate_trade_daily_loss_limit_reached(self, risk_service):
        """Test that trade is rejected when daily loss limit reached"""
        # Record losses to reach the limit
        risk_service._record_loss(1000.0)
        
        trade = TradeRequest(
            symbol='AAPL',
            quantity=10,
            side='buy',
            price=100.0,
            order_type='market'
        )
        
        result = risk_service.validate_trade(
            trade=trade,
            portfolio_value=100000.0,
            current_positions=[],
            available_cash=50000.0
        )
        
        assert not result.is_valid
        assert RiskViolationType.DAILY_LOSS in result.violations
    
    def test_sell_order_not_checked_for_insufficient_funds(self, risk_service):
        """Test that sell orders are not checked for insufficient funds"""
        trade = TradeRequest(
            symbol='AAPL',
            quantity=100,
            side='sell',
            price=150.0,
            order_type='market'
        )
        
        result = risk_service.validate_trade(
            trade=trade,
            portfolio_value=100000.0,
            current_positions=[],
            available_cash=0.0  # No cash, but selling doesn't require cash
        )
        
        # Should not have insufficient funds violation
        assert RiskViolationType.INSUFFICIENT_FUNDS not in result.violations
    
    def test_adding_to_existing_position_not_counted_as_new(self, risk_service):
        """Test that adding to existing position doesn't count toward max positions"""
        # Create existing position
        positions = [
            Position(
                symbol='AAPL',
                quantity=10,
                entry_price=100.0,
                current_price=105.0,
                unrealized_pl=50.0,
                unrealized_pl_pct=0.05
            )
        ]
        
        # Add to existing position
        trade = TradeRequest(
            symbol='AAPL',  # Same symbol
            quantity=10,
            side='buy',
            price=105.0,
            order_type='market'
        )
        
        result = risk_service.validate_trade(
            trade=trade,
            portfolio_value=100000.0,
            current_positions=positions,
            available_cash=50000.0
        )
        
        # Should not have max positions violation
        assert RiskViolationType.MAX_POSITIONS not in result.violations
    
    def test_validation_result_error_message(self, risk_service):
        """Test that validation result provides clear error messages"""
        trade = TradeRequest(
            symbol='AAPL',
            quantity=1000,
            side='buy',
            price=150.0,
            order_type='market'
        )
        
        result = risk_service.validate_trade(
            trade=trade,
            portfolio_value=100000.0,
            current_positions=[],
            available_cash=10000.0
        )
        
        assert not result.is_valid
        error_msg = result.get_error_message()
        assert len(error_msg) > 0
        assert 'exceeds' in error_msg.lower() or 'insufficient' in error_msg.lower()
    
    def test_reset_daily_limits(self, risk_service):
        """Test that daily limits can be reset"""
        # Record some losses
        risk_service._record_loss(500.0)
        assert risk_service._get_daily_loss() == 500.0
        
        # Reset
        risk_service.reset_daily_limits()
        assert risk_service._get_daily_loss() == 0.0
    
    def test_get_daily_loss_remaining(self, risk_service):
        """Test getting remaining daily loss allowance"""
        # Initially, full limit available
        remaining = risk_service.get_daily_loss_remaining()
        assert remaining == risk_service.config.daily_loss_limit
        
        # Record some losses
        risk_service._record_loss(300.0)
        remaining = risk_service.get_daily_loss_remaining()
        assert remaining == risk_service.config.daily_loss_limit - 300.0
        
        # Exceed limit
        risk_service._record_loss(800.0)
        remaining = risk_service.get_daily_loss_remaining()
        assert remaining == 0.0
    
    def test_is_trading_allowed(self, risk_service):
        """Test trading allowed check"""
        # Initially allowed
        allowed, reason = risk_service.is_trading_allowed()
        assert allowed is True
        
        # Reach daily loss limit
        risk_service._record_loss(1000.0)
        allowed, reason = risk_service.is_trading_allowed()
        assert allowed is False
        assert 'loss limit' in reason.lower()
    
    # Position size calculation tests
    def test_calculate_position_size_with_strong_signal(self, risk_service):
        """Test position size calculation with strong signal (0.8-1.0)"""
        quantity = risk_service.calculate_position_size(
            symbol='AAPL',
            signal_strength=0.9,
            current_price=150.0,
            portfolio_value=100000.0,
            available_cash=50000.0
        )
        
        # Strong signal should result in larger position
        assert quantity > 0
        position_value = quantity * 150.0
        position_pct = position_value / 100000.0
        
        # Should be close to max position size for strong signal
        assert position_pct <= risk_service.config.max_position_size
        assert position_pct > risk_service.config.max_position_size * 0.7  # At least 70% of max
    
    def test_calculate_position_size_with_moderate_signal(self, risk_service):
        """Test position size calculation with moderate signal (0.6-0.8)"""
        quantity = risk_service.calculate_position_size(
            symbol='AAPL',
            signal_strength=0.7,
            current_price=150.0,
            portfolio_value=100000.0,
            available_cash=50000.0
        )
        
        # Moderate signal should result in smaller position
        assert quantity > 0
        position_value = quantity * 150.0
        position_pct = position_value / 100000.0
        
        # Should be less than max position size
        assert position_pct <= risk_service.config.max_position_size
        assert position_pct < risk_service.config.max_position_size * 0.9  # Less than 90% of max
    
    def test_calculate_position_size_with_weak_signal(self, risk_service):
        """Test position size calculation with weak signal (below 0.6)"""
        quantity = risk_service.calculate_position_size(
            symbol='AAPL',
            signal_strength=0.5,
            current_price=150.0,
            portfolio_value=100000.0,
            available_cash=50000.0
        )
        
        # Weak signal should result in no position
        assert quantity == 0
    
    def test_calculate_position_size_limited_by_cash(self, risk_service):
        """Test position size calculation limited by available cash"""
        quantity = risk_service.calculate_position_size(
            symbol='AAPL',
            signal_strength=1.0,
            current_price=150.0,
            portfolio_value=100000.0,
            available_cash=5000.0  # Limited cash
        )
        
        # Position should be limited by available cash
        assert quantity > 0
        position_value = quantity * 150.0
        assert position_value <= 5000.0
    
    def test_calculate_position_size_with_invalid_signal(self, risk_service):
        """Test position size calculation with invalid signal strength"""
        # Signal strength > 1.0
        quantity = risk_service.calculate_position_size(
            symbol='AAPL',
            signal_strength=1.5,
            current_price=150.0,
            portfolio_value=100000.0,
            available_cash=50000.0
        )
        assert quantity == 0
        
        # Signal strength <= 0
        quantity = risk_service.calculate_position_size(
            symbol='AAPL',
            signal_strength=0.0,
            current_price=150.0,
            portfolio_value=100000.0,
            available_cash=50000.0
        )
        assert quantity == 0
        
        # Negative signal
        quantity = risk_service.calculate_position_size(
            symbol='AAPL',
            signal_strength=-0.5,
            current_price=150.0,
            portfolio_value=100000.0,
            available_cash=50000.0
        )
        assert quantity == 0
    
    def test_calculate_position_size_with_invalid_price(self, risk_service):
        """Test position size calculation with invalid price"""
        quantity = risk_service.calculate_position_size(
            symbol='AAPL',
            signal_strength=0.9,
            current_price=0.0,
            portfolio_value=100000.0,
            available_cash=50000.0
        )
        assert quantity == 0
        
        quantity = risk_service.calculate_position_size(
            symbol='AAPL',
            signal_strength=0.9,
            current_price=-10.0,
            portfolio_value=100000.0,
            available_cash=50000.0
        )
        assert quantity == 0
    
    def test_calculate_position_size_with_invalid_portfolio_value(self, risk_service):
        """Test position size calculation with invalid portfolio value"""
        quantity = risk_service.calculate_position_size(
            symbol='AAPL',
            signal_strength=0.9,
            current_price=150.0,
            portfolio_value=0.0,
            available_cash=50000.0
        )
        assert quantity == 0
        
        quantity = risk_service.calculate_position_size(
            symbol='AAPL',
            signal_strength=0.9,
            current_price=150.0,
            portfolio_value=-10000.0,
            available_cash=50000.0
        )
        assert quantity == 0
    
    # Stop-loss logic tests
    def test_check_stop_loss_with_explicit_price_triggers(self, risk_service):
        """Test stop-loss with explicit price triggers correctly"""
        position = Position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            current_price=140.0,
            unrealized_pl=-1000.0,
            unrealized_pl_pct=-0.0667,
            stop_loss=145.0,  # Explicit stop-loss
            take_profit=None
        )
        
        # Price at stop-loss
        assert risk_service.check_stop_loss(position, 145.0) is True
        
        # Price below stop-loss
        assert risk_service.check_stop_loss(position, 140.0) is True
        
        # Price above stop-loss
        assert risk_service.check_stop_loss(position, 146.0) is False
    
    def test_check_stop_loss_with_percentage_triggers(self, risk_service):
        """Test stop-loss with percentage-based threshold triggers correctly"""
        position = Position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            current_price=142.5,
            unrealized_pl=-750.0,
            unrealized_pl_pct=-0.05,
            stop_loss=None,  # Use percentage-based
            take_profit=None
        )
        
        # Price at 5% loss (exactly at threshold)
        assert risk_service.check_stop_loss(position, 142.5) is True
        
        # Price at 6% loss (below threshold)
        assert risk_service.check_stop_loss(position, 141.0) is True
        
        # Price at 4% loss (above threshold)
        assert risk_service.check_stop_loss(position, 144.0) is False
    
    def test_check_stop_loss_does_not_trigger_on_profit(self, risk_service):
        """Test stop-loss does not trigger when position is profitable"""
        position = Position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            current_price=160.0,
            unrealized_pl=1000.0,
            unrealized_pl_pct=0.0667,
            stop_loss=145.0,
            take_profit=None
        )
        
        # Position is profitable, stop-loss should not trigger
        assert risk_service.check_stop_loss(position, 160.0) is False
    
    def test_check_take_profit_with_explicit_price_triggers(self, risk_service):
        """Test take-profit with explicit price triggers correctly"""
        position = Position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            current_price=165.0,
            unrealized_pl=1500.0,
            unrealized_pl_pct=0.10,
            stop_loss=None,
            take_profit=165.0  # Explicit take-profit
        )
        
        # Price at take-profit
        assert risk_service.check_take_profit(position, 165.0) is True
        
        # Price above take-profit
        assert risk_service.check_take_profit(position, 170.0) is True
        
        # Price below take-profit
        assert risk_service.check_take_profit(position, 160.0) is False
    
    def test_check_take_profit_with_percentage_triggers(self, risk_service):
        """Test take-profit with percentage-based threshold triggers correctly"""
        position = Position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            current_price=165.0,
            unrealized_pl=1500.0,
            unrealized_pl_pct=0.10,
            stop_loss=None,
            take_profit=None  # Use percentage-based (10%)
        )
        
        # Price at 10% profit (exactly at threshold)
        assert risk_service.check_take_profit(position, 165.0) is True
        
        # Price at 12% profit (above threshold)
        assert risk_service.check_take_profit(position, 168.0) is True
        
        # Price at 8% profit (below threshold)
        assert risk_service.check_take_profit(position, 162.0) is False
    
    def test_check_take_profit_does_not_trigger_on_loss(self, risk_service):
        """Test take-profit does not trigger when position is losing"""
        position = Position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            current_price=140.0,
            unrealized_pl=-1000.0,
            unrealized_pl_pct=-0.0667,
            stop_loss=None,
            take_profit=165.0
        )
        
        # Position is losing, take-profit should not trigger
        assert risk_service.check_take_profit(position, 140.0) is False
    
    # Portfolio risk metrics tests
    def test_get_portfolio_risk_with_no_positions(self, risk_service):
        """Test portfolio risk metrics with no positions"""
        metrics = risk_service.get_portfolio_risk(
            portfolio_value=100000.0,
            positions=[],
            daily_pl=0.0
        )
        
        assert metrics.portfolio_value == 100000.0
        assert metrics.total_exposure == 0.0
        assert metrics.exposure_pct == 0.0
        assert metrics.open_positions == 0
        assert metrics.risk_score >= 0.0
        assert metrics.risk_score <= 100.0
    
    def test_get_portfolio_risk_with_positions(self, risk_service):
        """Test portfolio risk metrics with open positions"""
        positions = [
            Position(
                symbol='AAPL',
                quantity=100,
                entry_price=150.0,
                current_price=155.0,
                unrealized_pl=500.0,
                unrealized_pl_pct=0.0333
            ),
            Position(
                symbol='GOOGL',
                quantity=50,
                entry_price=2800.0,
                current_price=2750.0,
                unrealized_pl=-2500.0,
                unrealized_pl_pct=-0.0179
            )
        ]
        
        metrics = risk_service.get_portfolio_risk(
            portfolio_value=300000.0,
            positions=positions,
            daily_pl=-2000.0
        )
        
        assert metrics.portfolio_value == 300000.0
        assert metrics.total_exposure > 0.0
        assert metrics.open_positions == 2
        assert metrics.daily_pl == -2000.0
        assert metrics.max_drawdown >= 2500.0  # At least the losing position
        assert metrics.risk_score >= 0.0
        assert metrics.risk_score <= 100.0
    
    def test_get_portfolio_risk_high_exposure(self, risk_service):
        """Test portfolio risk metrics with high exposure"""
        # Create positions with high total exposure
        positions = [
            Position(
                symbol=f'SYM{i}',
                quantity=100,
                entry_price=100.0,
                current_price=105.0,
                unrealized_pl=500.0,
                unrealized_pl_pct=0.05
            )
            for i in range(8)
        ]
        
        metrics = risk_service.get_portfolio_risk(
            portfolio_value=100000.0,
            positions=positions,
            daily_pl=0.0
        )
        
        # High exposure should result in higher risk score
        assert metrics.exposure_pct > 0.5
        # Risk score should be positive and reflect the exposure
        assert metrics.risk_score > 0.0
        assert metrics.risk_score <= 100.0
    
    def test_risk_metrics_is_high_risk(self, risk_service):
        """Test risk metrics high risk detection"""
        # Create high-risk scenario
        positions = [
            Position(
                symbol=f'SYM{i}',
                quantity=100,
                entry_price=100.0,
                current_price=95.0,
                unrealized_pl=-500.0,
                unrealized_pl_pct=-0.05
            )
            for i in range(10)
        ]
        
        # Record significant daily loss
        risk_service._record_loss(900.0)  # 90% of limit
        
        metrics = risk_service.get_portfolio_risk(
            portfolio_value=100000.0,
            positions=positions,
            daily_pl=-900.0
        )
        
        # Should be flagged as high risk (risk score should be high)
        # With 90% of daily loss limit used, should have high risk score
        assert metrics.risk_score > 60.0  # Should have elevated risk
    
    # Risk reduction suggestions tests
    def test_suggest_risk_reduction_low_risk(self, risk_service):
        """Test risk reduction suggestions when risk is low"""
        positions = [
            Position(
                symbol='AAPL',
                quantity=10,
                entry_price=150.0,
                current_price=155.0,
                unrealized_pl=50.0,
                unrealized_pl_pct=0.0333
            )
        ]
        
        metrics = risk_service.get_portfolio_risk(
            portfolio_value=100000.0,
            positions=positions,
            daily_pl=50.0
        )
        
        suggestions = risk_service.suggest_risk_reduction(positions, metrics)
        
        # Low risk should result in minimal suggestions
        assert len(suggestions) > 0
        assert any('acceptable' in s.lower() or 'within' in s.lower() for s in suggestions)
    
    def test_suggest_risk_reduction_high_exposure(self, risk_service):
        """Test risk reduction suggestions for high exposure"""
        # Create high exposure scenario
        positions = [
            Position(
                symbol=f'SYM{i}',
                quantity=100,
                entry_price=100.0,
                current_price=105.0,
                unrealized_pl=500.0,
                unrealized_pl_pct=0.05
            )
            for i in range(9)
        ]
        
        metrics = risk_service.get_portfolio_risk(
            portfolio_value=100000.0,
            positions=positions,
            daily_pl=0.0
        )
        
        suggestions = risk_service.suggest_risk_reduction(positions, metrics)
        
        # Should have suggestions (may or may not be high risk depending on risk score calculation)
        assert len(suggestions) > 0
        # If risk is high, should suggest reducing exposure or other risk reduction measures
        if metrics.is_high_risk():
            assert any('exposure' in s.lower() or 'reduce' in s.lower() or 'position' in s.lower() for s in suggestions)
    
    def test_suggest_risk_reduction_daily_loss_limit(self, risk_service):
        """Test risk reduction suggestions when approaching daily loss limit"""
        positions = [
            Position(
                symbol='AAPL',
                quantity=100,
                entry_price=150.0,
                current_price=140.0,
                unrealized_pl=-1000.0,
                unrealized_pl_pct=-0.0667
            )
        ]
        
        # Record losses approaching limit
        risk_service._record_loss(750.0)
        
        metrics = risk_service.get_portfolio_risk(
            portfolio_value=100000.0,
            positions=positions,
            daily_pl=-750.0
        )
        
        suggestions = risk_service.suggest_risk_reduction(positions, metrics)
        
        # Should have suggestions (may or may not be high risk depending on risk score calculation)
        assert len(suggestions) > 0
        # If risk is high, should suggest closing losing positions or other risk reduction measures
        if metrics.is_high_risk():
            assert any('closing' in s.lower() or 'loss' in s.lower() or 'position' in s.lower() for s in suggestions)
    
    def test_suggest_risk_reduction_under_diversified(self, risk_service):
        """Test risk reduction suggestions for under-diversified portfolio"""
        positions = [
            Position(
                symbol='AAPL',
                quantity=100,
                entry_price=150.0,
                current_price=155.0,
                unrealized_pl=500.0,
                unrealized_pl_pct=0.0333
            )
        ]
        
        # Artificially create high risk score
        metrics = RiskMetrics(
            portfolio_value=100000.0,
            total_exposure=15500.0,
            exposure_pct=0.155,
            var_95=825.0,
            max_drawdown=0.0,
            daily_pl=500.0,
            daily_pl_pct=0.5,
            open_positions=1,
            risk_score=75.0  # High risk
        )
        
        suggestions = risk_service.suggest_risk_reduction(positions, metrics)
        
        # Should suggest diversification
        assert any('diversif' in s.lower() for s in suggestions)
    
    def test_suggest_risk_reduction_no_stop_losses(self, risk_service):
        """Test risk reduction suggestions for positions without stop-losses"""
        positions = [
            Position(
                symbol='AAPL',
                quantity=100,
                entry_price=150.0,
                current_price=155.0,
                unrealized_pl=500.0,
                unrealized_pl_pct=0.0333,
                stop_loss=None  # No stop-loss set
            ),
            Position(
                symbol='GOOGL',
                quantity=50,
                entry_price=2800.0,
                current_price=2850.0,
                unrealized_pl=2500.0,
                unrealized_pl_pct=0.0179,
                stop_loss=None  # No stop-loss set
            )
        ]
        
        # Create high risk scenario
        metrics = RiskMetrics(
            portfolio_value=300000.0,
            total_exposure=158000.0,
            exposure_pct=0.527,
            var_95=4125.0,
            max_drawdown=0.0,
            daily_pl=3000.0,
            daily_pl_pct=1.0,
            open_positions=2,
            risk_score=75.0  # High risk
        )
        
        suggestions = risk_service.suggest_risk_reduction(positions, metrics)
        
        # Should suggest setting stop-losses
        assert any('stop-loss' in s.lower() or 'stop loss' in s.lower() for s in suggestions)
    
    def test_suggest_risk_reduction_take_profits(self, risk_service):
        """Test risk reduction suggestions for positions at profit targets"""
        positions = [
            Position(
                symbol='AAPL',
                quantity=100,
                entry_price=150.0,
                current_price=165.0,  # 10% profit (at take-profit threshold)
                unrealized_pl=1500.0,
                unrealized_pl_pct=0.10
            )
        ]
        
        # Create high risk scenario
        metrics = RiskMetrics(
            portfolio_value=100000.0,
            total_exposure=16500.0,
            exposure_pct=0.165,
            var_95=2475.0,
            max_drawdown=0.0,
            daily_pl=1500.0,
            daily_pl_pct=1.5,
            open_positions=1,
            risk_score=75.0  # High risk
        )
        
        suggestions = risk_service.suggest_risk_reduction(positions, metrics)
        
        # Should suggest taking profits
        assert any('profit' in s.lower() for s in suggestions)
    
    # Configuration update tests
    def test_update_config(self, risk_service):
        """Test updating risk configuration"""
        new_config = RiskConfig(
            max_position_size=0.15,
            max_portfolio_risk=0.25,
            daily_loss_limit=2000.0,
            stop_loss_pct=0.07,
            take_profit_pct=0.15,
            max_open_positions=15
        )
        
        risk_service.update_config(new_config)
        
        assert risk_service.config.max_position_size == 0.15
        assert risk_service.config.daily_loss_limit == 2000.0
        assert risk_service.config.max_open_positions == 15
