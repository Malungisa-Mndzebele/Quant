"""Unit tests for strategy components."""
import pytest
from datetime import datetime
from typing import List

from src.strategies import Strategy, StrategyManager, MovingAverageCrossover
from src.models.signal import Signal, SignalAction, OrderType
from src.models.market_data import MarketData
from src.models.order import Order, OrderAction, OrderStatusEnum


class MockStrategy(Strategy):
    """Mock strategy for testing."""
    
    def __init__(self, name: str = "MockStrategy"):
        super().__init__(name)
        self.data_received = []
        self.signals_to_return = []
        self.orders_filled = []
    
    def on_data(self, market_data: MarketData) -> None:
        """Store received market data."""
        self.data_received.append(market_data)
    
    def generate_signals(self) -> List[Signal]:
        """Return pre-configured signals."""
        return self.signals_to_return
    
    def on_order_filled(self, order: Order) -> None:
        """Store filled orders."""
        self.orders_filled.append(order)


class TestStrategyBase:
    """Test Strategy base class."""
    
    def test_strategy_initialization(self):
        """Test strategy initialization with name and params."""
        strategy = MockStrategy("TestStrategy")
        
        assert strategy.name == "TestStrategy"
        assert strategy.params == {}
        assert strategy._state == {}
    
    def test_strategy_with_params(self):
        """Test strategy initialization with parameters."""
        strategy = MockStrategy("TestStrategy")
        strategy.params = {"param1": 10, "param2": "value"}
        
        assert strategy.params["param1"] == 10
        assert strategy.params["param2"] == "value"
    
    def test_strategy_state_management(self):
        """Test strategy state get/set operations."""
        strategy = MockStrategy()
        
        # Set state
        strategy.set_state("key1", "value1")
        strategy.set_state("key2", 42)
        
        # Get state
        assert strategy.get_state("key1") == "value1"
        assert strategy.get_state("key2") == 42
        assert strategy.get_state("nonexistent") is None
        assert strategy.get_state("nonexistent", "default") == "default"
    
    def test_strategy_reset_state(self):
        """Test resetting strategy state."""
        strategy = MockStrategy()
        
        strategy.set_state("key1", "value1")
        strategy.set_state("key2", 42)
        
        strategy.reset_state()
        
        assert strategy.get_state("key1") is None
        assert strategy.get_state("key2") is None
    
    def test_strategy_on_data(self):
        """Test on_data method receives market data."""
        strategy = MockStrategy()
        
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000
        )
        
        strategy.on_data(market_data)
        
        assert len(strategy.data_received) == 1
        assert strategy.data_received[0] == market_data
    
    def test_strategy_generate_signals(self):
        """Test generate_signals returns configured signals."""
        strategy = MockStrategy()
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        strategy.signals_to_return = [signal]
        
        signals = strategy.generate_signals()
        
        assert len(signals) == 1
        assert signals[0] == signal
    
    def test_strategy_on_order_filled(self):
        """Test on_order_filled callback."""
        strategy = MockStrategy()
        
        order = Order(
            order_id="123",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            status=OrderStatusEnum.FILLED,
            filled_quantity=100,
            filled_price=150.0
        )
        
        strategy.on_order_filled(order)
        
        assert len(strategy.orders_filled) == 1
        assert strategy.orders_filled[0] == order


class TestStrategyManager:
    """Test StrategyManager class."""
    
    def test_manager_initialization(self):
        """Test strategy manager initialization."""
        manager = StrategyManager()
        
        assert len(manager.strategies) == 0
    
    def test_register_strategy(self):
        """Test registering a strategy."""
        manager = StrategyManager()
        strategy = MockStrategy("Strategy1")
        
        manager.register_strategy(strategy)
        
        assert len(manager.strategies) == 1
        assert manager.strategies[0] == strategy
    
    def test_register_multiple_strategies(self):
        """Test registering multiple strategies."""
        manager = StrategyManager()
        strategy1 = MockStrategy("Strategy1")
        strategy2 = MockStrategy("Strategy2")
        
        manager.register_strategy(strategy1)
        manager.register_strategy(strategy2)
        
        assert len(manager.strategies) == 2
    
    def test_register_duplicate_name_raises_error(self):
        """Test registering strategy with duplicate name raises error."""
        manager = StrategyManager()
        strategy1 = MockStrategy("Strategy1")
        strategy2 = MockStrategy("Strategy1")
        
        manager.register_strategy(strategy1)
        
        with pytest.raises(ValueError) as exc_info:
            manager.register_strategy(strategy2)
        
        assert "already registered" in str(exc_info.value)
    
    def test_register_invalid_type_raises_error(self):
        """Test registering non-Strategy object raises error."""
        manager = StrategyManager()
        
        with pytest.raises(TypeError) as exc_info:
            manager.register_strategy("not a strategy")
        
        assert "Expected Strategy instance" in str(exc_info.value)
    
    def test_unregister_strategy(self):
        """Test unregistering a strategy."""
        manager = StrategyManager()
        strategy = MockStrategy("Strategy1")
        
        manager.register_strategy(strategy)
        result = manager.unregister_strategy("Strategy1")
        
        assert result is True
        assert len(manager.strategies) == 0
    
    def test_unregister_nonexistent_strategy(self):
        """Test unregistering non-existent strategy returns False."""
        manager = StrategyManager()
        
        result = manager.unregister_strategy("NonExistent")
        
        assert result is False
    
    def test_get_strategy(self):
        """Test retrieving a strategy by name."""
        manager = StrategyManager()
        strategy = MockStrategy("Strategy1")
        
        manager.register_strategy(strategy)
        retrieved = manager.get_strategy("Strategy1")
        
        assert retrieved == strategy
    
    def test_get_nonexistent_strategy_raises_error(self):
        """Test retrieving non-existent strategy raises error."""
        manager = StrategyManager()
        
        with pytest.raises(ValueError) as exc_info:
            manager.get_strategy("NonExistent")
        
        assert "not found" in str(exc_info.value)
    
    def test_run_strategies(self):
        """Test running all strategies with market data."""
        manager = StrategyManager()
        strategy1 = MockStrategy("Strategy1")
        strategy2 = MockStrategy("Strategy2")
        
        manager.register_strategy(strategy1)
        manager.register_strategy(strategy2)
        
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000
        )
        
        manager.run_strategies(market_data)
        
        assert len(strategy1.data_received) == 1
        assert len(strategy2.data_received) == 1
        assert strategy1.data_received[0] == market_data
        assert strategy2.data_received[0] == market_data
    
    def test_get_signals(self):
        """Test collecting signals from all strategies."""
        manager = StrategyManager()
        strategy1 = MockStrategy("Strategy1")
        strategy2 = MockStrategy("Strategy2")
        
        signal1 = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        signal2 = Signal(
            symbol="GOOGL",
            action=SignalAction.SELL,
            quantity=50,
            order_type=OrderType.MARKET
        )
        
        strategy1.signals_to_return = [signal1]
        strategy2.signals_to_return = [signal2]
        
        manager.register_strategy(strategy1)
        manager.register_strategy(strategy2)
        
        signals = manager.get_signals()
        
        assert len(signals) == 2
        assert signal1 in signals
        assert signal2 in signals
    
    def test_notify_order_filled(self):
        """Test notifying all strategies of filled order."""
        manager = StrategyManager()
        strategy1 = MockStrategy("Strategy1")
        strategy2 = MockStrategy("Strategy2")
        
        manager.register_strategy(strategy1)
        manager.register_strategy(strategy2)
        
        order = Order(
            order_id="123",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            status=OrderStatusEnum.FILLED,
            filled_quantity=100,
            filled_price=150.0
        )
        
        manager.notify_order_filled(order)
        
        assert len(strategy1.orders_filled) == 1
        assert len(strategy2.orders_filled) == 1
        assert strategy1.orders_filled[0] == order
        assert strategy2.orders_filled[0] == order
    
    def test_clear_strategies(self):
        """Test clearing all strategies."""
        manager = StrategyManager()
        strategy1 = MockStrategy("Strategy1")
        strategy2 = MockStrategy("Strategy2")
        
        manager.register_strategy(strategy1)
        manager.register_strategy(strategy2)
        
        manager.clear()
        
        assert len(manager.strategies) == 0


class TestMovingAverageCrossover:
    """Test MovingAverageCrossover strategy."""
    
    def test_initialization(self):
        """Test strategy initialization with default parameters."""
        strategy = MovingAverageCrossover()
        
        assert strategy.name == "MovingAverageCrossover"
        assert strategy.fast_period == 10
        assert strategy.slow_period == 30
        assert strategy.quantity == 100
    
    def test_initialization_with_custom_params(self):
        """Test strategy initialization with custom parameters."""
        strategy = MovingAverageCrossover(
            name="CustomMA",
            fast_period=5,
            slow_period=20,
            quantity=50
        )
        
        assert strategy.name == "CustomMA"
        assert strategy.fast_period == 5
        assert strategy.slow_period == 20
        assert strategy.quantity == 50
    
    def test_invalid_periods_raises_error(self):
        """Test that invalid period configuration raises error."""
        with pytest.raises(ValueError) as exc_info:
            MovingAverageCrossover(fast_period=30, slow_period=10)
        
        assert "must be less than" in str(exc_info.value)
    
    def test_fast_period_too_small_raises_error(self):
        """Test that fast period < 2 raises error."""
        with pytest.raises(ValueError) as exc_info:
            MovingAverageCrossover(fast_period=1, slow_period=10)
        
        assert "must be at least 2" in str(exc_info.value)
    
    def test_invalid_quantity_raises_error(self):
        """Test that invalid quantity raises error."""
        with pytest.raises(ValueError) as exc_info:
            MovingAverageCrossover(quantity=0)
        
        assert "must be positive" in str(exc_info.value)
    
    def test_on_data_builds_price_history(self):
        """Test that on_data builds price history."""
        strategy = MovingAverageCrossover(fast_period=3, slow_period=5)
        
        # Feed market data
        for i in range(5):
            market_data = MarketData(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=150.0 + i,
                high=152.0 + i,
                low=149.0 + i,
                close=151.0 + i,
                volume=1000000
            )
            strategy.on_data(market_data)
        
        # Check price history
        assert "AAPL" in strategy._price_history
        assert len(strategy._price_history["AAPL"]) == 5
    
    def test_generate_signals_insufficient_data(self):
        """Test that no signals generated with insufficient data."""
        strategy = MovingAverageCrossover(fast_period=3, slow_period=5)
        
        # Feed only 3 data points (need 5 for slow MA)
        for i in range(3):
            market_data = MarketData(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=150.0,
                high=152.0,
                low=149.0,
                close=151.0,
                volume=1000000
            )
            strategy.on_data(market_data)
        
        signals = strategy.generate_signals()
        
        assert len(signals) == 0
    
    def test_generate_buy_signal_on_crossover(self):
        """Test BUY signal generation on bullish crossover."""
        strategy = MovingAverageCrossover(fast_period=2, slow_period=3, quantity=100)
        
        # Create price sequence that causes bullish crossover
        # Need slow_period + 1 = 4 data points
        # Prices: 105, 100, 100, 105
        # At point 3 (prev): fast MA (2) = 100, slow MA (3) = 101.67
        # At point 4 (curr): fast MA (2) = 102.5, slow MA (3) = 101.67 (crossover!)
        prices = [105.0, 100.0, 100.0, 105.0]
        
        for price in prices:
            market_data = MarketData(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            )
            strategy.on_data(market_data)
        
        signals = strategy.generate_signals()
        
        # Should generate BUY signal
        assert len(signals) == 1
        assert signals[0].symbol == "AAPL"
        assert signals[0].action == SignalAction.BUY
        assert signals[0].quantity == 100
        assert signals[0].order_type == OrderType.MARKET
    
    def test_generate_sell_signal_on_crossover(self):
        """Test SELL signal generation on bearish crossover."""
        strategy = MovingAverageCrossover(fast_period=2, slow_period=3, quantity=100)
        
        # First create a position with bullish crossover
        prices = [105.0, 100.0, 100.0, 105.0]
        for price in prices:
            market_data = MarketData(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            )
            strategy.on_data(market_data)
        
        # Generate BUY signal and simulate position
        signals = strategy.generate_signals()
        assert len(signals) == 1
        
        # Now create bearish crossover
        # Continue with prices that cause fast MA to cross below slow MA
        prices = [105.0, 105.0, 95.0]
        for price in prices:
            market_data = MarketData(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            )
            strategy.on_data(market_data)
        
        signals = strategy.generate_signals()
        
        # Should generate SELL signal
        assert len(signals) == 1
        assert signals[0].symbol == "AAPL"
        assert signals[0].action == SignalAction.SELL
        assert signals[0].quantity == 100
    
    def test_no_duplicate_buy_signals(self):
        """Test that duplicate BUY signals are not generated."""
        strategy = MovingAverageCrossover(fast_period=2, slow_period=3, quantity=100)
        
        # Create bullish crossover
        prices = [105.0, 100.0, 100.0, 105.0]
        for price in prices:
            market_data = MarketData(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            )
            strategy.on_data(market_data)
        
        # First signal generation
        signals1 = strategy.generate_signals()
        assert len(signals1) == 1
        
        # Add more data (still bullish)
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=107.0,
            high=108.0,
            low=106.0,
            close=107.0,
            volume=1000000
        )
        strategy.on_data(market_data)
        
        # Second signal generation should not produce another BUY
        signals2 = strategy.generate_signals()
        assert len(signals2) == 0
    
    def test_on_order_filled_updates_position(self):
        """Test that on_order_filled updates position tracking."""
        strategy = MovingAverageCrossover()
        
        # Simulate BUY order filled
        buy_order = Order(
            order_id="123",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            status=OrderStatusEnum.FILLED,
            filled_quantity=100,
            filled_price=150.0
        )
        
        strategy.on_order_filled(buy_order)
        
        assert strategy.get_current_position("AAPL") == 100
        
        # Simulate SELL order filled
        sell_order = Order(
            order_id="124",
            symbol="AAPL",
            action=OrderAction.SELL,
            quantity=100,
            order_type=OrderType.MARKET,
            status=OrderStatusEnum.FILLED,
            filled_quantity=100,
            filled_price=155.0
        )
        
        strategy.on_order_filled(sell_order)
        
        assert strategy.get_current_position("AAPL") == 0
    
    def test_multiple_symbols(self):
        """Test strategy handles multiple symbols independently."""
        strategy = MovingAverageCrossover(fast_period=2, slow_period=3, quantity=100)
        
        # Feed data for AAPL
        for price in [105.0, 100.0, 100.0, 105.0]:
            market_data = MarketData(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            )
            strategy.on_data(market_data)
        
        # Feed data for GOOGL
        for price in [210.0, 200.0, 200.0, 210.0]:
            market_data = MarketData(
                symbol="GOOGL",
                timestamp=datetime.now(),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            )
            strategy.on_data(market_data)
        
        signals = strategy.generate_signals()
        
        # Should generate BUY signals for both symbols
        assert len(signals) == 2
        symbols = [s.symbol for s in signals]
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
    
    def test_reset_positions(self):
        """Test resetting position tracking."""
        strategy = MovingAverageCrossover()
        
        strategy._positions["AAPL"] = 100
        strategy._positions["GOOGL"] = 50
        
        strategy.reset_positions()
        
        assert len(strategy._positions) == 0
    
    def test_reset_history(self):
        """Test resetting price history."""
        strategy = MovingAverageCrossover()
        
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000
        )
        strategy.on_data(market_data)
        
        assert len(strategy._price_history) == 1
        
        strategy.reset_history()
        
        assert len(strategy._price_history) == 0
