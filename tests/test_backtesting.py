"""Unit tests for backtesting module."""
import pytest
from datetime import date, datetime
from typing import List
import pandas as pd

from src.backtesting import Backtester, BacktestResult
from src.strategies.base import Strategy
from src.models.signal import Signal, SignalAction
from src.models.order import OrderType
from src.models.market_data import MarketData
from src.data.base import MarketDataProvider


class MockDataProvider(MarketDataProvider):
    """Mock data provider for testing."""
    
    def __init__(self):
        self.historical_data = {}
    
    def get_quote(self, symbol: str):
        """Not used in backtesting."""
        pass
    
    def get_historical_data(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Return mock historical data."""
        if symbol in self.historical_data:
            return self.historical_data[symbol]
        return pd.DataFrame()
    
    def set_historical_data(self, symbol: str, data: pd.DataFrame):
        """Set mock historical data for a symbol."""
        self.historical_data[symbol] = data


class SimpleStrategy(Strategy):
    """Simple test strategy that generates signals based on price threshold."""
    
    def __init__(self, name: str = "SimpleStrategy", buy_threshold: float = 100.0, sell_threshold: float = 110.0):
        super().__init__(name)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.current_data = {}
        self.position = 0
    
    def on_data(self, market_data: MarketData) -> None:
        """Store current market data."""
        self.current_data[market_data.symbol] = market_data
    
    def generate_signals(self) -> List[Signal]:
        """Generate signals based on price thresholds."""
        signals = []
        
        for symbol, data in self.current_data.items():
            # Buy if price below threshold and no position
            if data.close <= self.buy_threshold and self.position == 0:
                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    quantity=10,
                    order_type=OrderType.MARKET
                ))
            # Sell if price above threshold and have position
            elif data.close >= self.sell_threshold and self.position > 0:
                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.SELL,
                    quantity=10,
                    order_type=OrderType.MARKET
                ))
        
        return signals
    
    def on_order_filled(self, order):
        """Update position tracking."""
        from src.models.order import OrderAction
        if order.action == OrderAction.BUY:
            self.position += order.filled_quantity
        elif order.action == OrderAction.SELL:
            self.position -= order.filled_quantity


class TestBacktester:
    """Test Backtester class."""
    
    def test_backtester_initialization(self):
        """Test backtester initialization."""
        data_provider = MockDataProvider()
        backtester = Backtester(data_provider)
        
        assert backtester.data_provider == data_provider
        assert backtester.portfolio is None
        assert backtester.strategy is None
        assert len(backtester.trades) == 0
    
    def test_run_backtest_basic(self):
        """Test basic backtest execution."""
        # Create mock data provider
        data_provider = MockDataProvider()
        
        # Create simple price data
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        
        df = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': prices,
            'Volume': [1000000] * len(prices)
        }, index=dates)
        
        data_provider.set_historical_data('AAPL', df)
        
        # Create backtester and strategy
        backtester = Backtester(data_provider)
        strategy = SimpleStrategy(buy_threshold=100.0, sell_threshold=110.0)
        
        # Run backtest
        result = backtester.run_backtest(
            strategy=strategy,
            symbols=['AAPL'],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            initial_capital=10000.0
        )
        
        # Verify result
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "SimpleStrategy"
        assert result.symbols == ['AAPL']
        assert result.initial_capital == 10000.0
        assert result.final_value > 0
    
    def test_run_backtest_with_trades(self):
        """Test backtest with actual trades."""
        # Create mock data provider
        data_provider = MockDataProvider()
        
        # Create price data that triggers buy and sell
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        prices = [100, 105, 110, 115, 120]  # Price rises from 100 to 120
        
        df = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': prices,
            'Volume': [1000000] * len(prices)
        }, index=dates)
        
        data_provider.set_historical_data('AAPL', df)
        
        # Create backtester and strategy
        backtester = Backtester(data_provider)
        strategy = SimpleStrategy(buy_threshold=100.0, sell_threshold=110.0)
        
        # Run backtest
        result = backtester.run_backtest(
            strategy=strategy,
            symbols=['AAPL'],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
            initial_capital=10000.0
        )
        
        # Should have executed trades (buy at 100, sell at 110+)
        assert result.num_trades > 0
        assert len(result.trade_history) > 0
    
    def test_run_backtest_portfolio_tracking(self):
        """Test that portfolio is tracked during backtest."""
        # Create mock data provider
        data_provider = MockDataProvider()
        
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        prices = [100, 105, 110, 115, 120]
        
        df = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': prices,
            'Volume': [1000000] * len(prices)
        }, index=dates)
        
        data_provider.set_historical_data('AAPL', df)
        
        # Create backtester and strategy
        backtester = Backtester(data_provider)
        strategy = SimpleStrategy(buy_threshold=100.0, sell_threshold=110.0)
        
        # Run backtest
        result = backtester.run_backtest(
            strategy=strategy,
            symbols=['AAPL'],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
            initial_capital=10000.0
        )
        
        # Portfolio should be created and tracked
        assert backtester.portfolio is not None
        assert backtester.portfolio.initial_value == 10000.0
    
    def test_run_backtest_no_data_raises_error(self):
        """Test that backtest with no data raises error."""
        data_provider = MockDataProvider()
        backtester = Backtester(data_provider)
        strategy = SimpleStrategy()
        
        with pytest.raises(ValueError) as exc_info:
            backtester.run_backtest(
                strategy=strategy,
                symbols=['AAPL'],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 10),
                initial_capital=10000.0
            )
        
        assert "No historical data available" in str(exc_info.value)
    
    def test_simulate_order_fill_market_order(self):
        """Test simulating market order fill."""
        data_provider = MockDataProvider()
        backtester = Backtester(data_provider)
        
        from src.models.order import Order, OrderAction, OrderType, OrderStatusEnum
        
        order = Order(
            order_id="TEST_1",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            timestamp=datetime.now()
        )
        
        filled_order = backtester._simulate_order_fill(order, 150.0)
        
        assert filled_order.status == OrderStatusEnum.FILLED
        assert filled_order.filled_price == 150.0
        assert filled_order.filled_quantity == 10
    
    def test_simulate_order_fill_limit_order_buy(self):
        """Test simulating limit buy order fill."""
        data_provider = MockDataProvider()
        backtester = Backtester(data_provider)
        
        from src.models.order import Order, OrderAction, OrderType, OrderStatusEnum
        
        # Limit buy at 150, current price 145 - should fill
        order = Order(
            order_id="TEST_1",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
            timestamp=datetime.now()
        )
        
        filled_order = backtester._simulate_order_fill(order, 145.0)
        
        assert filled_order.status == OrderStatusEnum.FILLED
        assert filled_order.filled_price == 150.0
    
    def test_simulate_order_fill_limit_order_not_filled(self):
        """Test limit order not filled when price doesn't meet threshold."""
        data_provider = MockDataProvider()
        backtester = Backtester(data_provider)
        
        from src.models.order import Order, OrderAction, OrderType, OrderStatusEnum
        
        # Limit buy at 150, current price 155 - should not fill
        order = Order(
            order_id="TEST_1",
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
            timestamp=datetime.now()
        )
        
        filled_order = backtester._simulate_order_fill(order, 155.0)
        
        assert filled_order.status == OrderStatusEnum.PENDING


class TestBacktestResult:
    """Test BacktestResult class."""
    
    def create_sample_result(self) -> BacktestResult:
        """Create a sample backtest result for testing."""
        from src.models.order import Trade, OrderAction
        
        trades = [
            Trade(
                trade_id="1",
                symbol="AAPL",
                action=OrderAction.BUY,
                quantity=10,
                price=100.0,
                timestamp=datetime(2024, 1, 1),
                pnl=None
            ),
            Trade(
                trade_id="2",
                symbol="AAPL",
                action=OrderAction.SELL,
                quantity=10,
                price=110.0,
                timestamp=datetime(2024, 1, 5),
                pnl=100.0
            )
        ]
        
        return BacktestResult(
            strategy_name="TestStrategy",
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            initial_capital=10000.0,
            final_value=11000.0,
            total_return=10.0,
            sharpe_ratio=1.5,
            max_drawdown=5.0,
            num_trades=2,
            win_rate=100.0,
            trade_history=trades
        )
    
    def test_backtest_result_initialization(self):
        """Test backtest result initialization."""
        result = self.create_sample_result()
        
        assert result.strategy_name == "TestStrategy"
        assert result.symbols == ["AAPL"]
        assert result.initial_capital == 10000.0
        assert result.final_value == 11000.0
        assert result.total_return == 10.0
        assert result.num_trades == 2
    
    def test_generate_report(self):
        """Test generating formatted report."""
        result = self.create_sample_result()
        
        report = result.generate_report()
        
        assert isinstance(report, str)
        assert "BACKTEST RESULTS" in report
        assert "TestStrategy" in report
        assert "10,000.00" in report  # Comma-formatted
        assert "11,000.00" in report  # Comma-formatted
        assert "10.00%" in report
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = self.create_sample_result()
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['strategy_name'] == "TestStrategy"
        assert result_dict['initial_capital'] == 10000.0
        assert result_dict['final_value'] == 11000.0
        assert 'trade_history' in result_dict
        assert len(result_dict['trade_history']) == 2
    
    def test_save_to_file(self, tmp_path):
        """Test saving result to JSON file."""
        result = self.create_sample_result()
        
        filepath = tmp_path / "backtest_result.json"
        result.save_to_file(str(filepath))
        
        assert filepath.exists()
        
        # Verify file content
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert data['strategy_name'] == "TestStrategy"
        assert data['initial_capital'] == 10000.0
    
    def test_export_trades_csv(self, tmp_path):
        """Test exporting trades to CSV."""
        result = self.create_sample_result()
        
        filepath = tmp_path / "trades.csv"
        result.export_trades_csv(str(filepath))
        
        assert filepath.exists()
        
        # Verify CSV content
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 3  # Header + 2 trades
        assert "trade_id,symbol,action,quantity,price,timestamp,pnl" in lines[0]
        assert "AAPL" in lines[1]
        assert "AAPL" in lines[2]


class TestPerformanceMetrics:
    """Test performance metric calculations in backtest."""
    
    def test_backtest_calculates_returns(self):
        """Test that backtest calculates total return correctly."""
        data_provider = MockDataProvider()
        
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        prices = [100, 105, 110, 115, 120]
        
        df = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': prices,
            'Volume': [1000000] * len(prices)
        }, index=dates)
        
        data_provider.set_historical_data('AAPL', df)
        
        backtester = Backtester(data_provider)
        strategy = SimpleStrategy(buy_threshold=100.0, sell_threshold=110.0)
        
        result = backtester.run_backtest(
            strategy=strategy,
            symbols=['AAPL'],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
            initial_capital=10000.0
        )
        
        # Total return should be calculated
        assert result.total_return is not None
        assert isinstance(result.total_return, float)
    
    def test_backtest_calculates_sharpe_ratio(self):
        """Test that backtest calculates Sharpe ratio."""
        data_provider = MockDataProvider()
        
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        prices = [100 + i for i in range(10)]
        
        df = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': prices,
            'Volume': [1000000] * len(prices)
        }, index=dates)
        
        data_provider.set_historical_data('AAPL', df)
        
        backtester = Backtester(data_provider)
        strategy = SimpleStrategy(buy_threshold=100.0, sell_threshold=110.0)
        
        result = backtester.run_backtest(
            strategy=strategy,
            symbols=['AAPL'],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            initial_capital=10000.0
        )
        
        # Sharpe ratio should be calculated
        assert result.sharpe_ratio is not None
        assert isinstance(result.sharpe_ratio, float)
    
    def test_backtest_calculates_max_drawdown(self):
        """Test that backtest calculates maximum drawdown."""
        data_provider = MockDataProvider()
        
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        prices = [100, 105, 110, 105, 100, 95, 100, 105, 110, 115]
        
        df = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': prices,
            'Volume': [1000000] * len(prices)
        }, index=dates)
        
        data_provider.set_historical_data('AAPL', df)
        
        backtester = Backtester(data_provider)
        strategy = SimpleStrategy(buy_threshold=100.0, sell_threshold=110.0)
        
        result = backtester.run_backtest(
            strategy=strategy,
            symbols=['AAPL'],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            initial_capital=10000.0
        )
        
        # Max drawdown should be calculated
        assert result.max_drawdown is not None
        assert isinstance(result.max_drawdown, float)
        assert result.max_drawdown >= 0
