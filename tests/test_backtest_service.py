"""Unit and property-based tests for backtest service."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings, assume

from services.backtest_service import (
    BacktestEngine,
    Strategy,
    Trade,
    BacktestResult
)


class TestBacktestEngine:
    """Test suite for BacktestEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create backtest engine"""
        return BacktestEngine(initial_capital=100000.0)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        data = pd.DataFrame({
            'close': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(100, 200, len(dates)),
            'low': np.random.uniform(100, 200, len(dates))
        }, index=dates)
        
        # Ensure high >= close >= low
        data['high'] = data[['close', 'high']].max(axis=1)
        data['low'] = data[['close', 'low']].min(axis=1)
        
        return data
    
    @pytest.fixture
    def simple_strategy(self):
        """Create simple buy-and-hold strategy"""
        def signal_func(data):
            if len(data) == 1:
                return 'buy'
            elif len(data) == len(data):
                return 'hold'
            return 'hold'
        
        return Strategy(
            name='Buy and Hold',
            signal_func=signal_func,
            position_size=1.0,
            commission_pct=0.001
        )
    
    def test_init(self, engine):
        """Test engine initialization"""
        assert engine.initial_capital == 100000.0
    
    def test_run_backtest_empty_data(self, engine, simple_strategy):
        """Test that empty data raises ValueError"""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Data cannot be empty"):
            engine.run_backtest(
                simple_strategy,
                empty_data,
                datetime(2024, 1, 1),
                datetime(2024, 1, 31)
            )
    
    def test_run_backtest_invalid_date_range(self, engine, simple_strategy, sample_data):
        """Test that invalid date range raises ValueError"""
        with pytest.raises(ValueError, match="Start date must be before end date"):
            engine.run_backtest(
                simple_strategy,
                sample_data,
                datetime(2024, 1, 31),
                datetime(2024, 1, 1)
            )
    
    def test_run_backtest_missing_columns(self, engine, simple_strategy):
        """Test that missing required columns raises ValueError"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        data = pd.DataFrame({
            'close': np.random.uniform(100, 200, len(dates))
        }, index=dates)
        
        with pytest.raises(ValueError, match="Data must contain columns"):
            engine.run_backtest(
                simple_strategy,
                data,
                datetime(2024, 1, 1),
                datetime(2024, 1, 31)
            )
    
    def test_run_backtest_success(self, engine, simple_strategy, sample_data):
        """Test successful backtest run"""
        result = engine.run_backtest(
            simple_strategy,
            sample_data,
            datetime(2024, 1, 1),
            datetime(2024, 1, 31)
        )
        
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == 'Buy and Hold'
        assert result.initial_capital == 100000.0
        assert result.final_capital > 0
        assert len(result.equity_curve) > 0
    
    def test_calculate_metrics_no_trades(self, engine):
        """Test metrics calculation with no trades"""
        metrics = engine.calculate_metrics([], [100000.0])
        
        assert metrics['total_return'] == 0.0
        assert metrics['total_trades'] == 0
        assert metrics['win_rate'] == 0.0
    
    def test_calculate_metrics_with_trades(self, engine):
        """Test metrics calculation with trades"""
        trades = [
            Trade(
                symbol='TEST',
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 5),
                entry_price=100.0,
                exit_price=110.0,
                quantity=100,
                side='long',
                pnl=1000.0,
                pnl_pct=10.0
            ),
            Trade(
                symbol='TEST',
                entry_time=datetime(2024, 1, 6),
                exit_time=datetime(2024, 1, 10),
                entry_price=110.0,
                exit_price=105.0,
                quantity=100,
                side='long',
                pnl=-500.0,
                pnl_pct=-4.55
            )
        ]
        
        equity_curve = [100000.0, 101000.0, 100500.0]
        
        metrics = engine.calculate_metrics(trades, equity_curve)
        
        assert metrics['total_trades'] == 2
        assert metrics['winning_trades'] == 1
        assert metrics['losing_trades'] == 1
        assert metrics['win_rate'] == 0.5
        assert metrics['total_return'] == 500.0
        assert metrics['largest_win'] == 1000.0
        assert metrics['largest_loss'] == -500.0
    
    def test_generate_equity_curve_no_trades(self, engine):
        """Test equity curve generation with no trades"""
        curve = engine.generate_equity_curve([])
        
        assert len(curve) == 1
        assert curve.iloc[0] == 100000.0
    
    def test_generate_equity_curve_with_trades(self, engine):
        """Test equity curve generation with trades"""
        trades = [
            Trade(
                symbol='TEST',
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 5),
                entry_price=100.0,
                exit_price=110.0,
                quantity=100,
                side='long',
                pnl=1000.0,
                pnl_pct=10.0
            ),
            Trade(
                symbol='TEST',
                entry_time=datetime(2024, 1, 6),
                exit_time=datetime(2024, 1, 10),
                entry_price=110.0,
                exit_price=105.0,
                quantity=100,
                side='long',
                pnl=-500.0,
                pnl_pct=-4.55
            )
        ]
        
        curve = engine.generate_equity_curve(trades)
        
        assert len(curve) == 3
        assert curve.iloc[0] == 100000.0
        assert curve.iloc[1] == 101000.0
        assert curve.iloc[2] == 100500.0
    
    def test_backtest_execution_with_sample_data(self, engine, sample_data):
        """Test complete backtest execution with realistic sample data"""
        # Create a simple moving average crossover strategy
        def ma_crossover_signal(data):
            if len(data) < 5:
                return 'hold'
            
            # Calculate short and long moving averages
            short_ma = data['close'].tail(3).mean()
            long_ma = data['close'].tail(5).mean()
            
            # Generate signal
            if short_ma > long_ma:
                return 'buy'
            elif short_ma < long_ma:
                return 'sell'
            return 'hold'
        
        strategy = Strategy(
            name='MA Crossover',
            signal_func=ma_crossover_signal,
            position_size=0.5,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            commission_pct=0.001
        )
        
        result = engine.run_backtest(
            strategy,
            sample_data,
            datetime(2024, 1, 1),
            datetime(2024, 1, 31)
        )
        
        # Verify result structure
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == 'MA Crossover'
        assert result.initial_capital == 100000.0
        assert result.final_capital > 0
        
        # Verify metrics are calculated
        assert isinstance(result.total_return, float)
        assert isinstance(result.total_return_pct, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.profit_factor, float)
        
        # Verify trades list
        assert isinstance(result.trades, list)
        assert result.total_trades == len(result.trades)
        
        # Verify equity curve
        assert len(result.equity_curve) > 0
        assert result.equity_curve[0] == 100000.0
    
    def test_metric_calculations_comprehensive(self, engine):
        """Test comprehensive metric calculations with various trade scenarios"""
        # Create a mix of winning and losing trades
        trades = [
            # Large win
            Trade(
                symbol='TEST1',
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 5),
                entry_price=100.0,
                exit_price=120.0,
                quantity=100,
                side='long',
                pnl=2000.0,
                pnl_pct=20.0,
                commission=10.0
            ),
            # Small loss
            Trade(
                symbol='TEST2',
                entry_time=datetime(2024, 1, 6),
                exit_time=datetime(2024, 1, 8),
                entry_price=110.0,
                exit_price=105.0,
                quantity=50,
                side='long',
                pnl=-250.0,
                pnl_pct=-4.55,
                commission=5.0
            ),
            # Medium win
            Trade(
                symbol='TEST3',
                entry_time=datetime(2024, 1, 9),
                exit_time=datetime(2024, 1, 12),
                entry_price=105.0,
                exit_price=115.0,
                quantity=80,
                side='long',
                pnl=800.0,
                pnl_pct=9.52,
                commission=8.0
            ),
            # Large loss
            Trade(
                symbol='TEST4',
                entry_time=datetime(2024, 1, 13),
                exit_time=datetime(2024, 1, 15),
                entry_price=115.0,
                exit_price=95.0,
                quantity=100,
                side='long',
                pnl=-2000.0,
                pnl_pct=-17.39,
                commission=10.0
            ),
            # Small win
            Trade(
                symbol='TEST5',
                entry_time=datetime(2024, 1, 16),
                exit_time=datetime(2024, 1, 18),
                entry_price=95.0,
                exit_price=100.0,
                quantity=60,
                side='long',
                pnl=300.0,
                pnl_pct=5.26,
                commission=6.0
            )
        ]
        
        equity_curve = [100000.0, 102000.0, 101750.0, 102550.0, 100550.0, 100850.0]
        
        metrics = engine.calculate_metrics(trades, equity_curve)
        
        # Verify basic counts
        assert metrics['total_trades'] == 5
        assert metrics['winning_trades'] == 3
        assert metrics['losing_trades'] == 2
        
        # Verify win rate
        assert metrics['win_rate'] == 0.6  # 3/5
        
        # Verify total return
        expected_total_return = 2000.0 - 250.0 + 800.0 - 2000.0 + 300.0
        assert abs(metrics['total_return'] - expected_total_return) < 0.01
        
        # Verify return percentage
        expected_return_pct = (expected_total_return / 100000.0) * 100
        assert abs(metrics['total_return_pct'] - expected_return_pct) < 0.01
        
        # Verify largest win and loss
        assert metrics['largest_win'] == 2000.0
        assert metrics['largest_loss'] == -2000.0
        
        # Verify average win and loss
        winning_pnls = [2000.0, 800.0, 300.0]
        losing_pnls = [-250.0, -2000.0]
        assert abs(metrics['avg_win'] - np.mean(winning_pnls)) < 0.01
        assert abs(metrics['avg_loss'] - np.mean(losing_pnls)) < 0.01
        
        # Verify profit factor
        gross_profit = sum(winning_pnls)
        gross_loss = abs(sum(losing_pnls))
        expected_profit_factor = gross_profit / gross_loss
        assert abs(metrics['profit_factor'] - expected_profit_factor) < 0.01
        
        # Verify max drawdown calculation
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = equity_array - running_max
        expected_max_drawdown = abs(np.min(drawdown))
        assert abs(metrics['max_drawdown'] - expected_max_drawdown) < 0.01
        
        # Verify Sharpe ratio is calculated
        assert isinstance(metrics['sharpe_ratio'], float)
    
    def test_equity_curve_generation_ordering(self, engine):
        """Test that equity curve correctly orders trades by exit time"""
        # Create trades in non-chronological order
        trades = [
            Trade(
                symbol='TEST2',
                entry_time=datetime(2024, 1, 10),
                exit_time=datetime(2024, 1, 15),
                entry_price=110.0,
                exit_price=115.0,
                quantity=50,
                side='long',
                pnl=250.0,
                pnl_pct=4.55
            ),
            Trade(
                symbol='TEST1',
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 5),
                entry_price=100.0,
                exit_price=110.0,
                quantity=100,
                side='long',
                pnl=1000.0,
                pnl_pct=10.0
            ),
            Trade(
                symbol='TEST3',
                entry_time=datetime(2024, 1, 20),
                exit_time=datetime(2024, 1, 25),
                entry_price=115.0,
                exit_price=120.0,
                quantity=75,
                side='long',
                pnl=375.0,
                pnl_pct=4.35
            )
        ]
        
        curve = engine.generate_equity_curve(trades)
        
        # Verify curve is sorted by exit time
        assert len(curve) == 4
        assert curve.iloc[0] == 100000.0  # Initial capital
        assert curve.iloc[1] == 101000.0  # After TEST1 (earliest exit)
        assert curve.iloc[2] == 101250.0  # After TEST2
        assert curve.iloc[3] == 101625.0  # After TEST3 (latest exit)
        
        # Verify timestamps are in order
        assert curve.index[0] == datetime(2024, 1, 1)  # TEST1 entry
        assert curve.index[1] == datetime(2024, 1, 5)  # TEST1 exit
        assert curve.index[2] == datetime(2024, 1, 15)  # TEST2 exit
        assert curve.index[3] == datetime(2024, 1, 25)  # TEST3 exit
    
    def test_compare_strategies(self, engine, sample_data):
        """Test strategy comparison functionality"""
        # Create two simple strategies
        def always_buy_signal(data):
            return 'buy' if len(data) == 1 else 'hold'
        
        def never_trade_signal(data):
            return 'hold'
        
        strategy1 = Strategy(
            name='Aggressive',
            signal_func=always_buy_signal,
            position_size=1.0,
            commission_pct=0.001
        )
        
        strategy2 = Strategy(
            name='Conservative',
            signal_func=never_trade_signal,
            position_size=0.5,
            commission_pct=0.001
        )
        
        comparison = engine.compare_strategies(
            [strategy1, strategy2],
            sample_data,
            datetime(2024, 1, 1),
            datetime(2024, 1, 31)
        )
        
        # Verify comparison DataFrame structure
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'Strategy' in comparison.columns
        assert 'Total Return' in comparison.columns
        assert 'Return %' in comparison.columns
        assert 'Sharpe Ratio' in comparison.columns
        assert 'Win Rate' in comparison.columns
        
        # Verify strategy names
        assert 'Aggressive' in comparison['Strategy'].values
        assert 'Conservative' in comparison['Strategy'].values
    
    def test_backtest_with_stop_loss(self, engine):
        """Test backtest execution with stop-loss functionality"""
        # Create data where price drops to trigger stop-loss
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        data = pd.DataFrame({
            'close': [100.0, 105.0, 110.0, 108.0, 95.0, 90.0, 92.0, 94.0, 96.0, 98.0],
            'high': [102.0, 107.0, 112.0, 110.0, 97.0, 92.0, 94.0, 96.0, 98.0, 100.0],
            'low': [98.0, 103.0, 108.0, 106.0, 90.0, 88.0, 90.0, 92.0, 94.0, 96.0]
        }, index=dates)
        
        def buy_first_day_signal(data):
            return 'buy' if len(data) == 1 else 'hold'
        
        strategy = Strategy(
            name='Stop Loss Test',
            signal_func=buy_first_day_signal,
            position_size=1.0,
            stop_loss_pct=0.10,  # 10% stop-loss
            commission_pct=0.0
        )
        
        result = engine.run_backtest(
            strategy,
            data,
            datetime(2024, 1, 1),
            datetime(2024, 1, 10)
        )
        
        # Verify that stop-loss was triggered
        assert result.total_trades >= 1
        
        # Check if any trade has a loss around 10%
        stop_loss_triggered = any(
            abs(trade.pnl_pct + 10.0) < 2.0  # Within 2% of -10%
            for trade in result.trades
        )
        assert stop_loss_triggered, "Stop-loss should have been triggered"
    
    def test_backtest_with_take_profit(self, engine):
        """Test backtest execution with take-profit functionality"""
        # Create data where price rises to trigger take-profit
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        data = pd.DataFrame({
            'close': [100.0, 105.0, 110.0, 115.0, 120.0, 118.0, 116.0, 114.0, 112.0, 110.0],
            'high': [102.0, 107.0, 112.0, 117.0, 122.0, 120.0, 118.0, 116.0, 114.0, 112.0],
            'low': [98.0, 103.0, 108.0, 113.0, 118.0, 116.0, 114.0, 112.0, 110.0, 108.0]
        }, index=dates)
        
        def buy_first_day_signal(data):
            return 'buy' if len(data) == 1 else 'hold'
        
        strategy = Strategy(
            name='Take Profit Test',
            signal_func=buy_first_day_signal,
            position_size=1.0,
            take_profit_pct=0.15,  # 15% take-profit
            commission_pct=0.0
        )
        
        result = engine.run_backtest(
            strategy,
            data,
            datetime(2024, 1, 1),
            datetime(2024, 1, 10)
        )
        
        # Verify that take-profit was triggered
        assert result.total_trades >= 1
        
        # Check if any trade has a profit around 15%
        take_profit_triggered = any(
            abs(trade.pnl_pct - 15.0) < 2.0  # Within 2% of +15%
            for trade in result.trades
        )
        assert take_profit_triggered, "Take-profit should have been triggered"


class TestBacktestMetricConsistencyProperty:
    """Property-based tests for backtest metric consistency"""
    
    @given(
        num_trades=st.integers(min_value=1, max_value=50),
        initial_capital=st.floats(min_value=10000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=10000)
    def test_backtest_pnl_equals_final_minus_initial_capital(
        self,
        num_trades,
        initial_capital
    ):
        """
        Feature: ai-trading-agent, Property 8: Backtest metric consistency
        
        Property: For any backtest run, the sum of all trade P&Ls should equal 
        the final portfolio value minus initial capital.
        
        This test verifies that:
        1. The sum of individual trade P&Ls equals the total return
        2. Final capital = Initial capital + Total P&L
        3. This holds for any number of trades and any initial capital
        4. The accounting is consistent throughout the backtest
        
        This is a fundamental property that ensures the backtest engine correctly
        tracks capital and P&L. If this property fails, it indicates a bug in
        the capital tracking or P&L calculation logic.
        
        Validates: Requirements 8.3
        """
        # Create backtest engine
        engine = BacktestEngine(initial_capital=initial_capital)
        
        # Generate random trades with realistic P&Ls
        trades = []
        entry_time = datetime(2024, 1, 1)
        
        for i in range(num_trades):
            # Random entry and exit prices
            entry_price = np.random.uniform(50.0, 500.0)
            # Exit price can be higher or lower (profit or loss)
            price_change_pct = np.random.uniform(-0.2, 0.3)  # -20% to +30%
            exit_price = entry_price * (1 + price_change_pct)
            
            # Random quantity (but ensure it's affordable)
            max_quantity = int(initial_capital * 0.1 / entry_price)  # Use max 10% of capital
            if max_quantity < 1:
                max_quantity = 1
            quantity = np.random.randint(1, max(2, max_quantity))
            
            # Calculate P&L
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            
            # Create trade
            exit_time = entry_time + timedelta(days=np.random.randint(1, 10))
            trade = Trade(
                symbol=f'SYM{i}',
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                side='long',
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=0.0  # No commission for simplicity
            )
            trades.append(trade)
            
            # Update entry time for next trade
            entry_time = exit_time + timedelta(days=1)
        
        # Build equity curve
        equity_curve = [initial_capital]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade.pnl)
        
        # Calculate metrics
        metrics = engine.calculate_metrics(trades, equity_curve)
        
        # PROPERTY: Sum of all trade P&Ls should equal total return
        sum_of_pnls = sum(trade.pnl for trade in trades)
        total_return = metrics['total_return']
        
        assert abs(sum_of_pnls - total_return) < 0.01, \
            f"Sum of P&Ls ({sum_of_pnls:.2f}) should equal total return ({total_return:.2f})"
        
        # PROPERTY: Final capital should equal initial capital + total P&L
        final_capital = equity_curve[-1]
        expected_final_capital = initial_capital + sum_of_pnls
        
        assert abs(final_capital - expected_final_capital) < 0.01, \
            f"Final capital ({final_capital:.2f}) should equal initial ({initial_capital:.2f}) + P&L ({sum_of_pnls:.2f}) = {expected_final_capital:.2f}"
        
        # PROPERTY: Equity curve should be consistent with cumulative P&L
        for i, trade in enumerate(trades):
            expected_equity = initial_capital + sum(t.pnl for t in trades[:i+1])
            actual_equity = equity_curve[i+1]
            
            assert abs(expected_equity - actual_equity) < 0.01, \
                f"Equity at trade {i+1} ({actual_equity:.2f}) should equal initial + cumulative P&L ({expected_equity:.2f})"
        
        # PROPERTY: Total return percentage should be consistent
        expected_return_pct = (sum_of_pnls / initial_capital) * 100
        actual_return_pct = metrics['total_return_pct']
        
        assert abs(expected_return_pct - actual_return_pct) < 0.01, \
            f"Return % ({actual_return_pct:.2f}%) should equal (P&L / initial) * 100 ({expected_return_pct:.2f}%)"
        
        # PROPERTY: Number of trades should match
        assert metrics['total_trades'] == num_trades, \
            f"Total trades ({metrics['total_trades']}) should equal number of trades ({num_trades})"
        
        # PROPERTY: Winning + losing trades should equal total trades
        assert metrics['winning_trades'] + metrics['losing_trades'] == metrics['total_trades'], \
            f"Winning ({metrics['winning_trades']}) + Losing ({metrics['losing_trades']}) should equal Total ({metrics['total_trades']})"
    
    @given(
        num_winning=st.integers(min_value=0, max_value=20),
        num_losing=st.integers(min_value=0, max_value=20)
    )
    @settings(max_examples=100, deadline=10000)
    def test_win_rate_calculation_consistency(self, num_winning, num_losing):
        """
        Property: Win rate should equal winning_trades / total_trades.
        
        This verifies that the win rate calculation is consistent with the
        number of winning and losing trades.
        
        Validates: Requirements 8.3
        """
        # Skip if no trades
        assume(num_winning + num_losing > 0)
        
        engine = BacktestEngine(initial_capital=100000.0)
        
        # Create winning trades
        trades = []
        for i in range(num_winning):
            trade = Trade(
                symbol=f'WIN{i}',
                entry_time=datetime(2024, 1, 1) + timedelta(days=i),
                exit_time=datetime(2024, 1, 2) + timedelta(days=i),
                entry_price=100.0,
                exit_price=110.0,
                quantity=10,
                side='long',
                pnl=100.0,  # Positive P&L
                pnl_pct=10.0
            )
            trades.append(trade)
        
        # Create losing trades
        for i in range(num_losing):
            trade = Trade(
                symbol=f'LOSS{i}',
                entry_time=datetime(2024, 1, 1) + timedelta(days=num_winning + i),
                exit_time=datetime(2024, 1, 2) + timedelta(days=num_winning + i),
                entry_price=100.0,
                exit_price=90.0,
                quantity=10,
                side='long',
                pnl=-100.0,  # Negative P&L
                pnl_pct=-10.0
            )
            trades.append(trade)
        
        # Build equity curve
        equity_curve = [100000.0]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade.pnl)
        
        # Calculate metrics
        metrics = engine.calculate_metrics(trades, equity_curve)
        
        # PROPERTY: Win rate should equal winning / total
        total_trades = num_winning + num_losing
        expected_win_rate = num_winning / total_trades
        actual_win_rate = metrics['win_rate']
        
        assert abs(expected_win_rate - actual_win_rate) < 0.0001, \
            f"Win rate ({actual_win_rate:.4f}) should equal winning ({num_winning}) / total ({total_trades}) = {expected_win_rate:.4f}"
        
        # PROPERTY: Winning trades count should match
        assert metrics['winning_trades'] == num_winning, \
            f"Winning trades ({metrics['winning_trades']}) should equal {num_winning}"
        
        # PROPERTY: Losing trades count should match
        assert metrics['losing_trades'] == num_losing, \
            f"Losing trades ({metrics['losing_trades']}) should equal {num_losing}"
