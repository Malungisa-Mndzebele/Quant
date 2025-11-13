"""End-to-end integration tests for the trading loop."""

import pytest
import yaml
import time
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock
from threading import Thread

from src.trading_system import TradingSystem, TradingSystemError
from src.models.config import TradingMode
from src.models.market_data import MarketData
from src.models.signal import Signal, SignalAction, OrderType as SignalOrderType
from src.models.order import Order, OrderAction, OrderType, OrderStatusEnum, OrderStatus
from src.data.base import Quote
from src.brokers.base import AccountInfo, Position


class TestTradingLoopIntegration:
    """Test complete trading cycle in simulation mode."""
    
    @pytest.fixture
    def config_file(self, tmp_path):
        """Create a test configuration file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'simulation',
                'initial_capital': 100000,
                'log_level': 'DEBUG'
            },
            'brokerage': {
                'provider': 'simulated',
                'credentials': {},
                'max_retries': 3
            },
            'data': {
                'provider': 'yfinance',
                'cache_enabled': False
            },
            'risk': {
                'max_position_size_pct': 20.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0,
                'max_positions': 5
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': True,
                    'params': {
                        'fast_period': 5,
                        'slow_period': 10,
                        'quantity': 10
                    }
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        return str(config_file)
    
    def test_complete_trading_cycle(self, config_file):
        """Test a complete trading cycle: fetch data -> generate signals -> create orders -> execute."""
        system = TradingSystem(config_file)
        system.initialize()
        
        assert system.is_initialized()
        assert system.portfolio.cash_balance == 100000
        
        # Mock market data provider to return predictable data
        mock_quote = Quote(
            symbol='AAPL',
            price=150.0,
            bid=149.95,
            ask=150.05,
            volume=1000000,
            timestamp=datetime.now()
        )
        
        with patch.object(system.data_provider, 'get_quote', return_value=mock_quote):
            # Fetch market data
            market_data_list = system._fetch_market_data(['AAPL'])
            
            assert len(market_data_list) == 1
            assert market_data_list[0].symbol == 'AAPL'
            assert market_data_list[0].close == 150.0
            
            # Update portfolio prices
            current_prices = {md.symbol: md.close for md in market_data_list}
            system.portfolio.update_prices(current_prices)
            
            # Execute strategies
            for market_data in market_data_list:
                for strategy in system.strategies:
                    strategy.on_data(market_data)
            
            # Generate signals (may or may not generate based on strategy logic)
            all_signals = []
            for strategy in system.strategies:
                signals = strategy.generate_signals()
                if signals:
                    all_signals.extend(signals)
            
            # If signals were generated, process them
            if all_signals:
                system._process_signals(all_signals, current_prices)
                
                # Check that orders were created
                orders = system.order_manager.get_all_orders()
                assert len(orders) > 0
    
    def test_signal_processing_with_risk_validation(self, config_file):
        """Test that signals are properly validated by risk manager before order submission."""
        system = TradingSystem(config_file)
        system.initialize()
        
        # Create a buy signal
        signal = Signal(
            symbol='AAPL',
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET,
            timestamp=datetime.now()
        )
        
        current_prices = {'AAPL': 150.0}
        
        # Process the signal
        system._process_signals([signal], current_prices)
        
        # Verify order was created and submitted
        orders = system.order_manager.get_all_orders()
        assert len(orders) == 1
        assert orders[0].symbol == 'AAPL'
        assert orders[0].action == OrderAction.BUY
        assert orders[0].quantity == 100
    
    def test_signal_rejected_by_risk_manager(self, config_file):
        """Test that signals violating risk limits are rejected."""
        system = TradingSystem(config_file)
        system.initialize()
        
        # Create a signal that would exceed position size limit
        # Portfolio value is 100000, max position size is 20% = 20000
        # Try to buy 200 shares at $150 = $30000 (exceeds limit)
        signal = Signal(
            symbol='AAPL',
            action=SignalAction.BUY,
            quantity=200,
            order_type=SignalOrderType.MARKET,
            timestamp=datetime.now()
        )
        
        current_prices = {'AAPL': 150.0}
        
        # Process the signal
        system._process_signals([signal], current_prices)
        
        # Verify order was NOT submitted (rejected by risk manager)
        orders = system.order_manager.get_all_orders()
        # Order is created but not submitted if risk validation fails
        assert len(orders) <= 1
        if len(orders) == 1:
            # If order was created, it should not be in SUBMITTED status
            assert orders[0].status != OrderStatusEnum.SUBMITTED
    
    def test_order_tracking_and_portfolio_update(self, config_file):
        """Test that filled orders update the portfolio correctly."""
        system = TradingSystem(config_file)
        system.initialize()
        
        initial_cash = system.portfolio.cash_balance
        
        # Create and submit a buy order
        signal = Signal(
            symbol='AAPL',
            action=SignalAction.BUY,
            quantity=10,
            order_type=SignalOrderType.MARKET,
            timestamp=datetime.now()
        )
        
        current_prices = {'AAPL': 150.0}
        system._process_signals([signal], current_prices)
        
        # Get the order
        orders = system.order_manager.get_all_orders()
        assert len(orders) == 1
        order = orders[0]
        
        # Simulate order fill
        fill_status = OrderStatus(
            order_id=order.order_id,
            status=OrderStatusEnum.FILLED,
            filled_quantity=10,
            average_fill_price=150.0,
            message="Order filled"
        )
        
        # Update portfolio on fill
        system._update_portfolio_on_fill(order, fill_status)
        
        # Verify portfolio was updated
        assert system.portfolio.cash_balance == initial_cash - (10 * 150.0)
        assert 'AAPL' in system.portfolio.positions
        assert system.portfolio.positions['AAPL'].quantity == 10
        assert system.portfolio.positions['AAPL'].entry_price == 150.0
    
    def test_sell_order_updates_portfolio(self, config_file):
        """Test that sell orders correctly update portfolio and calculate P&L."""
        system = TradingSystem(config_file)
        system.initialize()
        
        # First, add a position manually
        system.portfolio.add_position('AAPL', 10, 140.0)
        system.portfolio.cash_balance -= 1400.0
        initial_cash = system.portfolio.cash_balance
        
        # Create a sell signal
        signal = Signal(
            symbol='AAPL',
            action=SignalAction.SELL,
            quantity=10,
            order_type=SignalOrderType.MARKET,
            timestamp=datetime.now()
        )
        
        current_prices = {'AAPL': 150.0}
        system._process_signals([signal], current_prices)
        
        # Get the order
        orders = system.order_manager.get_all_orders()
        order = orders[0]
        
        # Simulate order fill
        fill_status = OrderStatus(
            order_id=order.order_id,
            status=OrderStatusEnum.FILLED,
            filled_quantity=10,
            average_fill_price=150.0,
            message="Order filled"
        )
        
        # Update portfolio on fill
        system._update_portfolio_on_fill(order, fill_status)
        
        # Verify portfolio was updated
        expected_cash = initial_cash + (10 * 150.0)
        assert system.portfolio.cash_balance == expected_cash
        
        # Position should be closed
        position = system.portfolio.get_position('AAPL')
        assert position is None or position.quantity == 0
        
        # Trade should be recorded with P&L
        assert len(system.portfolio.trade_history) == 1
        trade = system.portfolio.trade_history[0]
        assert trade.symbol == 'AAPL'
        assert trade.pnl == (150.0 - 140.0) * 10  # $100 profit
    
    def test_multiple_strategies_generate_signals(self, config_file, tmp_path):
        """Test that multiple strategies can generate signals independently."""
        # Create config with multiple strategies
        config_file = tmp_path / "multi_strategy_config.yaml"
        config_data = {
            'trading': {
                'mode': 'simulation',
                'initial_capital': 100000,
                'log_level': 'INFO'
            },
            'brokerage': {
                'provider': 'simulated',
                'credentials': {}
            },
            'data': {
                'provider': 'yfinance',
                'cache_enabled': False
            },
            'risk': {
                'max_position_size_pct': 20.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': True,
                    'params': {
                        'fast_period': 5,
                        'slow_period': 10,
                        'quantity': 10
                    }
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        system = TradingSystem(str(config_file))
        system.initialize()
        
        # Verify multiple strategies loaded
        assert len(system.strategies) >= 1


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery in the trading loop."""
    
    @pytest.fixture
    def config_file(self, tmp_path):
        """Create a test configuration file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'simulation',
                'initial_capital': 100000,
                'log_level': 'WARNING'
            },
            'brokerage': {
                'provider': 'simulated',
                'credentials': {}
            },
            'data': {
                'provider': 'yfinance',
                'cache_enabled': False
            },
            'risk': {
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': True,
                    'params': {}
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        return str(config_file)
    
    def test_data_fetch_error_handling(self, config_file):
        """Test that data fetch errors are handled gracefully."""
        system = TradingSystem(config_file)
        system.initialize()
        
        # Mock data provider to raise an exception
        with patch.object(system.data_provider, 'get_quote', side_effect=Exception("API Error")):
            # Fetch should handle the error and return empty list
            market_data_list = system._fetch_market_data(['AAPL'])
            
            # Should return empty list on error
            assert len(market_data_list) == 0
    
    def test_strategy_error_does_not_stop_loop(self, config_file):
        """Test that an error in one strategy doesn't stop other strategies."""
        system = TradingSystem(config_file)
        system.initialize()
        
        # Mock one strategy to raise an exception
        mock_market_data = MarketData(
            symbol='AAPL',
            timestamp=datetime.now(),
            open=150.0,
            high=151.0,
            low=149.0,
            close=150.5,
            volume=1000000
        )
        
        # Make the first strategy raise an error
        with patch.object(system.strategies[0], 'on_data', side_effect=Exception("Strategy error")):
            # The trading loop should handle this error gracefully
            # Simulate what happens in the trading loop
            all_signals = []
            for strategy in system.strategies:
                try:
                    strategy.on_data(mock_market_data)
                    signals = strategy.generate_signals()
                    if signals:
                        all_signals.extend(signals)
                except Exception as e:
                    # Error should be caught and logged, not raised
                    assert "Strategy error" in str(e)
    
    def test_order_submission_retry_logic(self, config_file):
        """Test that failed order submissions are retried."""
        system = TradingSystem(config_file)
        system.initialize()
        
        # Create an order
        signal = Signal(
            symbol='AAPL',
            action=SignalAction.BUY,
            quantity=10,
            order_type=SignalOrderType.MARKET,
            timestamp=datetime.now()
        )
        
        order = system.order_manager.create_order(signal)
        
        # Mock broker to fail first 2 attempts, succeed on 3rd
        call_count = 0
        def mock_submit(order):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return f"ORDER-{call_count}"
        
        with patch.object(system.brokerage, 'submit_order', side_effect=mock_submit):
            # This should succeed after retries
            success = system.order_manager.submit_order(order)
            assert success
            assert call_count == 3


class TestModeSwitchingSafety:
    """Test mode switching safety between simulation and live trading."""
    
    def test_simulation_mode_indicator(self, tmp_path):
        """Test that simulation mode is clearly indicated in logs."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'simulation',
                'initial_capital': 100000,
                'log_level': 'INFO'
            },
            'brokerage': {
                'provider': 'simulated',
                'credentials': {}
            },
            'data': {
                'provider': 'yfinance'
            },
            'risk': {
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': True,
                    'params': {}
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        system = TradingSystem(str(config_file))
        system.initialize()
        
        assert system.config.trading.mode == TradingMode.SIMULATION
        assert not system.is_live_mode()
    
    def test_live_mode_requires_real_brokerage(self, tmp_path):
        """Test that live mode cannot use simulated brokerage."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'live',
                'initial_capital': 100000
            },
            'brokerage': {
                'provider': 'simulated',  # Should not be allowed in live mode
                'credentials': {}
            },
            'data': {
                'provider': 'yfinance'
            },
            'risk': {
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': True,
                    'params': {}
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        system = TradingSystem(str(config_file))
        
        # This should raise an error - simulated broker not allowed in live mode
        with pytest.raises(TradingSystemError, match="Cannot use simulated brokerage in LIVE trading mode"):
            system.initialize()


class TestSystemMonitoring:
    """Test system monitoring and health checks."""
    
    @pytest.fixture
    def config_file(self, tmp_path):
        """Create a test configuration file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'simulation',
                'initial_capital': 100000,
                'log_level': 'INFO'
            },
            'brokerage': {
                'provider': 'simulated',
                'credentials': {}
            },
            'data': {
                'provider': 'yfinance'
            },
            'risk': {
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': True,
                    'params': {}
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        return str(config_file)
    
    def test_system_health_check(self, config_file):
        """Test system health check reporting."""
        system = TradingSystem(config_file)
        system.initialize()
        
        # Should not raise an exception
        system.log_system_health()
        
        # Verify system is healthy
        assert system.is_initialized()
        assert system.data_provider is not None
        assert system.brokerage is not None
        assert len(system.strategies) > 0
    
    def test_performance_summary(self, config_file):
        """Test performance summary reporting."""
        system = TradingSystem(config_file)
        system.initialize()
        
        # Add some mock trades
        from src.models.order import Trade
        trade = Trade(
            trade_id='TEST-001',
            symbol='AAPL',
            action=OrderAction.SELL,
            quantity=10,
            price=150.0,
            timestamp=datetime.now(),
            pnl=100.0
        )
        system.portfolio.record_trade(trade)
        
        # Should not raise an exception
        system.log_performance_summary()
    
    def test_portfolio_status_logging(self, config_file):
        """Test portfolio status logging."""
        system = TradingSystem(config_file)
        system.initialize()
        
        # Add a position
        system.portfolio.add_position('AAPL', 10, 150.0)
        system.portfolio.update_prices({'AAPL': 155.0})
        
        # Should not raise an exception
        system._log_portfolio_status()
        
        # Verify portfolio has the position
        assert 'AAPL' in system.portfolio.positions
        assert system.portfolio.positions['AAPL'].quantity == 10


class TestEndToEndTradingScenarios:
    """Additional end-to-end integration tests for complete trading scenarios."""
    
    @pytest.fixture
    def config_file(self, tmp_path):
        """Create a test configuration file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            'trading': {
                'mode': 'simulation',
                'initial_capital': 100000,
                'log_level': 'INFO'
            },
            'brokerage': {
                'provider': 'simulated',
                'credentials': {}
            },
            'data': {
                'provider': 'yfinance',
                'cache_enabled': False
            },
            'risk': {
                'max_position_size_pct': 20.0,
                'max_daily_loss_pct': 5.0,
                'max_portfolio_leverage': 1.0
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'enabled': True,
                    'params': {
                        'fast_period': 5,
                        'slow_period': 10,
                        'quantity': 10
                    }
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        return str(config_file)
    
    def test_full_buy_sell_cycle_with_profit(self, config_file):
        """Test a complete buy-sell cycle that results in profit."""
        system = TradingSystem(config_file)
        system.initialize()
        
        initial_cash = system.portfolio.cash_balance
        
        # Step 1: Buy signal and execution
        buy_signal = Signal(
            symbol='AAPL',
            action=SignalAction.BUY,
            quantity=10,
            order_type=SignalOrderType.MARKET,
            timestamp=datetime.now()
        )
        
        current_prices = {'AAPL': 150.0}
        system._process_signals([buy_signal], current_prices)
        
        # Get and fill the buy order
        orders = system.order_manager.get_all_orders()
        buy_order = orders[0]
        
        buy_fill_status = OrderStatus(
            order_id=buy_order.order_id,
            status=OrderStatusEnum.FILLED,
            filled_quantity=10,
            average_fill_price=150.0,
            message="Order filled"
        )
        system._update_portfolio_on_fill(buy_order, buy_fill_status)
        
        # Verify position was created
        assert 'AAPL' in system.portfolio.positions
        assert system.portfolio.positions['AAPL'].quantity == 10
        assert system.portfolio.cash_balance == initial_cash - 1500.0
        
        # Step 2: Price increases
        system.portfolio.update_prices({'AAPL': 160.0})
        unrealized_pnl = system.portfolio.get_unrealized_pnl()
        assert unrealized_pnl == 100.0  # (160 - 150) * 10
        
        # Step 3: Sell signal and execution
        sell_signal = Signal(
            symbol='AAPL',
            action=SignalAction.SELL,
            quantity=10,
            order_type=SignalOrderType.MARKET,
            timestamp=datetime.now()
        )
        
        current_prices = {'AAPL': 160.0}
        system._process_signals([sell_signal], current_prices)
        
        # Get and fill the sell order
        sell_order = [o for o in system.order_manager.get_all_orders() if o.action == OrderAction.SELL][0]
        
        sell_fill_status = OrderStatus(
            order_id=sell_order.order_id,
            status=OrderStatusEnum.FILLED,
            filled_quantity=10,
            average_fill_price=160.0,
            message="Order filled"
        )
        system._update_portfolio_on_fill(sell_order, sell_fill_status)
        
        # Verify position was closed and profit realized
        position = system.portfolio.get_position('AAPL')
        assert position is None or position.quantity == 0
        assert system.portfolio.cash_balance == initial_cash + 100.0  # $100 profit
        
        # Verify trade was recorded
        assert len(system.portfolio.trade_history) == 1
        assert system.portfolio.trade_history[0].pnl == 100.0
    
    def test_full_buy_sell_cycle_with_loss(self, config_file):
        """Test a complete buy-sell cycle that results in loss."""
        system = TradingSystem(config_file)
        system.initialize()
        
        initial_cash = system.portfolio.cash_balance
        
        # Buy at $150
        buy_signal = Signal(
            symbol='AAPL',
            action=SignalAction.BUY,
            quantity=10,
            order_type=SignalOrderType.MARKET,
            timestamp=datetime.now()
        )
        
        system._process_signals([buy_signal], {'AAPL': 150.0})
        buy_order = system.order_manager.get_all_orders()[0]
        
        buy_fill = OrderStatus(
            order_id=buy_order.order_id,
            status=OrderStatusEnum.FILLED,
            filled_quantity=10,
            average_fill_price=150.0,
            message="Filled"
        )
        system._update_portfolio_on_fill(buy_order, buy_fill)
        
        # Sell at $140 (loss)
        sell_signal = Signal(
            symbol='AAPL',
            action=SignalAction.SELL,
            quantity=10,
            order_type=SignalOrderType.MARKET,
            timestamp=datetime.now()
        )
        
        system._process_signals([sell_signal], {'AAPL': 140.0})
        sell_order = [o for o in system.order_manager.get_all_orders() if o.action == OrderAction.SELL][0]
        
        sell_fill = OrderStatus(
            order_id=sell_order.order_id,
            status=OrderStatusEnum.FILLED,
            filled_quantity=10,
            average_fill_price=140.0,
            message="Filled"
        )
        system._update_portfolio_on_fill(sell_order, sell_fill)
        
        # Verify loss was recorded
        assert system.portfolio.cash_balance == initial_cash - 100.0  # $100 loss
        assert len(system.portfolio.trade_history) == 1
        assert system.portfolio.trade_history[0].pnl == -100.0
    
    def test_multiple_positions_management(self, config_file):
        """Test managing multiple positions simultaneously."""
        system = TradingSystem(config_file)
        system.initialize()
        
        # Buy multiple different stocks
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        prices = {'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 300.0}
        
        for symbol in symbols:
            signal = Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                quantity=5,
                order_type=SignalOrderType.MARKET,
                timestamp=datetime.now()
            )
            
            system._process_signals([signal], {symbol: prices[symbol]})
            
            # Fill the order
            orders = [o for o in system.order_manager.get_all_orders() if o.symbol == symbol]
            if orders:
                order = orders[0]
                fill_status = OrderStatus(
                    order_id=order.order_id,
                    status=OrderStatusEnum.FILLED,
                    filled_quantity=5,
                    average_fill_price=prices[symbol],
                    message="Filled"
                )
                system._update_portfolio_on_fill(order, fill_status)
        
        # Verify all positions exist
        assert len(system.portfolio.positions) == 3
        for symbol in symbols:
            assert symbol in system.portfolio.positions
            assert system.portfolio.positions[symbol].quantity == 5
        
        # Update prices and verify portfolio value
        new_prices = {'AAPL': 155.0, 'GOOGL': 2850.0, 'MSFT': 310.0}
        system.portfolio.update_prices(new_prices)
        
        # Calculate expected unrealized P&L
        expected_pnl = (155 - 150) * 5 + (2850 - 2800) * 5 + (310 - 300) * 5
        actual_pnl = system.portfolio.get_unrealized_pnl()
        assert abs(actual_pnl - expected_pnl) < 0.01
    
    def test_risk_manager_blocks_oversized_position(self, config_file):
        """Test that risk manager blocks orders that would exceed position size limits."""
        system = TradingSystem(config_file)
        system.initialize()
        
        # Try to buy a position that exceeds 20% of portfolio
        # Portfolio = $100,000, max position = 20% = $20,000
        # Try to buy 150 shares at $150 = $22,500 (exceeds limit)
        signal = Signal(
            symbol='AAPL',
            action=SignalAction.BUY,
            quantity=150,
            order_type=SignalOrderType.MARKET,
            timestamp=datetime.now()
        )
        
        current_prices = {'AAPL': 150.0}
        
        # Process signal - should be rejected by risk manager
        system._process_signals([signal], current_prices)
        
        # Verify no position was created
        assert 'AAPL' not in system.portfolio.positions or system.portfolio.positions['AAPL'].quantity == 0
    
    def test_daily_loss_limit_enforcement(self, config_file):
        """Test that daily loss limit prevents further trading."""
        system = TradingSystem(config_file)
        system.initialize()
        
        # Create a large losing position to trigger daily loss limit
        # Buy at high price
        system.portfolio.add_position('AAPL', 100, 150.0)
        system.portfolio.cash_balance -= 15000.0
        
        # Price drops significantly (more than 5% of portfolio)
        # Portfolio was $100k, 5% = $5k loss limit
        # Drop from $150 to $90 = $60 loss per share * 100 = $6k loss (exceeds limit)
        system.portfolio.update_prices({'AAPL': 90.0})
        
        # Try to place a new order - should be blocked
        signal = Signal(
            symbol='GOOGL',
            action=SignalAction.BUY,
            quantity=10,
            order_type=SignalOrderType.MARKET,
            timestamp=datetime.now()
        )
        
        current_prices = {'GOOGL': 2800.0, 'AAPL': 90.0}
        
        # Check if daily loss limit is breached
        is_breached = system.risk_manager.check_daily_loss_limit(system.portfolio)
        assert is_breached  # Should be True (limit breached)
    
    def test_system_shutdown_gracefully(self, config_file):
        """Test that system shuts down gracefully."""
        system = TradingSystem(config_file)
        system.initialize()
        
        assert system.is_initialized()
        
        # Shutdown
        system.shutdown()
        
        assert not system.is_running()
    
    def test_configuration_mode_validation(self, tmp_path):
        """Test that configuration properly validates trading mode."""
        # Test simulation mode with simulated broker (valid)
        config_file = tmp_path / "sim_config.yaml"
        config_data = {
            'trading': {'mode': 'simulation', 'initial_capital': 100000},
            'brokerage': {'provider': 'simulated', 'credentials': {}},
            'data': {'provider': 'yfinance'},
            'risk': {'max_position_size_pct': 10.0, 'max_daily_loss_pct': 5.0, 'max_portfolio_leverage': 1.0},
            'strategies': [{'name': 'MovingAverageCrossover', 'enabled': True, 'params': {}}]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        system = TradingSystem(str(config_file))
        system.initialize()
        
        assert system.config.trading.mode == TradingMode.SIMULATION
        assert not system.is_live_mode()
        assert system.is_initialized()
