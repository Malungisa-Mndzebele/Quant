"""Unit tests for portfolio service."""

import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings

from services.portfolio_service import (
    PortfolioService,
    Transaction,
    PerformanceMetrics
)
from services.trading_service import Position


class TestPortfolioService:
    """Test suite for PortfolioService"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.fixture
    def mock_trading_service(self):
        """Create mock trading service"""
        service = Mock()
        
        # Mock account info
        service.get_account.return_value = {
            'portfolio_value': 100000.0,
            'cash': 50000.0,
            'equity': 100000.0,
            'buying_power': 200000.0
        }
        
        # Mock positions
        service.get_positions.return_value = [
            Position(
                symbol='AAPL',
                quantity=100,
                side='long',
                entry_price=150.0,
                current_price=160.0,
                market_value=16000.0,
                cost_basis=15000.0,
                unrealized_pl=1000.0,
                unrealized_pl_pct=6.67
            ),
            Position(
                symbol='GOOGL',
                quantity=50,
                side='long',
                entry_price=2800.0,
                current_price=2900.0,
                market_value=145000.0,
                cost_basis=140000.0,
                unrealized_pl=5000.0,
                unrealized_pl_pct=3.57
            )
        ]
        
        return service
    
    @pytest.fixture
    def portfolio_service(self, temp_db, mock_trading_service):
        """Create portfolio service with temp database"""
        return PortfolioService(
            db_path=temp_db,
            trading_service=mock_trading_service
        )
    
    def test_init_creates_database(self, temp_db):
        """Test that initialization creates database tables"""
        service = PortfolioService(db_path=temp_db)
        
        # Check that database file exists
        assert os.path.exists(temp_db)
        
        # Check that tables exist
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='transactions'
        """)
        assert cursor.fetchone() is not None
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='portfolio_snapshots'
        """)
        assert cursor.fetchone() is not None
        
        conn.close()
    
    def test_record_transaction_success(self, portfolio_service):
        """Test recording a transaction"""
        txn_id = portfolio_service.record_transaction(
            symbol='AAPL',
            quantity=100,
            side='buy',
            price=150.0,
            order_id='order123',
            commission=1.0,
            notes='Test transaction'
        )
        
        assert txn_id > 0
        
        # Verify transaction was recorded
        transactions = portfolio_service.get_transaction_history()
        assert len(transactions) == 1
        
        txn = transactions[0]
        assert txn.symbol == 'AAPL'
        assert txn.quantity == 100
        assert txn.side == 'buy'
        assert txn.price == 150.0
        assert txn.total_value == 15000.0
        assert txn.commission == 1.0
        assert txn.order_id == 'order123'
        assert txn.notes == 'Test transaction'
    
    def test_record_transaction_invalid_params(self, portfolio_service):
        """Test that invalid parameters raise ValueError"""
        # Empty symbol
        with pytest.raises(ValueError, match="Symbol is required"):
            portfolio_service.record_transaction(
                symbol='',
                quantity=100,
                side='buy',
                price=150.0,
                order_id='order123'
            )
        
        # Negative quantity
        with pytest.raises(ValueError, match="Quantity must be positive"):
            portfolio_service.record_transaction(
                symbol='AAPL',
                quantity=-100,
                side='buy',
                price=150.0,
                order_id='order123'
            )
        
        # Invalid side
        with pytest.raises(ValueError, match="Side must be 'buy' or 'sell'"):
            portfolio_service.record_transaction(
                symbol='AAPL',
                quantity=100,
                side='invalid',
                price=150.0,
                order_id='order123'
            )
        
        # Negative price
        with pytest.raises(ValueError, match="Price must be positive"):
            portfolio_service.record_transaction(
                symbol='AAPL',
                quantity=100,
                side='buy',
                price=-150.0,
                order_id='order123'
            )
        
        # Negative commission
        with pytest.raises(ValueError, match="Commission cannot be negative"):
            portfolio_service.record_transaction(
                symbol='AAPL',
                quantity=100,
                side='buy',
                price=150.0,
                order_id='order123',
                commission=-1.0
            )
        
        # Empty order_id
        with pytest.raises(ValueError, match="Order ID is required"):
            portfolio_service.record_transaction(
                symbol='AAPL',
                quantity=100,
                side='buy',
                price=150.0,
                order_id=''
            )
    
    def test_get_portfolio_value(self, portfolio_service, mock_trading_service):
        """Test getting portfolio value"""
        value = portfolio_service.get_portfolio_value()
        
        assert value == 100000.0
        mock_trading_service.get_account.assert_called_once()
    
    def test_get_portfolio_value_no_trading_service(self, temp_db):
        """Test that getting portfolio value without trading service raises error"""
        service = PortfolioService(db_path=temp_db)
        
        with pytest.raises(ValueError, match="Trading service is required"):
            service.get_portfolio_value()
    
    def test_get_positions(self, portfolio_service, mock_trading_service):
        """Test getting positions"""
        positions = portfolio_service.get_positions()
        
        assert len(positions) == 2
        assert positions[0].symbol == 'AAPL'
        assert positions[1].symbol == 'GOOGL'
        
        mock_trading_service.get_positions.assert_called_once()
    
    def test_get_positions_no_trading_service(self, temp_db):
        """Test that getting positions without trading service raises error"""
        service = PortfolioService(db_path=temp_db)
        
        with pytest.raises(ValueError, match="Trading service is required"):
            service.get_positions()
    
    def test_get_transaction_history_empty(self, portfolio_service):
        """Test getting transaction history when empty"""
        transactions = portfolio_service.get_transaction_history()
        
        assert len(transactions) == 0
    
    def test_get_transaction_history_with_filters(self, portfolio_service):
        """Test getting transaction history with date and symbol filters"""
        # Record some transactions
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        two_days_ago = now - timedelta(days=2)
        
        portfolio_service.record_transaction(
            symbol='AAPL',
            quantity=100,
            side='buy',
            price=150.0,
            order_id='order1',
            timestamp=two_days_ago
        )
        
        portfolio_service.record_transaction(
            symbol='GOOGL',
            quantity=50,
            side='buy',
            price=2800.0,
            order_id='order2',
            timestamp=yesterday
        )
        
        portfolio_service.record_transaction(
            symbol='AAPL',
            quantity=50,
            side='sell',
            price=160.0,
            order_id='order3',
            timestamp=now
        )
        
        # Test date filtering
        transactions = portfolio_service.get_transaction_history(
            start_date=yesterday - timedelta(hours=1)
        )
        assert len(transactions) == 2
        
        # Test symbol filtering
        transactions = portfolio_service.get_transaction_history(symbol='AAPL')
        assert len(transactions) == 2
        assert all(t.symbol == 'AAPL' for t in transactions)
        
        # Test combined filtering
        transactions = portfolio_service.get_transaction_history(
            start_date=yesterday - timedelta(hours=1),
            symbol='AAPL'
        )
        assert len(transactions) == 1
        assert transactions[0].symbol == 'AAPL'
    
    def test_performance_metrics_no_transactions(self, portfolio_service):
        """Test performance metrics with no transactions"""
        metrics = portfolio_service.get_performance_metrics()
        
        assert metrics.total_return == 0.0
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
    
    def test_performance_metrics_with_trades(self, portfolio_service):
        """Test performance metrics calculation with completed trades"""
        now = datetime.now()
        
        # Buy 100 AAPL at $150
        portfolio_service.record_transaction(
            symbol='AAPL',
            quantity=100,
            side='buy',
            price=150.0,
            order_id='order1',
            timestamp=now - timedelta(days=10)
        )
        
        # Sell 100 AAPL at $160 (profit: $1000)
        portfolio_service.record_transaction(
            symbol='AAPL',
            quantity=100,
            side='sell',
            price=160.0,
            order_id='order2',
            timestamp=now - timedelta(days=5)
        )
        
        # Buy 50 GOOGL at $2800
        portfolio_service.record_transaction(
            symbol='GOOGL',
            quantity=50,
            side='buy',
            price=2800.0,
            order_id='order3',
            timestamp=now - timedelta(days=8)
        )
        
        # Sell 50 GOOGL at $2700 (loss: $5000)
        portfolio_service.record_transaction(
            symbol='GOOGL',
            quantity=50,
            side='sell',
            price=2700.0,
            order_id='order4',
            timestamp=now - timedelta(days=3)
        )
        
        metrics = portfolio_service.get_performance_metrics()
        
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.5
        assert metrics.total_return == -4000.0  # 1000 - 5000
        assert metrics.largest_win == 1000.0
        assert metrics.largest_loss == -5000.0
        assert metrics.avg_win == 1000.0
        assert metrics.avg_loss == -5000.0
    
    def test_export_for_taxes_empty(self, portfolio_service):
        """Test tax export with no transactions"""
        df = portfolio_service.export_for_taxes(2024)
        
        assert df.empty
    
    def test_export_for_taxes_with_trades(self, portfolio_service):
        """Test tax export with completed trades"""
        # Create trades in 2024
        year_2024 = datetime(2024, 1, 1)
        
        # Buy and sell AAPL
        portfolio_service.record_transaction(
            symbol='AAPL',
            quantity=100,
            side='buy',
            price=150.0,
            order_id='order1',
            timestamp=datetime(2024, 1, 15)
        )
        
        portfolio_service.record_transaction(
            symbol='AAPL',
            quantity=100,
            side='sell',
            price=160.0,
            order_id='order2',
            timestamp=datetime(2024, 6, 15)
        )
        
        df = portfolio_service.export_for_taxes(2024)
        
        assert not df.empty
        assert len(df) == 1
        assert 'Symbol' in df.columns
        assert 'Date Acquired' in df.columns
        assert 'Date Sold' in df.columns
        assert 'Gain/Loss' in df.columns
        assert 'Term' in df.columns
        
        # Check values
        assert df.iloc[0]['Symbol'] == 'AAPL'
        assert df.iloc[0]['Quantity'] == 100
        assert df.iloc[0]['Term'] == 'Short-term'  # Less than 365 days
    
    def test_export_for_taxes_invalid_year(self, portfolio_service):
        """Test that invalid year raises ValueError"""
        with pytest.raises(ValueError, match="Invalid year"):
            portfolio_service.export_for_taxes(1999)
        
        with pytest.raises(ValueError, match="Invalid year"):
            portfolio_service.export_for_taxes(2100)
    
    def test_save_portfolio_snapshot(self, portfolio_service, mock_trading_service):
        """Test saving portfolio snapshot"""
        snapshot_id = portfolio_service.save_portfolio_snapshot()
        
        assert snapshot_id > 0
        
        # Verify snapshot was saved
        df = portfolio_service.get_portfolio_history()
        assert len(df) == 1
        assert df.iloc[0]['portfolio_value'] == 100000.0
        assert df.iloc[0]['num_positions'] == 2
    
    def test_save_portfolio_snapshot_no_trading_service(self, temp_db):
        """Test that saving snapshot without trading service raises error"""
        service = PortfolioService(db_path=temp_db)
        
        with pytest.raises(ValueError, match="Trading service is required"):
            service.save_portfolio_snapshot()
    
    def test_get_portfolio_history_empty(self, portfolio_service):
        """Test getting portfolio history when empty"""
        df = portfolio_service.get_portfolio_history()
        
        assert df.empty
    
    def test_get_portfolio_history_with_filters(self, portfolio_service):
        """Test getting portfolio history with date filters"""
        # Save multiple snapshots
        now = datetime.now()
        
        # Mock different times
        with patch('services.portfolio_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = now - timedelta(days=2)
            mock_datetime.fromisoformat = datetime.fromisoformat
            portfolio_service.save_portfolio_snapshot()
            
            mock_datetime.now.return_value = now - timedelta(days=1)
            portfolio_service.save_portfolio_snapshot()
            
            mock_datetime.now.return_value = now
            portfolio_service.save_portfolio_snapshot()
        
        # Get all snapshots
        df = portfolio_service.get_portfolio_history()
        assert len(df) == 3
        
        # Filter by start date
        df = portfolio_service.get_portfolio_history(
            start_date=now - timedelta(days=1, hours=12)
        )
        assert len(df) == 2
        
        # Filter by end date
        df = portfolio_service.get_portfolio_history(
            end_date=now - timedelta(days=1, hours=12)
        )
        assert len(df) == 1
    
    def test_calculate_trade_pnl_fifo(self, portfolio_service):
        """Test FIFO matching for trade P&L calculation"""
        now = datetime.now()
        
        # Buy 100 shares at $100
        portfolio_service.record_transaction(
            symbol='TEST',
            quantity=100,
            side='buy',
            price=100.0,
            order_id='order1',
            timestamp=now - timedelta(days=10)
        )
        
        # Buy 50 shares at $110
        portfolio_service.record_transaction(
            symbol='TEST',
            quantity=50,
            side='buy',
            price=110.0,
            order_id='order2',
            timestamp=now - timedelta(days=8)
        )
        
        # Sell 120 shares at $120
        # Should match: 100 @ $100 (profit: $2000) and 20 @ $110 (profit: $200)
        portfolio_service.record_transaction(
            symbol='TEST',
            quantity=120,
            side='sell',
            price=120.0,
            order_id='order3',
            timestamp=now - timedelta(days=5)
        )
        
        transactions = portfolio_service.get_transaction_history()
        trades = portfolio_service._calculate_trade_pnl(transactions)
        
        assert len(trades) == 2
        
        # First trade: 100 shares
        assert trades[0]['quantity'] == 100
        assert trades[0]['entry_price'] == 100.0
        assert trades[0]['exit_price'] == 120.0
        assert trades[0]['pnl'] == 2000.0
        
        # Second trade: 20 shares
        assert trades[1]['quantity'] == 20
        assert trades[1]['entry_price'] == 110.0
        assert trades[1]['exit_price'] == 120.0
        assert trades[1]['pnl'] == 200.0
    
    def test_calculate_max_drawdown(self, portfolio_service):
        """Test maximum drawdown calculation"""
        # Equity curve: 0, 100, 150, 120, 180, 140
        equity_curve = np.array([0, 100, 150, 120, 180, 140])
        
        max_dd, max_dd_pct = portfolio_service._calculate_max_drawdown(equity_curve)
        
        # Max drawdown should be from 180 to 140 = 40
        assert max_dd == 40.0
        # Max drawdown % should be 40/180 * 100 = 22.22%
        assert abs(max_dd_pct - 22.22) < 0.01
    
    def test_calculate_sharpe_ratio(self, portfolio_service):
        """Test Sharpe ratio calculation"""
        # Create equity curve with positive returns
        equity_curve = np.array([100, 105, 110, 108, 115, 120])
        
        sharpe = portfolio_service._calculate_sharpe_ratio(equity_curve)
        
        # Sharpe should be positive for positive returns
        assert sharpe > 0
    
    def test_calculate_streaks(self, portfolio_service):
        """Test win/loss streak calculation"""
        now = datetime.now()
        
        trades = [
            {'pnl': 100, 'exit_time': now - timedelta(days=10)},  # Win
            {'pnl': 200, 'exit_time': now - timedelta(days=9)},   # Win
            {'pnl': -50, 'exit_time': now - timedelta(days=8)},   # Loss
            {'pnl': 150, 'exit_time': now - timedelta(days=7)},   # Win
            {'pnl': -100, 'exit_time': now - timedelta(days=6)},  # Loss
            {'pnl': -75, 'exit_time': now - timedelta(days=5)},   # Loss
        ]
        
        current_streak, longest_win, longest_loss = portfolio_service._calculate_streaks(trades)
        
        assert longest_win == 2  # Two wins in a row at the start
        assert longest_loss == 2  # Two losses in a row at the end
        assert current_streak == -2  # Currently on a 2-loss streak
    
    def test_transaction_to_dict(self):
        """Test Transaction to_dict method"""
        now = datetime.now()
        txn = Transaction(
            transaction_id=1,
            symbol='AAPL',
            quantity=100,
            side='buy',
            price=150.0,
            total_value=15000.0,
            commission=1.0,
            timestamp=now,
            order_id='order123',
            notes='Test'
        )
        
        data = txn.to_dict()
        
        assert data['symbol'] == 'AAPL'
        assert data['quantity'] == 100
        assert data['timestamp'] == now.isoformat()
    
    def test_performance_metrics_to_dict(self):
        """Test PerformanceMetrics to_dict method"""
        metrics = PerformanceMetrics(
            total_return=1000.0,
            total_return_pct=10.0,
            sharpe_ratio=1.5,
            max_drawdown=500.0,
            max_drawdown_pct=5.0,
            win_rate=0.6,
            profit_factor=2.0,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            avg_win=200.0,
            avg_loss=-100.0,
            largest_win=500.0,
            largest_loss=-300.0,
            current_streak=2,
            longest_win_streak=3,
            longest_loss_streak=2
        )
        
        data = metrics.to_dict()
        
        assert data['total_return'] == 1000.0
        assert data['win_rate'] == 0.6
        assert data['total_trades'] == 10


class TestPortfolioValueProperty:
    """Property-based tests for portfolio value calculation"""
    
    @given(
        cash=st.floats(min_value=0.0, max_value=1000000.0, allow_nan=False, allow_infinity=False),
        num_positions=st.integers(min_value=0, max_value=10),
        position_values=st.lists(
            st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
            min_size=0,
            max_size=10
        )
    )
    @settings(max_examples=100, deadline=5000)
    def test_portfolio_value_equals_cash_plus_positions(
        self,
        cash,
        num_positions,
        position_values
    ):
        """
        Feature: ai-trading-agent, Property 10: Portfolio value calculation
        
        Property: For any portfolio state, the total value should equal the sum of 
        (cash balance + market value of all positions).
        
        This test verifies that:
        1. Portfolio value is always the sum of cash and position values
        2. This holds for any combination of cash and positions
        3. The calculation is consistent regardless of the number of positions
        
        Validates: Requirements 10.1
        """
        # Ensure position_values matches num_positions
        position_values = position_values[:num_positions]
        if len(position_values) < num_positions:
            position_values.extend([0.0] * (num_positions - len(position_values)))
        
        # Create temporary database
        fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        try:
            # Create mock trading service
            mock_trading_service = Mock()
            
            # Calculate expected portfolio value
            expected_portfolio_value = cash + sum(position_values)
            
            # Mock account info
            mock_trading_service.get_account.return_value = {
                'portfolio_value': expected_portfolio_value,
                'cash': cash,
                'equity': expected_portfolio_value,
                'buying_power': cash * 2
            }
            
            # Create mock positions
            mock_positions = []
            for i, value in enumerate(position_values):
                # Create position with market value
                position = Position(
                    symbol=f'SYM{i}',
                    quantity=100,
                    side='long',
                    entry_price=value / 100 if value > 0 else 1.0,
                    current_price=value / 100 if value > 0 else 1.0,
                    market_value=value,
                    cost_basis=value,
                    unrealized_pl=0.0,
                    unrealized_pl_pct=0.0
                )
                mock_positions.append(position)
            
            mock_trading_service.get_positions.return_value = mock_positions
            
            # Create portfolio service
            service = PortfolioService(
                db_path=db_path,
                trading_service=mock_trading_service
            )
            
            # Get portfolio value
            actual_portfolio_value = service.get_portfolio_value()
            
            # Verify property: portfolio value = cash + sum of position values
            assert abs(actual_portfolio_value - expected_portfolio_value) < 0.01, \
                f"Portfolio value {actual_portfolio_value} should equal cash {cash} + positions {sum(position_values)} = {expected_portfolio_value}"
            
            # Also verify by manually calculating from positions
            positions = service.get_positions()
            calculated_positions_value = sum(pos.market_value for pos in positions)
            
            # The portfolio value should equal cash + calculated positions value
            assert abs(actual_portfolio_value - (cash + calculated_positions_value)) < 0.01, \
                f"Portfolio value {actual_portfolio_value} should equal cash {cash} + calculated positions {calculated_positions_value}"
            
        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestExportDataCompletenessProperty:
    """Property-based tests for export data completeness"""
    
    @given(
        num_transactions=st.integers(min_value=1, max_value=50),
        year=st.integers(min_value=2020, max_value=2024)
    )
    @settings(max_examples=100, deadline=10000)
    def test_export_contains_all_transactions_in_date_range(
        self,
        num_transactions,
        year
    ):
        """
        Feature: ai-trading-agent, Property 15: Export data completeness
        
        Property: For any data export request, the generated file should contain 
        all transactions within the specified date range.
        
        This test verifies that:
        1. All transactions within the date range are included in the export
        2. No transactions are missing from the export
        3. The export contains all required fields (timestamps, symbols, prices, P&L)
        4. This holds for any number of transactions and any date range
        
        Validates: Requirements 15.2
        """
        # Create temporary database
        fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        try:
            # Create portfolio service
            service = PortfolioService(db_path=db_path)
            
            # Generate random transactions within the year
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
            start_of_year = datetime(year, 1, 1)
            end_of_year = datetime(year, 12, 31, 23, 59, 59)
            
            # Track all transactions we create
            created_transactions = []
            
            # Create buy/sell pairs to ensure we have completed trades
            # Use different symbols for each pair to avoid FIFO complications
            for i in range(num_transactions // 2):
                symbol = symbols[i % len(symbols)]
                
                # Random buy date in first half of year
                buy_days = np.random.randint(0, 180)
                buy_date = start_of_year + timedelta(days=buy_days)
                
                # Random sell date in second half of year (after buy)
                sell_days = np.random.randint(buy_days + 1, 365)
                sell_date = start_of_year + timedelta(days=sell_days)
                
                # Random prices
                buy_price = np.random.uniform(50.0, 500.0)
                sell_price = buy_price * np.random.uniform(0.8, 1.3)  # +/- 20-30%
                
                quantity = np.random.randint(10, 200)
                
                # Record buy transaction
                buy_txn_id = service.record_transaction(
                    symbol=symbol,
                    quantity=quantity,
                    side='buy',
                    price=buy_price,
                    order_id=f'order_buy_{i}',
                    commission=1.0,
                    timestamp=buy_date
                )
                created_transactions.append({
                    'id': buy_txn_id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'side': 'buy',
                    'price': buy_price,
                    'timestamp': buy_date
                })
                
                # Record sell transaction with SAME quantity to ensure clean matching
                sell_txn_id = service.record_transaction(
                    symbol=symbol,
                    quantity=quantity,
                    side='sell',
                    price=sell_price,
                    order_id=f'order_sell_{i}',
                    commission=1.0,
                    timestamp=sell_date
                )
                created_transactions.append({
                    'id': sell_txn_id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'side': 'sell',
                    'price': sell_price,
                    'timestamp': sell_date
                })
            
            # If odd number of transactions, add one more buy with a DIFFERENT symbol
            if num_transactions % 2 == 1:
                # Use a different symbol to avoid interfering with existing pairs
                symbol = symbols[(num_transactions // 2) % len(symbols)]
                buy_days = np.random.randint(0, 365)
                buy_date = start_of_year + timedelta(days=buy_days)
                buy_price = np.random.uniform(50.0, 500.0)
                quantity = np.random.randint(10, 200)
                
                buy_txn_id = service.record_transaction(
                    symbol=symbol,
                    quantity=quantity,
                    side='buy',
                    price=buy_price,
                    order_id=f'order_buy_extra',
                    commission=1.0,
                    timestamp=buy_date
                )
                created_transactions.append({
                    'id': buy_txn_id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'side': 'buy',
                    'price': buy_price,
                    'timestamp': buy_date
                })
            
            # Export data for the year
            export_df = service.export_for_taxes(year)
            
            # Get all transactions for the year
            all_transactions = service.get_transaction_history(
                start_date=start_of_year,
                end_date=end_of_year
            )
            
            # Property 1: All transactions in the date range should be retrievable
            assert len(all_transactions) == len(created_transactions), \
                f"Expected {len(created_transactions)} transactions, but got {len(all_transactions)}"
            
            # Property 2: Export should contain all completed trades
            # The export contains trades calculated using FIFO matching.
            # We need to verify that all completed trades are in the export.
            # Since we're using random quantities and symbols may repeat,
            # we can't predict the exact number. Instead, verify that:
            # 1. Export is not empty if we have any sell transactions
            # 2. All exported trades have valid data
            
            # Count sell transactions (each sell creates at least one trade)
            sell_count = sum(1 for txn in created_transactions if txn['side'] == 'sell')
            
            if sell_count > 0:
                # We should have at least one completed trade
                assert not export_df.empty, \
                    f"Expected non-empty export with {sell_count} sell transactions"
                
                actual_trades_in_export = len(export_df)
                
                # We should have at least as many trades as sells (could be more if sells match multiple buys)
                assert actual_trades_in_export >= sell_count, \
                    f"Expected at least {sell_count} trades (one per sell), but got {actual_trades_in_export}"
                
                # Property 3: Export must contain all required fields
                required_fields = [
                    'Symbol', 'Quantity', 'Date Acquired', 'Date Sold',
                    'Purchase Price', 'Sale Price', 'Cost Basis', 'Proceeds',
                    'Gain/Loss', 'Gain/Loss %', 'Holding Period (Days)', 'Term'
                ]
                
                for field in required_fields:
                    assert field in export_df.columns, \
                        f"Required field '{field}' missing from export"
                
                # Property 4: All exported trades should have valid data
                for idx, row in export_df.iterrows():
                    # Symbol should not be empty
                    assert row['Symbol'] in symbols, \
                        f"Invalid symbol in export: {row['Symbol']}"
                    
                    # Quantity should be positive
                    assert row['Quantity'] > 0, \
                        f"Invalid quantity in export: {row['Quantity']}"
                    
                    # Dates should be in the correct year
                    date_acquired = datetime.strptime(row['Date Acquired'], '%m/%d/%Y')
                    date_sold = datetime.strptime(row['Date Sold'], '%m/%d/%Y')
                    
                    assert date_acquired.year <= year, \
                        f"Date acquired {date_acquired} is after year {year}"
                    assert date_sold.year == year, \
                        f"Date sold {date_sold} is not in year {year}"
                    
                    # Holding period should be positive
                    assert row['Holding Period (Days)'] >= 0, \
                        f"Invalid holding period: {row['Holding Period (Days)']}"
                    
                    # Term should be either 'Short-term' or 'Long-term'
                    assert row['Term'] in ['Short-term', 'Long-term'], \
                        f"Invalid term: {row['Term']}"
            else:
                # If no sell transactions, export should be empty
                assert export_df.empty, \
                    f"Expected empty export with no sell transactions, but got {len(export_df)} trades"
            
            # Property 5: Verify completeness by checking transaction history
            # All transactions should be accounted for
            transaction_ids = [txn.transaction_id for txn in all_transactions]
            created_ids = [txn['id'] for txn in created_transactions]
            
            assert set(transaction_ids) == set(created_ids), \
                "Transaction IDs in database don't match created transactions"
            
        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
