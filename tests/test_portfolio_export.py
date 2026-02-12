"""Tests for portfolio export functionality.

Tests the CSV export functions for trades, portfolio history, and tax reports.
"""

import pytest
import pandas as pd
import io
from datetime import datetime, timedelta
from services.portfolio_service import PortfolioService
from services.trading_service import TradingService


@pytest.fixture
def portfolio_service():
    """Create a portfolio service instance for testing."""
    # Use a test database
    service = PortfolioService(db_path="data/database/portfolio_test_export.db")
    yield service
    
    # Cleanup
    import os
    if os.path.exists("data/database/portfolio_test_export.db"):
        os.remove("data/database/portfolio_test_export.db")


def test_export_trades_csv_empty(portfolio_service):
    """Test exporting trades when no transactions exist."""
    csv_data = portfolio_service.export_trades_csv()
    
    assert csv_data == "", "Should return empty string when no transactions"


def test_export_trades_csv_with_data(portfolio_service):
    """Test exporting trades with transaction data."""
    # Add some test transactions
    portfolio_service.record_transaction(
        symbol="AAPL",
        quantity=10,
        side="buy",
        price=150.0,
        order_id="test_order_1",
        commission=1.0,
        notes="Test buy"
    )
    
    portfolio_service.record_transaction(
        symbol="AAPL",
        quantity=5,
        side="sell",
        price=155.0,
        order_id="test_order_2",
        commission=1.0,
        notes="Test sell"
    )
    
    # Export to CSV
    csv_data = portfolio_service.export_trades_csv()
    
    # Verify CSV is not empty
    assert csv_data != "", "CSV should not be empty"
    
    # Parse CSV
    df = pd.read_csv(io.StringIO(csv_data))
    
    # Verify structure
    assert len(df) == 2, "Should have 2 transactions"
    assert 'Date' in df.columns
    assert 'Symbol' in df.columns
    assert 'Side' in df.columns
    assert 'Quantity' in df.columns
    assert 'Price' in df.columns
    assert 'Total Value' in df.columns
    assert 'Commission' in df.columns
    assert 'Order ID' in df.columns
    assert 'Notes' in df.columns
    
    # Verify data
    assert df['Symbol'].iloc[0] == 'AAPL'
    assert df['Side'].iloc[0] == 'SELL'  # Most recent first
    assert df['Quantity'].iloc[0] == 5
    assert df['Price'].iloc[0] == 155.0


def test_export_trades_csv_with_date_range(portfolio_service):
    """Test exporting trades with date range filtering."""
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    two_days_ago = now - timedelta(days=2)
    
    # Add transactions at different times
    portfolio_service.record_transaction(
        symbol="AAPL",
        quantity=10,
        side="buy",
        price=150.0,
        order_id="old_order",
        timestamp=two_days_ago
    )
    
    portfolio_service.record_transaction(
        symbol="GOOGL",
        quantity=5,
        side="buy",
        price=2800.0,
        order_id="recent_order",
        timestamp=now
    )
    
    # Export only recent transactions
    csv_data = portfolio_service.export_trades_csv(
        start_date=yesterday,
        end_date=now
    )
    
    df = pd.read_csv(io.StringIO(csv_data))
    
    # Should only have the recent transaction
    assert len(df) == 1, "Should only have 1 recent transaction"
    assert df['Symbol'].iloc[0] == 'GOOGL'


def test_export_portfolio_history_csv_empty(portfolio_service):
    """Test exporting portfolio history when no snapshots exist."""
    csv_data = portfolio_service.export_portfolio_history_csv()
    
    assert csv_data == "", "Should return empty string when no history"


def test_export_portfolio_history_csv_with_data(portfolio_service):
    """Test exporting portfolio history with snapshot data."""
    # Mock trading service for snapshot
    class MockTradingService:
        def get_account(self):
            return {
                'portfolio_value': 10000.0,
                'cash': 5000.0,
                'equity': 10000.0
            }
        
        def get_positions(self):
            return []
    
    portfolio_service.trading_service = MockTradingService()
    
    # Save a snapshot
    portfolio_service.save_portfolio_snapshot()
    
    # Export to CSV
    csv_data = portfolio_service.export_portfolio_history_csv()
    
    # Verify CSV is not empty
    assert csv_data != "", "CSV should not be empty"
    
    # Parse CSV
    df = pd.read_csv(io.StringIO(csv_data))
    
    # Verify structure
    assert len(df) >= 1, "Should have at least 1 snapshot"
    assert 'Date' in df.columns
    assert 'Portfolio Value' in df.columns
    assert 'Cash' in df.columns
    assert 'Equity' in df.columns
    assert 'Positions Value' in df.columns
    assert 'Number of Positions' in df.columns
    
    # Verify data
    assert df['Portfolio Value'].iloc[0] == 10000.0
    assert df['Cash'].iloc[0] == 5000.0


def test_export_tax_report_csv_empty(portfolio_service):
    """Test exporting tax report when no trades exist."""
    csv_data = portfolio_service.export_tax_report_csv(2024)
    
    assert csv_data == "", "Should return empty string when no trades"


def test_export_tax_report_csv_with_data(portfolio_service):
    """Test exporting tax report with completed trades."""
    # Add buy and sell transactions to create a completed trade
    buy_date = datetime(2024, 1, 15)
    sell_date = datetime(2024, 6, 15)
    
    portfolio_service.record_transaction(
        symbol="AAPL",
        quantity=10,
        side="buy",
        price=150.0,
        order_id="buy_order",
        timestamp=buy_date
    )
    
    portfolio_service.record_transaction(
        symbol="AAPL",
        quantity=10,
        side="sell",
        price=160.0,
        order_id="sell_order",
        timestamp=sell_date
    )
    
    # Export tax report
    csv_data = portfolio_service.export_tax_report_csv(2024)
    
    # Verify CSV is not empty
    assert csv_data != "", "CSV should not be empty"
    
    # Parse CSV
    df = pd.read_csv(io.StringIO(csv_data))
    
    # Verify structure
    assert len(df) >= 1, "Should have at least 1 completed trade"
    assert 'Symbol' in df.columns
    assert 'Quantity' in df.columns
    assert 'Date Acquired' in df.columns
    assert 'Date Sold' in df.columns
    assert 'Purchase Price' in df.columns
    assert 'Sale Price' in df.columns
    assert 'Cost Basis' in df.columns
    assert 'Proceeds' in df.columns
    assert 'Gain/Loss' in df.columns
    assert 'Gain/Loss %' in df.columns
    assert 'Holding Period (Days)' in df.columns
    assert 'Term' in df.columns
    
    # Verify data for AAPL trade
    aapl_trades = df[df['Symbol'] == 'AAPL']
    assert len(aapl_trades) >= 1, "Should have at least 1 AAPL trade"
    assert aapl_trades.iloc[0]['Quantity'] == 10
    assert aapl_trades.iloc[0]['Term'] == 'Short-term'  # Less than 365 days


def test_export_csv_special_characters(portfolio_service):
    """Test that exports handle special characters correctly."""
    # Add transaction with special characters in notes
    portfolio_service.record_transaction(
        symbol="AAPL",
        quantity=10,
        side="buy",
        price=150.0,
        order_id="test_order",
        notes="Test with comma, quotes \"test\", and newline\n"
    )
    
    # Export to CSV
    csv_data = portfolio_service.export_trades_csv()
    
    # Should be able to parse without errors
    df = pd.read_csv(io.StringIO(csv_data))
    
    assert len(df) == 1
    assert 'Notes' in df.columns


def test_export_csv_large_dataset(portfolio_service):
    """Test exporting a large number of transactions."""
    # Add many transactions
    for i in range(100):
        portfolio_service.record_transaction(
            symbol=f"STOCK{i % 10}",
            quantity=10,
            side="buy" if i % 2 == 0 else "sell",
            price=100.0 + i,
            order_id=f"order_{i}"
        )
    
    # Export to CSV
    csv_data = portfolio_service.export_trades_csv()
    
    df = pd.read_csv(io.StringIO(csv_data))
    
    # Verify all transactions are exported
    assert len(df) == 100, "Should export all 100 transactions"


def test_export_invalid_year(portfolio_service):
    """Test that invalid year raises ValueError."""
    with pytest.raises(ValueError):
        portfolio_service.export_tax_report_csv(1999)
    
    with pytest.raises(ValueError):
        portfolio_service.export_tax_report_csv(2100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
