"""
Unit tests for Watchlist Service
"""

import pytest
import os
import json
import tempfile
from datetime import datetime
from hypothesis import given, strategies as st, settings
from services.watchlist_service import WatchlistService, Watchlist


@pytest.fixture
def temp_storage():
    """Create temporary storage file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def watchlist_service(temp_storage):
    """Create watchlist service with temporary storage."""
    return WatchlistService(storage_path=temp_storage)


def test_create_watchlist(watchlist_service):
    """Test creating a new watchlist."""
    watchlist = watchlist_service.create_watchlist("Tech Stocks", ["AAPL", "GOOGL"])
    
    assert watchlist.name == "Tech Stocks"
    assert watchlist.symbols == ["AAPL", "GOOGL"]
    assert isinstance(watchlist.created_at, datetime)
    assert isinstance(watchlist.updated_at, datetime)


def test_create_duplicate_watchlist(watchlist_service):
    """Test creating a watchlist with duplicate name raises error."""
    watchlist_service.create_watchlist("Tech Stocks")
    
    with pytest.raises(ValueError, match="already exists"):
        watchlist_service.create_watchlist("Tech Stocks")


def test_delete_watchlist(watchlist_service):
    """Test deleting a watchlist."""
    watchlist_service.create_watchlist("Tech Stocks")
    
    result = watchlist_service.delete_watchlist("Tech Stocks")
    assert result is True
    
    # Verify it's deleted
    assert watchlist_service.get_watchlist("Tech Stocks") is None


def test_delete_nonexistent_watchlist(watchlist_service):
    """Test deleting a non-existent watchlist returns False."""
    result = watchlist_service.delete_watchlist("Nonexistent")
    assert result is False


def test_rename_watchlist(watchlist_service):
    """Test renaming a watchlist."""
    watchlist_service.create_watchlist("Old Name")
    
    result = watchlist_service.rename_watchlist("Old Name", "New Name")
    assert result is True
    
    # Verify old name is gone
    assert watchlist_service.get_watchlist("Old Name") is None
    
    # Verify new name exists
    watchlist = watchlist_service.get_watchlist("New Name")
    assert watchlist is not None
    assert watchlist.name == "New Name"


def test_rename_to_existing_name(watchlist_service):
    """Test renaming to an existing name raises error."""
    watchlist_service.create_watchlist("Name1")
    watchlist_service.create_watchlist("Name2")
    
    with pytest.raises(ValueError, match="already exists"):
        watchlist_service.rename_watchlist("Name1", "Name2")


def test_add_symbol(watchlist_service):
    """Test adding a symbol to watchlist."""
    watchlist_service.create_watchlist("Tech Stocks")
    
    result = watchlist_service.add_symbol("Tech Stocks", "AAPL")
    assert result is True
    
    watchlist = watchlist_service.get_watchlist("Tech Stocks")
    assert "AAPL" in watchlist.symbols


def test_add_symbol_uppercase_conversion(watchlist_service):
    """Test that symbols are converted to uppercase."""
    watchlist_service.create_watchlist("Tech Stocks")
    
    watchlist_service.add_symbol("Tech Stocks", "aapl")
    
    watchlist = watchlist_service.get_watchlist("Tech Stocks")
    assert "AAPL" in watchlist.symbols


def test_add_duplicate_symbol(watchlist_service):
    """Test adding duplicate symbol returns False."""
    watchlist_service.create_watchlist("Tech Stocks", ["AAPL"])
    
    result = watchlist_service.add_symbol("Tech Stocks", "AAPL")
    assert result is False


def test_remove_symbol(watchlist_service):
    """Test removing a symbol from watchlist."""
    watchlist_service.create_watchlist("Tech Stocks", ["AAPL", "GOOGL"])
    
    result = watchlist_service.remove_symbol("Tech Stocks", "AAPL")
    assert result is True
    
    watchlist = watchlist_service.get_watchlist("Tech Stocks")
    assert "AAPL" not in watchlist.symbols
    assert "GOOGL" in watchlist.symbols


def test_remove_nonexistent_symbol(watchlist_service):
    """Test removing non-existent symbol returns False."""
    watchlist_service.create_watchlist("Tech Stocks", ["AAPL"])
    
    result = watchlist_service.remove_symbol("Tech Stocks", "GOOGL")
    assert result is False


def test_get_all_watchlists(watchlist_service):
    """Test getting all watchlists."""
    watchlist_service.create_watchlist("Tech")
    watchlist_service.create_watchlist("Finance")
    
    watchlists = watchlist_service.get_all_watchlists()
    assert len(watchlists) == 2
    
    names = [wl.name for wl in watchlists]
    assert "Tech" in names
    assert "Finance" in names


def test_get_watchlist_names(watchlist_service):
    """Test getting watchlist names."""
    watchlist_service.create_watchlist("Tech")
    watchlist_service.create_watchlist("Finance")
    
    names = watchlist_service.get_watchlist_names()
    assert len(names) == 2
    assert "Tech" in names
    assert "Finance" in names


def test_export_watchlist(watchlist_service):
    """Test exporting a watchlist."""
    watchlist_service.create_watchlist("Tech Stocks", ["AAPL", "GOOGL"])
    
    export_data = watchlist_service.export_watchlist("Tech Stocks")
    
    assert export_data is not None
    assert export_data['name'] == "Tech Stocks"
    assert export_data['symbols'] == ["AAPL", "GOOGL"]
    assert 'created_at' in export_data
    assert 'updated_at' in export_data


def test_export_nonexistent_watchlist(watchlist_service):
    """Test exporting non-existent watchlist returns None."""
    export_data = watchlist_service.export_watchlist("Nonexistent")
    assert export_data is None


def test_import_watchlist(watchlist_service):
    """Test importing a watchlist."""
    data = {
        'name': 'Imported',
        'symbols': ['AAPL', 'GOOGL'],
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    
    result = watchlist_service.import_watchlist(data)
    assert result is True
    
    watchlist = watchlist_service.get_watchlist('Imported')
    assert watchlist is not None
    assert watchlist.name == 'Imported'
    assert watchlist.symbols == ['AAPL', 'GOOGL']


def test_import_duplicate_without_overwrite(watchlist_service):
    """Test importing duplicate without overwrite raises error."""
    watchlist_service.create_watchlist("Existing")
    
    data = {
        'name': 'Existing',
        'symbols': ['AAPL'],
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    
    with pytest.raises(ValueError, match="already exists"):
        watchlist_service.import_watchlist(data, overwrite=False)


def test_import_with_overwrite(watchlist_service):
    """Test importing with overwrite replaces existing."""
    watchlist_service.create_watchlist("Existing", ["OLD"])
    
    data = {
        'name': 'Existing',
        'symbols': ['NEW'],
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    
    result = watchlist_service.import_watchlist(data, overwrite=True)
    assert result is True
    
    watchlist = watchlist_service.get_watchlist('Existing')
    assert watchlist.symbols == ['NEW']


def test_persistence(temp_storage):
    """Test that watchlists persist across service instances."""
    # Create watchlist with first service instance
    service1 = WatchlistService(storage_path=temp_storage)
    service1.create_watchlist("Tech", ["AAPL", "GOOGL"])
    
    # Create new service instance and verify data persists
    service2 = WatchlistService(storage_path=temp_storage)
    watchlist = service2.get_watchlist("Tech")
    
    assert watchlist is not None
    assert watchlist.name == "Tech"
    assert watchlist.symbols == ["AAPL", "GOOGL"]


def test_export_all_watchlists(watchlist_service):
    """Test exporting all watchlists."""
    watchlist_service.create_watchlist("Tech", ["AAPL"])
    watchlist_service.create_watchlist("Finance", ["JPM"])
    
    export_data = watchlist_service.export_all_watchlists()
    
    assert len(export_data) == 2
    assert "Tech" in export_data
    assert "Finance" in export_data
    assert export_data["Tech"]["symbols"] == ["AAPL"]
    assert export_data["Finance"]["symbols"] == ["JPM"]


def test_import_all_watchlists(watchlist_service):
    """Test importing multiple watchlists."""
    data = {
        "Tech": {
            'name': 'Tech',
            'symbols': ['AAPL'],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        },
        "Finance": {
            'name': 'Finance',
            'symbols': ['JPM'],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
    }
    
    count = watchlist_service.import_all_watchlists(data)
    assert count == 2
    
    assert watchlist_service.get_watchlist("Tech") is not None
    assert watchlist_service.get_watchlist("Finance") is not None


def test_watchlist_symbol_uniqueness(watchlist_service):
    """Test that symbols in a watchlist are unique."""
    # This is enforced by the add_symbol method returning False for duplicates
    watchlist_service.create_watchlist("Test", ["AAPL"])
    
    # Try to add duplicate
    result = watchlist_service.add_symbol("Test", "AAPL")
    assert result is False
    
    # Verify only one instance
    watchlist = watchlist_service.get_watchlist("Test")
    assert watchlist.symbols.count("AAPL") == 1


# Property-Based Tests

@given(
    watchlist_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    symbols=st.lists(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu',))),
        min_size=1,
        max_size=50
    )
)
@settings(max_examples=100)
def test_property_watchlist_symbol_uniqueness(watchlist_name, symbols):
    """
    Property 17: Watchlist symbol uniqueness
    
    For any watchlist, each stock symbol should appear at most once in that watchlist.
    
    **Validates: Requirements 17.2**
    
    Feature: ai-trading-agent, Property 17: Watchlist symbol uniqueness
    """
    # Create temporary storage for this test
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        # Create a fresh service for each test
        service = WatchlistService(storage_path=temp_path)
        
        # Create watchlist
        service.create_watchlist(watchlist_name)
        
        # Add all symbols
        for symbol in symbols:
            service.add_symbol(watchlist_name, symbol)
        
        # Get the watchlist
        watchlist = service.get_watchlist(watchlist_name)
        
        # Property: Each symbol should appear at most once
        # This means the list should have no duplicates
        assert len(watchlist.symbols) == len(set(watchlist.symbols)), \
            f"Watchlist contains duplicate symbols: {watchlist.symbols}"
        
        # Additional check: verify each symbol appears exactly once
        for symbol in watchlist.symbols:
            count = watchlist.symbols.count(symbol)
            assert count == 1, \
                f"Symbol {symbol} appears {count} times in watchlist, expected 1"
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
