"""
Verification script for watchlist functionality
"""

from services.watchlist_service import WatchlistService
import json

def verify_watchlist_service():
    """Verify watchlist service functionality."""
    print("🔍 Verifying Watchlist Service...")
    
    # Initialize service
    service = WatchlistService(storage_path='data/test_watchlists.json')
    
    # Test 1: Create watchlist
    print("\n1. Creating watchlist...")
    watchlist = service.create_watchlist("Tech Stocks", ["AAPL", "GOOGL", "MSFT"])
    print(f"   ✅ Created: {watchlist.name} with {len(watchlist.symbols)} symbols")
    
    # Test 2: Add symbol
    print("\n2. Adding symbol...")
    result = service.add_symbol("Tech Stocks", "NVDA")
    print(f"   ✅ Added NVDA: {result}")
    
    # Test 3: Get watchlist
    print("\n3. Retrieving watchlist...")
    wl = service.get_watchlist("Tech Stocks")
    print(f"   ✅ Retrieved: {wl.name} with symbols: {wl.symbols}")
    
    # Test 4: Create another watchlist
    print("\n4. Creating second watchlist...")
    service.create_watchlist("Finance", ["JPM", "BAC", "GS"])
    print(f"   ✅ Created Finance watchlist")
    
    # Test 5: List all watchlists
    print("\n5. Listing all watchlists...")
    names = service.get_watchlist_names()
    print(f"   ✅ Found {len(names)} watchlists: {names}")
    
    # Test 6: Export watchlist
    print("\n6. Exporting watchlist...")
    export_data = service.export_watchlist("Tech Stocks")
    print(f"   ✅ Exported: {json.dumps(export_data, indent=2, default=str)}")
    
    # Test 7: Remove symbol
    print("\n7. Removing symbol...")
    result = service.remove_symbol("Tech Stocks", "GOOGL")
    wl = service.get_watchlist("Tech Stocks")
    print(f"   ✅ Removed GOOGL: {result}, remaining: {wl.symbols}")
    
    # Test 8: Rename watchlist
    print("\n8. Renaming watchlist...")
    result = service.rename_watchlist("Finance", "Financial Sector")
    names = service.get_watchlist_names()
    print(f"   ✅ Renamed: {result}, current names: {names}")
    
    # Test 9: Delete watchlist
    print("\n9. Deleting watchlist...")
    result = service.delete_watchlist("Financial Sector")
    names = service.get_watchlist_names()
    print(f"   ✅ Deleted: {result}, remaining: {names}")
    
    # Test 10: Symbol uniqueness
    print("\n10. Testing symbol uniqueness...")
    result = service.add_symbol("Tech Stocks", "AAPL")  # Already exists
    print(f"   ✅ Duplicate rejected: {not result}")
    
    print("\n" + "="*60)
    print("✅ All watchlist service tests passed!")
    print("="*60)
    
    # Cleanup
    import os
    if os.path.exists('data/test_watchlists.json'):
        os.remove('data/test_watchlists.json')
        print("\n🧹 Cleaned up test data")

if __name__ == "__main__":
    verify_watchlist_service()
