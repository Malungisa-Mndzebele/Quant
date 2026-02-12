"""Quick test status check - runs fast smoke tests only."""

import sys

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing core imports...")
    try:
        from services.portfolio_service import PortfolioService
        from services.trading_service import TradingService
        from services.paper_trading_service import PaperTradingService
        from services.market_data_service import MarketDataService
        from services.risk_service import RiskService
        from services.alert_service import AlertService
        from services.backtest_service import BacktestEngine
        from ai.inference import AIEngine
        print("✓ All core imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key services."""
    print("\nTesting basic functionality...")
    try:
        from services.paper_trading_service import PaperTradingService
        from services.portfolio_service import PortfolioService
        
        # Test paper trading
        paper = PaperTradingService(initial_capital=100000.0)
        account = paper.get_account()
        assert account.cash == 100000.0
        print("✓ Paper trading service works")
        
        # Test portfolio
        portfolio = PortfolioService(trading_service=paper)
        portfolio.record_transaction("AAPL", 10, 150.0, "buy")
        history = portfolio.get_transaction_history()
        assert len(history) > 0
        print("✓ Portfolio service works")
        
        return True
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("QUICK TEST STATUS CHECK")
    print("="*70)
    
    results = []
    results.append(("Core Imports", test_imports()))
    results.append(("Basic Functionality", test_basic_functionality()))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("="*70))
    if all_passed:
        print("✓ All quick tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
