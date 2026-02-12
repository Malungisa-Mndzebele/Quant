"""
Verification script for the backtesting page.

This script verifies that the backtesting page is properly implemented
and can be imported without errors.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_backtest_page():
    """Verify the backtesting page implementation"""
    
    print("=" * 60)
    print("Backtesting Page Verification")
    print("=" * 60)
    
    try:
        # Test 1: Import the page module
        print("\n✓ Test 1: Importing backtesting page module...")
        # Import using importlib to handle the numeric prefix
        import importlib.util
        spec = importlib.util.spec_from_file_location("backtest_page", "pages/5_backtest.py")
        backtest_page = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backtest_page)
        print("  SUCCESS: Module imported successfully")
        
        # Test 2: Check for required strategy functions
        print("\n✓ Test 2: Checking strategy functions...")
        required_strategies = [
            'simple_ma_crossover',
            'rsi_strategy',
            'momentum_strategy',
            'bollinger_bands_strategy'
        ]
        
        for strategy in required_strategies:
            if hasattr(backtest_page, strategy):
                print(f"  ✓ Found strategy: {strategy}")
            else:
                print(f"  ✗ Missing strategy: {strategy}")
                return False
        
        print("  SUCCESS: All strategy functions found")
        
        # Test 3: Check STRATEGIES dictionary
        print("\n✓ Test 3: Checking STRATEGIES configuration...")
        if hasattr(backtest_page, 'STRATEGIES'):
            strategies = backtest_page.STRATEGIES
            print(f"  Found {len(strategies)} strategies:")
            for name, config in strategies.items():
                print(f"    - {name}: {config['description'][:50]}...")
            print("  SUCCESS: STRATEGIES dictionary is properly configured")
        else:
            print("  ✗ STRATEGIES dictionary not found")
            return False
        
        # Test 4: Check helper functions
        print("\n✓ Test 4: Checking helper functions...")
        required_functions = [
            'fetch_historical_data',
            'plot_equity_curve',
            'plot_drawdown',
            'plot_trade_distribution'
        ]
        
        for func in required_functions:
            if hasattr(backtest_page, func):
                print(f"  ✓ Found function: {func}")
            else:
                print(f"  ✗ Missing function: {func}")
                return False
        
        print("  SUCCESS: All helper functions found")
        
        # Test 5: Test strategy function with sample data
        print("\n✓ Test 5: Testing strategy function with sample data...")
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample data
        dates = pd.date_range(start=datetime.now() - timedelta(days=100), periods=100, freq='D')
        sample_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Test each strategy
        for strategy_name, strategy_config in strategies.items():
            try:
                signal = strategy_config['func'](sample_data)
                if signal in ['buy', 'sell', 'hold']:
                    print(f"  ✓ {strategy_name} returned valid signal: {signal}")
                else:
                    print(f"  ✗ {strategy_name} returned invalid signal: {signal}")
                    return False
            except Exception as e:
                print(f"  ✗ {strategy_name} raised error: {e}")
                return False
        
        print("  SUCCESS: All strategies work with sample data")
        
        print("\n" + "=" * 60)
        print("✅ ALL VERIFICATION TESTS PASSED")
        print("=" * 60)
        print("\nThe backtesting page is ready to use!")
        print("\nTo run the page:")
        print("  streamlit run pages/5_backtest.py")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install streamlit pandas numpy plotly")
        return False
    
    except Exception as e:
        print(f"\n✗ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_backtest_page()
    sys.exit(0 if success else 1)
