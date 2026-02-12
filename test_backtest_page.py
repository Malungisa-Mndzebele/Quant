"""
Simple test for the backtesting page functionality.

Tests the core strategy functions without requiring Streamlit runtime.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import importlib.util


def test_strategies():
    """Test all strategy functions with sample data"""
    
    print("Testing Backtesting Page Strategies")
    print("=" * 60)
    
    # Load the module
    spec = importlib.util.spec_from_file_location("backtest_page", "pages/5_backtest.py")
    backtest_page = importlib.util.module_from_spec(spec)
    
    # Suppress Streamlit warnings during import
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(backtest_page)
    
    # Create sample data with realistic price movements
    dates = pd.date_range(start=datetime.now() - timedelta(days=100), periods=100, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.randn(100) * 0.02  # 2% daily volatility
    prices = 100 * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    strategies = backtest_page.STRATEGIES
    
    print(f"\nTesting {len(strategies)} strategies with 100 days of sample data\n")
    
    all_passed = True
    
    for strategy_name, strategy_config in strategies.items():
        try:
            # Test with full dataset
            signal = strategy_config['func'](sample_data)
            
            if signal not in ['buy', 'sell', 'hold']:
                print(f"✗ {strategy_name}: Invalid signal '{signal}'")
                all_passed = False
                continue
            
            # Test with minimal data
            minimal_signal = strategy_config['func'](sample_data.head(5))
            
            if minimal_signal not in ['buy', 'sell', 'hold']:
                print(f"✗ {strategy_name}: Invalid signal with minimal data")
                all_passed = False
                continue
            
            # Test with trending data
            trending_data = sample_data.copy()
            trending_data['close'] = np.linspace(90, 110, 100)
            trending_signal = strategy_config['func'](trending_data)
            
            if trending_signal not in ['buy', 'sell', 'hold']:
                print(f"✗ {strategy_name}: Invalid signal with trending data")
                all_passed = False
                continue
            
            print(f"✓ {strategy_name}")
            print(f"  - Random data: {signal}")
            print(f"  - Minimal data: {minimal_signal}")
            print(f"  - Trending data: {trending_signal}")
            
        except Exception as e:
            print(f"✗ {strategy_name}: Error - {str(e)}")
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ All strategy tests passed!")
        return True
    else:
        print("❌ Some strategy tests failed")
        return False


def test_strategy_parameters():
    """Test that strategy parameters are properly configured"""
    
    print("\nTesting Strategy Parameters")
    print("=" * 60)
    
    # Load the module
    spec = importlib.util.spec_from_file_location("backtest_page", "pages/5_backtest.py")
    backtest_page = importlib.util.module_from_spec(spec)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(backtest_page)
    
    strategies = backtest_page.STRATEGIES
    
    all_valid = True
    
    for strategy_name, strategy_config in strategies.items():
        params = strategy_config['default_params']
        
        # Check required parameters
        required_params = ['position_size', 'stop_loss_pct', 'take_profit_pct', 'commission_pct']
        
        for param in required_params:
            if param not in params:
                print(f"✗ {strategy_name}: Missing parameter '{param}'")
                all_valid = False
            elif not isinstance(params[param], (int, float)):
                print(f"✗ {strategy_name}: Parameter '{param}' is not numeric")
                all_valid = False
            elif params[param] < 0:
                print(f"✗ {strategy_name}: Parameter '{param}' is negative")
                all_valid = False
        
        if all_valid:
            print(f"✓ {strategy_name}: All parameters valid")
    
    print("=" * 60)
    
    if all_valid:
        print("✅ All parameter tests passed!")
        return True
    else:
        print("❌ Some parameter tests failed")
        return False


if __name__ == "__main__":
    test1_passed = test_strategies()
    test2_passed = test_strategy_parameters()
    
    if test1_passed and test2_passed:
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe backtesting page is working correctly!")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        sys.exit(1)
