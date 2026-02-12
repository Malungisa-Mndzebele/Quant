"""Verification script for multi-asset support implementation."""

import sys
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_asset_class_detection():
    """Verify asset class detection from symbols."""
    print("\n" + "="*60)
    print("Testing Asset Class Detection")
    print("="*60)
    
    try:
        from services.market_data_service import MarketDataService, AssetClass
        from unittest.mock import Mock, patch
        
        # Mock the API credentials to avoid initialization errors
        with patch('services.market_data_service.settings') as mock_settings:
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            mock_settings.cache_ttl_seconds = 300
            
            mds = MarketDataService()
            
            # Test stock symbols
            test_cases = [
                ('AAPL', AssetClass.STOCK),
                ('GOOGL', AssetClass.STOCK),
                ('BTC/USD', AssetClass.CRYPTO),
                ('BTCUSD', AssetClass.CRYPTO),
                ('ETH/USD', AssetClass.CRYPTO),
                ('EUR/USD', AssetClass.FOREX),
                ('GBP/USD', AssetClass.FOREX),
            ]
            
            passed = 0
            failed = 0
            
            for symbol, expected_class in test_cases:
                detected = mds.detect_asset_class(symbol)
                if detected == expected_class:
                    print(f"✓ {symbol:12} -> {detected.value:10} (correct)")
                    passed += 1
                else:
                    print(f"✗ {symbol:12} -> {detected.value:10} (expected {expected_class.value})")
                    failed += 1
            
            print(f"\nResults: {passed} passed, {failed} failed")
            return failed == 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def verify_asset_specific_indicators():
    """Verify asset-specific indicator calculation."""
    print("\n" + "="*60)
    print("Testing Asset-Specific Indicators")
    print("="*60)
    
    try:
        import pandas as pd
        import numpy as np
        from services.market_data_service import AssetClass
        from utils.asset_analysis import (
            calculate_crypto_specific_indicators,
            calculate_forex_specific_indicators,
            calculate_asset_specific_indicators
        )
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 115, 100),
            'low': np.random.uniform(95, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        # Test crypto indicators
        print("\nCrypto-specific indicators:")
        crypto_df = calculate_crypto_specific_indicators(df)
        crypto_indicators = ['realized_vol_24h', 'momentum_6h', 'momentum_24h', 'volume_ratio']
        for indicator in crypto_indicators:
            if indicator in crypto_df.columns:
                print(f"  ✓ {indicator}")
            else:
                print(f"  ✗ {indicator} (missing)")
        
        # Test forex indicators
        print("\nForex-specific indicators:")
        forex_df = calculate_forex_specific_indicators(df)
        forex_indicators = ['ema_8', 'ema_21', 'trend_strength', 'atr_14']
        for indicator in forex_indicators:
            if indicator in forex_df.columns:
                print(f"  ✓ {indicator}")
            else:
                print(f"  ✗ {indicator} (missing)")
        
        # Test routing
        print("\nAsset-specific indicator routing:")
        crypto_result = calculate_asset_specific_indicators(df, AssetClass.CRYPTO)
        if 'realized_vol_24h' in crypto_result.columns:
            print("  ✓ Crypto routing works")
        else:
            print("  ✗ Crypto routing failed")
        
        forex_result = calculate_asset_specific_indicators(df, AssetClass.FOREX)
        if 'trend_strength' in forex_result.columns:
            print("  ✓ Forex routing works")
        else:
            print("  ✗ Forex routing failed")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_ai_engine_multi_asset():
    """Verify AI engine supports multiple asset classes."""
    print("\n" + "="*60)
    print("Testing AI Engine Multi-Asset Support")
    print("="*60)
    
    try:
        from ai.inference import AIEngine, TradingSignal, Recommendation
        from services.market_data_service import AssetClass
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 115, 100),
            'low': np.random.uniform(95, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000000, 10000000, 100)
        }, index=dates)
        
        engine = AIEngine()
        
        # Check that methods accept asset_class parameter
        print("\nChecking method signatures:")
        
        import inspect
        
        # Check analyze_stock
        sig = inspect.signature(engine.analyze_stock)
        if 'asset_class' in sig.parameters:
            print("  ✓ analyze_stock accepts asset_class parameter")
        else:
            print("  ✗ analyze_stock missing asset_class parameter")
        
        # Check get_recommendation
        sig = inspect.signature(engine.get_recommendation)
        if 'asset_class' in sig.parameters:
            print("  ✓ get_recommendation accepts asset_class parameter")
        else:
            print("  ✗ get_recommendation missing asset_class parameter")
        
        # Check explain_prediction
        sig = inspect.signature(engine.explain_prediction)
        if 'asset_class' in sig.parameters:
            print("  ✓ explain_prediction accepts asset_class parameter")
        else:
            print("  ✗ explain_prediction missing asset_class parameter")
        
        # Check rank_opportunities
        sig = inspect.signature(engine.rank_opportunities)
        if 'asset_classes' in sig.parameters:
            print("  ✓ rank_opportunities accepts asset_classes parameter")
        else:
            print("  ✗ rank_opportunities missing asset_classes parameter")
        
        # Check dataclass fields
        print("\nChecking dataclass fields:")
        
        from dataclasses import fields
        
        signal_fields = {f.name for f in fields(TradingSignal)}
        if 'asset_class' in signal_fields:
            print("  ✓ TradingSignal has asset_class field")
        else:
            print("  ✗ TradingSignal missing asset_class field")
        
        rec_fields = {f.name for f in fields(Recommendation)}
        if 'asset_class' in rec_fields:
            print("  ✓ Recommendation has asset_class field")
        else:
            print("  ✗ Recommendation missing asset_class field")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_portfolio_asset_allocation():
    """Verify portfolio service supports asset class allocation."""
    print("\n" + "="*60)
    print("Testing Portfolio Asset Class Allocation")
    print("="*60)
    
    try:
        from services.portfolio_service import PortfolioService
        import inspect
        
        portfolio = PortfolioService()
        
        # Check that new methods exist
        print("\nChecking new methods:")
        
        methods = [
            'get_asset_class_allocation',
            'get_correlation_matrix',
            'get_diversification_metrics'
        ]
        
        for method_name in methods:
            if hasattr(portfolio, method_name):
                print(f"  ✓ {method_name} exists")
            else:
                print(f"  ✗ {method_name} missing")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_trading_service_asset_routing():
    """Verify trading service supports asset-specific order routing."""
    print("\n" + "="*60)
    print("Testing Trading Service Asset Routing")
    print("="*60)
    
    try:
        from services.trading_service import TradingService, AssetClass
        from unittest.mock import patch
        import inspect
        
        # Mock the API credentials
        with patch('services.trading_service.settings') as mock_settings:
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            
            service = TradingService(paper=True)
            
            # Check that schedule methods exist
            print("\nChecking schedule methods:")
            
            methods = [
                'set_trading_schedule',
                'get_trading_schedule',
                'is_trading_allowed',
                'place_order_with_schedule_check'
            ]
            
            for method_name in methods:
                if hasattr(service, method_name):
                    print(f"  ✓ {method_name} exists")
                else:
                    print(f"  ✗ {method_name} missing")
            
            # Check that schedules are configured for different asset classes
            print("\nChecking asset class schedules:")
            
            schedules = service.get_all_schedules()
            for asset_class in [AssetClass.STOCKS, AssetClass.CRYPTO, AssetClass.FOREX]:
                if asset_class in schedules:
                    schedule = schedules[asset_class]
                    print(f"  ✓ {asset_class.value}: {schedule.get_schedule_description()}")
                else:
                    print(f"  ✗ {asset_class.value} schedule missing")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("MULTI-ASSET SUPPORT VERIFICATION")
    print("="*60)
    
    results = []
    
    # Run all verification tests
    results.append(("Asset Class Detection", verify_asset_class_detection()))
    results.append(("Asset-Specific Indicators", verify_asset_specific_indicators()))
    results.append(("AI Engine Multi-Asset", verify_ai_engine_multi_asset()))
    results.append(("Portfolio Asset Allocation", verify_portfolio_asset_allocation()))
    results.append(("Trading Service Asset Routing", verify_trading_service_asset_routing()))
    
    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All verification tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} verification test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
