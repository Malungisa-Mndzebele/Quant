#!/usr/bin/env python
"""Test script to verify backtest service import."""

import sys

# Clear any cached imports
if 'services.backtest_service' in sys.modules:
    del sys.modules['services.backtest_service']

try:
    from services.backtest_service import BacktestEngine, Strategy, Trade, BacktestResult
    print("✓ Successfully imported all classes from backtest_service")
    
    # Test instantiation
    engine = BacktestEngine()
    print(f"✓ Created BacktestEngine with initial capital: ${engine.initial_capital:,.2f}")
    
    print("\n✓ All imports successful!")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
