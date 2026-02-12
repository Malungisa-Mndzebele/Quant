"""
Comprehensive test suite runner for the Quantitative Trading System.

This script runs all verification scripts and provides a complete status report.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✓ {description} - PASSED")
            if result.stdout:
                print(result.stdout[:500])
            return True
        else:
            print(f"✗ {description} - FAILED")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⚠ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ {description} - ERROR: {e}")
        return False

def main():
    """Run full test suite."""
    print("="*70)
    print("QUANTITATIVE TRADING SYSTEM - FULL TEST SUITE")
    print("="*70)
    
    results = {}
    
    # 1. Unit tests
    results['Unit Tests'] = run_command(
        'pytest tests/ -v --tb=short -x',
        'Unit Tests (pytest)'
    )
    
    # 2. Verification scripts
    verification_scripts = [
        ('verify_setup.py', 'Setup Verification'),
        ('verify_export_functionality.py', 'Export Functionality'),
        ('verify_paper_trading.py', 'Paper Trading'),
        ('verify_risk_service.py', 'Risk Service'),
        ('verify_alert_service.py', 'Alert Service'),
        ('verify_watchlist.py', 'Watchlist Service'),
        ('verify_personalization.py', 'Personalization'),
        ('verify_error_recovery.py', 'Error Recovery'),
        ('verify_multi_asset_support.py', 'Multi-Asset Support'),
        ('verify_scenario_service.py', 'Scenario Service'),
        ('verify_strategies.py', 'Strategy Configuration'),
        ('verify_trading_schedule.py', 'Trading Schedule'),
    ]
    
    for script, description in verification_scripts:
        if Path(script).exists():
            results[description] = run_command(
                f'python {script}',
                description
            )
        else:
            print(f"⚠ {description} - Script not found: {script}")
            results[description] = None
    
    # 3. Import tests
    results['Core Imports'] = run_command(
        'python -c "from services.portfolio_service import PortfolioService; '
        'from services.trading_service import TradingService; '
        'from services.market_data_service import MarketDataService; '
        'print(\'✓ All core imports successful\')"',
        'Core Module Imports'
    )
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)
    
    for test_name, status in results.items():
        if status is True:
            print(f"✓ {test_name}")
        elif status is False:
            print(f"✗ {test_name}")
        else:
            print(f"⚠ {test_name} (skipped)")
    
    print("\n" + "="*70)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print(f"Success Rate: {(passed/total*100):.1f}%")
    print("="*70)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
