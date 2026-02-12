"""
Verification script for trading schedule management.

This script demonstrates and verifies the trading schedule functionality.
"""

from datetime import time as dt_time, datetime
from services.trading_service import TradingSchedule, AssetClass


def verify_schedule_creation():
    """Verify schedule creation and validation"""
    print("=" * 60)
    print("1. Schedule Creation and Validation")
    print("=" * 60)
    
    # Create valid schedule
    schedule = TradingSchedule(
        active_days={0, 1, 2, 3, 4},
        start_time=dt_time(9, 30),
        end_time=dt_time(16, 0),
        asset_class=AssetClass.STOCKS
    )
    
    print(f"Schedule created: {schedule.get_schedule_description()}")
    
    # Validate
    errors = schedule.validate()
    if errors:
        print(f"❌ Validation failed: {errors}")
        return False
    else:
        print("✅ Schedule validation passed")
    
    return True


def verify_trading_allowed_logic():
    """Verify is_trading_allowed logic"""
    print("\n" + "=" * 60)
    print("2. Trading Allowed Logic")
    print("=" * 60)
    
    schedule = TradingSchedule(
        active_days={0, 1, 2, 3, 4},  # Mon-Fri
        start_time=dt_time(9, 30),
        end_time=dt_time(16, 0),
        asset_class=AssetClass.STOCKS
    )
    
    # Test during trading hours (Monday 10:00 AM)
    check_time = datetime(2024, 1, 1, 10, 0)  # Monday
    is_allowed = schedule.is_trading_allowed(check_time)
    print(f"Monday 10:00 AM: {'✅ Allowed' if is_allowed else '❌ Not allowed'}")
    
    if not is_allowed:
        return False
    
    # Test before trading hours (Monday 8:00 AM)
    check_time = datetime(2024, 1, 1, 8, 0)
    is_allowed = schedule.is_trading_allowed(check_time)
    print(f"Monday 8:00 AM: {'❌ Allowed (should not be)' if is_allowed else '✅ Not allowed'}")
    
    if is_allowed:
        return False
    
    # Test on weekend (Saturday 10:00 AM)
    check_time = datetime(2024, 1, 6, 10, 0)  # Saturday
    is_allowed = schedule.is_trading_allowed(check_time)
    print(f"Saturday 10:00 AM: {'❌ Allowed (should not be)' if is_allowed else '✅ Not allowed'}")
    
    if is_allowed:
        return False
    
    return True


def verify_schedule_crossing_midnight():
    """Verify schedules that cross midnight"""
    print("\n" + "=" * 60)
    print("3. Schedule Crossing Midnight")
    print("=" * 60)
    
    schedule = TradingSchedule(
        active_days={0, 1, 2, 3, 4, 5, 6},
        start_time=dt_time(23, 0),  # 11:00 PM
        end_time=dt_time(2, 0),     # 2:00 AM
        asset_class=AssetClass.CRYPTO
    )
    
    print(f"Schedule: {schedule.get_schedule_description()}")
    
    # Test at 11:30 PM (should be allowed)
    check_time = datetime(2024, 1, 1, 23, 30)
    is_allowed = schedule.is_trading_allowed(check_time)
    print(f"11:30 PM: {'✅ Allowed' if is_allowed else '❌ Not allowed'}")
    
    if not is_allowed:
        return False
    
    # Test at 1:00 AM (should be allowed)
    check_time = datetime(2024, 1, 1, 1, 0)
    is_allowed = schedule.is_trading_allowed(check_time)
    print(f"1:00 AM: {'✅ Allowed' if is_allowed else '❌ Not allowed'}")
    
    if not is_allowed:
        return False
    
    # Test at 3:00 AM (should not be allowed)
    check_time = datetime(2024, 1, 1, 3, 0)
    is_allowed = schedule.is_trading_allowed(check_time)
    print(f"3:00 AM: {'❌ Allowed (should not be)' if is_allowed else '✅ Not allowed'}")
    
    if is_allowed:
        return False
    
    return True


def verify_asset_class_schedules():
    """Verify different schedules for different asset classes"""
    print("\n" + "=" * 60)
    print("4. Asset Class Specific Schedules")
    print("=" * 60)
    
    # Stock schedule (Mon-Fri, 9:30 AM - 4:00 PM)
    stock_schedule = TradingSchedule(
        active_days={0, 1, 2, 3, 4},
        start_time=dt_time(9, 30),
        end_time=dt_time(16, 0),
        asset_class=AssetClass.STOCKS
    )
    
    # Crypto schedule (24/7)
    crypto_schedule = TradingSchedule(
        active_days={0, 1, 2, 3, 4, 5, 6},
        start_time=dt_time(0, 0),
        end_time=dt_time(23, 59),
        asset_class=AssetClass.CRYPTO
    )
    
    # Forex schedule (Mon-Fri, 24 hours)
    forex_schedule = TradingSchedule(
        active_days={0, 1, 2, 3, 4},
        start_time=dt_time(0, 0),
        end_time=dt_time(23, 59),
        asset_class=AssetClass.FOREX
    )
    
    print(f"Stocks: {stock_schedule.get_schedule_description()}")
    print(f"Crypto: {crypto_schedule.get_schedule_description()}")
    print(f"Forex: {forex_schedule.get_schedule_description()}")
    
    # Test Saturday at 10:00 AM
    check_time = datetime(2024, 1, 6, 10, 0)  # Saturday
    
    stock_allowed = stock_schedule.is_trading_allowed(check_time)
    crypto_allowed = crypto_schedule.is_trading_allowed(check_time)
    forex_allowed = forex_schedule.is_trading_allowed(check_time)
    
    print(f"\nSaturday 10:00 AM:")
    print(f"  Stocks: {'❌ Allowed (should not be)' if stock_allowed else '✅ Not allowed'}")
    print(f"  Crypto: {'✅ Allowed' if crypto_allowed else '❌ Not allowed'}")
    print(f"  Forex: {'❌ Allowed (should not be)' if forex_allowed else '✅ Not allowed'}")
    
    if stock_allowed or not crypto_allowed or forex_allowed:
        return False
    
    return True


def verify_schedule_validation():
    """Verify schedule validation catches errors"""
    print("\n" + "=" * 60)
    print("5. Schedule Validation")
    print("=" * 60)
    
    # Invalid: No active days
    invalid_schedule = TradingSchedule(
        active_days=set(),
        start_time=dt_time(9, 30),
        end_time=dt_time(16, 0),
        asset_class=AssetClass.STOCKS
    )
    
    errors = invalid_schedule.validate()
    if errors:
        print(f"✅ Correctly caught error: {errors[0]}")
    else:
        print("❌ Failed to catch validation error")
        return False
    
    # Invalid: Same start and end time
    invalid_schedule2 = TradingSchedule(
        active_days={0, 1, 2, 3, 4},
        start_time=dt_time(9, 30),
        end_time=dt_time(9, 30),
        asset_class=AssetClass.STOCKS
    )
    
    errors = invalid_schedule2.validate()
    if errors:
        print(f"✅ Correctly caught error: {errors[0]}")
    else:
        print("❌ Failed to catch validation error")
        return False
    
    return True


def main():
    """Run all verification tests"""
    print("\n" + "=" * 60)
    print("Trading Schedule Management Verification")
    print("=" * 60)
    
    tests = [
        ("Schedule Creation", verify_schedule_creation),
        ("Trading Allowed Logic", verify_trading_allowed_logic),
        ("Midnight Crossing", verify_schedule_crossing_midnight),
        ("Asset Class Schedules", verify_asset_class_schedules),
        ("Schedule Validation", verify_schedule_validation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All verification tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
