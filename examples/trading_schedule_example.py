"""
Example: Trading Schedule Management

This example demonstrates how to configure and use trading schedules
for automated trading with different asset classes.
"""

from datetime import time as dt_time, datetime
from services.trading_service import TradingService, TradingSchedule, AssetClass


def main():
    """Demonstrate trading schedule management"""
    
    # Initialize trading service (paper trading mode)
    service = TradingService(
        api_key='your_api_key',
        api_secret='your_api_secret',
        paper=True
    )
    
    print("=" * 60)
    print("Trading Schedule Management Example")
    print("=" * 60)
    
    # 1. View default schedules
    print("\n1. Default Schedules:")
    print("-" * 60)
    schedules = service.get_all_schedules()
    for asset_class, schedule in schedules.items():
        print(f"\n{asset_class.value.upper()}:")
        print(f"  {schedule.get_schedule_description()}")
    
    # 2. Check if trading is currently allowed
    print("\n2. Current Trading Status:")
    print("-" * 60)
    for asset_class in [AssetClass.STOCKS, AssetClass.CRYPTO, AssetClass.FOREX]:
        is_allowed = service.is_trading_allowed(asset_class)
        status = "ALLOWED" if is_allowed else "NOT ALLOWED"
        print(f"{asset_class.value.upper()}: {status}")
    
    # 3. Create custom schedule for stocks (shorter trading hours)
    print("\n3. Setting Custom Stock Schedule:")
    print("-" * 60)
    custom_stock_schedule = TradingSchedule(
        active_days={0, 1, 2, 3, 4},  # Mon-Fri
        start_time=dt_time(10, 0),    # 10:00 AM
        end_time=dt_time(15, 0),      # 3:00 PM
        asset_class=AssetClass.STOCKS,
        timezone="US/Eastern",
        enabled=True
    )
    
    # Validate before setting
    errors = custom_stock_schedule.validate()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        service.set_trading_schedule(AssetClass.STOCKS, custom_stock_schedule)
        print(f"Custom schedule set: {custom_stock_schedule.get_schedule_description()}")
    
    # 4. Validate schedule against market hours
    print("\n4. Schedule Validation:")
    print("-" * 60)
    warnings = service.validate_schedule_against_market_hours(AssetClass.STOCKS)
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("No warnings - schedule is within normal market hours")
    
    # 5. Enable automated trading
    print("\n5. Automated Trading Mode:")
    print("-" * 60)
    print(f"Automated trading enabled: {service.is_automated_trading_enabled()}")
    
    service.enable_automated_trading()
    print(f"After enabling: {service.is_automated_trading_enabled()}")
    
    # 6. Try to place order with schedule enforcement
    print("\n6. Order Placement with Schedule Check:")
    print("-" * 60)
    
    try:
        # This will check if trading is allowed based on current time
        order = service.place_order_with_schedule_check(
            symbol='AAPL',
            qty=10,
            side='buy',
            asset_class=AssetClass.STOCKS,
            override_schedule=False  # Enforce schedule
        )
        print(f"Order placed successfully: {order.order_id}")
    except ValueError as e:
        print(f"Order rejected by schedule: {e}")
    
    # 7. Place order with schedule override (manual trading)
    print("\n7. Manual Order (Override Schedule):")
    print("-" * 60)
    
    try:
        order = service.place_order_with_schedule_check(
            symbol='AAPL',
            qty=10,
            side='buy',
            asset_class=AssetClass.STOCKS,
            override_schedule=True  # Bypass schedule check
        )
        print(f"Manual order placed: {order.order_id}")
    except Exception as e:
        print(f"Order failed: {e}")
    
    # 8. Get comprehensive schedule status
    print("\n8. Complete Schedule Status:")
    print("-" * 60)
    status = service.get_schedule_status()
    
    print(f"Automated Trading: {status['automated_trading_enabled']}")
    print(f"Current Time: {status['current_time']}")
    print("\nSchedules by Asset Class:")
    
    for asset_class, schedule_info in status['schedules'].items():
        print(f"\n{asset_class.upper()}:")
        print(f"  Enabled: {schedule_info['enabled']}")
        print(f"  Trading Allowed Now: {schedule_info['trading_allowed_now']}")
        print(f"  Active Days: {schedule_info['active_days']}")
        print(f"  Hours: {schedule_info['start_time']} - {schedule_info['end_time']}")
        print(f"  Timezone: {schedule_info['timezone']}")
        if schedule_info['next_trading_window']:
            print(f"  Next Window: {schedule_info['next_trading_window']}")
    
    # 9. Create 24/7 crypto schedule
    print("\n9. 24/7 Crypto Trading Schedule:")
    print("-" * 60)
    crypto_schedule = TradingSchedule(
        active_days={0, 1, 2, 3, 4, 5, 6},  # All days
        start_time=dt_time(0, 0),           # Midnight
        end_time=dt_time(23, 59),           # 11:59 PM
        asset_class=AssetClass.CRYPTO,
        timezone="UTC",
        enabled=True
    )
    
    service.set_trading_schedule(AssetClass.CRYPTO, crypto_schedule)
    print(f"Crypto schedule: {crypto_schedule.get_schedule_description()}")
    print(f"Crypto trading allowed now: {service.is_trading_allowed(AssetClass.CRYPTO)}")
    
    # 10. Disable automated trading
    print("\n10. Disable Automated Trading:")
    print("-" * 60)
    service.disable_automated_trading()
    print(f"Automated trading enabled: {service.is_automated_trading_enabled()}")
    print("Note: With automated trading disabled, schedule checks are not enforced")
    
    print("\n" + "=" * 60)
    print("Example Complete")
    print("=" * 60)


if __name__ == '__main__':
    # Note: This example requires valid Alpaca API credentials
    # Set them in your .env file or pass them directly
    
    print("\nNote: This is a demonstration script.")
    print("To run with real API, update the credentials in the code.")
    print("\nFor now, showing the structure without actual API calls...\n")
    
    # Uncomment to run with real credentials:
    # main()
