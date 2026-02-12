# Trading Schedule Management Guide

## Overview

The Trading Schedule Management feature allows you to configure when automated trading is allowed to execute trades. This is essential for:

- **Risk Management**: Prevent trades during volatile market open/close periods
- **Asset-Specific Trading**: Different schedules for stocks, crypto, and forex
- **Compliance**: Ensure trading only occurs during approved hours
- **Strategy Optimization**: Trade only during high-liquidity periods

## Key Concepts

### Trading Schedule

A `TradingSchedule` defines when trading is allowed for a specific asset class:

- **Active Days**: Which days of the week trading is allowed (0=Monday, 6=Sunday)
- **Trading Hours**: Start and end times for trading (24-hour format)
- **Asset Class**: Stocks, Crypto, Forex, or Options
- **Timezone**: Timezone for the schedule (e.g., "US/Eastern", "UTC")
- **Enabled**: Whether the schedule is active

### Asset Classes

The system supports different schedules for different asset classes:

- **STOCKS**: Traditional stock market (default: Mon-Fri, 9:30 AM - 4:00 PM ET)
- **CRYPTO**: Cryptocurrency markets (default: 24/7)
- **FOREX**: Foreign exchange markets (default: Mon-Fri, 24 hours)
- **OPTIONS**: Options trading (configurable)

### Automated Trading Mode

When automated trading is enabled, the system enforces trading schedules:

- **Enabled**: Orders are checked against schedules before execution
- **Disabled**: Schedules are not enforced (manual trading mode)

## Basic Usage

### 1. Initialize Trading Service

```python
from services.trading_service import TradingService

service = TradingService(
    api_key='your_api_key',
    api_secret='your_api_secret',
    paper=True  # Use paper trading for testing
)
```

### 2. View Default Schedules

```python
from services.trading_service import AssetClass

# Get all schedules
schedules = service.get_all_schedules()

# Get specific schedule
stock_schedule = service.get_trading_schedule(AssetClass.STOCKS)
print(stock_schedule.get_schedule_description())
# Output: "Stocks: Mon, Tue, Wed, Thu, Fri, 09:30 AM - 04:00 PM (US/Eastern)"
```

### 3. Check if Trading is Allowed

```python
from datetime import datetime

# Check current time
is_allowed = service.is_trading_allowed(AssetClass.STOCKS)
print(f"Trading allowed now: {is_allowed}")

# Check specific time
check_time = datetime(2024, 1, 1, 10, 0)  # Monday 10:00 AM
is_allowed = service.is_trading_allowed(AssetClass.STOCKS, check_time)
```

### 4. Create Custom Schedule

```python
from services.trading_service import TradingSchedule, AssetClass
from datetime import time as dt_time

# Create custom schedule (shorter trading hours)
custom_schedule = TradingSchedule(
    active_days={0, 1, 2, 3, 4},  # Mon-Fri
    start_time=dt_time(10, 0),    # 10:00 AM
    end_time=dt_time(15, 0),      # 3:00 PM
    asset_class=AssetClass.STOCKS,
    timezone="US/Eastern",
    enabled=True
)

# Validate schedule
errors = custom_schedule.validate()
if errors:
    print(f"Validation errors: {errors}")
else:
    # Set the schedule
    service.set_trading_schedule(AssetClass.STOCKS, custom_schedule)
```

### 5. Enable Automated Trading

```python
# Enable automated trading (enforces schedules)
service.enable_automated_trading()

# Check status
print(f"Automated: {service.is_automated_trading_enabled()}")

# Disable automated trading
service.disable_automated_trading()
```

### 6. Place Orders with Schedule Check

```python
from services.trading_service import AssetClass

# Place order with schedule enforcement
try:
    order = service.place_order_with_schedule_check(
        symbol='AAPL',
        qty=10,
        side='buy',
        asset_class=AssetClass.STOCKS,
        override_schedule=False  # Enforce schedule
    )
    print(f"Order placed: {order.order_id}")
except ValueError as e:
    print(f"Order rejected: {e}")

# Manual order (bypass schedule)
order = service.place_order_with_schedule_check(
    symbol='AAPL',
    qty=10,
    side='buy',
    asset_class=AssetClass.STOCKS,
    override_schedule=True  # Bypass schedule check
)
```

## Advanced Features

### Schedule Validation

Validate schedules against typical market hours:

```python
warnings = service.validate_schedule_against_market_hours(AssetClass.STOCKS)

if warnings:
    print("Schedule warnings:")
    for warning in warnings:
        print(f"  - {warning}")
```

Common warnings:
- Trading before market open
- Trading after market close
- Trading on weekends (for stocks)
- Trading on weekends (for forex)

### Schedule Status

Get comprehensive status for all schedules:

```python
status = service.get_schedule_status()

print(f"Automated Trading: {status['automated_trading_enabled']}")
print(f"Current Time: {status['current_time']}")

for asset_class, info in status['schedules'].items():
    print(f"\n{asset_class}:")
    print(f"  Trading Allowed: {info['trading_allowed_now']}")
    print(f"  Next Window: {info['next_trading_window']}")
```

### Schedules Crossing Midnight

For 24-hour markets or schedules that cross midnight:

```python
# Schedule from 11:00 PM to 2:00 AM
overnight_schedule = TradingSchedule(
    active_days={0, 1, 2, 3, 4, 5, 6},
    start_time=dt_time(23, 0),  # 11:00 PM
    end_time=dt_time(2, 0),     # 2:00 AM
    asset_class=AssetClass.CRYPTO
)

# Trading is allowed from 11:00 PM to 2:00 AM
# Not allowed from 2:00 AM to 11:00 PM
```

## Common Scenarios

### Scenario 1: Conservative Stock Trading

Only trade during mid-day hours to avoid volatility:

```python
conservative_schedule = TradingSchedule(
    active_days={0, 1, 2, 3, 4},  # Mon-Fri only
    start_time=dt_time(10, 30),   # 10:30 AM (after open volatility)
    end_time=dt_time(15, 30),     # 3:30 PM (before close volatility)
    asset_class=AssetClass.STOCKS,
    timezone="US/Eastern"
)

service.set_trading_schedule(AssetClass.STOCKS, conservative_schedule)
service.enable_automated_trading()
```

### Scenario 2: 24/7 Crypto Trading

Enable round-the-clock cryptocurrency trading:

```python
crypto_schedule = TradingSchedule(
    active_days={0, 1, 2, 3, 4, 5, 6},  # All days
    start_time=dt_time(0, 0),           # Midnight
    end_time=dt_time(23, 59),           # 11:59 PM
    asset_class=AssetClass.CRYPTO,
    timezone="UTC"
)

service.set_trading_schedule(AssetClass.CRYPTO, crypto_schedule)
```

### Scenario 3: Forex Trading (Mon-Fri)

Forex markets are open 24 hours but closed on weekends:

```python
forex_schedule = TradingSchedule(
    active_days={0, 1, 2, 3, 4},  # Mon-Fri only
    start_time=dt_time(0, 0),     # Midnight
    end_time=dt_time(23, 59),     # 11:59 PM
    asset_class=AssetClass.FOREX,
    timezone="UTC"
)

service.set_trading_schedule(AssetClass.FOREX, forex_schedule)
```

### Scenario 4: Testing Schedule

For testing, create a schedule that's always active:

```python
test_schedule = TradingSchedule(
    active_days={0, 1, 2, 3, 4, 5, 6},
    start_time=dt_time(0, 0),
    end_time=dt_time(23, 59),
    asset_class=AssetClass.STOCKS,
    enabled=True
)

service.set_trading_schedule(AssetClass.STOCKS, test_schedule)
```

## Best Practices

### 1. Validate Before Setting

Always validate schedules before applying them:

```python
errors = schedule.validate()
if errors:
    print(f"Invalid schedule: {errors}")
    return

service.set_trading_schedule(asset_class, schedule)
```

### 2. Check Market Hours

Validate against typical market hours to avoid issues:

```python
warnings = service.validate_schedule_against_market_hours(AssetClass.STOCKS)
if warnings:
    print("Warning: Schedule may be outside normal market hours")
    for warning in warnings:
        print(f"  - {warning}")
```

### 3. Use Override for Manual Trading

When placing manual orders, use `override_schedule=True`:

```python
# Manual order - bypass schedule
order = service.place_order_with_schedule_check(
    symbol='AAPL',
    qty=10,
    side='buy',
    override_schedule=True  # Manual trading
)
```

### 4. Monitor Schedule Status

Regularly check schedule status in your application:

```python
status = service.get_schedule_status()

if not status['automated_trading_enabled']:
    print("Warning: Automated trading is disabled")

for asset_class, info in status['schedules'].items():
    if not info['trading_allowed_now']:
        print(f"{asset_class}: Trading not allowed until {info['next_trading_window']}")
```

### 5. Test with Paper Trading

Always test schedules with paper trading first:

```python
# Use paper trading for testing
service = TradingService(
    api_key='your_api_key',
    api_secret='your_api_secret',
    paper=True  # Paper trading mode
)
```

## Requirements Validation

This implementation satisfies the following requirements:

### Requirement 18.1
✅ **Trading hour configuration**: Users can set active trading hours and days for each asset class

### Requirement 18.2
✅ **Schedule enforcement**: When automated trading is enabled, orders are only executed during configured hours

### Requirement 18.3
✅ **Resume trading**: System automatically resumes trading when schedule window begins

### Requirement 18.4
✅ **Asset-specific schedules**: Different schedules for stocks, crypto, and forex

### Requirement 18.5
✅ **Schedule validation**: System warns users when schedules conflict with typical market hours

## Troubleshooting

### Issue: Orders Rejected Outside Hours

**Problem**: Orders are being rejected even though you want to trade

**Solution**: Either disable automated trading or use `override_schedule=True`:

```python
# Option 1: Disable automated trading
service.disable_automated_trading()

# Option 2: Override schedule for specific order
order = service.place_order_with_schedule_check(
    symbol='AAPL',
    qty=10,
    side='buy',
    override_schedule=True
)
```

### Issue: Schedule Not Working

**Problem**: Schedule doesn't seem to be enforced

**Solution**: Ensure automated trading is enabled:

```python
# Check status
if not service.is_automated_trading_enabled():
    service.enable_automated_trading()
```

### Issue: Validation Errors

**Problem**: Schedule validation fails

**Solution**: Check common issues:

```python
# Ensure active days are specified
schedule.active_days = {0, 1, 2, 3, 4}  # Not empty

# Ensure start and end times are different
schedule.start_time != schedule.end_time

# Ensure asset class matches
schedule.asset_class == AssetClass.STOCKS
```

## API Reference

See the full API documentation in the code:

- `TradingSchedule`: Schedule configuration dataclass
- `AssetClass`: Asset class enumeration
- `TradingService.set_trading_schedule()`: Set schedule for asset class
- `TradingService.get_trading_schedule()`: Get schedule for asset class
- `TradingService.is_trading_allowed()`: Check if trading is allowed
- `TradingService.enable_automated_trading()`: Enable schedule enforcement
- `TradingService.disable_automated_trading()`: Disable schedule enforcement
- `TradingService.place_order_with_schedule_check()`: Place order with schedule check
- `TradingService.validate_schedule_against_market_hours()`: Validate schedule
- `TradingService.get_schedule_status()`: Get comprehensive schedule status

## Examples

See `examples/trading_schedule_example.py` for a complete working example.
