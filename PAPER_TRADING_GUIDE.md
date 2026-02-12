# Paper Trading Guide

## Overview

The Paper Trading feature allows you to practice trading with virtual money using real-time market prices. This is perfect for testing strategies, learning the platform, and building confidence before risking real capital.

## Key Features

- **Virtual Portfolio**: Start with configurable virtual capital (default: $100,000)
- **Real-Time Prices**: All trades execute at actual market prices
- **Separate Tracking**: Paper trading performance is tracked independently from live trading
- **Visual Distinction**: Clear indicators show when you're in paper trading mode
- **Session Management**: Save and load paper trading sessions
- **Full Functionality**: All trading features available (market orders, limit orders, positions, etc.)

## Getting Started

### Initialize Paper Trading Service

```python
from services.paper_trading_service import PaperTradingService
from services.market_data_service import MarketDataService

# Initialize market data service
market_data = MarketDataService(
    api_key='your_api_key',
    api_secret='your_secret_key'
)

# Initialize paper trading with $100,000 virtual capital
paper_trading = PaperTradingService(
    market_data_service=market_data,
    initial_capital=100000.0
)
```

### Check Trading Mode

```python
# Verify you're in paper trading mode
assert paper_trading.is_paper_trading() == True
assert paper_trading.get_trading_mode() == 'paper'
```

## Trading Operations

### Place Market Orders

```python
# Buy 10 shares of AAPL at market price
order = paper_trading.place_order(
    symbol='AAPL',
    qty=10,
    side='buy',
    order_type='market'
)

print(f"Order ID: {order.order_id}")
print(f"Status: {order.status}")
print(f"Filled at: ${order.filled_avg_price:.2f}")
```

### Place Limit Orders

```python
# Place limit buy order at $150
order = paper_trading.place_order(
    symbol='AAPL',
    qty=10,
    side='buy',
    order_type='limit',
    limit_price=150.0
)

# Order will be pending if limit price is below market
if order.status == OrderStatus.PENDING:
    print("Order pending - waiting for price to reach limit")
```

### Sell Shares

```python
# Sell 5 shares of AAPL
order = paper_trading.place_order(
    symbol='AAPL',
    qty=5,
    side='sell',
    order_type='market'
)
```

### Close Entire Position

```python
# Close all shares of AAPL
order = paper_trading.close_position('AAPL')
print(f"Closed {order.quantity} shares")
```

## Portfolio Management

### View Account Information

```python
account = paper_trading.get_account()

print(f"Account ID: {account['account_id']}")
print(f"Cash: ${account['cash']:.2f}")
print(f"Portfolio Value: ${account['portfolio_value']:.2f}")
print(f"Total Return: ${account['total_return']:.2f} ({account['total_return_pct']:.2f}%)")
print(f"Mode: {account['mode']}")  # Shows "PAPER TRADING"
```

### View Open Positions

```python
positions = paper_trading.get_positions()

for position in positions:
    print(f"\n{position.symbol}:")
    print(f"  Quantity: {position.quantity}")
    print(f"  Entry Price: ${position.entry_price:.2f}")
    print(f"  Current Price: ${position.current_price:.2f}")
    print(f"  P&L: ${position.unrealized_pl:.2f} ({position.unrealized_pl_pct:.2f}%)")
```

### Get Performance Summary

```python
summary = paper_trading.get_performance_summary()

print(f"Initial Capital: ${summary['initial_capital']:.2f}")
print(f"Current Value: ${summary['current_value']:.2f}")
print(f"Total Return: ${summary['total_return']:.2f} ({summary['total_return_pct']:.2f}%)")
print(f"Open Positions: {summary['num_positions']}")
print(f"Winning Positions: {summary['winning_positions']}")
print(f"Losing Positions: {summary['losing_positions']}")
```

## Order Management

### Check Order Status

```python
# Get order status by ID
order_status = paper_trading.get_order_status(order_id)

print(f"Status: {order_status.status}")
print(f"Filled Quantity: {order_status.filled_qty}/{order_status.quantity}")
if order_status.filled_avg_price:
    print(f"Filled Price: ${order_status.filled_avg_price:.2f}")
```

### Cancel Pending Orders

```python
# Cancel a pending limit order
try:
    paper_trading.cancel_order(order_id)
    print("Order cancelled successfully")
except ValueError as e:
    print(f"Cannot cancel: {e}")
```

## Session Management

### Save Current Session

```python
# Save session with custom name
session_file = paper_trading.save_session('my_strategy_test')
print(f"Session saved to: {session_file}")

# Or save with auto-generated name
session_file = paper_trading.save_session()
```

### Reset Account

```python
# Reset to initial capital
paper_trading.reset_account()

# Or reset with new capital amount
paper_trading.reset_account(initial_capital=50000.0)
```

## Best Practices

### 1. Test Strategies Thoroughly

Use paper trading to test your strategies over different market conditions before going live:

```python
# Test a strategy
def test_strategy():
    # Buy signal
    order = paper_trading.place_order('AAPL', 10, 'buy', 'market')
    
    # Monitor position
    positions = paper_trading.get_positions()
    
    # Sell signal
    if should_sell(positions):
        paper_trading.close_position('AAPL')
    
    # Check performance
    summary = paper_trading.get_performance_summary()
    return summary['total_return_pct']
```

### 2. Track Performance Metrics

Regularly check your performance to understand what's working:

```python
def check_performance():
    summary = paper_trading.get_performance_summary()
    
    if summary['total_return_pct'] < -5:
        print("Strategy underperforming - review approach")
    elif summary['total_return_pct'] > 10:
        print("Strategy performing well - consider live trading")
```

### 3. Use Realistic Position Sizes

Practice with position sizes you would actually use in live trading:

```python
# Calculate position size based on portfolio percentage
account = paper_trading.get_account()
position_size_pct = 0.10  # 10% of portfolio
position_value = account['portfolio_value'] * position_size_pct

# Get current price
quote = market_data.get_latest_quote('AAPL')
current_price = (quote.bid_price + quote.ask_price) / 2

# Calculate shares to buy
shares = int(position_value / current_price)

order = paper_trading.place_order('AAPL', shares, 'buy', 'market')
```

### 4. Save Sessions Regularly

Save your paper trading sessions to track progress over time:

```python
# Save at end of each trading day
session_name = f"session_{datetime.now().strftime('%Y%m%d')}"
paper_trading.save_session(session_name)
```

## Visual Distinction from Live Trading

The paper trading service provides clear indicators that you're in simulation mode:

1. **Account Mode**: All account info includes `'mode': 'PAPER TRADING'`
2. **Account ID**: Paper trading accounts have IDs starting with `'paper_'`
3. **Order IDs**: Paper trading orders have IDs starting with `'paper_'`
4. **Separate Database**: Paper trading uses a completely separate database

### Example: Checking Trading Mode

```python
account = paper_trading.get_account()

if account['mode'] == 'PAPER TRADING':
    print("✓ You are in PAPER TRADING mode")
    print("  No real money is at risk")
else:
    print("⚠ WARNING: You are in LIVE TRADING mode")
```

## Common Scenarios

### Scenario 1: Testing a Day Trading Strategy

```python
# Start fresh
paper_trading.reset_account(initial_capital=25000.0)

# Execute multiple trades
symbols = ['AAPL', 'GOOGL', 'MSFT']

for symbol in symbols:
    # Buy
    paper_trading.place_order(symbol, 10, 'buy', 'market')
    
    # Wait for profit target or stop loss
    # ... (implement your logic)
    
    # Sell
    paper_trading.close_position(symbol)

# Check results
summary = paper_trading.get_performance_summary()
print(f"Day Trading P&L: ${summary['total_return']:.2f}")
```

### Scenario 2: Testing Risk Management

```python
# Test stop-loss strategy
def test_stop_loss():
    # Buy shares
    order = paper_trading.place_order('AAPL', 100, 'buy', 'market')
    entry_price = order.filled_avg_price
    
    # Set stop loss at 5%
    stop_loss_price = entry_price * 0.95
    
    # Monitor position
    while True:
        positions = paper_trading.get_positions()
        position = next((p for p in positions if p.symbol == 'AAPL'), None)
        
        if position and position.current_price <= stop_loss_price:
            # Stop loss triggered
            paper_trading.close_position('AAPL')
            print(f"Stop loss triggered at ${position.current_price:.2f}")
            break
```

### Scenario 3: Testing Portfolio Allocation

```python
# Test diversified portfolio
def build_portfolio():
    account = paper_trading.get_account()
    total_value = account['portfolio_value']
    
    # Allocate 20% to each of 5 stocks
    allocation = 0.20
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    for symbol in symbols:
        # Calculate position size
        position_value = total_value * allocation
        
        # Get current price
        quote = market_data.get_latest_quote(symbol)
        price = (quote.bid_price + quote.ask_price) / 2
        
        # Buy shares
        shares = int(position_value / price)
        paper_trading.place_order(symbol, shares, 'buy', 'market')
    
    # Check allocation
    positions = paper_trading.get_positions()
    for pos in positions:
        pct = (pos.market_value / total_value) * 100
        print(f"{pos.symbol}: {pct:.1f}%")
```

## Troubleshooting

### Issue: "Insufficient funds" error

**Solution**: Check your available cash before placing orders:

```python
account = paper_trading.get_account()
print(f"Available cash: ${account['cash']:.2f}")

# Calculate required cash
required = qty * estimated_price
if required > account['cash']:
    print(f"Need ${required:.2f}, have ${account['cash']:.2f}")
```

### Issue: "Insufficient shares" error

**Solution**: Check your position before selling:

```python
positions = paper_trading.get_positions()
position = next((p for p in positions if p.symbol == 'AAPL'), None)

if position:
    print(f"Available shares: {position.quantity}")
else:
    print("No position in AAPL")
```

### Issue: Limit order not filling

**Solution**: Limit orders only fill when the market price reaches your limit:

```python
# Check order status
order_status = paper_trading.get_order_status(order_id)

if order_status.status == OrderStatus.PENDING:
    print(f"Order pending - limit price: ${order_status.limit_price:.2f}")
    
    # Get current market price
    quote = market_data.get_latest_quote(order_status.symbol)
    current_price = (quote.bid_price + quote.ask_price) / 2
    print(f"Current market price: ${current_price:.2f}")
    
    # Cancel if needed
    paper_trading.cancel_order(order_id)
```

## Integration with Live Trading

When you're ready to transition from paper to live trading:

1. **Review Performance**: Ensure consistent positive results in paper trading
2. **Start Small**: Begin with smaller position sizes in live trading
3. **Use Same Strategy**: Apply the exact same strategy you tested
4. **Monitor Closely**: Watch live trades more carefully initially
5. **Keep Paper Trading**: Continue paper trading new strategies

### Example: Switching Between Modes

```python
# Paper trading
paper_service = PaperTradingService(
    market_data_service=market_data,
    initial_capital=100000.0
)

# Live trading (when ready)
live_service = TradingService(
    api_key='your_api_key',
    api_secret='your_secret_key',
    paper=False  # LIVE MODE
)

# Always check mode before trading
if paper_service.is_paper_trading():
    print("✓ Safe to experiment - paper trading mode")
else:
    print("⚠ CAUTION - live trading mode")
```

## Summary

Paper trading is an essential tool for:
- Learning the platform without risk
- Testing new strategies
- Building confidence
- Practicing risk management
- Validating trading ideas

Always use paper trading to test thoroughly before risking real capital!
