# Backtesting Page Quick Start Guide

## What is Backtesting?

Backtesting is the process of testing a trading strategy using historical data to see how it would have performed in the past. This helps you evaluate whether a strategy is worth using with real money.

## Accessing the Backtesting Page

Run the page directly:
```bash
streamlit run pages/5_backtest.py
```

Or access it through the main AI Trading Agent app navigation.

## Quick Start: Running Your First Backtest

### Step 1: Choose a Strategy
In the sidebar, select one of the pre-built strategies:
- **MA Crossover**: Good for trending markets
- **RSI Mean Reversion**: Good for range-bound markets
- **Momentum**: Good for volatile markets
- **Bollinger Bands**: Good for mean-reverting markets

### Step 2: Enter a Stock Symbol
Type a stock symbol (e.g., AAPL, TSLA, MSFT) in the "Stock Symbol" field.

### Step 3: Set Date Range
- **Start Date**: When to begin the backtest (default: 1 year ago)
- **End Date**: When to end the backtest (default: today)

Tip: Use at least 6 months of data for reliable results.

### Step 4: Set Initial Capital
Enter how much money you want to start with (default: $100,000).

### Step 5: Configure Parameters (Optional)
Adjust these settings to fine-tune the strategy:
- **Position Size**: How much of your capital to use per trade (default: 90-95%)
- **Stop Loss**: Maximum loss before automatically closing (default: 3-5%)
- **Take Profit**: Target profit before automatically closing (default: 8-12%)
- **Commission**: Trading fees per trade (default: 0.1%)

### Step 6: Run the Backtest
Click the "🚀 Run Backtest" button and wait for results.

## Understanding the Results

### Performance Metrics

**Total Return**: How much money you made or lost
- Example: $5,000 (5.0%) means you made $5,000 profit on $100,000

**Sharpe Ratio**: Risk-adjusted return (higher is better)
- < 1.0: Poor risk-adjusted returns
- 1.0-2.0: Good risk-adjusted returns
- > 2.0: Excellent risk-adjusted returns

**Max Drawdown**: Largest peak-to-trough decline
- Example: -$10,000 (-10%) means at worst, you were down $10,000

**Win Rate**: Percentage of profitable trades
- Example: 60% means 6 out of 10 trades were profitable

**Profit Factor**: Gross profit divided by gross loss
- < 1.0: Losing strategy
- 1.0-2.0: Decent strategy
- > 2.0: Strong strategy

### Charts

**Equity Curve**: Shows your portfolio value over time
- Upward slope = making money
- Downward slope = losing money
- Smooth line = consistent performance
- Jagged line = volatile performance

**Drawdown Chart**: Shows how much you're down from peak
- Deeper valleys = larger losses
- Flatter line = more stable strategy

**Trade P&L Distribution**: Shows spread of trade profits/losses
- Right-skewed = more winning trades
- Left-skewed = more losing trades
- Centered at zero = balanced wins/losses

### Trade-by-Trade Breakdown

Review every trade executed:
- Entry and exit times
- Entry and exit prices
- Profit/loss for each trade
- Commission costs

Download as CSV for detailed analysis in Excel.

## Comparing Multiple Strategies

Want to find the best strategy? Use comparison mode:

1. Check "Compare Multiple Strategies" in the sidebar
2. Configure parameters (applied to all strategies)
3. Click "Run Backtest"
4. View side-by-side comparison

The comparison shows:
- Which strategy had the highest return
- Which strategy had the best Sharpe ratio
- Which strategy had the lowest drawdown
- Which strategy had the most trades

## Tips for Better Backtesting

### Do's ✅
- Use at least 6-12 months of historical data
- Test multiple strategies to find what works
- Include realistic commission costs
- Consider both return AND risk (Sharpe ratio)
- Test on different stocks and market conditions

### Don'ts ❌
- Don't over-optimize parameters (curve fitting)
- Don't ignore transaction costs
- Don't assume past performance = future results
- Don't use too short of a time period
- Don't test only on one stock

## Strategy Descriptions

### MA Crossover
**When it works**: Trending markets (strong up or down trends)
**When it fails**: Choppy, sideways markets
**Best for**: Patient traders, medium-term holds

### RSI Mean Reversion
**When it works**: Range-bound markets (price bounces between levels)
**When it fails**: Strong trending markets
**Best for**: Active traders, short-term trades

### Momentum
**When it works**: Volatile markets with strong moves
**When it fails**: Low volatility, quiet markets
**Best for**: Aggressive traders, momentum plays

### Bollinger Bands
**When it works**: Mean-reverting markets
**When it fails**: Breakout scenarios
**Best for**: Technical traders, volatility plays

## Example Workflow

### Testing AAPL with MA Crossover

1. Select "MA Crossover" strategy
2. Enter "AAPL" as symbol
3. Set date range: Jan 1, 2023 - Dec 31, 2023
4. Set initial capital: $100,000
5. Keep default parameters
6. Click "Run Backtest"

**Interpreting Results**:
- If Total Return > 0: Strategy was profitable
- If Sharpe Ratio > 1.5: Good risk-adjusted returns
- If Win Rate > 50%: More winners than losers
- If Max Drawdown < 20%: Acceptable risk

### Comparing All Strategies on TSLA

1. Check "Compare Multiple Strategies"
2. Enter "TSLA" as symbol
3. Set date range: Last 6 months
4. Set initial capital: $50,000
5. Click "Run Backtest"

**Finding the Winner**:
- Look at Return % column (highest = most profitable)
- Look at Sharpe Ratio (highest = best risk-adjusted)
- Look at Max Drawdown % (lowest = least risky)
- Consider trade-off between return and risk

## Common Questions

**Q: Why did my strategy make no trades?**
A: The strategy conditions weren't met during the test period. Try:
- Longer date range
- Different stock
- Adjusted parameters

**Q: Why is my Sharpe ratio negative?**
A: The strategy lost money. Negative Sharpe means negative returns.

**Q: What's a good win rate?**
A: 50-60% is typical. Higher is better, but profit factor matters more.

**Q: Should I use the strategy with the highest return?**
A: Not necessarily. Consider risk (drawdown, Sharpe ratio) too.

**Q: How do I know if results are realistic?**
A: Include commission costs, use reasonable position sizes, and test on multiple stocks/periods.

## Next Steps

After backtesting:

1. **Paper Trade**: Test the strategy with virtual money in real-time
2. **Monitor Performance**: Track how the strategy performs going forward
3. **Adjust Parameters**: Fine-tune based on results
4. **Start Small**: If going live, start with small position sizes
5. **Review Regularly**: Check performance weekly/monthly

## Troubleshooting

**"No data available"**
- Check that symbol is valid (e.g., AAPL not Apple)
- Ensure date range is reasonable (not too far back)
- Verify Alpaca API credentials are configured

**"Error fetching data"**
- Check internet connection
- Verify API credentials in .env file
- Try a different symbol

**Page is slow**
- Reduce date range (test shorter periods)
- Don't use comparison mode for very long periods
- Close other browser tabs

## Summary

The backtesting page helps you:
- Test strategies before risking real money
- Compare different approaches
- Understand strategy strengths and weaknesses
- Make data-driven trading decisions

Remember: Past performance doesn't guarantee future results, but it's a valuable tool for strategy evaluation!
