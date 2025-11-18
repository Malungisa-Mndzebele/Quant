# Trading Strategy Guide

## Available Strategies

### 1. Moving Average Crossover

**Type:** Trend Following  
**Difficulty:** Beginner  
**Best For:** Trending markets

**Description:**
Uses two moving averages (fast and slow) to identify trend changes. Generates buy signals when the fast MA crosses above the slow MA, and sell signals when it crosses below.

**Parameters:**
- `fast_period`: Fast moving average period (default: 10)
- `slow_period`: Slow moving average period (default: 30)
- `quantity`: Shares per trade (default: 100)

**Configuration:**
```yaml
strategies:
  - name: "MovingAverageCrossover"
    enabled: true
    params:
      fast_period: 10
      slow_period: 30
      quantity: 100
```

**Pros:**
- Simple and easy to understand
- Works well in trending markets
- Clear entry and exit signals

**Cons:**
- Lags behind price action
- Generates false signals in ranging markets
- Can miss early trend reversals

---

### 2. Bollinger Bands Mean Reversion ⭐ NEW

**Type:** Mean Reversion  
**Difficulty:** Intermediate  
**Best For:** Ranging/sideways markets

**Description:**
Identifies overbought and oversold conditions using Bollinger Bands and RSI. Trades on the principle that prices tend to revert to their mean after extreme movements.

**Strategy Logic:**
1. **BUY Signal:**
   - Price touches or goes below lower Bollinger Band
   - AND RSI < 30 (oversold condition)

2. **SELL Signal:**
   - Price touches or goes above upper Bollinger Band
   - AND RSI > 70 (overbought condition)
   - OR price crosses back above middle band (mean reversion)

**Parameters:**
- `symbols`: List of symbols to trade
- `bb_period`: Bollinger Bands period (default: 20)
- `bb_std`: Standard deviations for bands (default: 2.0)
- `rsi_period`: RSI calculation period (default: 14)
- `rsi_oversold`: RSI oversold threshold (default: 30)
- `rsi_overbought`: RSI overbought threshold (default: 70)
- `quantity`: Shares per trade (default: 100)

**Configuration:**
```yaml
strategies:
  - name: "BollingerMeanReversion"
    enabled: true
    params:
      symbols: ["AAPL", "GOOGL", "MSFT"]
      bb_period: 20
      bb_std: 2.0
      rsi_period: 14
      rsi_oversold: 30
      rsi_overbought: 70
      quantity: 100
```

**Pros:**
- Effective in ranging markets
- Multiple confirmation signals (BB + RSI)
- Clear risk/reward setup
- Automatic exit on mean reversion

**Cons:**
- Can lose money in strong trends
- Requires sufficient volatility
- May generate false signals in choppy markets

**Parameter Tuning Guide:**

| Market Condition | BB Period | BB Std | RSI Oversold/Overbought |
|-----------------|-----------|--------|-------------------------|
| Low Volatility  | 20-25     | 1.5-2.0| 30/70                  |
| High Volatility | 20        | 2.5-3.0| 20/80                  |
| Trending        | 25-30     | 2.5    | 25/75                  |
| Ranging         | 15-20     | 2.0    | 30/70                  |

**Best Practices:**
- Backtest on at least 1 year of data
- Start with standard parameters (20, 2.0, 14)
- Adjust for individual stock volatility
- Use with stocks that have clear support/resistance
- Avoid during major news events

---

## Strategy Selection Guide

### Choose Moving Average Crossover if:
- ✅ Market is trending (up or down)
- ✅ You prefer simple, easy-to-understand strategies
- ✅ You want fewer but potentially larger moves
- ✅ You're comfortable with some lag in signals

### Choose Bollinger Mean Reversion if:
- ✅ Market is ranging/sideways
- ✅ You want to capture short-term reversals
- ✅ You prefer multiple confirmation signals
- ✅ You're trading stocks with clear volatility patterns

### Combine Both if:
- ✅ You want to trade in all market conditions
- ✅ You have sufficient capital for multiple strategies
- ✅ You want diversification across strategy types
- ✅ You can monitor and adjust parameters regularly

---

## Creating Custom Strategies

### Step 1: Use the Template

Start with `examples/custom_strategy_template.py`:

```python
from src.strategies.base import Strategy
from src.models.signal import Signal, SignalAction
from src.models.market_data import MarketData

class MyCustomStrategy(Strategy):
    def __init__(self, param1, param2):
        super().__init__(name="MyCustomStrategy")
        self.param1 = param1
        self.param2 = param2
    
    def on_data(self, market_data: MarketData) -> None:
        # Process incoming data
        pass
    
    def generate_signals(self) -> List[Signal]:
        # Generate trading signals
        return []
    
    def on_order_filled(self, order) -> None:
        # Handle filled orders
        pass
    
    def reset_state(self) -> None:
        # Reset for backtesting
        pass
```

### Step 2: Implement Your Logic

Add your trading logic to the methods above.

### Step 3: Register in Trading System

Add your strategy to `src/trading_system.py` in the `_create_strategy` method.

### Step 4: Add to Configuration

Add your strategy to `config.yaml`:

```yaml
strategies:
  - name: "MyCustomStrategy"
    enabled: true
    params:
      param1: value1
      param2: value2
```

### Step 5: Backtest

Test your strategy thoroughly:

```bash
python main.py backtest \\
  --start-date 2024-01-01 \\
  --end-date 2024-12-31 \\
  --symbols AAPL \\
  --strategy MyCustomStrategy \\
  --output results/my_strategy
```

---

## Backtesting Best Practices

### 1. Use Sufficient Data
- Minimum: 6 months
- Recommended: 1-2 years
- Ideal: 3-5 years including different market conditions

### 2. Test Multiple Symbols
- Don't optimize for just one stock
- Test on at least 5-10 different symbols
- Include different sectors and volatilities

### 3. Walk-Forward Testing
- Train on first 70% of data
- Test on remaining 30%
- Repeat with rolling windows

### 4. Key Metrics to Monitor
- **Total Return**: Overall profitability
- **Sharpe Ratio**: Risk-adjusted returns (>1.0 is good)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

### 5. Avoid Overfitting
- Don't over-optimize parameters
- Keep strategies simple
- Test on out-of-sample data
- Be skeptical of "too good" results

---

## Risk Management

### Position Sizing
```yaml
risk:
  max_position_size_pct: 10.0  # Max 10% per position
```

### Daily Loss Limits
```yaml
risk:
  max_daily_loss_pct: 5.0  # Stop at 5% daily loss
```

### Diversification
- Trade multiple symbols
- Use multiple strategies
- Vary position sizes based on confidence

### Stop Losses
- Always use stop losses
- Set based on volatility (e.g., 2x ATR)
- Don't move stops against your position

---

## Performance Monitoring

### Daily Review
- Check win rate
- Review largest losses
- Monitor drawdown
- Verify strategy is executing correctly

### Weekly Review
- Compare to benchmarks (S&P 500)
- Analyze strategy performance by symbol
- Review risk metrics
- Adjust parameters if needed

### Monthly Review
- Full performance analysis
- Strategy comparison
- Risk-adjusted returns
- Decision to continue/modify/stop

---

## Common Pitfalls

### 1. Over-Trading
- Too many signals = high costs
- Solution: Increase signal thresholds

### 2. Under-Diversification
- All eggs in one basket
- Solution: Trade multiple symbols/strategies

### 3. Ignoring Transaction Costs
- Commissions and slippage add up
- Solution: Factor in 0.1-0.5% per trade

### 4. Emotional Override
- Manually overriding the system
- Solution: Trust the backtest, follow the plan

### 5. Insufficient Testing
- Going live too quickly
- Solution: Minimum 2 weeks simulation mode

---

## Resources

### Books
- "Algorithmic Trading" by Ernest Chan
- "Quantitative Trading" by Ernest Chan
- "Trading Systems" by Emilio Tomasini

### Websites
- QuantConnect (backtesting platform)
- TradingView (charting and ideas)
- Investopedia (education)

### Tools
- Python libraries: pandas, numpy, ta-lib
- Backtesting frameworks: backtrader, zipline
- Data sources: yfinance, Alpha Vantage

---

## Getting Help

### Troubleshooting
1. Check logs in `logs/` directory
2. Verify configuration in `config.yaml`
3. Run in simulation mode first
4. Review strategy parameters

### Support
- GitHub Issues for bugs
- Documentation in `docs/`
- Examples in `examples/`
- Community forums

---

**Remember:** Past performance does not guarantee future results. Always test thoroughly in simulation mode before risking real capital.
