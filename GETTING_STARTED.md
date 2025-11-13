# Getting Started with Quant Trading System

## 🚀 Quick Start Guide

### Option 1: Web Interface (Recommended for Beginners)

The easiest way to get started is using the web interface:

1. **Start the web server**:
   ```bash
   python web_app.py
   ```

2. **Open your browser**:
   Navigate to `http://localhost:5000`

3. **Configure your system**:
   - Set trading mode (Simulation recommended)
   - Add symbols to trade (e.g., AAPL, GOOGL, MSFT)
   - Set initial capital
   - Click "Save Configuration"

4. **Start trading**:
   - Click "Start System"
   - Watch your portfolio in real-time
   - Monitor positions and P&L

5. **Run a backtest** (optional):
   - Scroll to Backtesting section
   - Select date range
   - Enter symbols
   - Click "Run Backtest"

### Option 2: Command Line Interface

For advanced users who prefer the terminal:

1. **Validate configuration**:
   ```bash
   python main.py config
   ```

2. **Run a backtest**:
   ```bash
   python main.py backtest --start-date 2024-01-01 --end-date 2024-12-31 --symbols AAPL
   ```

3. **Start trading**:
   ```bash
   python main.py run
   ```

## 📊 Your First Strategy

The system comes with a Moving Average Crossover strategy pre-configured:

- **Fast MA**: 10 periods
- **Slow MA**: 30 periods
- **Logic**: Buy when fast MA crosses above slow MA, sell when it crosses below

### How it works:

1. System fetches market data every 60 seconds
2. Strategy calculates moving averages
3. Generates buy/sell signals on crossovers
4. Risk manager validates orders
5. Orders are executed through the broker
6. Portfolio is updated automatically

## 🎯 Recommended Learning Path

### Day 1: Simulation Mode
- Start with web interface
- Use simulation mode only
- Trade 2-3 symbols (AAPL, GOOGL)
- Monitor for a few hours
- Review logs in `logs/` directory

### Day 2: Backtesting
- Run backtests on historical data
- Try different date ranges
- Experiment with different symbols
- Analyze performance metrics

### Day 3: Configuration
- Adjust strategy parameters
- Modify risk limits
- Try different update intervals
- Test with more symbols

### Day 4: Custom Strategy
- Study `examples/custom_strategy_template.py`
- Create your own strategy
- Backtest it thoroughly
- Compare with default strategy

### Week 2+: Advanced Topics
- Multiple strategies simultaneously
- Advanced risk management
- Performance optimization
- Consider live trading (with caution!)

## 🛡️ Safety First

### Before Live Trading:

1. ✅ Test in simulation mode for at least 2 weeks
2. ✅ Run extensive backtests
3. ✅ Understand all risk parameters
4. ✅ Start with small capital
5. ✅ Monitor closely for first week
6. ✅ Have stop-loss limits configured

### Red Flags to Watch:

- ⚠️ Frequent order rejections
- ⚠️ Unexpected position sizes
- ⚠️ Rapid portfolio value changes
- ⚠️ Strategy generating too many signals
- ⚠️ Daily loss limit being hit

## 📈 Understanding the Dashboard

### Portfolio Overview
- **Total Value**: Current portfolio worth (cash + positions)
- **Cash**: Available buying power
- **Positions Value**: Market value of all holdings
- **Unrealized P&L**: Profit/loss on open positions
- **Total Return**: Overall performance since start

### Positions Table
- **Symbol**: Stock ticker
- **Quantity**: Number of shares held
- **Avg Price**: Average purchase price
- **Current Price**: Latest market price
- **Market Value**: Current position worth
- **Unrealized P&L**: Profit/loss on position
- **P&L %**: Percentage return

## 🔧 Common Configurations

### Conservative Trading
```yaml
risk:
  max_position_size_pct: 5.0      # Small positions
  max_daily_loss_pct: 2.0         # Tight stop
  max_positions: 3                # Few positions
```

### Moderate Trading
```yaml
risk:
  max_position_size_pct: 10.0     # Medium positions
  max_daily_loss_pct: 5.0         # Standard stop
  max_positions: 5                # Moderate diversification
```

### Aggressive Trading
```yaml
risk:
  max_position_size_pct: 20.0     # Large positions
  max_daily_loss_pct: 10.0        # Wider stop
  max_positions: 10               # High diversification
```

## 📚 Next Steps

1. **Read the full documentation**: [README.md](README.md)
2. **Explore example strategies**: `examples/` directory
3. **Review test cases**: `tests/` directory
4. **Check troubleshooting guide**: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
5. **Join the community**: Share your strategies and learn from others

## 💡 Pro Tips

1. **Start small**: Use simulation mode with small capital
2. **Log everything**: Review logs regularly to understand system behavior
3. **Backtest thoroughly**: Test strategies on multiple time periods
4. **Monitor closely**: Watch the system especially in first few days
5. **Iterate slowly**: Make one change at a time and observe results
6. **Stay informed**: Keep up with market news that affects your symbols
7. **Have a plan**: Know when to stop trading (daily loss limit, time of day, etc.)

## ❓ Need Help?

- Check logs in `logs/` directory
- Run `python main.py config` to validate setup
- Review [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- Test with simulation mode first
- Start with fewer symbols and simpler strategies

## ⚠️ Important Reminders

- This is educational software
- Trading involves substantial risk
- Past performance ≠ future results
- Always test in simulation first
- Never trade with money you can't afford to lose
- The authors are not responsible for financial losses

---

**Ready to start?** Run `python web_app.py` and open `http://localhost:5000` in your browser!
