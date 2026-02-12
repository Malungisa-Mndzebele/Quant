# Backtesting Page Implementation

## Overview

The backtesting page (`pages/5_backtest.py`) provides comprehensive strategy backtesting functionality for the AI Trading Agent. Users can test trading strategies against historical data, evaluate performance metrics, and compare multiple strategies.

## Features Implemented

### ✅ Strategy Selection and Configuration
- **4 Pre-built Strategies**:
  - MA Crossover: Moving average crossover strategy
  - RSI Mean Reversion: RSI-based oversold/overbought strategy
  - Momentum: Momentum-based trend following
  - Bollinger Bands: Mean reversion using Bollinger Bands

- **Configurable Parameters**:
  - Position size (10-100%)
  - Stop-loss percentage (0-20%)
  - Take-profit percentage (0-50%)
  - Commission percentage (0-1%)

### ✅ Date Range and Capital Configuration
- Start and end date selection
- Initial capital input ($1,000 - $10,000,000)
- Symbol selection for any stock

### ✅ Backtest Execution
- Progress indicator during backtest
- Real-time status updates
- Historical data fetching from Alpaca API
- Strategy simulation with transaction costs

### ✅ Performance Metrics Display
- **Key Metrics**:
  - Total return ($ and %)
  - Sharpe ratio
  - Maximum drawdown ($ and %)
  - Win rate
  - Total trades
  - Winning/losing trades
  - Profit factor
  - Average win/loss
  - Largest win/loss

### ✅ Interactive Visualizations
- **Equity Curve**: Portfolio value over time
- **Drawdown Chart**: Drawdown percentage over time
- **Trade P&L Distribution**: Histogram of trade profits/losses
- **Strategy Comparison Charts**: Side-by-side performance comparison

### ✅ Trade-by-Trade Breakdown
- Detailed table of all trades
- Entry/exit times and prices
- Quantity and P&L for each trade
- Commission costs
- CSV export functionality

### ✅ Strategy Comparison Mode
- Compare all strategies simultaneously
- Side-by-side metrics comparison
- Visual comparison charts
- Identify best-performing strategy

## Strategy Descriptions

### 1. MA Crossover
**Logic**: Buy when 20-day MA crosses above 50-day MA, sell on opposite crossover

**Default Parameters**:
- Position Size: 95%
- Stop Loss: 5%
- Take Profit: 10%
- Commission: 0.1%

**Best For**: Trending markets, medium-term trades

### 2. RSI Mean Reversion
**Logic**: Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)

**Default Parameters**:
- Position Size: 90%
- Stop Loss: 3%
- Take Profit: 8%
- Commission: 0.1%

**Best For**: Range-bound markets, short-term trades

### 3. Momentum
**Logic**: Buy on strong upward momentum (>5%), sell on strong downward momentum (<-5%)

**Default Parameters**:
- Position Size: 85%
- Stop Loss: 4%
- Take Profit: 12%
- Commission: 0.1%

**Best For**: Volatile markets, momentum plays

### 4. Bollinger Bands
**Logic**: Buy when price touches lower band, sell when price touches upper band

**Default Parameters**:
- Position Size: 90%
- Stop Loss: 4%
- Take Profit: 8%
- Commission: 0.1%

**Best For**: Mean-reverting markets, volatility plays

## Usage Guide

### Running a Single Strategy Backtest

1. **Select Strategy**: Choose from the dropdown in the sidebar
2. **Enter Symbol**: Input the stock symbol (e.g., AAPL)
3. **Set Date Range**: Choose start and end dates
4. **Configure Capital**: Set initial capital amount
5. **Adjust Parameters**: Fine-tune position size, stop-loss, etc.
6. **Run Backtest**: Click "Run Backtest" button
7. **Review Results**: Analyze metrics, charts, and trade details

### Comparing Multiple Strategies

1. **Enable Comparison Mode**: Check "Compare Multiple Strategies"
2. **Configure Parameters**: Set parameters (applied to all strategies)
3. **Run Backtest**: Click "Run Backtest" button
4. **Compare Results**: View side-by-side comparison table and charts

### Exporting Results

- Click "Download Trades as CSV" to export trade details
- Includes all trade information for external analysis
- Filename includes symbol and date for easy organization

## Technical Implementation

### Data Flow

```
User Input → Fetch Historical Data → Run Strategy → Calculate Metrics → Display Results
```

### Key Components

1. **Strategy Functions**: Pure functions that take DataFrame and return signals
2. **Data Fetching**: Integration with MarketDataService for historical data
3. **Backtest Engine**: Uses BacktestService for strategy simulation
4. **Visualization**: Plotly charts for interactive analysis
5. **State Management**: Streamlit session state for results persistence

### Strategy Function Interface

```python
def strategy_function(data: pd.DataFrame) -> str:
    """
    Strategy function interface.
    
    Args:
        data: Historical price data with columns:
              - close: Closing prices
              - high: High prices
              - low: Low prices
              - volume: Trading volume
    
    Returns:
        Signal: 'buy', 'sell', or 'hold'
    """
    # Strategy logic here
    return 'buy'  # or 'sell' or 'hold'
```

### Performance Metrics Calculation

All metrics are calculated by the BacktestEngine:
- **Total Return**: Sum of all trade P&Ls
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

## Requirements Validation

### ✅ Requirement 8.1: Strategy Selection and Configuration
- Multiple pre-built strategies available
- Configurable parameters for each strategy
- Date range and initial capital inputs

### ✅ Requirement 8.2: Backtest Execution
- Historical data simulation
- Transaction costs included
- Progress indicator during execution

### ✅ Requirement 8.3: Performance Metrics
- Total return, Sharpe ratio, max drawdown
- Win rate, profit factor
- Trade statistics

### ✅ Requirement 8.4: Visualization
- Equity curve chart
- Drawdown chart
- Trade P&L distribution
- Trade-by-trade breakdown

### ✅ Requirement 8.5: Strategy Comparison
- Compare multiple strategies
- Side-by-side metrics
- Visual comparison charts

## Testing

### Verification Tests
- ✅ Module imports successfully
- ✅ All strategy functions present
- ✅ STRATEGIES dictionary configured
- ✅ Helper functions available
- ✅ Strategies work with sample data
- ✅ Parameters properly configured

### Strategy Tests
- ✅ All strategies return valid signals
- ✅ Strategies handle minimal data
- ✅ Strategies handle trending data
- ✅ Parameters are numeric and valid

## Running the Page

### Standalone Mode
```bash
streamlit run pages/5_backtest.py
```

### From Main App
The page is automatically available in the Streamlit multi-page app navigation.

## Future Enhancements

### Potential Additions
1. **Custom Strategy Builder**: Allow users to create custom strategies
2. **Walk-Forward Analysis**: Test strategy robustness
3. **Monte Carlo Simulation**: Assess strategy risk
4. **Optimization**: Parameter optimization tools
5. **More Strategies**: Additional pre-built strategies
6. **Multi-Symbol Backtesting**: Test portfolio strategies
7. **Advanced Metrics**: Sortino ratio, Calmar ratio, etc.
8. **Strategy Templates**: Save and load custom strategies

## Dependencies

- `streamlit`: Web interface
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `plotly`: Interactive charts
- `services.backtest_service`: Backtesting engine
- `services.market_data_service`: Historical data fetching

## File Structure

```
pages/5_backtest.py
├── Strategy Functions
│   ├── simple_ma_crossover()
│   ├── rsi_strategy()
│   ├── momentum_strategy()
│   └── bollinger_bands_strategy()
├── STRATEGIES Dictionary
├── Helper Functions
│   ├── fetch_historical_data()
│   ├── plot_equity_curve()
│   ├── plot_drawdown()
│   └── plot_trade_distribution()
└── Main UI Logic
```

## Notes

- All strategies use daily timeframe data
- Commission costs are included in P&L calculations
- Stop-loss and take-profit are optional (can be set to 0)
- Comparison mode uses same parameters for all strategies
- Results are stored in session state for persistence
- Historical data is fetched from Alpaca API

## Troubleshooting

### No Data Available
- Check that the symbol is valid
- Ensure date range is within available data
- Verify Alpaca API credentials are configured

### Strategy Not Generating Trades
- Check if date range is sufficient (need enough data for indicators)
- Verify strategy parameters are reasonable
- Review strategy logic for the specific market conditions

### Performance Issues
- Reduce date range for faster backtests
- Use comparison mode sparingly (tests all strategies)
- Consider caching historical data

## Conclusion

The backtesting page provides a comprehensive tool for evaluating trading strategies before risking real capital. With multiple pre-built strategies, configurable parameters, detailed metrics, and interactive visualizations, users can thoroughly test and compare different approaches to find what works best for their trading style.
