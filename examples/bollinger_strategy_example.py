"""
Example: Bollinger Bands Mean Reversion Strategy

This example demonstrates how to use the Bollinger Bands Mean Reversion strategy
for algorithmic trading.

Strategy Overview:
- Uses Bollinger Bands to identify price extremes
- Confirms with RSI for overbought/oversold conditions
- Trades on mean reversion principle
- Exits when price returns to mean or reaches opposite extreme

Configuration Example:
Add this to your config.yaml:

strategies:
  - name: "BollingerMeanReversion"
    enabled: true
    params:
      symbols: ["AAPL", "GOOGL", "MSFT"]
      bb_period: 20          # Bollinger Bands period
      bb_std: 2.0            # Standard deviations
      rsi_period: 14         # RSI calculation period
      rsi_oversold: 30       # RSI oversold threshold
      rsi_overbought: 70     # RSI overbought threshold
      quantity: 100          # Shares per trade

Strategy Logic:
1. BUY when:
   - Price touches or goes below lower Bollinger Band
   - AND RSI < 30 (oversold)
   
2. SELL when:
   - Price touches or goes above upper Bollinger Band
   - AND RSI > 70 (overbought)
   - OR price crosses back above middle band (mean reversion)

Best Practices:
- Works best in ranging/sideways markets
- Less effective in strong trending markets
- Combine with trend filters for better results
- Backtest thoroughly before live trading
- Start with conservative position sizes

Parameter Tuning:
- Shorter BB period (10-15): More sensitive, more signals
- Longer BB period (25-30): Less sensitive, fewer signals
- Wider bands (2.5-3.0 std): Fewer but stronger signals
- Tighter bands (1.5-2.0 std): More frequent signals
- RSI thresholds: Adjust based on market volatility

Example Backtest:
python main.py backtest \\
  --start-date 2024-01-01 \\
  --end-date 2024-12-31 \\
  --symbols AAPL GOOGL MSFT \\
  --strategy BollingerMeanReversion \\
  --output results/bollinger_backtest \\
  --export-trades \\
  --visualize
"""

from src.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy

# Example 1: Conservative parameters (fewer trades, higher confidence)
conservative_strategy = BollingerMeanReversionStrategy(
    symbols=["AAPL", "GOOGL"],
    bb_period=25,           # Longer period for smoother bands
    bb_std=2.5,             # Wider bands for extreme moves only
    rsi_period=14,
    rsi_oversold=25,        # More extreme oversold
    rsi_overbought=75,      # More extreme overbought
    quantity=50             # Smaller position size
)

# Example 2: Aggressive parameters (more trades, faster signals)
aggressive_strategy = BollingerMeanReversionStrategy(
    symbols=["AAPL", "GOOGL", "MSFT", "TSLA"],
    bb_period=15,           # Shorter period for faster response
    bb_std=1.5,             # Tighter bands for more signals
    rsi_period=10,          # Faster RSI
    rsi_oversold=35,        # Less extreme thresholds
    rsi_overbought=65,
    quantity=100
)

# Example 3: Balanced parameters (recommended starting point)
balanced_strategy = BollingerMeanReversionStrategy(
    symbols=["AAPL", "GOOGL", "MSFT"],
    bb_period=20,           # Standard BB period
    bb_std=2.0,             # Standard deviation
    rsi_period=14,          # Standard RSI period
    rsi_oversold=30,        # Classic oversold level
    rsi_overbought=70,      # Classic overbought level
    quantity=100
)

# Example 4: Volatile market parameters
volatile_market_strategy = BollingerMeanReversionStrategy(
    symbols=["TSLA", "NVDA"],
    bb_period=20,
    bb_std=3.0,             # Wider bands for volatile stocks
    rsi_period=14,
    rsi_oversold=20,        # More extreme thresholds
    rsi_overbought=80,
    quantity=50             # Smaller size for volatile stocks
)

print("Bollinger Mean Reversion Strategy Examples")
print("=" * 50)
print("\nConservative Strategy:")
print(conservative_strategy.get_strategy_info())
print("\nAggressive Strategy:")
print(aggressive_strategy.get_strategy_info())
print("\nBalanced Strategy (Recommended):")
print(balanced_strategy.get_strategy_info())
print("\nVolatile Market Strategy:")
print(volatile_market_strategy.get_strategy_info())
