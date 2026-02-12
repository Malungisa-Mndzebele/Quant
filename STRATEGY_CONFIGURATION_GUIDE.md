# Strategy Configuration Guide

## Overview

The AI Trading Agent includes a comprehensive strategy configuration system that allows you to customize trading behavior according to your investment philosophy and risk tolerance. This guide explains how to use the strategy configuration features.

## Quick Start

### Using Preset Strategies

The system includes three preset strategies optimized for different trading styles:

```python
from config.strategies import get_strategy_preset

# Get a preset strategy
conservative = get_strategy_preset("conservative")
moderate = get_strategy_preset("moderate")
aggressive = get_strategy_preset("aggressive")
```

### Strategy Characteristics

**Conservative Strategy:**
- High confidence threshold (75%)
- Small position sizes (5% max)
- Tight stop losses (3%)
- Fewer trades (5 per day max)
- Focus on capital preservation

**Moderate Strategy:**
- Balanced confidence threshold (60%)
- Moderate position sizes (10% max)
- Standard stop losses (5%)
- Moderate trading frequency (10 per day max)
- Balance between growth and safety

**Aggressive Strategy:**
- Lower confidence threshold (50%)
- Larger position sizes (15% max)
- Wider stop losses (8%)
- More frequent trading (20 per day max)
- Focus on maximizing returns

## Creating Custom Strategies

### Method 1: Start from a Preset

The easiest way to create a custom strategy is to start from a preset and override specific parameters:

```python
from config.strategies import create_custom_strategy

# Create custom strategy based on moderate preset
my_strategy = create_custom_strategy(
    "My Custom Strategy",
    base_style="moderate",
    min_confidence=0.65,
    max_position_size=0.08,
    use_trailing_stop=True
)
```

### Method 2: Create from Scratch

For full control, create a strategy from scratch:

```python
from config.strategies import StrategyParameters

strategy = StrategyParameters(
    name="My Strategy",
    min_confidence=0.7,
    signal_threshold=0.8,
    lstm_weight=0.4,
    rf_weight=0.4,
    sentiment_weight=0.2,
    max_position_size=0.1,
    max_portfolio_risk=0.02,
    stop_loss_pct=0.05,
    take_profit_pct=0.10,
    max_trades_per_day=10
)
```

## Strategy Parameters

### Signal Generation

- **min_confidence** (0.0-1.0): Minimum confidence to generate a signal
- **signal_threshold** (0.0-1.0): Threshold for strong signals (must be >= min_confidence)

### Model Weights

These weights determine how much each model contributes to the final signal. They must sum to 1.0:

- **lstm_weight** (0.0-1.0): Weight for LSTM price predictions
- **rf_weight** (0.0-1.0): Weight for Random Forest classifications
- **sentiment_weight** (0.0-1.0): Weight for sentiment analysis

### Risk Parameters

- **max_position_size** (0.0-1.0): Maximum % of portfolio per position
- **max_portfolio_risk** (0.0-1.0): Maximum % of portfolio at risk
- **stop_loss_pct** (0.0-1.0): Stop loss percentage from entry
- **take_profit_pct** (0.0-1.0): Take profit percentage from entry

### Trading Behavior

- **max_trades_per_day** (>0): Maximum number of trades per day
- **min_holding_period** (>0): Minimum holding period in days
- **max_holding_period** (>0): Maximum holding period in days
- **use_trailing_stop** (bool): Enable trailing stop loss
- **trailing_stop_pct** (0.0-1.0): Trailing stop percentage (if enabled)
- **rebalance_frequency** (>0): Days between portfolio rebalancing

## Indicator Configuration

Each strategy includes configuration for technical indicators:

```python
from config.strategies import IndicatorConfig, IndicatorType

# Create custom indicator configuration
rsi_config = IndicatorConfig(
    indicator_type=IndicatorType.RSI.value,
    enabled=True,
    weight=0.9,  # Higher weight = more influence
    parameters={"period": 14}
)

# Add to strategy
strategy.indicators.append(rsi_config)
```

Available indicators:
- **SMA**: Simple Moving Average
- **EMA**: Exponential Moving Average
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **BOLLINGER_BANDS**: Bollinger Bands
- **ATR**: Average True Range

## Saving and Loading Strategies

### Save a Strategy

```python
from config.strategies import strategy_manager

# Save strategy to file
success = strategy_manager.save_strategy(my_strategy)
```

Strategies are saved as JSON files in `data/strategies/` directory.

### Load a Strategy

```python
# Load strategy by name (without .json extension)
loaded_strategy = strategy_manager.load_strategy("my_custom_strategy")
```

### List Saved Strategies

```python
# Get list of all saved strategies
strategies = strategy_manager.list_strategies()
for name in strategies:
    print(f"- {name}")
```

### Delete a Strategy

```python
# Delete a saved strategy
success = strategy_manager.delete_strategy("my_custom_strategy")
```

## Validation

All strategies are automatically validated before saving or use:

```python
# Manually validate a strategy
is_valid, errors = strategy.validate()
if not is_valid:
    print(f"Validation errors: {errors}")
```

Common validation rules:
- Confidence thresholds must be between 0.0 and 1.0
- Model weights must sum to 1.0
- Risk parameters must be positive and within valid ranges
- Holding periods must be positive
- min_confidence must be <= signal_threshold

## Example Workflows

### Workflow 1: Customize a Preset

```python
from config.strategies import get_strategy_preset, strategy_manager

# 1. Start with a preset
strategy = get_strategy_preset("moderate")

# 2. Customize it
strategy.name = "My Moderate Strategy"
strategy.min_confidence = 0.65
strategy.max_position_size = 0.08

# 3. Validate
is_valid, errors = strategy.validate()
if is_valid:
    # 4. Save
    strategy_manager.save_strategy(strategy)
```

### Workflow 2: Create and Test Custom Strategy

```python
from config.strategies import create_custom_strategy, strategy_manager

# 1. Create custom strategy
strategy = create_custom_strategy(
    "My Balanced Strategy",
    base_style="moderate",
    min_confidence=0.7,
    signal_threshold=0.8,
    max_position_size=0.08,
    use_trailing_stop=True,
    trailing_stop_pct=0.03
)

# 2. Save it
strategy_manager.save_strategy(strategy)

# 3. Later, load and use it
loaded = strategy_manager.load_strategy("my_balanced_strategy")
```

### Workflow 3: Compare Strategies

```python
from config.strategies import strategy_manager

# Get all preset strategies
presets = strategy_manager.get_preset_strategies()

# Compare key parameters
for name, strategy in presets.items():
    print(f"{name}:")
    print(f"  Confidence: {strategy.min_confidence}")
    print(f"  Position Size: {strategy.max_position_size:.1%}")
    print(f"  Stop Loss: {strategy.stop_loss_pct:.1%}")
```

## Integration with AI Trading Agent

Once you've created and saved a strategy, you can use it in the AI trading agent:

```python
from config.strategies import strategy_manager
from ai.inference import AIEngine

# Load your strategy
strategy = strategy_manager.load_strategy("my_custom_strategy")

# Configure AI engine with strategy parameters
ai_engine = AIEngine(
    model_dir="data/models",
    min_confidence=strategy.min_confidence,
    lstm_weight=strategy.lstm_weight,
    rf_weight=strategy.rf_weight,
    sentiment_weight=strategy.sentiment_weight
)

# Use strategy risk parameters for trading
from services.risk_service import RiskService
from config.settings import RiskConfig

risk_config = RiskConfig(
    max_position_size=strategy.max_position_size,
    max_portfolio_risk=strategy.max_portfolio_risk,
    stop_loss_pct=strategy.stop_loss_pct,
    take_profit_pct=strategy.take_profit_pct,
    max_open_positions=strategy.max_trades_per_day
)

risk_service = RiskService(risk_config)
```

## Best Practices

1. **Start with Presets**: Begin with a preset strategy that matches your risk tolerance
2. **Make Small Changes**: Adjust one or two parameters at a time
3. **Validate Always**: Always validate strategies before using them
4. **Backtest First**: Test custom strategies with backtesting before live trading
5. **Document Changes**: Use descriptive names for custom strategies
6. **Version Control**: Save different versions as you iterate
7. **Monitor Performance**: Track how different strategies perform over time

## Troubleshooting

### Validation Errors

**"min_confidence cannot be greater than signal_threshold"**
- Solution: Increase signal_threshold or decrease min_confidence

**"Model weights must sum to 1.0"**
- Solution: Ensure lstm_weight + rf_weight + sentiment_weight = 1.0

**"max_position_size must be between 0.0 and 1.0"**
- Solution: Use decimal values (e.g., 0.1 for 10%, not 10)

### Loading Errors

**"Strategy file not found"**
- Solution: Check the filename (without .json extension)
- Verify the file exists in `data/strategies/` directory

**"Loaded strategy validation failed"**
- Solution: The saved strategy has invalid parameters
- Delete and recreate the strategy with valid parameters

## Advanced Topics

### Custom Indicator Weights

Adjust how much each indicator influences trading decisions:

```python
# Increase RSI influence
for indicator in strategy.indicators:
    if indicator.indicator_type == "rsi":
        indicator.weight = 1.0  # Maximum influence
```

### Dynamic Strategy Adjustment

Strategies can be modified at runtime based on market conditions:

```python
# Reduce risk during volatile markets
if market_volatility > threshold:
    strategy.max_position_size *= 0.5
    strategy.stop_loss_pct *= 0.8
```

### Strategy Backtesting

Before using a custom strategy in live trading, backtest it:

```python
from services.backtest_service import BacktestEngine

backtest = BacktestEngine(initial_capital=100000)
results = backtest.run_backtest(
    strategy=my_strategy,
    start=start_date,
    end=end_date
)

print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
```

## Support

For questions or issues with strategy configuration:
1. Check this guide for common solutions
2. Review the validation error messages
3. Examine the example strategies in `verify_strategies.py`
4. Consult the requirements document for parameter constraints

## Related Documentation

- [AI Trading Agent README](AI_TRADING_AGENT_README.md)
- [Risk Service Implementation](RISK_SERVICE_IMPLEMENTATION.md)
- [Backtesting Guide](BACKTEST_PAGE_GUIDE.md)
- [Requirements Document](.kiro/specs/ai-trading-agent/requirements.md)
