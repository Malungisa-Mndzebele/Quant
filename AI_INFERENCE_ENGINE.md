# AI Inference Engine

## Overview

The AI Inference Engine (`ai/inference.py`) is the core component that combines multiple ML models to generate trading signals and recommendations. It provides a unified interface for analyzing stocks and generating actionable trading insights.

## Key Components

### AIEngine

The main inference engine that orchestrates model predictions and generates trading signals.

**Features:**
- Loads and manages multiple ML models (LSTM, Random Forest)
- Combines predictions using ensemble methods
- Generates trading signals with confidence scores
- Provides detailed explanations for predictions
- Ranks multiple stocks by opportunity strength

### Data Classes

#### TradingSignal
Represents a trading recommendation with:
- `symbol`: Stock symbol
- `action`: 'buy', 'sell', or 'hold'
- `confidence`: Confidence score (0.0 to 1.0)
- `target_price`: Suggested target price
- `stop_loss`: Suggested stop-loss price
- `reasoning`: Detailed reasoning dictionary
- `model_predictions`: Individual model predictions

#### Recommendation
Comprehensive trading recommendation with:
- All TradingSignal fields
- `expected_return`: Expected return percentage
- `risk_level`: 'low', 'medium', or 'high'
- `reasoning`: List of human-readable reasons
- `technical_factors`: Technical indicator values
- `model_scores`: Individual model scores

#### RankedStock
Stock ranked by trading opportunity:
- `symbol`: Stock symbol
- `score`: Opportunity score (0-100)
- `action`: Recommended action
- `confidence`: Confidence level
- `expected_return`: Expected return
- `risk_level`: Risk assessment
- `key_factors`: Top reasons for ranking

## Usage Examples

### Basic Usage

```python
from ai.inference import create_default_engine
import pandas as pd

# Create engine
engine = create_default_engine()

# Load models (if available)
engine.load_models()

# Prepare data (OHLCV DataFrame)
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Generate trading signal
signal = engine.analyze_stock('AAPL', data)
print(f"Action: {signal.action}")
print(f"Confidence: {signal.confidence:.2%}")
print(f"Target: ${signal.target_price:.2f}")
```

### Get Detailed Recommendation

```python
# Get comprehensive recommendation
recommendation = engine.get_recommendation('AAPL', data)

print(f"Action: {recommendation.action}")
print(f"Confidence: {recommendation.confidence:.2%}")
print(f"Expected Return: {recommendation.expected_return:.2%}")
print(f"Risk Level: {recommendation.risk_level}")
print("\nReasons:")
for reason in recommendation.reasoning:
    print(f"  - {reason}")
```

### Explain Prediction

```python
# Get detailed explanation
explanation = engine.explain_prediction('AAPL', data)

print("Model Contributions:")
for model, contrib in explanation['model_contributions'].items():
    print(f"  {model}: {contrib['contribution']:.3f}")

print("\nTechnical Analysis:")
print(f"  RSI: {explanation['technical_analysis']['rsi']['value']:.1f}")
print(f"  Interpretation: {explanation['technical_analysis']['rsi']['interpretation']}")
```

### Rank Multiple Stocks

```python
# Analyze and rank multiple stocks
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
data_dict = {
    'AAPL': aapl_data,
    'GOOGL': googl_data,
    'MSFT': msft_data,
    'TSLA': tsla_data
}

# Get top 3 opportunities
ranked = engine.rank_opportunities(symbols, data_dict, top_n=3)

for stock in ranked:
    print(f"{stock.symbol}: Score {stock.score:.1f} - {stock.action.upper()}")
    print(f"  Confidence: {stock.confidence:.2%}")
    print(f"  Expected Return: {stock.expected_return:.2%}")
    print(f"  Risk: {stock.risk_level}")
```

## Model Requirements

The engine expects trained models in the `data/models/` directory:

- **LSTM Model**: `lstm_model.h5` and `lstm_model_config.json`
- **Random Forest Model**: `rf_model.pkl`, `rf_model_encoder.pkl`, and `rf_model_config.json`

Models can be trained using the training scripts in `ai/training.py`.

## Data Requirements

Input data must be a pandas DataFrame with the following columns:
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

Minimum 60 data points required for analysis.

## Ensemble Configuration

The engine uses weighted ensemble prediction:

```python
engine = AIEngine(
    model_dir='data/models',
    ensemble_weights={
        'lstm': 0.4,  # 40% weight for LSTM predictions
        'rf': 0.6     # 60% weight for Random Forest
    }
)
```

## Technical Indicators

The engine automatically calculates:
- Simple Moving Average (SMA 20, 50)
- Exponential Moving Average (EMA 12, 26)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Average True Range (ATR)

## Signal Generation Logic

1. **Load Models**: Load trained LSTM and Random Forest models
2. **Calculate Indicators**: Compute technical indicators from price data
3. **Generate Predictions**: Get predictions from each model
4. **Ensemble**: Combine predictions using weighted average
5. **Determine Action**: Convert ensemble score to buy/sell/hold
6. **Calculate Targets**: Set target price and stop-loss levels
7. **Assess Risk**: Evaluate risk based on volatility and confidence
8. **Generate Reasoning**: Create human-readable explanations

## Confidence Scoring

Confidence is calculated based on:
- Model agreement (higher when models agree)
- Prediction strength (distance from neutral)
- Technical indicator alignment
- Historical model performance

## Risk Assessment

Risk level is determined by:
- Volatility (ATR and Bollinger Band width)
- RSI extremes (overbought/oversold)
- Model confidence
- Price momentum

## Integration with Trading System

The AI engine integrates with other components:

```python
from services.market_data_service import MarketDataService
from ai.inference import create_default_engine

# Get market data
market_service = MarketDataService()
data = market_service.get_bars('AAPL', timeframe='1Day', limit=100)

# Analyze with AI
engine = create_default_engine()
engine.load_models()
recommendation = engine.get_recommendation('AAPL', data)

# Use recommendation for trading decision
if recommendation.action == 'buy' and recommendation.confidence > 0.7:
    # Execute trade
    pass
```

## Error Handling

The engine handles various error conditions:
- Missing models (graceful degradation)
- Insufficient data (raises ValueError)
- Model prediction failures (logs and continues)
- Invalid data format (raises ValueError)

## Logging

The engine uses Python's logging module:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Use engine
engine = create_default_engine()
# Logs will show detailed information about predictions
```

## Performance Considerations

- Models are loaded lazily (only when needed)
- Technical indicators are cached during analysis
- Ensemble weights can be tuned for performance
- Batch analysis supported via `rank_opportunities()`

## Future Enhancements

Potential improvements:
- Additional ML models (XGBoost, Transformer)
- Sentiment analysis integration
- Multi-timeframe analysis
- Custom indicator support
- Real-time model updates
- A/B testing framework
