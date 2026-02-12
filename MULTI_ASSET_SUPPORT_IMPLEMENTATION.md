# Multi-Asset Support Implementation

## Overview

This document describes the implementation of multi-asset support for the AI Trading Agent, enabling trading across stocks, cryptocurrencies, and forex markets.

## Implementation Date

February 12, 2026

## Requirements Addressed

- **Requirement 20.1**: Asset class selection and display
- **Requirement 20.2**: Asset-specific models and indicators
- **Requirement 20.3**: Asset class allocation display
- **Requirement 20.4**: Asset-specific order routing
- **Requirement 20.5**: Correlation analysis between asset classes

## Components Modified

### 1. Market Data Service (`services/market_data_service.py`)

**Enhancements:**
- Added `AssetClass` enum (STOCK, CRYPTO, FOREX)
- Implemented `detect_asset_class()` method for automatic asset class detection from symbols
- Extended `get_latest_quote()` to support crypto and forex
- Extended `get_bars()` to support crypto and forex with asset-specific data clients
- Added asset class parameter to all data fetching methods
- Configured separate data clients for stocks and crypto

**Asset Class Detection Logic:**
- Stock symbols: Standard format (e.g., AAPL, GOOGL)
- Crypto symbols: Contains BTC, ETH, LTC, etc. or has / separator (e.g., BTC/USD, BTCUSD)
- Forex symbols: Currency pairs with / separator (e.g., EUR/USD, GBP/USD)

### 2. AI Inference Engine (`ai/inference.py`)

**Enhancements:**
- Added `asset_class` field to `TradingSignal` dataclass
- Added `asset_class` field to `Recommendation` dataclass
- Added `asset_class` field to `RankedStock` dataclass
- Updated `analyze_stock()` to accept and use asset class parameter
- Updated `get_recommendation()` to accept and use asset class parameter
- Updated `explain_prediction()` to include asset-specific indicators
- Updated `rank_opportunities()` to support asset class mapping
- Modified `_assess_risk()` to adjust risk scoring based on asset class
- Modified `_generate_reasoning()` to include asset-specific analysis

**Asset-Specific Risk Assessment:**
- Crypto: Higher base risk due to 24/7 trading and volatility
- Forex: Standard risk assessment
- Stocks: Standard risk assessment with market hours consideration

### 3. Asset Analysis Utilities (`utils/asset_analysis.py`)

**New Functions:**
- `calculate_crypto_specific_indicators()`: 24h volatility, momentum, VWAP
- `calculate_forex_specific_indicators()`: EMA trends, ATR percentage
- `calculate_asset_specific_indicators()`: Routes to appropriate indicator calculator
- `calculate_correlation_matrix()`: Cross-asset correlation analysis
- `calculate_portfolio_allocation()`: Allocation by asset class
- `get_asset_class_risk_metrics()`: Asset-adjusted risk calculations
- `suggest_asset_allocation()`: Risk-based allocation recommendations
- `calculate_diversification_ratio()`: Portfolio diversification scoring

**Crypto-Specific Indicators:**
- `realized_vol_24h`: 24-hour rolling volatility
- `momentum_6h`: 6-hour price momentum
- `momentum_24h`: 24-hour price momentum
- `vwap`: Volume-weighted average price
- `volume_ratio`: Current volume vs 24h average

**Forex-Specific Indicators:**
- `ema_8`, `ema_21`, `ema_55`: Multiple EMA timeframes
- `trend_strength`: Trend strength indicator
- `atr_14`: 14-period Average True Range
- `atr_pct`: ATR as percentage of price

### 4. Portfolio Service (`services/portfolio_service.py`)

**New Methods:**
- `get_asset_class_allocation()`: Calculate allocation by asset class
- `get_correlation_matrix()`: Calculate correlation between positions
- `get_diversification_metrics()`: Comprehensive diversification analysis

**Diversification Metrics:**
- Asset class allocation percentages
- Correlation matrix between positions
- Diversification ratio (higher = better diversification)
- Concentration risk (largest position as % of portfolio)
- Number of positions

### 5. Trading Service (`services/trading_service.py`)

**Existing Features Leveraged:**
- `TradingSchedule` class with asset class support
- `place_order_with_schedule_check()` for asset-specific routing
- Pre-configured schedules for STOCKS, CRYPTO, and FOREX
- Schedule validation and enforcement

**Asset-Specific Trading Hours:**
- **Stocks**: Mon-Fri, 9:30 AM - 4:00 PM ET
- **Crypto**: 24/7 (all days, all hours)
- **Forex**: Mon-Fri, 24 hours (closed weekends)

## Testing

### Test Coverage

Created comprehensive test suite in `tests/test_multi_asset_support.py`:

1. **Asset Class Detection Tests**
   - Stock symbol detection
   - Crypto symbol detection
   - Forex symbol detection

2. **Asset-Specific Indicator Tests**
   - Crypto indicator calculation
   - Forex indicator calculation
   - Indicator routing by asset class

3. **AI Engine Multi-Asset Tests**
   - Analysis with asset class parameter
   - Recommendations with asset class
   - Risk assessment by asset class

4. **Portfolio Allocation Tests**
   - Asset class allocation calculation
   - Diversification metrics
   - Correlation matrix generation

5. **Trading Service Routing Tests**
   - Schedule enforcement by asset class
   - Order placement with asset class
   - 24/7 crypto trading validation

### Verification Script

Created `verify_multi_asset_support.py` for manual verification:
- All 5 verification tests passed
- Validates asset detection, indicators, AI engine, portfolio, and trading service

## Usage Examples

### 1. Detecting Asset Class

```python
from services.market_data_service import MarketDataService

mds = MarketDataService()

# Detect asset class from symbol
asset_class = mds.detect_asset_class('BTC/USD')  # Returns AssetClass.CRYPTO
asset_class = mds.detect_asset_class('AAPL')     # Returns AssetClass.STOCK
asset_class = mds.detect_asset_class('EUR/USD')  # Returns AssetClass.FOREX
```

### 2. Fetching Multi-Asset Data

```python
# Fetch stock data
stock_data = mds.get_bars('AAPL', timeframe='1Day', asset_class=AssetClass.STOCK)

# Fetch crypto data
crypto_data = mds.get_bars('BTC/USD', timeframe='1Hour', asset_class=AssetClass.CRYPTO)

# Fetch forex data (when supported)
# forex_data = mds.get_bars('EUR/USD', timeframe='1Hour', asset_class=AssetClass.FOREX)
```

### 3. AI Analysis with Asset Class

```python
from ai.inference import AIEngine
from services.market_data_service import AssetClass

engine = AIEngine()

# Analyze stock
stock_signal = engine.analyze_stock('AAPL', stock_data, AssetClass.STOCK)

# Analyze crypto
crypto_signal = engine.analyze_stock('BTC/USD', crypto_data, AssetClass.CRYPTO)

# Get recommendation with asset-specific reasoning
recommendation = engine.get_recommendation('BTC/USD', crypto_data, AssetClass.CRYPTO)
```

### 4. Portfolio Asset Allocation

```python
from services.portfolio_service import PortfolioService

portfolio = PortfolioService(trading_service=trading_service)

# Get allocation by asset class
allocation = portfolio.get_asset_class_allocation()
# Returns: {'stock': 60.0, 'crypto': 30.0, 'forex': 10.0}

# Get diversification metrics
metrics = portfolio.get_diversification_metrics()
# Returns: {
#   'asset_class_allocation': {...},
#   'correlation_matrix': {...},
#   'diversification_ratio': 1.5,
#   'concentration_risk': 25.0,
#   'num_positions': 10
# }
```

### 5. Asset-Specific Order Routing

```python
from services.trading_service import TradingService, AssetClass

service = TradingService(paper=True)
service.enable_automated_trading()

# Place stock order (only during market hours)
order = service.place_order_with_schedule_check(
    symbol='AAPL',
    qty=100,
    side='buy',
    asset_class=AssetClass.STOCKS
)

# Place crypto order (24/7)
order = service.place_order_with_schedule_check(
    symbol='BTC/USD',
    qty=1,
    side='buy',
    asset_class=AssetClass.CRYPTO
)
```

## Key Features

### 1. Automatic Asset Detection
- Symbols are automatically classified by format
- No manual asset class specification required for common formats
- Supports override for edge cases

### 2. Asset-Specific Analysis
- Crypto: 24h volatility, momentum indicators
- Forex: Trend strength, pip-based volatility
- Stocks: Standard technical indicators

### 3. Risk-Adjusted Metrics
- Crypto: Higher volatility thresholds
- Forex: Lower volatility expectations
- Stocks: Standard risk assessment

### 4. Schedule Enforcement
- Stocks: Market hours only
- Crypto: 24/7 trading
- Forex: 24/5 trading

### 5. Portfolio Diversification
- Cross-asset correlation analysis
- Asset class allocation tracking
- Concentration risk monitoring

## Limitations and Future Enhancements

### Current Limitations

1. **Forex Support**: Alpaca API forex support is limited
   - Detection works but data fetching not fully implemented
   - Marked as NotImplementedError in code

2. **Asset-Specific Models**: Currently uses same ML models for all assets
   - Future: Train separate models for crypto, forex, stocks

3. **Correlation Calculation**: Requires sufficient historical data
   - May fail for newly added positions

### Future Enhancements

1. **Asset-Specific ML Models**
   - Train LSTM models specifically for crypto volatility
   - Train RF models for forex trend following
   - Ensemble models weighted by asset class

2. **Advanced Forex Support**
   - Full forex data integration when Alpaca adds support
   - Carry trade analysis
   - Interest rate differential tracking

3. **Options Trading**
   - Add OPTIONS asset class
   - Greeks calculation
   - Options-specific strategies

4. **Cross-Asset Strategies**
   - Pairs trading across asset classes
   - Hedging strategies
   - Correlation-based rebalancing

5. **Enhanced Diversification**
   - Sector-level diversification within stocks
   - Crypto category diversification (DeFi, Layer 1, etc.)
   - Geographic diversification for forex

## Verification Results

All verification tests passed:
- ✓ Asset Class Detection (7/7 symbols correctly classified)
- ✓ Asset-Specific Indicators (all indicators calculated correctly)
- ✓ AI Engine Multi-Asset (all methods support asset class parameter)
- ✓ Portfolio Asset Allocation (all new methods implemented)
- ✓ Trading Service Asset Routing (schedules configured for all asset classes)

## Conclusion

The multi-asset support implementation successfully extends the AI Trading Agent to handle stocks, cryptocurrencies, and forex markets with asset-specific analysis, risk assessment, and order routing. The system maintains backward compatibility while adding powerful new capabilities for diversified trading strategies.
