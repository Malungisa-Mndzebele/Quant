# Quantitative Trading System

A Python-based algorithmic trading platform for developing, backtesting, and executing quantitative trading strategies in the stock market.

## Features

- **Strategy Development**: Create custom trading strategies using a simple Python interface
- **Backtesting**: Test strategies against historical data with comprehensive performance metrics
- **Live Trading**: Execute trades automatically through brokerage APIs (Public, Moomoo)
- **Simulation Mode**: Paper trading with live data for risk-free strategy testing
- **Risk Management**: Built-in position sizing, daily loss limits, and leverage controls
- **Portfolio Tracking**: Real-time position monitoring and performance analytics
- **Data Integration**: Support for multiple market data providers (yfinance, Alpha Vantage)
- **Flexible Configuration**: YAML-based configuration with environment variable support

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Trading System](#running-the-trading-system)
  - [Backtesting Strategies](#backtesting-strategies)
  - [Validating Configuration](#validating-configuration)
- [Creating Custom Strategies](#creating-custom-strategies)
- [Risk Management](#risk-management)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For equity curve visualization in backtests:

```bash
pip install matplotlib
```

## Quick Start

### 1. Configure the System

Copy the example configuration and customize it:

```bash
cp .env.example .env
```

Edit `config.yaml` to set your trading parameters:

```yaml
trading:
  mode: "simulation"  # Start with simulation mode
  initial_capital: 100000.0
  symbols: ["AAPL", "GOOGL", "MSFT"]
```

### 2. Validate Configuration

```bash
python main.py config
```

### 3. Run a Backtest

Test a strategy with historical data:

```bash
python main.py backtest \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --symbols AAPL GOOGL \
  --strategy MovingAverageCrossover \
  --output results/backtest_2024
```

### 4. Start Trading (Simulation Mode)

```bash
python main.py run
```

Press `CTRL+C` to stop gracefully.

## Configuration

The system is configured through `config.yaml`. Here's a complete example:

```yaml
trading:
  mode: "simulation"              # "simulation" or "live"
  initial_capital: 100000.0       # Starting capital
  symbols: ["AAPL", "GOOGL"]      # Stocks to trade
  update_interval_seconds: 60     # Data fetch interval
  log_level: "INFO"               # Logging level

brokerage:
  provider: "simulated"           # "simulated", "public", or "moomoo"
  credentials:
    api_key: "${BROKERAGE_API_KEY}"
    api_secret: "${BROKERAGE_API_SECRET}"
  max_retries: 3

data:
  provider: "yfinance"            # "yfinance" or "alphaavantage"
  cache_enabled: true
  cache_dir: "./data/cache"
  cache_ttl_seconds: 3600

risk:
  max_position_size_pct: 10.0     # Max 10% per position
  max_daily_loss_pct: 5.0         # Stop trading at 5% daily loss
  max_portfolio_leverage: 1.0     # No leverage

strategies:
  - name: "MovingAverageCrossover"
    enabled: true
    params:
      fast_period: 10
      slow_period: 30
```

### Environment Variables

Store sensitive credentials in `.env`:

```bash
BROKERAGE_API_KEY=your_api_key_here
BROKERAGE_API_SECRET=your_api_secret_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

## Usage

### Running the Trading System

#### Simulation Mode (Paper Trading)

```bash
python main.py run
```

#### Live Trading Mode

**⚠️ WARNING: Live mode uses real money!**

```bash
python main.py run --mode live
```

You'll be prompted to confirm before live trading begins.

#### Custom Configuration File

```bash
python main.py run --config my_config.yaml
```

### Backtesting Strategies

#### Basic Backtest

```bash
python main.py backtest \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --symbols AAPL
```

#### Backtest with Results Export

```bash
python main.py backtest \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --symbols AAPL GOOGL MSFT \
  --strategy MovingAverageCrossover \
  --output results/my_backtest \
  --export-trades \
  --visualize
```

This will generate:
- `results/my_backtest.json` - Full backtest results
- `results/my_backtest.csv` - Trade history
- `results/my_backtest.png` - Equity curve chart

#### Backtest with Custom Capital

```bash
python main.py backtest \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --symbols AAPL \
  --capital 50000
```

### Validating Configuration

Check if your configuration is valid:

```bash
python main.py config
```

This will display:
- Configuration validation status
- Summary of all settings
- Enabled strategies
- Risk parameters

## Creating Custom Strategies

### Strategy Template

Create a new file in `src/strategies/`:

```python
from typing import List
from src.strategies.base import Strategy
from src.models.signal import Signal, SignalAction
from src.models.market_data import MarketData

class MyCustomStrategy(Strategy):
    """My custom trading strategy."""
    
    def __init__(self, param1: int = 10, param2: float = 0.5):
        super().__init__(name="MyCustomStrategy")
        self.param1 = param1
        self.param2 = param2
        self.data_buffer = {}
    
    def on_data(self, market_data: MarketData) -> None:
        """Process new market data."""
        symbol = market_data.symbol
        
        # Store data for analysis
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []
        
        self.data_buffer[symbol].append(market_data)
        
        # Keep only recent data
        if len(self.data_buffer[symbol]) > 100:
            self.data_buffer[symbol].pop(0)
    
    def generate_signals(self) -> List[Signal]:
        """Generate trading signals based on strategy logic."""
        signals = []
        
        for symbol, data_list in self.data_buffer.items():
            if len(data_list) < self.param1:
                continue
            
            # Your strategy logic here
            # Example: Simple price momentum
            recent_prices = [d.close for d in data_list[-self.param1:]]
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            if price_change > self.param2:
                # Buy signal
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    quantity=100,
                    order_type="MARKET",
                    timestamp=data_list[-1].timestamp
                )
                signals.append(signal)
            
            elif price_change < -self.param2:
                # Sell signal
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.SELL,
                    quantity=100,
                    order_type="MARKET",
                    timestamp=data_list[-1].timestamp
                )
                signals.append(signal)
        
        return signals
    
    def on_order_filled(self, order) -> None:
        """Handle order fill notifications."""
        # Update strategy state if needed
        pass
    
    def reset_state(self) -> None:
        """Reset strategy state for backtesting."""
        self.data_buffer = {}
```

### Register Your Strategy

1. Add your strategy to `src/trading_system.py` in the `_create_strategy` method:

```python
def _create_strategy(self, strategy_config) -> Optional[Strategy]:
    strategy_name = strategy_config.name.lower()
    
    if strategy_name == 'mycustomstrategy':
        return MyCustomStrategy(
            param1=strategy_config.params.get('param1', 10),
            param2=strategy_config.params.get('param2', 0.5)
        )
    # ... existing strategies
```

2. Add to `config.yaml`:

```yaml
strategies:
  - name: "MyCustomStrategy"
    enabled: true
    params:
      param1: 20
      param2: 0.03
```

### Example: Moving Average Crossover Strategy

See `src/strategies/moving_average_crossover.py` for a complete example of a working strategy.

## Risk Management

The system includes built-in risk controls:

### Position Size Limits

Prevents any single position from exceeding a percentage of portfolio value:

```yaml
risk:
  max_position_size_pct: 10.0  # Max 10% per position
```

### Daily Loss Limits

Stops trading if daily losses exceed threshold:

```yaml
risk:
  max_daily_loss_pct: 5.0  # Stop at 5% daily loss
```

### Leverage Controls

Limits portfolio leverage:

```yaml
risk:
  max_portfolio_leverage: 1.0  # No leverage (1.0 = cash only)
```

### Symbol Restrictions

Restrict trading to specific symbols:

```yaml
risk:
  allowed_symbols: ["AAPL", "GOOGL", "MSFT"]
```

### Maximum Positions

Limit number of concurrent positions:

```yaml
risk:
  max_positions: 5
```

## Troubleshooting

### Common Issues

#### Configuration Errors

**Problem**: `Configuration Error: Brokerage API key is required`

**Solution**: Set environment variables in `.env`:
```bash
BROKERAGE_API_KEY=your_key
BROKERAGE_API_SECRET=your_secret
```

#### Data Provider Issues

**Problem**: `Failed to fetch data for AAPL`

**Solutions**:
- Check internet connection
- Verify symbol is valid
- For Alpha Vantage, check API key and rate limits
- Try switching to yfinance provider

#### No Trades in Backtest

**Problem**: Backtest completes with 0 trades

**Possible Causes**:
- Strategy parameters too conservative
- Insufficient historical data
- Date range too short
- Strategy logic not triggering signals

**Solutions**:
- Adjust strategy parameters
- Use longer date range
- Add debug logging to strategy
- Verify data is being fetched correctly

#### Live Trading Not Starting

**Problem**: System fails to start in live mode

**Solutions**:
- Verify brokerage credentials are correct
- Check brokerage account is active
- Ensure sufficient account balance
- Review logs in `logs/` directory

### Logging

Logs are stored in the `logs/` directory:

- `trading_system_simulation.log` - All simulation mode logs
- `trading_system_live.log` - All live mode logs
- `*_errors.log` - Error-only logs

Adjust log level in `config.yaml`:

```yaml
trading:
  log_level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Getting Help

1. Check logs in `logs/` directory
2. Validate configuration: `python main.py config`
3. Test with simulation mode first
4. Review strategy logic and parameters

## Project Structure

```
quant-trading-system/
├── main.py                      # CLI entry point
├── config.yaml                  # System configuration
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
│
├── src/
│   ├── trading_system.py        # Main system orchestration
│   │
│   ├── strategies/              # Trading strategies
│   │   ├── base.py              # Strategy base class
│   │   └── moving_average_crossover.py
│   │
│   ├── data/                    # Market data providers
│   │   ├── base.py
│   │   ├── yfinance_provider.py
│   │   └── cache.py
│   │
│   ├── execution/               # Order management
│   │   └── order_manager.py
│   │
│   ├── risk/                    # Risk management
│   │   └── risk_manager.py
│   │
│   ├── brokers/                 # Brokerage integrations
│   │   ├── base.py
│   │   ├── simulated_adapter.py
│   │   ├── public_adapter.py
│   │   └── moomoo_adapter.py
│   │
│   ├── backtesting/             # Backtesting engine
│   │   ├── backtester.py
│   │   └── result.py
│   │
│   ├── models/                  # Data models
│   │   ├── market_data.py
│   │   ├── signal.py
│   │   ├── order.py
│   │   ├── portfolio.py
│   │   └── config.py
│   │
│   └── utils/                   # Utilities
│       └── config_loader.py
│
├── tests/                       # Unit and integration tests
│   ├── test_strategies.py
│   ├── test_backtesting.py
│   ├── test_order_management.py
│   └── ...
│
├── data/                        # Data storage
│   └── cache/                   # Cached market data
│
└── logs/                        # System logs
    ├── trading_system_simulation.log
    └── trading_system_live.log
```

## Testing

Run all tests:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_strategies.py
```

Run with coverage:

```bash
pytest tests/ --cov=src
```

## License

This project is provided as-is for educational and research purposes.

## Disclaimer

**IMPORTANT**: This software is for educational purposes only. Trading stocks involves substantial risk of loss. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred through the use of this software.

Always test strategies thoroughly in simulation mode before considering live trading.
