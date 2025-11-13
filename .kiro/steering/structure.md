# Project Structure

## Entry Point

- **main.py** - CLI entry point with commands: `run`, `backtest`, `config`

## Configuration

- **config.yaml** - System configuration (trading mode, symbols, risk parameters, strategies)
- **.env** - Environment variables for API credentials (gitignored)
- **.env.example** - Template for environment variables

## Source Code (`src/`)

### Core System
- **trading_system.py** - Main orchestration class that coordinates all components

### Strategies (`src/strategies/`)
- **base.py** - Abstract `Strategy` base class and `StrategyManager`
- **moving_average_crossover.py** - Example strategy implementation
- All strategies inherit from `Strategy` and implement: `on_data()`, `generate_signals()`, `on_order_filled()`, `reset_state()`

### Data Providers (`src/data/`)
- **base.py** - Abstract `MarketDataProvider` interface
- **yfinance_provider.py** - Yahoo Finance implementation
- **cache.py** - Data caching layer with TTL support

### Brokers (`src/brokers/`)
- **base.py** - Abstract `BrokerageAdapter` interface with `OrderStatus`, `AccountInfo`, `Position` dataclasses
- **factory.py** - Factory pattern for creating broker adapters
- **simulated_adapter.py** - Paper trading implementation
- **public_adapter.py** - Public.com API integration
- **moomoo_adapter.py** - Moomoo API integration
- **credentials.py** - Credential management
- **errors.py** - Brokerage-specific exceptions

### Execution (`src/execution/`)
- **order_manager.py** - Order lifecycle management, submission, and tracking

### Risk Management (`src/risk/`)
- **risk_manager.py** - Position sizing, loss limits, leverage controls, order validation

### Models (`src/models/`)
Data classes and enums for:
- **market_data.py** - Market data representation
- **signal.py** - Trading signals with `SignalAction` and `OrderType` enums
- **order.py** - Order objects with `OrderAction` and `OrderStatusEnum`
- **portfolio.py** - Portfolio tracking with positions and trades
- **position.py** - Position representation
- **performance.py** - Performance metrics
- **config.py** - Configuration models with `TradingMode` enum

### Backtesting (`src/backtesting/`)
- **backtester.py** - Backtesting engine
- **result.py** - Backtest results with performance metrics and reporting

### Utilities (`src/utils/`)
- **config_loader.py** - Configuration loading and validation

## Tests (`tests/`)

Test files mirror source structure:
- **test_strategies.py**
- **test_backtesting.py**
- **test_brokerage.py**
- **test_data_feed.py**
- **test_order_management.py**
- **test_portfolio.py**
- **test_risk_management.py**
- **test_system_initialization.py**
- **test_trading_loop.py**

## Data & Logs

- **data/cache/** - Cached market data (gitignored)
- **logs/** - System logs with rotation (gitignored)
  - `trading_system_simulation.log` - All simulation logs
  - `trading_system_live.log` - All live trading logs
  - `*_errors.log` - Error-only logs

## Examples (`examples/`)

- **custom_strategy_template.py** - Template for creating new strategies
- **rsi_strategy_example.py** - Example RSI strategy implementation

## Documentation (`docs/`)

- **TROUBLESHOOTING.md** - Common issues and solutions

## Architecture Patterns

### Base Classes
All major components use abstract base classes:
- `Strategy` for trading strategies
- `MarketDataProvider` for data sources
- `BrokerageAdapter` for broker integrations

### Factory Pattern
- `BrokerageFactory` creates appropriate broker adapter based on config

### Dependency Injection
- Components receive dependencies through constructor injection
- `TradingSystem` orchestrates component initialization in correct order

### Configuration-Driven
- All behavior controlled through `config.yaml`
- Environment variables for sensitive credentials
- Validation on startup

### Error Handling
- Custom exception classes: `TradingSystemError`, `BrokerageError`, `ConfigurationError`
- Comprehensive logging with context
- Graceful degradation where possible
