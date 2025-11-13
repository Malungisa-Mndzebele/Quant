# Technology Stack

## Language & Runtime

- **Python 3.10+** (minimum required version)
- Type hints used throughout codebase

## Core Dependencies

- **pandas** (>=2.0.0) - Data manipulation and analysis
- **numpy** (>=1.24.0) - Numerical computations
- **yfinance** (>=0.2.0) - Market data provider
- **pyyaml** (>=6.0) - Configuration file parsing
- **python-dotenv** (>=1.0.0) - Environment variable management
- **pytest** (>=7.4.0) - Testing framework

## Optional Dependencies

- **matplotlib** - Equity curve visualization in backtests

## Common Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running the System
```bash
# Simulation mode (default)
python main.py run

# Live trading mode (requires confirmation)
python main.py run --mode live

# Custom config file
python main.py run --config my_config.yaml
```

### Backtesting
```bash
# Basic backtest
python main.py backtest --start-date 2024-01-01 --end-date 2024-12-31 --symbols AAPL

# With results export and visualization
python main.py backtest --start-date 2024-01-01 --end-date 2024-12-31 --symbols AAPL GOOGL --output results/backtest --export-trades --visualize
```

### Configuration Validation
```bash
python main.py config
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_strategies.py

# Run with coverage
pytest tests/ --cov=src
```

## Configuration Management

- **Primary config**: `config.yaml` - System configuration with YAML format
- **Secrets**: `.env` - Environment variables for API keys and credentials (never commit)
- **Template**: `.env.example` - Example environment variables file

## Logging

- Logs stored in `logs/` directory
- Separate log files for simulation and live modes
- Rotating file handlers (10MB max, 5 backups)
- Error-only logs in separate files
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
