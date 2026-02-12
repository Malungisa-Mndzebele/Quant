# AI Trading Agent

An AI-powered trading agent with real-time market data integration, machine learning-based recommendations, sentiment analysis, and automated trading capabilities.

## Project Structure

```
.
├── ai/                          # Machine learning components
│   ├── models/                  # ML model implementations
│   ├── features.py             # Feature engineering
│   ├── training.py             # Model training pipeline
│   └── inference.py            # Real-time prediction
├── services/                    # Business logic services
│   ├── market_data_service.py  # Real-time data fetching
│   ├── trading_service.py      # Order execution
│   ├── portfolio_service.py    # Portfolio management
│   ├── sentiment_service.py    # News sentiment analysis
│   ├── risk_service.py         # Risk management
│   └── backtest_service.py     # Backtesting engine
├── utils/                       # Utility functions
│   ├── indicators.py           # Technical indicators
│   ├── validators.py           # Input validation
│   └── formatters.py           # Data formatting
├── config/                      # Configuration
│   ├── settings.py             # Application settings
│   ├── logging_config.py       # Logging configuration
│   └── strategies.py           # Trading strategy configs
├── data/                        # Data storage
│   ├── models/                 # Trained ML models
│   ├── cache/                  # Cached market data
│   └── database/               # SQLite database
├── tests/                       # Test suite
└── pages/                       # Streamlit pages
```

## Setup Instructions

### 1. Install Dependencies

```bash
# Install AI Trading Agent dependencies
pip install -r ai_requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.ai.example .env.ai

# Edit .env.ai and add your API keys:
# - ALPACA_API_KEY: Get from https://alpaca.markets/
# - ALPACA_SECRET_KEY: Get from https://alpaca.markets/
# - NEWS_API_KEY: Get from https://newsapi.org/
```

### 3. Verify Configuration

The application will validate your configuration on startup. Required settings:
- Alpaca API credentials (for market data and trading)
- News API key (optional, for sentiment analysis)
- Risk management parameters (configured in .env.ai)

### 4. Run the Application

```bash
# Start the AI Trading Agent
streamlit run app.py
```

## Features

- **Real-time Market Data**: Live price feeds via Alpaca API
- **AI Recommendations**: ML-based buy/sell/hold signals
- **Sentiment Analysis**: News sentiment integration using FinBERT
- **Automated Trading**: Optional automated execution mode
- **Risk Management**: Configurable position sizing and stop-loss
- **Portfolio Tracking**: Real-time P&L and performance metrics
- **Backtesting**: Test strategies on historical data
- **Paper Trading**: Practice with virtual money

## Configuration

Key configuration options in `.env.ai`:

### Trading Mode
- `PAPER_TRADING=true`: Use paper trading (recommended for testing)
- `PAPER_TRADING=false`: Use live trading (real money)

### Risk Management
- `MAX_POSITION_SIZE=0.1`: Maximum 10% of portfolio per position
- `DAILY_LOSS_LIMIT=1000`: Stop trading after $1000 daily loss
- `STOP_LOSS_PCT=0.05`: Automatic 5% stop-loss on positions

### Feature Flags
- `ENABLE_AUTOMATED_TRADING=false`: Disable automated trading by default
- `ENABLE_SENTIMENT_ANALYSIS=true`: Enable news sentiment analysis
- `ENABLE_PAPER_TRADING=true`: Enable paper trading mode

## Security

- API credentials are encrypted at rest using AES-256
- Encryption key stored in `data/.encryption_key` (never commit this file)
- All API calls use HTTPS
- Credentials never logged or displayed

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai --cov=services --cov=utils

# Run property-based tests
pytest -v tests/
```

## Logging

Logs are stored in the `logs/` directory:
- `trading_agent.log`: All application logs
- `errors.log`: Error logs only
- `trades.log`: Trade execution audit trail

## Development

This is a standalone AI trading agent system, separate from the existing option pricing application. The two systems can coexist in the same repository.

## Disclaimer

This software is for educational purposes only. Trading involves risk of loss. Always test thoroughly with paper trading before using real money. The authors are not responsible for any financial losses.

## License

See LICENSE file for details.
