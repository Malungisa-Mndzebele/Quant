# Quick Start Guide - AI Trading Agent

## Prerequisites

- Python 3.10 or higher
- pip package manager
- API keys from Alpaca and NewsAPI (see below)

## Step 1: Install Dependencies

```bash
pip install -r ai_requirements.txt
```

This will install all required packages including:
- Streamlit (web interface)
- alpaca-py (market data and trading)
- TensorFlow (machine learning)
- scikit-learn (ML algorithms)
- transformers (sentiment analysis)
- And more...

## Step 2: Get API Keys

### Alpaca API (Required)
1. Sign up at https://alpaca.markets/
2. Create a paper trading account (free)
3. Generate API keys from the dashboard
4. Copy your API Key and Secret Key

### NewsAPI (Optional)
1. Sign up at https://newsapi.org/
2. Get your free API key (500 requests/day)
3. Copy your API key

## Step 3: Configure Environment

```bash
# Copy the example environment file
cp .env.ai.example .env.ai

# Edit .env.ai with your favorite text editor
# Add your API keys:
ALPACA_API_KEY=your_actual_api_key_here
ALPACA_SECRET_KEY=your_actual_secret_key_here
NEWS_API_KEY=your_news_api_key_here

# Keep paper trading enabled for testing
PAPER_TRADING=true
```

## Step 4: Verify Setup

```bash
python verify_setup.py
```

This will check:
- ✓ Directory structure
- ✓ Configuration files
- ✓ Environment variables
- ✓ Python imports
- ✓ Settings validation

## Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at http://localhost:8501

## First Time Usage

1. **Dashboard**: View real-time market data and portfolio overview
2. **Trading Page**: Get AI recommendations and execute trades
3. **Portfolio Page**: Track your positions and performance
4. **Analytics Page**: View AI model performance and insights
5. **Backtest Page**: Test strategies on historical data

## Important Notes

### Paper Trading vs Live Trading

**Always start with paper trading!**

- Paper trading uses virtual money (no risk)
- Set `PAPER_TRADING=true` in .env.ai
- Test thoroughly before switching to live trading
- To enable live trading: `PAPER_TRADING=false` (use with caution!)

### Risk Management

Default risk settings (configured in .env.ai):
- Max position size: 10% of portfolio
- Daily loss limit: $1,000
- Stop-loss: 5% per position
- Max open positions: 10

Adjust these based on your risk tolerance.

### Automated Trading

Automated trading is **disabled by default** for safety:
- Set `ENABLE_AUTOMATED_TRADING=false` in .env.ai
- The AI will provide recommendations but won't execute automatically
- Enable only after thorough testing in paper trading mode

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the correct directory
cd /path/to/optlib

# Reinstall dependencies
pip install -r ai_requirements.txt
```

### API Connection Errors
- Check your API keys in .env.ai
- Verify your internet connection
- Check Alpaca API status: https://status.alpaca.markets/

### Configuration Errors
```bash
# Run verification script
python verify_setup.py

# Check for error messages and fix accordingly
```

## Next Steps

1. **Train Models**: Run model training scripts (coming in later tasks)
2. **Customize Strategy**: Adjust parameters in config/settings.py
3. **Add Watchlists**: Create watchlists of stocks to monitor
4. **Set Alerts**: Configure price and signal alerts
5. **Review Performance**: Analyze AI model accuracy and portfolio returns

## Getting Help

- Check the full documentation: AI_TRADING_AGENT_README.md
- Review the design document: .kiro/specs/ai-trading-agent/design.md
- Review the requirements: .kiro/specs/ai-trading-agent/requirements.md

## Safety Reminders

⚠️ **Important Safety Guidelines:**

1. Always test with paper trading first
2. Never invest more than you can afford to lose
3. Understand the risks of automated trading
4. Monitor your positions regularly
5. Set appropriate stop-losses
6. Start with small position sizes
7. Review AI recommendations before executing
8. Keep your API keys secure

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consult with a financial advisor before making investment decisions.

---

**Ready to start?** Run `streamlit run app.py` and begin your AI trading journey! 🚀
