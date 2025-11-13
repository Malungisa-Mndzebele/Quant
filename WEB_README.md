# Web Interface for Quant Trading System

A modern web interface to configure, monitor, and control your quantitative trading system.

## Features

- **Real-time Dashboard**: Monitor portfolio value, positions, and P&L in real-time
- **System Control**: Start/stop the trading system with a single click
- **Configuration Management**: Edit trading parameters, symbols, and risk settings
- **Backtesting**: Run historical backtests and view performance metrics
- **Live Updates**: Automatic status updates every 5 seconds when system is running

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install flask flask-cors
   ```

2. **Start the web server**:
   ```bash
   python web_app.py
   ```

3. **Open your browser**:
   Navigate to `http://localhost:5000`

## Usage

### Starting the Trading System

1. Configure your settings in the "Configuration" section
2. Click "Save Configuration"
3. Click "Start System" to begin trading
4. Monitor your portfolio in real-time

### Running a Backtest

1. Scroll to the "Backtesting" section
2. Select start and end dates
3. Enter symbols to test (comma-separated)
4. Click "Run Backtest"
5. View results including returns, Sharpe ratio, and drawdown

### Configuration Options

- **Trading Mode**: Simulation (paper trading) or Live (real money)
- **Symbols**: Comma-separated list of stock symbols to trade
- **Update Interval**: How often to check for new signals (in seconds)
- **Initial Capital**: Starting portfolio value

## API Endpoints

The web app exposes a REST API:

- `GET /api/config` - Get current configuration
- `POST /api/config` - Update configuration
- `POST /api/start` - Start the trading system
- `POST /api/stop` - Stop the trading system
- `GET /api/status` - Get current system status
- `POST /api/backtest` - Run a backtest

## Security Notes

⚠️ **Important**: This web interface is for local development only. Do not expose it to the internet without proper authentication and security measures.

For production use, consider:
- Adding authentication (JWT, OAuth, etc.)
- Using HTTPS
- Implementing rate limiting
- Adding input validation
- Setting up proper CORS policies

## Troubleshooting

**Port already in use?**
Edit `web_app.py` and change the port:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to 5001
```

**System won't start?**
Check the console logs for errors. Common issues:
- Invalid configuration
- Missing API credentials for live mode
- Network connectivity issues

## Architecture

```
web_app.py          # Flask backend server
├── templates/
│   └── index.html  # Main dashboard UI
└── static/
    ├── style.css   # Styling
    └── app.js      # Frontend JavaScript
```

The web app runs the trading system in a background thread, allowing you to control it through the browser while it executes trades.
