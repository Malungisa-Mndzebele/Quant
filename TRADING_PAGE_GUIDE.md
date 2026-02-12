# Trading Page User Guide

## Overview

The Trading Page (`pages/2_trading.py`) is a comprehensive interface for stock analysis and trading with AI-powered recommendations. It integrates real-time market data, technical analysis, AI recommendations, and order execution capabilities.

## Features

### 1. Stock Search and Selection
- **Symbol Input**: Enter any stock symbol (e.g., AAPL, GOOGL, MSFT)
- **Search Button**: Click to load data for the selected symbol
- **Real-time Data**: Fetches current quotes and historical data

### 2. Current Price Display
- **Live Price**: Shows the current mid-price (average of bid/ask)
- **Daily Change**: Displays price change and percentage change from previous close
- **Bid/Ask Prices**: Shows current bid and ask prices
- **Real-time Updates**: Prices refresh automatically

### 3. Price Chart
- **Candlestick Chart**: Interactive price chart with OHLC data
- **Moving Averages**: Displays SMA 20 and SMA 50 overlays
- **Zoom and Pan**: Interactive chart controls
- **Historical Data**: Shows 60 days of price history

### 4. Technical Indicators
Displays key technical indicators:
- **SMA 20/50**: Simple Moving Averages
- **EMA 12**: Exponential Moving Average
- **RSI**: Relative Strength Index (overbought/oversold indicator)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Upper and lower bands for volatility

### 5. AI Recommendation
- **Action**: BUY, SELL, or HOLD recommendation
- **Confidence Score**: AI confidence level (0-100%)
- **Target Price**: Suggested price target
- **Stop Loss**: Recommended stop-loss level
- **Expected Return**: Projected return percentage
- **Risk Level**: Low, Medium, or High risk assessment
- **Detailed Analysis**: Bullet-point reasoning for the recommendation
- **Model Scores**: Individual scores from LSTM and Random Forest models

### 6. Manual Trading

#### Account Information
- **Portfolio Value**: Total account value
- **Available Cash**: Cash available for trading
- **Trading Mode**: Paper or Live trading indicator

#### Order Form
- **Action**: Select BUY or SELL
- **Quantity**: Enter number of shares (1-10,000)
- **Order Type**: Choose Market or Limit order
- **Limit Price**: Set limit price (for limit orders only)

#### Order Workflow
1. **Preview Order**: Click to review order details before submission
2. **Order Preview**: Shows symbol, action, quantity, type, price, and total value
3. **Risk Validation**: Automatically validates against risk limits
4. **Submit Order**: Execute the trade after preview
5. **Order Confirmation**: Displays order status and details

#### Risk Validation
The system automatically checks:
- Position size limits
- Available cash
- Maximum open positions
- Daily loss limits
- Portfolio risk exposure

If validation fails, you'll see:
- Error message explaining the violation
- Suggested quantity (if applicable)

### 7. Automated Trading

⚠️ **Risk Warning**: Automated trading executes trades without manual approval.

#### Features
- **Toggle Switch**: Enable/disable automated trading
- **Risk Warning**: Clear warning about automated trading risks
- **Active Indicator**: Visual indicator when automated trading is active
- **Continuous Monitoring**: AI monitors markets and executes high-confidence signals
- **Risk Enforcement**: All risk limits are enforced automatically
- **Trade Notifications**: Receive notifications for all automated trades

#### Best Practices
1. **Test in Paper Mode**: Always test strategies in paper trading first
2. **Set Conservative Limits**: Use conservative risk limits initially
3. **Monitor Regularly**: Check automated trades frequently
4. **Start Small**: Begin with small position sizes
5. **Review Performance**: Regularly review automated trading performance

### 8. Recent Orders
- **Order History**: Shows last 10 orders
- **Order Details**: Time, symbol, side, quantity, type, status, filled quantity, and price
- **Status Tracking**: Real-time order status updates
- **Sortable Table**: Interactive table with all order information

## Requirements Covered

This page implements all requirements from task 16.1:

✅ **Requirement 2.1**: AI generates recommendations with confidence scores  
✅ **Requirement 2.2**: Technical indicators analyzed and displayed  
✅ **Requirement 2.3**: Reasoning and key factors shown for recommendations  
✅ **Requirement 5.1**: Trade button to execute suggested actions  
✅ **Requirement 5.2**: Order preview with price, quantity, and cost  
✅ **Requirement 5.3**: Order submission to broker API with confirmation  
✅ **Requirement 5.4**: Portfolio display updated after trades  
✅ **Requirement 6.1**: Automated mode with confirmation and risk warnings  
✅ **Requirement 6.2**: Continuous market monitoring in automated mode  
✅ **Requirement 6.3**: Position size limits and risk parameters respected  
✅ **Requirement 6.4**: Decision rationale logged and user notified  

## Usage Instructions

### Running the Page

```bash
streamlit run pages/2_trading.py
```

Or from the main app:
```bash
streamlit run app.py
```
Then navigate to "Trading" in the sidebar.

### Basic Trading Workflow

1. **Search for a Stock**
   - Enter symbol in search box
   - Click "Search" button

2. **Review Analysis**
   - Check current price and daily change
   - Review technical indicators
   - Read AI recommendation and reasoning

3. **Place a Trade**
   - Select BUY or SELL
   - Enter quantity
   - Choose order type (Market or Limit)
   - Click "Preview Order"
   - Review order details
   - Click "Submit Order"

4. **Monitor Order**
   - View order confirmation
   - Check order status
   - Review in Recent Orders table

### Automated Trading Workflow

1. **Enable Automated Trading**
   - Read and understand risk warning
   - Toggle "Enable Automated Trading"
   - Confirm you understand the risks

2. **Monitor Activity**
   - Check Recent Orders for automated trades
   - Review portfolio performance
   - Adjust risk limits if needed

3. **Disable When Needed**
   - Toggle off to stop automated trading
   - Existing positions remain open
   - No new trades will be executed

## Configuration

### API Keys Required
- **Alpaca API Key**: For market data and trading
- **Alpaca Secret Key**: For authentication
- **News API Key**: For sentiment analysis (optional)

### Risk Settings
Configure in `config/settings.py`:
- `max_position_size`: Maximum position as % of portfolio
- `max_portfolio_risk`: Maximum portfolio risk exposure
- `daily_loss_limit`: Maximum daily loss in dollars
- `stop_loss_pct`: Stop-loss percentage
- `take_profit_pct`: Take-profit percentage
- `max_open_positions`: Maximum number of open positions

## Troubleshooting

### Common Issues

**"Failed to initialize services"**
- Check API credentials in `.env` file
- Verify internet connection
- Ensure Alpaca API is accessible

**"Insufficient data for analysis"**
- Stock may be newly listed
- Try a different symbol
- Check if market data is available

**"Trade validation failed"**
- Check error message for specific violation
- Review risk limits in settings
- Ensure sufficient cash available
- Consider suggested quantity if provided

**"Error generating recommendation"**
- AI models may not be loaded
- Check model files in `data/models/`
- Review logs for specific errors

**"Order submission failed"**
- Verify trading mode (paper vs live)
- Check market hours
- Ensure symbol is tradeable
- Review broker API status

### Getting Help

1. Check application logs for detailed error messages
2. Review configuration in `config/settings.py`
3. Verify API credentials and connectivity
4. Test in paper trading mode first
5. Consult the main README.md for setup instructions

## Safety Features

The trading page includes multiple safety features:

1. **Risk Validation**: All trades validated before submission
2. **Order Preview**: Review before executing
3. **Position Limits**: Automatic enforcement of position size limits
4. **Daily Loss Limits**: Trading disabled if daily loss limit reached
5. **Stop-Loss Protection**: Automatic stop-loss triggers
6. **Paper Trading Mode**: Test without real money
7. **Clear Mode Indicators**: Visual distinction between paper and live
8. **Risk Warnings**: Prominent warnings for automated trading

## Best Practices

1. **Always Start with Paper Trading**: Test strategies before using real money
2. **Set Conservative Limits**: Use conservative risk limits initially
3. **Review Recommendations**: Don't blindly follow AI recommendations
4. **Monitor Positions**: Regularly check open positions and P&L
5. **Use Stop-Losses**: Always set stop-loss orders
6. **Diversify**: Don't put all capital in one position
7. **Stay Informed**: Keep up with market news and conditions
8. **Review Performance**: Regularly analyze trading performance
9. **Adjust Strategies**: Refine based on results
10. **Know Your Risk Tolerance**: Trade within your comfort level

## Disclaimer

⚠️ **Important**: This is a trading tool that involves financial risk. Past performance does not guarantee future results. Always verify AI recommendations before trading. The developers are not responsible for trading losses. Use at your own risk.

## Next Steps

After using the trading page, you may want to:
- Review portfolio performance in the Portfolio page
- Analyze AI model performance in the Analytics page
- Backtest strategies in the Backtesting page
- Set up alerts in the Alerts page
- Customize strategies in the Strategy Builder

---

For more information, see:
- [Main README](README.md)
- [AI Trading Agent README](AI_TRADING_AGENT_README.md)
- [Design Document](.kiro/specs/ai-trading-agent/design.md)
- [Requirements Document](.kiro/specs/ai-trading-agent/requirements.md)
