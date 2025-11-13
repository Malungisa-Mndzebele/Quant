# Live Trading Setup Guide

## ⚠️ IMPORTANT WARNING

**Live trading involves real money and real risk. Please read this entire guide before connecting your brokerage account.**

- Always test strategies thoroughly in simulation mode first (minimum 2 weeks recommended)
- Start with small amounts you can afford to lose
- Monitor the system closely, especially in the first few days
- Understand that past performance does not guarantee future results
- The developers are not responsible for any financial losses

## Prerequisites

Before setting up live trading:

1. ✅ Tested strategies in simulation mode extensively
2. ✅ Understand the risks of algorithmic trading
3. ✅ Have a funded brokerage account (Public.com or Moomoo)
4. ✅ Obtained API credentials from your broker
5. ✅ Set appropriate risk limits in configuration
6. ✅ Have a plan for monitoring and intervention

## Supported Brokers

### Public.com

**Getting API Credentials:**
1. Log into your Public.com account
2. Navigate to Settings → API Access
3. Create a new API key
4. Copy both the API Key and API Secret immediately
5. Store them securely (you won't be able to see the secret again)

**Permissions Required:**
- Read account information
- Place orders
- View positions
- Cancel orders

### Moomoo

**Getting API Credentials:**
1. Log into your Moomoo account
2. Go to Settings → API Management
3. Apply for API access (may require approval)
4. Once approved, generate API credentials
5. Download and save your credentials securely

**Permissions Required:**
- Account query
- Order placement
- Position query
- Order cancellation

## Setup Instructions

### Step 1: Access the Web Interface

1. Start the web server:
   ```bash
   python web_app.py
   ```

2. Open your browser to: http://localhost:5000

### Step 2: Configure Brokerage Account

1. Scroll to the **"🔐 Brokerage Account"** section

2. Select your brokerage provider:
   - Public.com
   - Moomoo
   - Simulated (for testing)

3. Enter your credentials:
   - **API Key / Username**: Your brokerage API key
   - **API Secret / Password**: Your brokerage API secret

4. **Security Options:**
   - ☑️ Check "Save credentials securely" to store in encrypted .env file
   - ☐ Leave unchecked for session-only storage (more secure but requires re-entry)

### Step 3: Test Connection

1. Click **"Test Connection"** button

2. Wait for verification (usually 2-5 seconds)

3. Successful connection will show:
   - ✓ Connection Successful!
   - Your Account ID
   - Current account balance

4. If connection fails:
   - Verify credentials are correct
   - Check your internet connection
   - Ensure API access is enabled in your brokerage account
   - Review error message for specific issues

### Step 4: Save Credentials

1. After successful test, click **"Save Credentials"**

2. Credentials will be saved:
   - If "Save securely" checked: Stored in `.env` file (encrypted)
   - If unchecked: Stored in memory for current session only

3. Confirmation message will appear when saved successfully

### Step 5: Configure Trading Mode

1. In the **"Configuration"** section:
   - Change **Trading Mode** to "Live"
   - A warning box will appear - read it carefully

2. Set your trading parameters:
   - **Symbols**: Stocks you want to trade (e.g., AAPL, GOOGL)
   - **Update Interval**: How often to check for signals (60 seconds recommended)
   - **Initial Capital**: Not used in live mode (uses actual account balance)

3. Click **"Save Configuration"**

### Step 6: Set Risk Limits

**CRITICAL: Always set appropriate risk limits before live trading!**

Edit `config.yaml` to set:

```yaml
risk:
  max_position_size_pct: 5.0      # Max 5% per position (conservative)
  max_daily_loss_pct: 2.0         # Stop trading at 2% daily loss
  max_portfolio_leverage: 1.0     # No leverage
  max_positions: 3                # Maximum 3 positions at once
  allowed_symbols:                # Only trade these symbols
    - AAPL
    - GOOGL
    - MSFT
```

**Recommended Settings for Beginners:**
- Position size: 5% or less
- Daily loss limit: 2-3%
- No leverage (1.0)
- Limited number of positions (3-5)
- Whitelist specific symbols only

### Step 7: Start Live Trading

1. Review all settings one final time

2. Click the green **"Start System"** button

3. Confirm you want to start in live mode (if prompted)

4. Monitor the dashboard:
   - Status badge turns green
   - Portfolio metrics update in real-time
   - Positions appear as trades execute

## Monitoring Your System

### Real-Time Dashboard

The dashboard auto-refreshes every 5 seconds showing:

- **Total Portfolio Value**: Current worth of all holdings + cash
- **Cash Balance**: Available buying power
- **Positions Value**: Market value of all open positions
- **Unrealized P&L**: Profit/loss on open positions
- **Total Return**: Overall performance percentage
- **Number of Positions**: Count of open positions

### Position Tracking

The positions table shows:
- Symbol and quantity
- Average entry price
- Current market price
- Market value
- Unrealized P&L ($ and %)

### What to Watch For

**Normal Behavior:**
- Positions opening and closing based on strategy signals
- Gradual portfolio value changes
- Occasional rejected orders (due to risk limits)

**Warning Signs:**
- Rapid, large position changes
- Frequent order rejections
- Approaching daily loss limit
- Unexpected symbols being traded
- System errors in logs

## Emergency Procedures

### Stopping the System

1. Click the red **"Stop System"** button
2. System will stop generating new signals
3. Existing positions remain open
4. You can manually close positions through your broker

### Manual Intervention

If you need to intervene:

1. Stop the system first
2. Log into your brokerage account directly
3. Manually close positions if needed
4. Review logs in `logs/trading_system_live.log`
5. Adjust configuration before restarting

### Daily Loss Limit Hit

When daily loss limit is reached:
- System automatically stops trading
- No new orders will be placed
- Existing positions remain open
- System will resume next trading day

## Security Best Practices

### Credential Storage

**Most Secure (Recommended):**
1. Save credentials to `.env` file
2. Add `.env` to `.gitignore` (already done)
3. Never commit `.env` to version control
4. Use API keys with minimal required permissions

**Session Only:**
- Credentials stored in memory only
- Must re-enter after restart
- More secure but less convenient

### API Key Permissions

**Enable Only:**
- Read account information
- Place orders
- View positions
- Cancel orders

**Disable:**
- Withdraw funds
- Transfer funds
- Change account settings
- Access personal information

### Network Security

- Run on trusted network only
- Use firewall to restrict access
- Don't expose web interface to internet
- Use HTTPS in production (not included in dev server)

## Troubleshooting

### Connection Issues

**Problem**: "Authentication failed"
- **Solution**: Verify API credentials are correct
- Check if API access is enabled in brokerage account
- Ensure API key hasn't expired

**Problem**: "Network timeout"
- **Solution**: Check internet connection
- Verify brokerage API is operational
- Try again in a few minutes

### Order Rejections

**Problem**: Orders being rejected
- **Possible Causes**:
  - Insufficient funds
  - Position size exceeds limit
  - Symbol not allowed
  - Market closed
  - Invalid order parameters

- **Solutions**:
  - Check account balance
  - Review risk limits in config
  - Verify symbol is tradeable
  - Check market hours
  - Review logs for specific error

### System Not Trading

**Problem**: System running but no trades
- **Possible Causes**:
  - Strategy not generating signals
  - Risk limits blocking orders
  - Market conditions not met
  - Insufficient data

- **Solutions**:
  - Review strategy parameters
  - Check risk configuration
  - Verify market data is updating
  - Review logs for details

## Performance Monitoring

### Daily Review

Check these metrics daily:
- Total return vs. benchmark
- Number of trades executed
- Win rate
- Largest gain/loss
- Risk limit hits

### Weekly Review

Analyze:
- Strategy performance by symbol
- Order fill quality
- Slippage and fees impact
- Risk-adjusted returns
- System uptime and errors

### Monthly Review

Evaluate:
- Overall strategy effectiveness
- Need for parameter adjustments
- Risk limit appropriateness
- Comparison to simulation results
- Decision to continue/modify/stop

## Logs and Records

### Log Files

- `logs/trading_system_live.log` - All live trading activity
- `logs/trading_system_live_errors.log` - Errors only

### What's Logged

- Every order submission
- Order fills and rejections
- Risk limit checks
- Account balance updates
- System errors and warnings

### Compliance

Keep logs for:
- Tax reporting
- Performance analysis
- Audit trail
- Troubleshooting

## Legal and Compliance

### Disclaimer

This software is provided "as is" without warranty. The developers are not:
- Financial advisors
- Responsible for trading losses
- Liable for system errors or bugs
- Guaranteeing any returns

### Your Responsibilities

You are responsible for:
- Understanding the risks
- Complying with regulations
- Paying taxes on gains
- Monitoring the system
- Making informed decisions

### Regulations

Be aware of:
- Pattern day trader rules (US)
- Wash sale rules
- Tax reporting requirements
- Broker-specific rules
- Local trading regulations

## Getting Help

### Before Contacting Support

1. Check logs in `logs/` directory
2. Review this guide thoroughly
3. Test in simulation mode
4. Verify configuration is correct

### Support Channels

- GitHub Issues: For bugs and feature requests
- Documentation: README.md and other guides
- Community: Share experiences with other users

## Recommended Reading

- [GETTING_STARTED.md](GETTING_STARTED.md) - General system guide
- [WEB_README.md](WEB_README.md) - Web interface documentation
- [README.md](README.md) - Complete system documentation
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - Common issues

---

## Final Checklist Before Going Live

- [ ] Tested strategies for at least 2 weeks in simulation
- [ ] Reviewed and understood all risks
- [ ] Set conservative risk limits
- [ ] Obtained and tested brokerage API credentials
- [ ] Verified account has sufficient funds
- [ ] Configured monitoring and alerts
- [ ] Have plan for daily monitoring
- [ ] Know how to stop system in emergency
- [ ] Understand tax implications
- [ ] Starting with small position sizes
- [ ] Ready to monitor closely for first week

**Only proceed if you can check ALL boxes above.**

---

**Remember: Start small, monitor closely, and never trade with money you can't afford to lose.**
