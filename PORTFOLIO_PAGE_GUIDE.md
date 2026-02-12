# Portfolio Page Implementation Guide

## Overview

The Portfolio Page (`pages/3_portfolio.py`) provides comprehensive portfolio management and performance tracking capabilities for the AI Trading Agent. This page displays open positions, performance metrics, transaction history, and position management tools.

## Features Implemented

### 1. Portfolio Summary
- **Portfolio Value**: Total value of all positions and cash
- **Cash**: Available cash balance
- **Equity**: Total equity value
- **Buying Power**: Available buying power for new trades
- **Trading Mode Indicator**: Shows whether in paper or live trading mode

### 2. Open Positions Display
- **Positions Table**: Shows all open positions with:
  - Symbol
  - Quantity
  - Entry Price
  - Current Price
  - Market Value
  - Cost Basis
  - Unrealized P&L (dollars and percentage)
  - Position side (long/short)
- **Color-coded P&L**: Green for profits, red for losses
- **Real-time updates**: Positions reflect current market prices

### 3. Portfolio Allocation Chart
- **Pie Chart**: Visual representation of portfolio allocation by position
- **Interactive**: Hover to see detailed allocation percentages and values
- **Automatic calculation**: Based on current market values

### 4. Portfolio Performance Chart
- **Time Series Chart**: Shows portfolio value over time
- **Multiple Time Periods**: 1D, 1W, 1M, 3M, 1Y, YTD, All
- **Interactive**: Zoom and pan capabilities
- **Fill area**: Visual representation of portfolio growth

### 5. Performance Metrics
Comprehensive performance statistics including:

#### Main Metrics
- **Total Return**: Absolute and percentage return
- **Sharpe Ratio**: Risk-adjusted return metric
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of winning trades

#### Trade Statistics
- **Total Trades**: Number of completed trades
- **Winning Trades**: Number of profitable trades
- **Losing Trades**: Number of unprofitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

#### Average Metrics
- **Average Win**: Average profit per winning trade
- **Average Loss**: Average loss per losing trade
- **Largest Win**: Biggest single trade profit
- **Largest Loss**: Biggest single trade loss

#### Streaks
- **Current Streak**: Current winning or losing streak
- **Longest Win Streak**: Longest consecutive winning trades
- **Longest Loss Streak**: Longest consecutive losing trades

### 6. Transaction History
- **Filterable Table**: View transactions by time period
- **Detailed Information**: Date, symbol, side, quantity, price, total value, commission
- **Color-coded**: Buy orders in green, sell orders in red
- **Export Capability**: Download transactions as CSV file

### 7. Position Management
Interactive position management tools:

#### Close Entire Position
- One-click close with confirmation dialog
- Shows position details before closing
- Displays expected P&L impact

#### Partial Close by Percentage
- Close a percentage of position (1-100%)
- Useful for taking partial profits or reducing exposure
- Immediate execution

#### Partial Close by Quantity
- Close specific number of shares
- Precise control over position sizing
- Validates against available quantity

### 8. Data Export
- **CSV Export**: Download transaction history
- **Customizable Period**: Export data for selected time range
- **Tax Reporting**: Formatted for tax preparation

## Technical Implementation

### Services Used
- **TradingService**: Fetches positions, account info, executes closes
- **PortfolioService**: Calculates metrics, retrieves history, manages transactions
- **MarketDataService**: Provides real-time price updates

### Key Functions

#### `get_services()`
Initializes and caches all required services using Streamlit's `@st.cache_resource` decorator.

#### `format_currency(value: float) -> str`
Formats numeric values as currency with proper formatting.

#### `format_percentage(value: float) -> str`
Formats numeric values as percentages with +/- sign.

#### `get_period_dates(period: str) -> tuple`
Converts period strings (1D, 1W, etc.) to start and end datetime objects.

#### `display_positions_table(positions: List[Position])`
Renders the positions table with styled P&L values.

#### `display_performance_chart(portfolio_service, period)`
Creates and displays the portfolio performance time series chart.

#### `display_performance_metrics(metrics: PerformanceMetrics)`
Displays all performance metrics in an organized grid layout.

#### `display_transaction_history(portfolio_service, period)`
Shows transaction history table with filtering and styling.

#### `display_allocation_chart(positions: List[Position])`
Creates and displays the portfolio allocation pie chart.

#### `display_position_management(trading_service, positions)`
Provides interactive position management interface with close options.

## Usage

### Running the Portfolio Page

```bash
streamlit run pages/3_portfolio.py
```

### Accessing from Main App

The portfolio page is automatically available in the Streamlit multi-page app navigation when running:

```bash
streamlit run app.py
```

### Navigation

1. Start the application
2. Navigate to "Portfolio" in the sidebar
3. View positions, metrics, and history
4. Use position management tools as needed

## Requirements Validation

This implementation satisfies all requirements from task 17.1:

✅ **Display all open positions with P&L**
- Implemented in `display_positions_table()`
- Shows all position details with color-coded P&L

✅ **Show portfolio performance chart over time**
- Implemented in `display_performance_chart()`
- Multiple time period options
- Interactive Plotly chart

✅ **Display performance metrics (return, Sharpe, drawdown)**
- Implemented in `display_performance_metrics()`
- Comprehensive metrics including Sharpe ratio, drawdown, win rate, etc.

✅ **Show transaction history table**
- Implemented in `display_transaction_history()`
- Filterable by time period
- Exportable to CSV

✅ **Add position management (close, modify stop-loss)**
- Implemented in `display_position_management()`
- Close entire position
- Close by percentage
- Close by quantity
- Confirmation dialogs for safety

✅ **Display allocation by sector/asset class**
- Implemented in `display_allocation_chart()`
- Pie chart showing allocation by position
- Interactive with hover details

## Error Handling

The page includes comprehensive error handling:

- **Service Initialization**: Graceful failure with error messages
- **Data Fetching**: Try-catch blocks with user-friendly error messages
- **Position Management**: Validation before executing closes
- **Empty States**: Informative messages when no data is available

## Session State Management

The page uses Streamlit session state for:
- `selected_position`: Currently selected position for management
- `show_close_confirmation`: Controls confirmation dialog visibility
- `performance_period`: Remembers selected time period

## Styling and UX

- **Color Coding**: Green for profits, red for losses
- **Emojis**: Visual indicators for different sections and states
- **Responsive Layout**: Wide layout for better data visualization
- **Loading States**: Spinners during data fetching and operations
- **Confirmation Dialogs**: Safety checks before destructive operations

## Future Enhancements

Potential improvements for future versions:

1. **Stop-Loss Management**: Add/modify stop-loss orders for positions
2. **Take-Profit Management**: Set take-profit targets
3. **Position Notes**: Add notes to positions for tracking strategy
4. **Performance Comparison**: Compare against benchmarks (S&P 500, etc.)
5. **Risk Metrics**: Add VaR, beta, correlation analysis
6. **Sector Allocation**: Group positions by sector/industry
7. **Tax Loss Harvesting**: Identify tax loss harvesting opportunities
8. **Performance Attribution**: Analyze what contributed to returns

## Testing

To verify the implementation:

```bash
# Run verification script
python verify_portfolio_page.py

# Run import test
python test_portfolio_page_import.py

# Run the page
streamlit run pages/3_portfolio.py
```

## Dependencies

Required packages:
- streamlit >= 1.28.0
- pandas >= 2.0.0
- plotly >= 5.17.0
- alpaca-py (for trading service)

## Related Files

- `services/portfolio_service.py`: Portfolio data and calculations
- `services/trading_service.py`: Position and order management
- `services/market_data_service.py`: Real-time price data
- `pages/2_trading.py`: Trading page for executing new trades
- `pages/1_ai_dashboard.py`: Dashboard overview

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify API credentials are configured correctly
3. Ensure all services are initialized properly
4. Review the requirements validation section above

## Conclusion

The Portfolio Page provides a comprehensive interface for monitoring and managing trading positions. It integrates seamlessly with the AI Trading Agent's other components and provides all the functionality needed for effective portfolio management.
