"""
AI Trading Agent - Portfolio Page

This page displays portfolio positions, performance metrics, transaction history,
and provides position management capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import time
import io
from typing import Optional, List, Dict, Any

from services.portfolio_service import PortfolioService, PerformanceMetrics
from services.trading_service import TradingService, Position
from services.market_data_service import MarketDataService
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Portfolio - AI Trading Agent",
    page_icon="💼",
    layout="wide"
)

# Initialize session state
if 'selected_position' not in st.session_state:
    st.session_state.selected_position = None

if 'show_close_confirmation' not in st.session_state:
    st.session_state.show_close_confirmation = False

if 'performance_period' not in st.session_state:
    st.session_state.performance_period = '1M'


# Initialize services
@st.cache_resource
def get_services():
    """Initialize and cache services"""
    try:
        trading_service = TradingService()
        portfolio_service = PortfolioService(trading_service=trading_service)
        market_service = MarketDataService()
        
        return trading_service, portfolio_service, market_service
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        logger.error(f"Service initialization error: {e}")
        return None, None, None


def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:+.2f}%"


def get_period_dates(period: str) -> tuple:
    """Get start and end dates for a period"""
    end_date = datetime.now()
    
    if period == '1D':
        start_date = end_date - timedelta(days=1)
    elif period == '1W':
        start_date = end_date - timedelta(weeks=1)
    elif period == '1M':
        start_date = end_date - timedelta(days=30)
    elif period == '3M':
        start_date = end_date - timedelta(days=90)
    elif period == '1Y':
        start_date = end_date - timedelta(days=365)
    elif period == 'YTD':
        start_date = datetime(end_date.year, 1, 1)
    else:  # All
        start_date = None
    
    return start_date, end_date


def display_positions_table(positions: List[Position]):
    """Display positions in a formatted table"""
    if not positions:
        st.info("No open positions")
        return
    
    # Create DataFrame
    positions_data = []
    for pos in positions:
        positions_data.append({
            'Symbol': pos.symbol,
            'Quantity': pos.quantity,
            'Entry Price': format_currency(pos.entry_price),
            'Current Price': format_currency(pos.current_price),
            'Market Value': format_currency(pos.market_value),
            'Cost Basis': format_currency(pos.cost_basis),
            'P&L': format_currency(pos.unrealized_pl),
            'P&L %': format_percentage(pos.unrealized_pl_pct * 100),
            'Side': pos.side.upper()
        })
    
    df = pd.DataFrame(positions_data)
    
    # Style the dataframe
    def style_pnl(val):
        """Style P&L cells"""
        if isinstance(val, str) and ('$' in val or '%' in val):
            # Extract numeric value
            numeric_val = float(val.replace('$', '').replace('%', '').replace(',', '').replace('+', ''))
            if numeric_val > 0:
                return 'color: #28a745; font-weight: bold'
            elif numeric_val < 0:
                return 'color: #dc3545; font-weight: bold'
        return ''
    
    styled_df = df.style.applymap(style_pnl, subset=['P&L', 'P&L %'])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def display_performance_chart(portfolio_service: PortfolioService, period: str):
    """Display portfolio performance chart"""
    try:
        start_date, end_date = get_period_dates(period)
        
        # Get portfolio history
        history = portfolio_service.get_portfolio_history(start_date, end_date)
        
        if history.empty:
            st.info("No portfolio history available. Portfolio snapshots are saved periodically.")
            return
        
        # Create chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=history['timestamp'],
            y=history['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ))
        
        fig.update_layout(
            title=f'Portfolio Performance ({period})',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=400,
            hovermode='x unified',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying performance chart: {e}")
        logger.error(f"Performance chart error: {e}")


def display_performance_metrics(metrics: PerformanceMetrics):
    """Display performance metrics in a grid"""
    st.markdown("### 📊 Performance Metrics")
    
    # Row 1: Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            format_currency(metrics.total_return),
            format_percentage(metrics.total_return_pct)
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.sharpe_ratio:.2f}",
            help="Risk-adjusted return metric"
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            format_currency(metrics.max_drawdown),
            format_percentage(-metrics.max_drawdown_pct)
        )
    
    with col4:
        st.metric(
            "Win Rate",
            f"{metrics.win_rate:.1%}",
            help="Percentage of winning trades"
        )
    
    # Row 2: Trade statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", metrics.total_trades)
    
    with col2:
        st.metric("Winning Trades", metrics.winning_trades)
    
    with col3:
        st.metric("Losing Trades", metrics.losing_trades)
    
    with col4:
        st.metric("Profit Factor", f"{metrics.profit_factor:.2f}")
    
    # Row 3: Average metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Win", format_currency(metrics.avg_win))
    
    with col2:
        st.metric("Avg Loss", format_currency(metrics.avg_loss))
    
    with col3:
        st.metric("Largest Win", format_currency(metrics.largest_win))
    
    with col4:
        st.metric("Largest Loss", format_currency(metrics.largest_loss))
    
    # Streaks
    st.markdown("### 🔥 Streaks")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        streak_emoji = "🟢" if metrics.current_streak > 0 else "🔴" if metrics.current_streak < 0 else "⚪"
        st.metric(
            "Current Streak",
            f"{streak_emoji} {abs(metrics.current_streak)}",
            help="Current winning or losing streak"
        )
    
    with col2:
        st.metric("Longest Win Streak", f"🟢 {metrics.longest_win_streak}")
    
    with col3:
        st.metric("Longest Loss Streak", f"🔴 {metrics.longest_loss_streak}")


def display_transaction_history(portfolio_service: PortfolioService, period: str):
    """Display transaction history table"""
    try:
        start_date, end_date = get_period_dates(period)
        
        transactions = portfolio_service.get_transaction_history(start_date, end_date)
        
        if not transactions:
            st.info("No transactions in selected period")
            return
        
        # Create DataFrame
        txn_data = []
        for txn in transactions:
            txn_data.append({
                'Date': txn.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Symbol': txn.symbol,
                'Side': txn.side.upper(),
                'Quantity': txn.quantity,
                'Price': format_currency(txn.price),
                'Total Value': format_currency(txn.total_value),
                'Commission': format_currency(txn.commission),
                'Order ID': txn.order_id[:8] + '...'
            })
        
        df = pd.DataFrame(txn_data)
        
        # Style buy/sell
        def style_side(val):
            if val == 'BUY':
                return 'color: #28a745; font-weight: bold'
            elif val == 'SELL':
                return 'color: #dc3545; font-weight: bold'
            return ''
        
        styled_df = df.style.applymap(style_side, subset=['Side'])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error displaying transaction history: {e}")
        logger.error(f"Transaction history error: {e}")


def display_allocation_chart(positions: List[Position]):
    """Display portfolio allocation by position"""
    if not positions:
        st.info("No positions to display")
        return
    
    # Calculate allocation
    total_value = sum(abs(pos.market_value) for pos in positions)
    
    if total_value == 0:
        st.info("No allocation to display")
        return
    
    symbols = [pos.symbol for pos in positions]
    values = [abs(pos.market_value) for pos in positions]
    percentages = [(v / total_value * 100) for v in values]
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=symbols,
        values=values,
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>Allocation: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Portfolio Allocation',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_position_management(
    trading_service: TradingService,
    positions: List[Position]
):
    """Display position management interface"""
    if not positions:
        st.info("No positions to manage")
        return
    
    st.markdown("### ⚙️ Position Management")
    
    # Select position
    position_symbols = [pos.symbol for pos in positions]
    selected_symbol = st.selectbox(
        "Select Position",
        options=position_symbols,
        key="position_select"
    )
    
    # Find selected position
    selected_pos = next((p for p in positions if p.symbol == selected_symbol), None)
    
    if not selected_pos:
        return
    
    # Display position details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Symbol:** {selected_pos.symbol}")
        st.markdown(f"**Quantity:** {selected_pos.quantity}")
    
    with col2:
        st.markdown(f"**Entry Price:** {format_currency(selected_pos.entry_price)}")
        st.markdown(f"**Current Price:** {format_currency(selected_pos.current_price)}")
    
    with col3:
        st.markdown(f"**Market Value:** {format_currency(selected_pos.market_value)}")
        pnl_color = "🟢" if selected_pos.unrealized_pl >= 0 else "🔴"
        st.markdown(f"**P&L:** {pnl_color} {format_currency(selected_pos.unrealized_pl)} ({format_percentage(selected_pos.unrealized_pl_pct * 100)})")
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔴 Close Position", use_container_width=True, type="primary"):
            st.session_state.show_close_confirmation = True
            st.session_state.selected_position = selected_pos
    
    with col2:
        close_percentage = st.number_input(
            "Close %",
            min_value=1,
            max_value=100,
            value=50,
            step=10,
            key="close_pct"
        )
        
        if st.button(f"Close {close_percentage}%", use_container_width=True):
            try:
                with st.spinner(f"Closing {close_percentage}% of {selected_pos.symbol}..."):
                    order = trading_service.close_position(
                        symbol=selected_pos.symbol,
                        percentage=close_percentage
                    )
                    st.success(f"✅ Close order submitted: {order.order_id[:8]}...")
                    time.sleep(1)
                    st.rerun()
            except Exception as e:
                st.error(f"Error closing position: {e}")
                logger.error(f"Position close error: {e}")
    
    with col3:
        close_quantity = st.number_input(
            "Close Qty",
            min_value=1,
            max_value=selected_pos.quantity,
            value=min(10, selected_pos.quantity),
            step=1,
            key="close_qty"
        )
        
        if st.button(f"Close {close_quantity} shares", use_container_width=True):
            try:
                with st.spinner(f"Closing {close_quantity} shares of {selected_pos.symbol}..."):
                    order = trading_service.close_position(
                        symbol=selected_pos.symbol,
                        qty=close_quantity
                    )
                    st.success(f"✅ Close order submitted: {order.order_id[:8]}...")
                    time.sleep(1)
                    st.rerun()
            except Exception as e:
                st.error(f"Error closing position: {e}")
                logger.error(f"Position close error: {e}")
    
    # Close confirmation dialog
    if st.session_state.show_close_confirmation and st.session_state.selected_position:
        pos = st.session_state.selected_position
        
        st.warning(f"""
        ⚠️ **Confirm Position Close**
        
        You are about to close your entire position in **{pos.symbol}**:
        - Quantity: {pos.quantity} shares
        - Current Value: {format_currency(pos.market_value)}
        - Unrealized P&L: {format_currency(pos.unrealized_pl)} ({format_percentage(pos.unrealized_pl_pct * 100)})
        
        This action cannot be undone.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Confirm Close", use_container_width=True, type="primary"):
                try:
                    with st.spinner(f"Closing position in {pos.symbol}..."):
                        order = trading_service.close_position(symbol=pos.symbol)
                        st.success(f"✅ Position closed: {order.order_id[:8]}...")
                        st.session_state.show_close_confirmation = False
                        st.session_state.selected_position = None
                        time.sleep(1)
                        st.rerun()
                except Exception as e:
                    st.error(f"Error closing position: {e}")
                    logger.error(f"Position close error: {e}")
        
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_close_confirmation = False
                st.session_state.selected_position = None
                st.rerun()


# Main page
st.title("💼 Portfolio")
st.markdown("View your positions, performance metrics, and transaction history")

# Get services
trading_service, portfolio_service, market_service = get_services()

if not all([trading_service, portfolio_service, market_service]):
    st.error("Failed to initialize services. Please check your configuration.")
    st.stop()

# Get account info
try:
    account = trading_service.get_account()
    portfolio_value = account['portfolio_value']
    cash = account['cash']
    equity = account['equity']
    buying_power = account['buying_power']
except Exception as e:
    st.error(f"Error fetching account info: {e}")
    logger.error(f"Account info error: {e}")
    st.stop()

# Portfolio summary
st.markdown("## 💰 Portfolio Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Portfolio Value", format_currency(portfolio_value))

with col2:
    st.metric("Cash", format_currency(cash))

with col3:
    st.metric("Equity", format_currency(equity))

with col4:
    st.metric("Buying Power", format_currency(buying_power))

# Trading mode indicator
trading_mode = trading_service.get_trading_mode()
mode_color = "🟢" if trading_mode == "paper" else "🔴"
st.info(f"{mode_color} Trading Mode: **{trading_mode.upper()}**")

# Get positions
try:
    positions = trading_service.get_positions()
except Exception as e:
    st.error(f"Error fetching positions: {e}")
    logger.error(f"Positions fetch error: {e}")
    positions = []

# Open positions
st.markdown("## 📊 Open Positions")
display_positions_table(positions)

# Portfolio allocation
if positions:
    st.markdown("## 🥧 Portfolio Allocation")
    display_allocation_chart(positions)

# Performance chart
st.markdown("## 📈 Portfolio Performance")

# Period selector
period_options = ['1D', '1W', '1M', '3M', '1Y', 'YTD', 'All']
selected_period = st.selectbox(
    "Time Period",
    options=period_options,
    index=period_options.index(st.session_state.performance_period),
    key="period_selector"
)
st.session_state.performance_period = selected_period

display_performance_chart(portfolio_service, selected_period)

# Performance metrics
st.markdown("## 📊 Performance Analysis")

try:
    start_date, end_date = get_period_dates(selected_period)
    metrics = portfolio_service.get_performance_metrics(start_date, end_date)
    display_performance_metrics(metrics)
except Exception as e:
    st.error(f"Error calculating performance metrics: {e}")
    logger.error(f"Performance metrics error: {e}")

# Transaction history
st.markdown("## 📜 Transaction History")

# Period selector for transactions
txn_period = st.selectbox(
    "Transaction Period",
    options=period_options,
    index=2,  # Default to 1M
    key="txn_period_selector"
)

display_transaction_history(portfolio_service, txn_period)

# Export section
st.markdown("## 📥 Export Data")
st.markdown("Export your trading data for analysis, record-keeping, or tax purposes.")

# Create tabs for different export types
export_tab1, export_tab2, export_tab3 = st.tabs([
    "📊 Trades Export",
    "📈 Portfolio History",
    "🧾 Tax Report"
])

# Tab 1: Trades Export
with export_tab1:
    st.markdown("### Export Transaction History")
    st.markdown("Export all your trades with detailed information including prices, quantities, and commissions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_trades_period = st.selectbox(
            "Select Time Period",
            options=period_options,
            index=2,  # Default to 1M
            key="export_trades_period"
        )
    
    with col2:
        # Custom date range option
        use_custom_dates_trades = st.checkbox("Use custom date range", key="custom_dates_trades")
    
    if use_custom_dates_trades:
        col1, col2 = st.columns(2)
        with col1:
            custom_start_trades = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=30),
                key="custom_start_trades"
            )
        with col2:
            custom_end_trades = st.date_input(
                "End Date",
                value=datetime.now(),
                key="custom_end_trades"
            )
        
        start_date_trades = datetime.combine(custom_start_trades, datetime.min.time())
        end_date_trades = datetime.combine(custom_end_trades, datetime.max.time())
    else:
        start_date_trades, end_date_trades = get_period_dates(export_trades_period)
    
    if st.button("📥 Generate Trades Export", key="export_trades_btn", use_container_width=True):
        try:
            with st.spinner("Generating trades export..."):
                csv_data = portfolio_service.export_trades_csv(start_date_trades, end_date_trades)
                
                if csv_data:
                    # Display preview
                    st.success(f"✅ Export generated successfully!")
                    
                    # Show preview
                    preview_df = pd.read_csv(io.StringIO(csv_data))
                    st.markdown(f"**Preview** (showing first 10 rows of {len(preview_df)} total):")
                    st.dataframe(preview_df.head(10), use_container_width=True)
                    
                    # Download button
                    filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("No trades found in the selected period")
        except Exception as e:
            st.error(f"Error generating trades export: {e}")
            logger.error(f"Trades export error: {e}")

# Tab 2: Portfolio History Export
with export_tab2:
    st.markdown("### Export Portfolio History")
    st.markdown("Export historical snapshots of your portfolio value and positions over time.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_history_period = st.selectbox(
            "Select Time Period",
            options=period_options,
            index=2,  # Default to 1M
            key="export_history_period"
        )
    
    with col2:
        # Custom date range option
        use_custom_dates_history = st.checkbox("Use custom date range", key="custom_dates_history")
    
    if use_custom_dates_history:
        col1, col2 = st.columns(2)
        with col1:
            custom_start_history = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=30),
                key="custom_start_history"
            )
        with col2:
            custom_end_history = st.date_input(
                "End Date",
                value=datetime.now(),
                key="custom_end_history"
            )
        
        start_date_history = datetime.combine(custom_start_history, datetime.min.time())
        end_date_history = datetime.combine(custom_end_history, datetime.max.time())
    else:
        start_date_history, end_date_history = get_period_dates(export_history_period)
    
    if st.button("📥 Generate Portfolio History Export", key="export_history_btn", use_container_width=True):
        try:
            with st.spinner("Generating portfolio history export..."):
                csv_data = portfolio_service.export_portfolio_history_csv(start_date_history, end_date_history)
                
                if csv_data:
                    # Display preview
                    st.success(f"✅ Export generated successfully!")
                    
                    # Show preview
                    preview_df = pd.read_csv(io.StringIO(csv_data))
                    st.markdown(f"**Preview** (showing first 10 rows of {len(preview_df)} total):")
                    st.dataframe(preview_df.head(10), use_container_width=True)
                    
                    # Download button
                    filename = f"portfolio_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("No portfolio history found in the selected period")
        except Exception as e:
            st.error(f"Error generating portfolio history export: {e}")
            logger.error(f"Portfolio history export error: {e}")

# Tab 3: Tax Report Export
with export_tab3:
    st.markdown("### Export Tax Report")
    st.markdown("Generate a tax report with all realized gains and losses for the selected year.")
    
    st.info("📋 This report includes all closed trades with cost basis, proceeds, and holding periods formatted for tax reporting.")
    
    # Year selector
    current_year = datetime.now().year
    tax_year = st.selectbox(
        "Select Tax Year",
        options=list(range(current_year, current_year - 10, -1)),
        index=0,
        key="tax_year_selector"
    )
    
    if st.button("📥 Generate Tax Report", key="export_tax_btn", use_container_width=True):
        try:
            with st.spinner(f"Generating tax report for {tax_year}..."):
                csv_data = portfolio_service.export_tax_report_csv(tax_year)
                
                if csv_data:
                    # Display preview
                    st.success(f"✅ Tax report generated successfully!")
                    
                    # Show preview
                    preview_df = pd.read_csv(io.StringIO(csv_data))
                    st.markdown(f"**Preview** (showing first 10 rows of {len(preview_df)} total):")
                    st.dataframe(preview_df.head(10), use_container_width=True)
                    
                    # Calculate summary statistics
                    st.markdown("### 📊 Tax Year Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    # Parse gain/loss values (remove $ and commas)
                    gains_losses = preview_df['Gain/Loss'].str.replace('$', '').str.replace(',', '').astype(float)
                    
                    with col1:
                        total_gain_loss = gains_losses.sum()
                        st.metric("Total Gain/Loss", f"${total_gain_loss:,.2f}")
                    
                    with col2:
                        short_term = preview_df[preview_df['Term'] == 'Short-term']
                        if not short_term.empty:
                            st_gains = short_term['Gain/Loss'].str.replace('$', '').str.replace(',', '').astype(float).sum()
                            st.metric("Short-term Gain/Loss", f"${st_gains:,.2f}")
                        else:
                            st.metric("Short-term Gain/Loss", "$0.00")
                    
                    with col3:
                        long_term = preview_df[preview_df['Term'] == 'Long-term']
                        if not long_term.empty:
                            lt_gains = long_term['Gain/Loss'].str.replace('$', '').str.replace(',', '').astype(float).sum()
                            st.metric("Long-term Gain/Loss", f"${lt_gains:,.2f}")
                        else:
                            st.metric("Long-term Gain/Loss", "$0.00")
                    
                    # Download button
                    filename = f"tax_report_{tax_year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    st.download_button(
                        label="⬇️ Download Tax Report",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.warning("⚠️ **Disclaimer:** This report is for informational purposes only. Please consult with a tax professional for accurate tax filing.")
                else:
                    st.info(f"No closed trades found for tax year {tax_year}")
        except Exception as e:
            st.error(f"Error generating tax report: {e}")
            logger.error(f"Tax report export error: {e}")

# Position management
if positions:
    st.markdown("## ⚙️ Manage Positions")
    display_position_management(trading_service, positions)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>AI Trading Agent | Portfolio tracking and management | Past performance does not guarantee future results</p>
</div>
""", unsafe_allow_html=True)
