"""
Backtesting page for testing trading strategies against historical data.

This page allows users to:
- Select and configure trading strategies
- Set date range and initial capital
- Run backtests with progress tracking
- View performance metrics and equity curves
- Compare multiple strategies
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging

from services.backtest_service import BacktestEngine, Strategy, BacktestResult
from services.market_data_service import MarketDataService
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Backtesting - AI Trading Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Strategy Backtesting")
st.markdown("Test your trading strategies against historical data to evaluate performance before risking real capital.")

# Initialize session state
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = []
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False


# Define available strategies
def simple_ma_crossover(data: pd.DataFrame) -> str:
    """Simple moving average crossover strategy"""
    if len(data) < 50:
        return 'hold'
    
    # Calculate moving averages
    data = data.copy()
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    
    # Get last two rows
    if len(data) < 2:
        return 'hold'
    
    current = data.iloc[-1]
    previous = data.iloc[-2]
    
    # Check for crossover
    if previous['sma_20'] <= previous['sma_50'] and current['sma_20'] > current['sma_50']:
        return 'buy'
    elif previous['sma_20'] >= previous['sma_50'] and current['sma_20'] < current['sma_50']:
        return 'sell'
    
    return 'hold'


def rsi_strategy(data: pd.DataFrame) -> str:
    """RSI-based mean reversion strategy"""
    if len(data) < 14:
        return 'hold'
    
    # Calculate RSI
    data = data.copy()
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    current_rsi = rsi.iloc[-1]
    
    # RSI signals
    if current_rsi < 30:  # Oversold
        return 'buy'
    elif current_rsi > 70:  # Overbought
        return 'sell'
    
    return 'hold'


def momentum_strategy(data: pd.DataFrame) -> str:
    """Momentum-based strategy"""
    if len(data) < 20:
        return 'hold'
    
    data = data.copy()
    
    # Calculate momentum (rate of change)
    data['momentum'] = data['close'].pct_change(periods=10) * 100
    
    current_momentum = data['momentum'].iloc[-1]
    
    # Momentum signals
    if current_momentum > 5:  # Strong upward momentum
        return 'buy'
    elif current_momentum < -5:  # Strong downward momentum
        return 'sell'
    
    return 'hold'


def bollinger_bands_strategy(data: pd.DataFrame) -> str:
    """Bollinger Bands mean reversion strategy"""
    if len(data) < 20:
        return 'hold'
    
    data = data.copy()
    
    # Calculate Bollinger Bands
    data['sma'] = data['close'].rolling(window=20).mean()
    data['std'] = data['close'].rolling(window=20).std()
    data['upper_band'] = data['sma'] + (2 * data['std'])
    data['lower_band'] = data['sma'] - (2 * data['std'])
    
    current = data.iloc[-1]
    
    # Bollinger Band signals
    if current['close'] < current['lower_band']:  # Price below lower band
        return 'buy'
    elif current['close'] > current['upper_band']:  # Price above upper band
        return 'sell'
    
    return 'hold'


# Strategy definitions
STRATEGIES = {
    'MA Crossover': {
        'func': simple_ma_crossover,
        'description': 'Buy when 20-day MA crosses above 50-day MA, sell on opposite crossover',
        'default_params': {
            'position_size': 0.95,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'commission_pct': 0.001
        }
    },
    'RSI Mean Reversion': {
        'func': rsi_strategy,
        'description': 'Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)',
        'default_params': {
            'position_size': 0.90,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.08,
            'commission_pct': 0.001
        }
    },
    'Momentum': {
        'func': momentum_strategy,
        'description': 'Buy on strong upward momentum, sell on strong downward momentum',
        'default_params': {
            'position_size': 0.85,
            'stop_loss_pct': 0.04,
            'take_profit_pct': 0.12,
            'commission_pct': 0.001
        }
    },
    'Bollinger Bands': {
        'func': bollinger_bands_strategy,
        'description': 'Buy when price touches lower band, sell when price touches upper band',
        'default_params': {
            'position_size': 0.90,
            'stop_loss_pct': 0.04,
            'take_profit_pct': 0.08,
            'commission_pct': 0.001
        }
    }
}


# Sidebar - Strategy Configuration
st.sidebar.header("⚙️ Backtest Configuration")

# Strategy selection
selected_strategy = st.sidebar.selectbox(
    "Select Strategy",
    options=list(STRATEGIES.keys()),
    help="Choose a trading strategy to backtest"
)

st.sidebar.markdown(f"**Description:** {STRATEGIES[selected_strategy]['description']}")

# Symbol input
symbol = st.sidebar.text_input(
    "Stock Symbol",
    value="AAPL",
    help="Enter the stock symbol to backtest"
).upper()

# Date range
st.sidebar.subheader("Date Range")
col1, col2 = st.sidebar.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365),
        max_value=datetime.now()
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )

# Initial capital
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000.0,
    max_value=10000000.0,
    value=100000.0,
    step=1000.0,
    help="Starting capital for the backtest"
)

# Strategy parameters
st.sidebar.subheader("Strategy Parameters")

default_params = STRATEGIES[selected_strategy]['default_params']

position_size = st.sidebar.slider(
    "Position Size (%)",
    min_value=10,
    max_value=100,
    value=int(default_params['position_size'] * 100),
    step=5,
    help="Percentage of capital to use per trade"
) / 100

stop_loss_pct = st.sidebar.slider(
    "Stop Loss (%)",
    min_value=0,
    max_value=20,
    value=int(default_params['stop_loss_pct'] * 100),
    step=1,
    help="Stop loss percentage from entry price"
) / 100

take_profit_pct = st.sidebar.slider(
    "Take Profit (%)",
    min_value=0,
    max_value=50,
    value=int(default_params['take_profit_pct'] * 100),
    step=1,
    help="Take profit percentage from entry price"
) / 100

commission_pct = st.sidebar.number_input(
    "Commission (%)",
    min_value=0.0,
    max_value=1.0,
    value=default_params['commission_pct'] * 100,
    step=0.01,
    help="Commission percentage per trade"
) / 100

# Comparison mode toggle
st.sidebar.markdown("---")
comparison_mode = st.sidebar.checkbox(
    "Compare Multiple Strategies",
    value=st.session_state.comparison_mode,
    help="Run and compare multiple strategies"
)
st.session_state.comparison_mode = comparison_mode

# Run backtest button
run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)


def fetch_historical_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch historical data for backtesting"""
    try:
        # Initialize market data service
        market_service = MarketDataService()
        
        # Fetch bars
        bars = market_service.get_bars(
            symbol=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )
        
        if bars.empty:
            st.error(f"No data available for {symbol} in the specified date range")
            return pd.DataFrame()
        
        return bars
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        logger.error(f"Error fetching historical data: {e}", exc_info=True)
        return pd.DataFrame()


def plot_equity_curve(result: BacktestResult):
    """Plot equity curve"""
    fig = go.Figure()
    
    # Equity curve
    fig.add_trace(go.Scatter(
        y=result.equity_curve,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    # Initial capital line
    fig.add_hline(
        y=result.initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Capital"
    )
    
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Trade Number",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_drawdown(result: BacktestResult):
    """Plot drawdown chart"""
    equity_array = np.array(result.equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = ((equity_array - running_max) / running_max) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=drawdown,
        mode='lines',
        name='Drawdown',
        line=dict(color='#d62728', width=2),
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.1)'
    ))
    
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Trade Number",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=300
    )
    
    return fig


def plot_trade_distribution(result: BacktestResult):
    """Plot trade P&L distribution"""
    if not result.trades:
        return None
    
    pnls = [trade.pnl for trade in result.trades]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=pnls,
        nbinsx=30,
        name='Trade P&L',
        marker_color='#2ca02c'
    ))
    
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Break Even"
    )
    
    fig.update_layout(
        title="Trade P&L Distribution",
        xaxis_title="P&L ($)",
        yaxis_title="Number of Trades",
        height=300
    )
    
    return fig


# Main content
if run_backtest:
    if not symbol:
        st.error("Please enter a stock symbol")
    elif start_date >= end_date:
        st.error("Start date must be before end date")
    else:
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Fetch data
            status_text.text("📥 Fetching historical data...")
            progress_bar.progress(20)
            
            data = fetch_historical_data(symbol, start_date, end_date)
            
            if data.empty:
                st.stop()
            
            progress_bar.progress(40)
            
            # Run backtest
            if comparison_mode:
                status_text.text("🔄 Running backtests for all strategies...")
                
                # Create strategies
                strategies = []
                for strategy_name, strategy_config in STRATEGIES.items():
                    strategy = Strategy(
                        name=strategy_name,
                        signal_func=strategy_config['func'],
                        position_size=position_size,
                        stop_loss_pct=stop_loss_pct if stop_loss_pct > 0 else None,
                        take_profit_pct=take_profit_pct if take_profit_pct > 0 else None,
                        commission_pct=commission_pct
                    )
                    strategies.append(strategy)
                
                # Run comparison
                engine = BacktestEngine(initial_capital=initial_capital)
                comparison_df = engine.compare_strategies(
                    strategies=strategies,
                    data=data,
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.min.time())
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Backtest complete!")
                
                # Store results
                st.session_state.comparison_results = comparison_df
                
            else:
                status_text.text(f"🔄 Running backtest for {selected_strategy}...")
                
                # Create strategy
                strategy = Strategy(
                    name=selected_strategy,
                    signal_func=STRATEGIES[selected_strategy]['func'],
                    position_size=position_size,
                    stop_loss_pct=stop_loss_pct if stop_loss_pct > 0 else None,
                    take_profit_pct=take_profit_pct if take_profit_pct > 0 else None,
                    commission_pct=commission_pct
                )
                
                # Run backtest
                engine = BacktestEngine(initial_capital=initial_capital)
                result = engine.run_backtest(
                    strategy=strategy,
                    data=data,
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.min.time())
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Backtest complete!")
                
                # Store result
                st.session_state.backtest_results.append(result)
                st.session_state.current_result = result
        
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            logger.error(f"Backtest error: {e}", exc_info=True)
        
        finally:
            progress_bar.empty()
            status_text.empty()


# Display results
if comparison_mode and 'comparison_results' in st.session_state:
    st.header("📊 Strategy Comparison")
    
    comparison_df = st.session_state.comparison_results
    
    # Format the dataframe
    formatted_df = comparison_df.copy()
    formatted_df['Total Return'] = formatted_df['Total Return'].apply(lambda x: f"${x:,.2f}")
    formatted_df['Return %'] = formatted_df['Return %'].apply(lambda x: f"{x:.2f}%")
    formatted_df['Sharpe Ratio'] = formatted_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
    formatted_df['Max Drawdown'] = formatted_df['Max Drawdown'].apply(lambda x: f"${x:,.2f}")
    formatted_df['Max Drawdown %'] = formatted_df['Max Drawdown %'].apply(lambda x: f"{x:.2f}%")
    formatted_df['Win Rate'] = formatted_df['Win Rate'].apply(lambda x: f"{x:.1%}")
    formatted_df['Profit Factor'] = formatted_df['Profit Factor'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(formatted_df, use_container_width=True, hide_index=True)
    
    # Comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Return comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comparison_df['Strategy'],
            y=comparison_df['Return %'],
            marker_color=['#2ca02c' if x > 0 else '#d62728' for x in comparison_df['Return %']]
        ))
        fig.update_layout(
            title="Return Comparison",
            xaxis_title="Strategy",
            yaxis_title="Return (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sharpe ratio comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comparison_df['Strategy'],
            y=comparison_df['Sharpe Ratio'],
            marker_color='#1f77b4'
        ))
        fig.update_layout(
            title="Sharpe Ratio Comparison",
            xaxis_title="Strategy",
            yaxis_title="Sharpe Ratio",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

elif 'current_result' in st.session_state:
    result = st.session_state.current_result
    
    # Performance metrics
    st.header("📈 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"${result.total_return:,.2f}",
            f"{result.total_return_pct:.2f}%"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{result.sharpe_ratio:.2f}"
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            f"${result.max_drawdown:,.2f}",
            f"-{result.max_drawdown_pct:.2f}%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Win Rate",
            f"{result.win_rate:.1%}"
        )
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "Total Trades",
            result.total_trades
        )
    
    with col6:
        st.metric(
            "Winning Trades",
            result.winning_trades
        )
    
    with col7:
        st.metric(
            "Losing Trades",
            result.losing_trades
        )
    
    with col8:
        st.metric(
            "Profit Factor",
            f"{result.profit_factor:.2f}"
        )
    
    # Charts
    st.header("📊 Performance Charts")
    
    # Equity curve
    st.plotly_chart(plot_equity_curve(result), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Drawdown
        st.plotly_chart(plot_drawdown(result), use_container_width=True)
    
    with col2:
        # Trade distribution
        fig = plot_trade_distribution(result)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Trade-by-trade breakdown
    st.header("📋 Trade-by-Trade Breakdown")
    
    if result.trades:
        trades_data = []
        for i, trade in enumerate(result.trades, 1):
            trades_data.append({
                'Trade #': i,
                'Entry Time': trade.entry_time.strftime('%Y-%m-%d %H:%M'),
                'Exit Time': trade.exit_time.strftime('%Y-%m-%d %H:%M'),
                'Entry Price': f"${trade.entry_price:.2f}",
                'Exit Price': f"${trade.exit_price:.2f}",
                'Quantity': trade.quantity,
                'P&L': f"${trade.pnl:.2f}",
                'P&L %': f"{trade.pnl_pct:.2f}%",
                'Commission': f"${trade.commission:.2f}"
            })
        
        trades_df = pd.DataFrame(trades_data)
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
        
        # Export trades
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trades as CSV",
            data=csv,
            file_name=f"backtest_trades_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No trades were executed during this backtest period.")
    
    # Additional statistics
    st.header("📊 Additional Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Win/Loss Statistics")
        st.write(f"**Average Win:** ${result.avg_win:,.2f}")
        st.write(f"**Average Loss:** ${result.avg_loss:,.2f}")
        st.write(f"**Largest Win:** ${result.largest_win:,.2f}")
        st.write(f"**Largest Loss:** ${result.largest_loss:,.2f}")
    
    with col2:
        st.subheader("Capital Statistics")
        st.write(f"**Initial Capital:** ${result.initial_capital:,.2f}")
        st.write(f"**Final Capital:** ${result.final_capital:,.2f}")
        st.write(f"**Peak Capital:** ${max(result.equity_curve):,.2f}")
        st.write(f"**Lowest Capital:** ${min(result.equity_curve):,.2f}")

else:
    # Show instructions
    st.info("👈 Configure your backtest parameters in the sidebar and click 'Run Backtest' to begin.")
    
    st.markdown("""
    ### How to Use
    
    1. **Select a Strategy**: Choose from pre-built strategies or create your own
    2. **Configure Parameters**: Set position size, stop-loss, take-profit, and commission
    3. **Choose Date Range**: Select the historical period to test
    4. **Set Initial Capital**: Define your starting capital
    5. **Run Backtest**: Click the button to execute the backtest
    6. **Analyze Results**: Review performance metrics, charts, and trade details
    
    ### Available Strategies
    
    - **MA Crossover**: Classic moving average crossover strategy
    - **RSI Mean Reversion**: Buy oversold, sell overbought conditions
    - **Momentum**: Follow strong price momentum
    - **Bollinger Bands**: Mean reversion using Bollinger Bands
    
    ### Tips for Better Backtesting
    
    - Use at least 1 year of historical data for reliable results
    - Test multiple strategies to find what works best
    - Consider transaction costs and slippage
    - Don't over-optimize parameters (avoid curve fitting)
    - Validate results with out-of-sample testing
    """)
