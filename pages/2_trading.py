"""
AI Trading Agent - Trading Page

This page provides stock search, AI recommendations, manual trading,
and automated trading capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any, List
import time

from services.market_data_service import MarketDataService, Quote
from services.trading_service import TradingService, Order, OrderStatus
from services.risk_service import RiskService, TradeRequest, Position as RiskPosition
from ai.inference import AIEngine, Recommendation
from utils.indicators import sma, ema, rsi, macd, bollinger_bands
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Trading - AI Trading Agent",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'AAPL'

if 'order_preview' not in st.session_state:
    st.session_state.order_preview = None

if 'last_order' not in st.session_state:
    st.session_state.last_order = None

if 'automated_trading_enabled' not in st.session_state:
    st.session_state.automated_trading_enabled = False

if 'trade_quantity' not in st.session_state:
    st.session_state.trade_quantity = 10

if 'order_type' not in st.session_state:
    st.session_state.order_type = 'market'

if 'limit_price' not in st.session_state:
    st.session_state.limit_price = 0.0


# Initialize services
@st.cache_resource
def get_services():
    """Initialize and cache services"""
    try:
        market_service = MarketDataService()
        trading_service = TradingService()
        risk_service = RiskService()
        ai_engine = AIEngine()
        ai_engine.load_models()
        
        return market_service, trading_service, risk_service, ai_engine
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        logger.error(f"Service initialization error: {e}")
        return None, None, None, None


def get_stock_data(market_service: MarketDataService, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
    """Fetch historical stock data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = market_service.get_bars(
            symbol=symbol,
            timeframe='1Day',
            start=start_date,
            end=end_date,
            use_cache=True
        )
        
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


def calculate_technical_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate technical indicators for display"""
    try:
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        indicators = {}
        
        # Moving averages
        indicators['SMA 20'] = float(sma(close, 20)[-1])
        indicators['SMA 50'] = float(sma(close, 50)[-1])
        indicators['EMA 12'] = float(ema(close, 12)[-1])
        
        # RSI
        rsi_values = rsi(close, 14)
        indicators['RSI'] = float(rsi_values[-1])
        
        # MACD
        macd_line, signal_line, histogram = macd(close)
        indicators['MACD'] = float(macd_line[-1])
        indicators['MACD Signal'] = float(signal_line[-1])
        
        # Bollinger Bands
        upper, middle, lower = bollinger_bands(close, 20, 2.0)
        indicators['BB Upper'] = float(upper[-1])
        indicators['BB Middle'] = float(middle[-1])
        indicators['BB Lower'] = float(lower[-1])
        
        return indicators
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}


def display_price_chart(data: pd.DataFrame, symbol: str):
    """Display price chart with technical indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price'
    ))
    
    # Add moving averages
    close = data['close'].values
    sma_20 = sma(close, 20)
    sma_50 = sma(close, 50)
    
    fig.add_trace(go.Scatter(
        x=data.index[-len(sma_20):],
        y=sma_20,
        name='SMA 20',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index[-len(sma_50):],
        y=sma_50,
        name='SMA 50',
        line=dict(color='blue', width=1)
    ))
    
    fig.update_layout(
        title=f'{symbol} Price Chart',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        height=400,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_recommendation(recommendation: Recommendation):
    """Display AI recommendation with styling"""
    # Action badge
    action_colors = {
        'buy': ('🟢', '#28a745', 'BUY'),
        'sell': ('🔴', '#dc3545', 'SELL'),
        'hold': ('🟡', '#ffc107', 'HOLD')
    }
    
    emoji, color, text = action_colors.get(recommendation.action, ('⚪', '#6c757d', 'UNKNOWN'))
    
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {color}20; border-left: 5px solid {color};">
        <h2 style="margin: 0; color: {color};">{emoji} {text}</h2>
        <p style="font-size: 24px; margin: 10px 0;">Confidence: {recommendation.confidence:.0%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Details
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${recommendation.current_price:.2f}")
    
    with col2:
        if recommendation.target_price:
            st.metric("Target Price", f"${recommendation.target_price:.2f}")
        else:
            st.metric("Target Price", "N/A")
    
    with col3:
        if recommendation.stop_loss:
            st.metric("Stop Loss", f"${recommendation.stop_loss:.2f}")
        else:
            st.metric("Stop Loss", "N/A")
    
    with col4:
        if recommendation.expected_return:
            st.metric("Expected Return", f"{recommendation.expected_return:.1%}")
        else:
            st.metric("Expected Return", "N/A")
    
    # Risk level
    risk_colors = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
    risk_emoji = risk_colors.get(recommendation.risk_level, '⚪')
    st.markdown(f"**Risk Level:** {risk_emoji} {recommendation.risk_level.upper()}")
    
    # Reasoning
    st.markdown("### 📊 Analysis")
    for reason in recommendation.reasoning:
        st.markdown(f"- {reason}")
    
    # Model scores
    if recommendation.model_scores:
        st.markdown("### 🤖 Model Scores")
        score_cols = st.columns(len(recommendation.model_scores))
        for idx, (model, score) in enumerate(recommendation.model_scores.items()):
            with score_cols[idx]:
                st.metric(model.upper(), f"{score:.2f}")


def display_order_preview(
    symbol: str,
    quantity: int,
    side: str,
    order_type: str,
    price: float,
    limit_price: Optional[float] = None
):
    """Display order preview before submission"""
    st.markdown("### 📋 Order Preview")
    
    # Calculate order value
    execution_price = limit_price if order_type == 'limit' and limit_price else price
    order_value = quantity * execution_price
    
    # Display order details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Symbol:** {symbol}")
        st.markdown(f"**Action:** {side.upper()}")
        st.markdown(f"**Quantity:** {quantity} shares")
    
    with col2:
        st.markdown(f"**Order Type:** {order_type.upper()}")
        st.markdown(f"**Price:** ${execution_price:.2f}")
        st.markdown(f"**Total Value:** ${order_value:.2f}")
    
    # Warning for large orders
    if order_value > 10000:
        st.warning(f"⚠️ Large order: ${order_value:,.2f}")
    
    return order_value


def display_order_confirmation(order: Order):
    """Display order confirmation"""
    status_colors = {
        OrderStatus.FILLED: '🟢',
        OrderStatus.PENDING: '🟡',
        OrderStatus.REJECTED: '🔴',
        OrderStatus.CANCELLED: '⚪'
    }
    
    emoji = status_colors.get(order.status, '⚪')
    
    st.success(f"{emoji} Order {order.status.value.upper()}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Order ID:** {order.order_id[:8]}...")
        st.markdown(f"**Symbol:** {order.symbol}")
    
    with col2:
        st.markdown(f"**Quantity:** {order.quantity}")
        st.markdown(f"**Side:** {order.side.upper()}")
    
    with col3:
        if order.filled_avg_price:
            st.markdown(f"**Filled Price:** ${order.filled_avg_price:.2f}")
            st.markdown(f"**Filled Qty:** {order.filled_qty}")
        else:
            st.markdown(f"**Status:** {order.status.value}")


# Main page
st.title("📈 Trading")
st.markdown("Search stocks, view AI recommendations, and execute trades")

# Get services
market_service, trading_service, risk_service, ai_engine = get_services()

if not all([market_service, trading_service, risk_service, ai_engine]):
    st.error("Failed to initialize services. Please check your configuration.")
    st.stop()

# Stock search section
st.markdown("## 🔍 Stock Search")

col1, col2 = st.columns([3, 1])

with col1:
    symbol_input = st.text_input(
        "Enter stock symbol",
        value=st.session_state.selected_symbol,
        placeholder="e.g., AAPL, GOOGL, MSFT",
        key="symbol_search"
    ).upper()

with col2:
    if st.button("🔍 Search", use_container_width=True):
        if symbol_input:
            st.session_state.selected_symbol = symbol_input
            st.session_state.order_preview = None
            st.session_state.last_order = None
            st.rerun()

symbol = st.session_state.selected_symbol

# Fetch data
with st.spinner(f"Loading data for {symbol}..."):
    try:
        # Get current quote
        quote = market_service.get_latest_quote(symbol)
        current_price = quote.mid_price
        
        # Get historical data
        data = get_stock_data(market_service, symbol)
        
        if data is None or len(data) < 60:
            st.error(f"Insufficient data for {symbol}. Need at least 60 days of history.")
            st.stop()
        
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")
        st.stop()

# Display current price
st.markdown("## 💰 Current Price")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Price", f"${current_price:.2f}")

with col2:
    daily_change = data['close'].iloc[-1] - data['close'].iloc[-2]
    daily_change_pct = (daily_change / data['close'].iloc[-2]) * 100
    st.metric("Daily Change", f"${daily_change:.2f}", f"{daily_change_pct:+.2f}%")

with col3:
    st.metric("Bid", f"${quote.bid_price:.2f}")

with col4:
    st.metric("Ask", f"${quote.ask_price:.2f}")

# Price chart
st.markdown("## 📊 Price Chart")
display_price_chart(data, symbol)

# Technical indicators
st.markdown("## 📈 Technical Indicators")
indicators = calculate_technical_indicators(data)

if indicators:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("SMA 20", f"${indicators.get('SMA 20', 0):.2f}")
        st.metric("SMA 50", f"${indicators.get('SMA 50', 0):.2f}")
    
    with col2:
        st.metric("EMA 12", f"${indicators.get('EMA 12', 0):.2f}")
        rsi_val = indicators.get('RSI', 50)
        st.metric("RSI", f"{rsi_val:.1f}")
    
    with col3:
        st.metric("MACD", f"{indicators.get('MACD', 0):.2f}")
        st.metric("MACD Signal", f"{indicators.get('MACD Signal', 0):.2f}")
    
    with col4:
        st.metric("BB Upper", f"${indicators.get('BB Upper', 0):.2f}")
        st.metric("BB Lower", f"${indicators.get('BB Lower', 0):.2f}")

# AI Recommendation
st.markdown("## 🤖 AI Recommendation")

with st.spinner("Generating AI recommendation..."):
    try:
        recommendation = ai_engine.get_recommendation(symbol, data)
        display_recommendation(recommendation)
    except Exception as e:
        st.error(f"Error generating recommendation: {e}")
        logger.error(f"Recommendation error: {e}")
        recommendation = None

# Trading section
st.markdown("## 💼 Manual Trading")

# Get account info
try:
    account = trading_service.get_account()
    available_cash = account['cash']
    portfolio_value = account['portfolio_value']
except Exception as e:
    st.error(f"Error fetching account info: {e}")
    available_cash = 0
    portfolio_value = 0

# Display account info
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Portfolio Value", f"${portfolio_value:,.2f}")

with col2:
    st.metric("Available Cash", f"${available_cash:,.2f}")

with col3:
    trading_mode = trading_service.get_trading_mode()
    mode_color = "🟢" if trading_mode == "paper" else "🔴"
    st.metric("Trading Mode", f"{mode_color} {trading_mode.upper()}")

# Trade form
st.markdown("### 📝 Place Order")

col1, col2, col3 = st.columns(3)

with col1:
    trade_side = st.selectbox(
        "Action",
        options=['buy', 'sell'],
        index=0 if recommendation and recommendation.action == 'buy' else 1 if recommendation and recommendation.action == 'sell' else 0
    )

with col2:
    trade_quantity = st.number_input(
        "Quantity",
        min_value=1,
        max_value=10000,
        value=st.session_state.trade_quantity,
        step=1
    )
    st.session_state.trade_quantity = trade_quantity

with col3:
    order_type = st.selectbox(
        "Order Type",
        options=['market', 'limit'],
        index=0 if st.session_state.order_type == 'market' else 1
    )
    st.session_state.order_type = order_type

# Limit price input for limit orders
limit_price = None
if order_type == 'limit':
    limit_price = st.number_input(
        "Limit Price",
        min_value=0.01,
        value=float(current_price),
        step=0.01,
        format="%.2f"
    )
    st.session_state.limit_price = limit_price

# Preview and submit buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("👁️ Preview Order", use_container_width=True):
        st.session_state.order_preview = {
            'symbol': symbol,
            'quantity': trade_quantity,
            'side': trade_side,
            'order_type': order_type,
            'price': current_price,
            'limit_price': limit_price
        }

with col2:
    submit_disabled = st.session_state.order_preview is None
    if st.button("✅ Submit Order", use_container_width=True, disabled=submit_disabled, type="primary"):
        # Validate trade with risk service
        try:
            # Get current positions
            positions = trading_service.get_positions()
            risk_positions = [
                RiskPosition(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    entry_price=pos.entry_price,
                    current_price=pos.current_price,
                    unrealized_pl=pos.unrealized_pl,
                    unrealized_pl_pct=pos.unrealized_pl_pct
                )
                for pos in positions
            ]
            
            # Create trade request
            trade_request = TradeRequest(
                symbol=symbol,
                quantity=trade_quantity,
                side=trade_side,
                price=current_price,
                order_type=order_type
            )
            
            # Validate
            validation = risk_service.validate_trade(
                trade=trade_request,
                portfolio_value=portfolio_value,
                current_positions=risk_positions,
                available_cash=available_cash
            )
            
            if not validation.is_valid:
                st.error(f"❌ Trade validation failed: {validation.get_error_message()}")
                if validation.suggested_quantity:
                    st.info(f"💡 Suggested quantity: {validation.suggested_quantity} shares")
            else:
                # Submit order
                with st.spinner("Submitting order..."):
                    order = trading_service.place_order(
                        symbol=symbol,
                        qty=trade_quantity,
                        side=trade_side,
                        order_type=order_type,
                        limit_price=limit_price
                    )
                    
                    st.session_state.last_order = order
                    st.session_state.order_preview = None
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error submitting order: {e}")
            logger.error(f"Order submission error: {e}")

# Display order preview
if st.session_state.order_preview:
    preview = st.session_state.order_preview
    display_order_preview(
        symbol=preview['symbol'],
        quantity=preview['quantity'],
        side=preview['side'],
        order_type=preview['order_type'],
        price=preview['price'],
        limit_price=preview['limit_price']
    )

# Display last order confirmation
if st.session_state.last_order:
    st.markdown("### ✅ Order Confirmation")
    display_order_confirmation(st.session_state.last_order)
    
    # Clear button
    if st.button("Clear"):
        st.session_state.last_order = None
        st.rerun()

# Automated trading section
st.markdown("## 🤖 Automated Trading")

st.warning("""
⚠️ **Risk Warning**: Automated trading will execute trades based on AI recommendations without manual approval.
This carries significant risk and should only be enabled after thorough testing in paper trading mode.
""")

col1, col2 = st.columns([1, 3])

with col1:
    automated_enabled = st.toggle(
        "Enable Automated Trading",
        value=st.session_state.automated_trading_enabled,
        help="Allow AI to execute trades automatically"
    )
    st.session_state.automated_trading_enabled = automated_enabled

with col2:
    if automated_enabled:
        st.error("🔴 AUTOMATED TRADING ACTIVE")
        st.markdown("""
        **Active Settings:**
        - AI will monitor markets continuously
        - Trades will be executed based on high-confidence signals
        - Risk limits will be enforced
        - You will receive notifications for all trades
        """)
    else:
        st.info("🟢 Automated trading is disabled. Enable to allow AI to trade automatically.")

# Recent orders
st.markdown("## 📜 Recent Orders")

try:
    recent_orders = trading_service.get_orders(status='all', limit=10)
    
    if recent_orders:
        orders_data = []
        for order in recent_orders:
            orders_data.append({
                'Time': order.submitted_at.strftime('%Y-%m-%d %H:%M:%S'),
                'Symbol': order.symbol,
                'Side': order.side.upper(),
                'Quantity': order.quantity,
                'Type': order.order_type.upper(),
                'Status': order.status.value.upper(),
                'Filled': f"{order.filled_qty}/{order.quantity}",
                'Price': f"${order.filled_avg_price:.2f}" if order.filled_avg_price else "N/A"
            })
        
        orders_df = pd.DataFrame(orders_data)
        st.dataframe(orders_df, use_container_width=True, hide_index=True)
    else:
        st.info("No recent orders")
        
except Exception as e:
    st.error(f"Error fetching orders: {e}")
    logger.error(f"Orders fetch error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>AI Trading Agent | Always verify recommendations before trading | Past performance does not guarantee future results</p>
</div>
""", unsafe_allow_html=True)
