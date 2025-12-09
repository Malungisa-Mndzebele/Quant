"""Main Streamlit application for Quantitative Trading System."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
import numpy as np
from typing import List, Dict
import copy
import io

import config
from services.pricing_service import (
    calculate_option_price,
    calculate_implied_volatility,
    MODEL_CONFIGS,
    PricingResult
)
from services.api_service import fetch_option_chain, fetch_historical_data
from services.visualization_service import create_payoff_diagram
from utils.formatters import (
    format_price,
    format_greek,
    format_percentage
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.APP_LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .greek-label {
        font-weight: 600;
        color: #555;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def render_header():
    """Render the application header."""
    st.markdown(f'<div class="main-header">{config.APP_ICON} {config.APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Option Pricing and Analysis Platform</div>', unsafe_allow_html=True)
    
    if config.USE_MOCK_DATA:
        st.info("📊 **Demo Mode**: Using mock data. TDAmeritrade API key not required.")


def render_sidebar():
    """Render the sidebar with navigation and settings."""
    st.sidebar.header("Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["Option Pricing", "Market Data", "Implied Volatility", "About"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("Settings")
    
    # Display precision settings
    st.sidebar.text(f"Price Decimals: {config.PRICE_DECIMALS}")
    st.sidebar.text(f"Greek Decimals: {config.GREEK_DECIMALS}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    st.sidebar.success("✓ All systems operational" if config.USE_MOCK_DATA else "⚠ API Key Required")
    
    return page


def initialize_strategy_state():
    """Initialize session state for strategy builder."""
    if 'strategy_positions' not in st.session_state:
        st.session_state.strategy_positions = []
    if 'saved_strategies' not in st.session_state:
        st.session_state.saved_strategies = {}
    if 'last_pricing_result' not in st.session_state:
        st.session_state.last_pricing_result = None
    if 'last_pricing_params' not in st.session_state:
        st.session_state.last_pricing_params = None
    if 'last_pricing_model' not in st.session_state:
        st.session_state.last_pricing_model = None


def export_pricing_to_csv(result: PricingResult, params: dict, model: str) -> str:
    """Export pricing results to CSV format.
    
    Args:
        result: PricingResult object with calculated values
        params: Dictionary of input parameters
        model: Name of the pricing model used
        
    Returns:
        CSV string with all calculated values and headers
    """
    # Create dataframe with results
    data = {
        'Model': [MODEL_CONFIGS[model]['display_name']],
        'Option_Type': ['Call' if params.get('option_type') == 'c' else 'Put'],
        'Option_Price': [result.value],
        'Delta': [result.delta],
        'Gamma': [result.gamma],
        'Theta': [result.theta],
        'Vega': [result.vega],
        'Rho': [result.rho]
    }
    
    # Add input parameters
    for key, value in params.items():
        if key != 'option_type':
            param_info = config.PARAMETER_INFO.get(key, {})
            display_name = param_info.get('display_name', key)
            data[f'Input_{display_name.replace(" ", "_")}'] = [value]
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def export_chart_to_png(fig: go.Figure) -> bytes:
    """Export Plotly chart to PNG format.
    
    Args:
        fig: Plotly figure object
        
    Returns:
        PNG image as bytes
    """
    # Export to PNG using Plotly's built-in export
    img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
    return img_bytes


def calculate_strategy_metrics(positions: List[Dict]) -> Dict:
    """Calculate strategy metrics including total cost, max profit, max loss, and breakeven points.
    
    Args:
        positions: List of option positions
        
    Returns:
        Dictionary with strategy metrics
    """
    if not positions:
        return {
            'total_cost': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'breakeven_points': []
        }
    
    # Calculate total cost (initial cash flow)
    total_cost = sum(pos['quantity'] * pos['premium'] for pos in positions)
    
    # Determine price range for analysis
    strikes = [pos['strike'] for pos in positions]
    min_strike = min(strikes)
    max_strike = max(strikes)
    
    # Create price range for payoff calculation
    price_range = np.linspace(min_strike * 0.5, max_strike * 1.5, 1000)
    
    # Calculate total payoff at each price point
    total_payoff = np.zeros_like(price_range)
    
    for pos in positions:
        option_type = pos['option_type']
        strike = pos['strike']
        premium = pos['premium']
        quantity = pos['quantity']
        
        # Calculate intrinsic value at expiration
        if option_type == 'c':
            intrinsic_value = np.maximum(price_range - strike, 0)
        else:
            intrinsic_value = np.maximum(strike - price_range, 0)
        
        # Calculate position payoff
        position_payoff = quantity * (intrinsic_value - premium)
        total_payoff += position_payoff
    
    # Find max profit and max loss
    max_profit = np.max(total_payoff)
    max_loss = np.min(total_payoff)
    
    # Find breakeven points (where payoff crosses zero)
    breakeven_points = []
    for i in range(len(total_payoff) - 1):
        if (total_payoff[i] <= 0 and total_payoff[i + 1] > 0) or \
           (total_payoff[i] >= 0 and total_payoff[i + 1] < 0):
            # Linear interpolation to find exact breakeven
            x1, x2 = price_range[i], price_range[i + 1]
            y1, y2 = total_payoff[i], total_payoff[i + 1]
            breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
            breakeven_points.append(breakeven)
    
    return {
        'total_cost': total_cost,
        'max_profit': max_profit if not np.isinf(max_profit) else float('inf'),
        'max_loss': max_loss if not np.isinf(max_loss) else float('-inf'),
        'breakeven_points': breakeven_points
    }


def render_strategy_builder():
    """Render the strategy builder section."""
    st.header("🎯 Strategy Builder")
    st.info("Build and analyze multi-leg option strategies by adding multiple positions.")
    
    # Position input form
    with st.expander("➕ Add New Position", expanded=len(st.session_state.strategy_positions) == 0):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            option_type = st.selectbox(
                "Option Type",
                ["Call", "Put"],
                key="new_option_type"
            )
        
        with col2:
            strike = st.number_input(
                "Strike Price",
                min_value=0.01,
                value=100.0,
                step=1.0,
                key="new_strike"
            )
        
        with col3:
            premium = st.number_input(
                "Premium",
                min_value=0.01,
                value=5.0,
                step=0.1,
                key="new_premium"
            )
        
        with col4:
            quantity = st.number_input(
                "Quantity",
                min_value=-100,
                max_value=100,
                value=1,
                step=1,
                help="Positive for long, negative for short",
                key="new_quantity"
            )
        
        if st.button("➕ Add Position", type="primary", use_container_width=True):
            if quantity != 0:
                position = {
                    'option_type': 'c' if option_type == "Call" else 'p',
                    'strike': strike,
                    'premium': premium,
                    'quantity': quantity
                }
                st.session_state.strategy_positions.append(position)
                st.success(f"✓ Added {'Long' if quantity > 0 else 'Short'} {abs(quantity)} {option_type} @ ${strike}")
                st.rerun()
            else:
                st.warning("⚠ Quantity cannot be zero")
    
    # Display current positions
    if st.session_state.strategy_positions:
        st.subheader("Current Positions")
        
        # Create dataframe for display
        positions_data = []
        for i, pos in enumerate(st.session_state.strategy_positions):
            positions_data.append({
                'Position': i + 1,
                'Type': 'Call' if pos['option_type'] == 'c' else 'Put',
                'Strike': f"${pos['strike']:.2f}",
                'Premium': f"${pos['premium']:.2f}",
                'Quantity': pos['quantity'],
                'Side': 'Long' if pos['quantity'] > 0 else 'Short',
                'Cost': f"${pos['quantity'] * pos['premium']:.2f}"
            })
        
        df = pd.DataFrame(positions_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Position management buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            position_to_delete = st.selectbox(
                "Select position to delete",
                range(1, len(st.session_state.strategy_positions) + 1),
                format_func=lambda x: f"Position {x}"
            )
        
        with col2:
            if st.button("🗑️ Delete Position", use_container_width=True):
                st.session_state.strategy_positions.pop(position_to_delete - 1)
                st.success(f"✓ Deleted position {position_to_delete}")
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.strategy_positions = []
                st.success("✓ Cleared all positions")
                st.rerun()
        
        # Calculate and display strategy metrics
        st.markdown("---")
        st.subheader("Strategy Analysis")
        
        metrics = calculate_strategy_metrics(st.session_state.strategy_positions)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Cost",
                f"${metrics['total_cost']:.2f}",
                help="Net premium paid (positive) or received (negative)"
            )
        
        with col2:
            max_profit_display = "Unlimited" if np.isinf(metrics['max_profit']) else f"${metrics['max_profit']:.2f}"
            st.metric(
                "Max Profit",
                max_profit_display,
                help="Maximum possible profit at expiration"
            )
        
        with col3:
            max_loss_display = "Unlimited" if np.isinf(abs(metrics['max_loss'])) else f"${metrics['max_loss']:.2f}"
            st.metric(
                "Max Loss",
                max_loss_display,
                help="Maximum possible loss at expiration"
            )
        
        with col4:
            breakeven_display = f"{len(metrics['breakeven_points'])} point(s)"
            st.metric(
                "Breakeven Points",
                breakeven_display,
                help="Price points where profit/loss is zero"
            )
        
        # Display breakeven points
        if metrics['breakeven_points']:
            st.markdown("**Breakeven Prices:**")
            breakeven_cols = st.columns(len(metrics['breakeven_points']))
            for i, be in enumerate(metrics['breakeven_points']):
                with breakeven_cols[i]:
                    st.info(f"${be:.2f}")
        
        # Display payoff diagram
        st.markdown("---")
        st.subheader("Payoff Diagram")
        
        try:
            fig = create_payoff_diagram(st.session_state.strategy_positions)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error creating payoff diagram: {str(e)}")
            logger.error(f"Payoff diagram error: {e}", exc_info=True)
        
        # Save/Load strategy functionality
        st.markdown("---")
        st.subheader("Save/Load Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            strategy_name = st.text_input(
                "Strategy Name",
                placeholder="e.g., Iron Condor, Bull Call Spread",
                key="strategy_name"
            )
            
            if st.button("💾 Save Strategy", use_container_width=True):
                if strategy_name:
                    st.session_state.saved_strategies[strategy_name] = copy.deepcopy(st.session_state.strategy_positions)
                    st.success(f"✓ Saved strategy '{strategy_name}'")
                else:
                    st.warning("⚠ Please enter a strategy name")
        
        with col2:
            if st.session_state.saved_strategies:
                selected_strategy = st.selectbox(
                    "Load Saved Strategy",
                    list(st.session_state.saved_strategies.keys()),
                    key="load_strategy"
                )
                
                if st.button("📂 Load Strategy", use_container_width=True):
                    st.session_state.strategy_positions = copy.deepcopy(st.session_state.saved_strategies[selected_strategy])
                    st.success(f"✓ Loaded strategy '{selected_strategy}'")
                    st.rerun()
            else:
                st.info("No saved strategies yet")
    
    else:
        st.info("👆 Add positions above to start building your strategy")


def render_option_pricing_page():
    """Render the main option pricing page."""
    initialize_strategy_state()
    
    st.header("📈 Option Pricing Calculator")
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Select Pricing Model",
            list(MODEL_CONFIGS.keys()),
            format_func=lambda x: MODEL_CONFIGS[x]['display_name']
        )
    
    with col2:
        option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
    
    # Display model description
    st.info(f"**{MODEL_CONFIGS[selected_model]['display_name']}**: {MODEL_CONFIGS[selected_model]['description']}")
    
    # Parameter inputs
    st.subheader("Input Parameters")
    params = {}
    params['option_type'] = 'c' if option_type == "Call" else 'p'
    
    # Get required parameters for selected model
    required_params = MODEL_CONFIGS[selected_model]['parameters']
    
    # Create input fields based on model requirements
    cols = st.columns(3)
    col_idx = 0
    
    for param in required_params:
        if param == 'option_type':
            continue  # Already handled
            
        param_info = config.PARAMETER_INFO.get(param, {})
        
        with cols[col_idx % 3]:
            if param in ['fs', 'x', 'f1', 'f2']:
                params[param] = st.number_input(
                    param_info.get('display_name', param),
                    min_value=param_info.get('min', 0.01),
                    max_value=param_info.get('max', 10000.0),
                    value=param_info.get('default', 100.0),
                    step=1.0,
                    help=param_info.get('description', '')
                )
            elif param in ['t', 'ta']:
                params[param] = st.number_input(
                    param_info.get('display_name', param),
                    min_value=param_info.get('min', 0.001),
                    max_value=param_info.get('max', 100.0),
                    value=param_info.get('default', 1.0),
                    step=0.01,
                    help=param_info.get('description', '')
                )
            elif param in ['r', 'q', 'rf', 'v', 'v1', 'v2']:
                params[param] = st.number_input(
                    param_info.get('display_name', param),
                    min_value=param_info.get('min', 0.0),
                    max_value=param_info.get('max', 2.0),
                    value=param_info.get('default', 0.05),
                    step=0.01,
                    format="%.4f",
                    help=param_info.get('description', '')
                )
            elif param == 'corr':
                params[param] = st.slider(
                    param_info.get('display_name', param),
                    min_value=-1.0,
                    max_value=1.0,
                    value=param_info.get('default', 0.5),
                    step=0.01,
                    help=param_info.get('description', '')
                )
        
        col_idx += 1
    
    # Calculate button
    st.markdown("---")
    if st.button("🧮 Calculate Option Price", type="primary", use_container_width=True):
        try:
            with st.spinner("Calculating..."):
                result = calculate_option_price(selected_model, params)
                
                # Store result in session state for export
                st.session_state.last_pricing_result = result
                st.session_state.last_pricing_params = params.copy()
                st.session_state.last_pricing_model = selected_model
                
                # Display results
                st.success("✓ Calculation Complete")
                
                # Main price display
                st.markdown("### Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Option Price",
                        format_price(result.value),
                        help="Theoretical fair value of the option"
                    )
                
                with col2:
                    st.metric(
                        "Delta (Δ)",
                        format_greek(result.delta),
                        help="Rate of change of option price w.r.t. underlying price"
                    )
                
                with col3:
                    st.metric(
                        "Gamma (Γ)",
                        format_greek(result.gamma),
                        help="Rate of change of delta w.r.t. underlying price"
                    )
                
                # Greeks display
                st.markdown("### Greeks")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Theta (Θ)",
                        format_greek(result.theta),
                        help="Rate of change of option price w.r.t. time"
                    )
                
                with col2:
                    st.metric(
                        "Vega (ν)",
                        format_greek(result.vega),
                        help="Rate of change of option price w.r.t. volatility"
                    )
                
                with col3:
                    st.metric(
                        "Rho (ρ)",
                        format_greek(result.rho),
                        help="Rate of change of option price w.r.t. interest rate"
                    )
                
                # Create visualization
                greeks_fig = render_greeks_chart(result)
                
                # Export section
                st.markdown("---")
                st.markdown("### 📥 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV Export
                    csv_data = export_pricing_to_csv(result, params, selected_model)
                    st.download_button(
                        label="📄 Download Results (CSV)",
                        data=csv_data,
                        file_name=f"option_pricing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="Export all calculated values and input parameters to CSV"
                    )
                
                with col2:
                    # Chart Export
                    try:
                        png_data = export_chart_to_png(greeks_fig)
                        st.download_button(
                            label="📊 Download Chart (PNG)",
                            data=png_data,
                            file_name=f"greeks_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            use_container_width=True,
                            help="Export Greeks visualization as high-resolution PNG"
                        )
                    except Exception as export_error:
                        st.warning(f"⚠ Chart export requires kaleido package: pip install kaleido")
                        logger.warning(f"Chart export error: {export_error}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Pricing calculation error: {e}", exc_info=True)
    
    # Display export buttons for previously calculated results
    elif st.session_state.last_pricing_result is not None:
        st.info("💡 Calculate an option price to see results and export options")
        
        with st.expander("📥 Export Previous Results", expanded=False):
            result = st.session_state.last_pricing_result
            params = st.session_state.last_pricing_params
            model = st.session_state.last_pricing_model
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                csv_data = export_pricing_to_csv(result, params, model)
                st.download_button(
                    label="📄 Download Results (CSV)",
                    data=csv_data,
                    file_name=f"option_pricing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Export all calculated values and input parameters to CSV"
                )
            
            with col2:
                # Recreate chart for export
                greeks_fig = render_greeks_chart(result, return_fig=True)
                try:
                    png_data = export_chart_to_png(greeks_fig)
                    st.download_button(
                        label="📊 Download Chart (PNG)",
                        data=png_data,
                        file_name=f"greeks_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True,
                        help="Export Greeks visualization as high-resolution PNG"
                    )
                except Exception as export_error:
                    st.warning(f"⚠ Chart export requires kaleido package: pip install kaleido")
                    logger.warning(f"Chart export error: {export_error}")
    
    # Strategy Builder Section
    st.markdown("---")
    render_strategy_builder()


def render_greeks_chart(result: PricingResult, return_fig: bool = False):
    """Render a chart showing the Greeks.
    
    Args:
        result: PricingResult object with Greeks values
        return_fig: If True, return the figure object instead of displaying
        
    Returns:
        Plotly figure object if return_fig is True, otherwise None
    """
    if not return_fig:
        st.markdown("### Greeks Visualization")
    
    greeks_data = {
        'Greek': ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
        'Value': [
            abs(result.delta),
            abs(result.gamma) * 10,  # Scale gamma for visibility
            abs(result.theta),
            abs(result.vega),
            abs(result.rho)
        ],
        'Actual': [result.delta, result.gamma, result.theta, result.vega, result.rho]
    }
    
    df = pd.DataFrame(greeks_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Greek'],
        y=df['Value'],
        text=[f"{v:.4f}" for v in df['Actual']],
        textposition='auto',
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ))
    
    fig.update_layout(
        title="Option Greeks (Absolute Values, Gamma scaled 10x)",
        xaxis_title="Greek",
        yaxis_title="Absolute Value",
        height=400,
        template=config.CHART_TEMPLATE
    )
    
    if return_fig:
        return fig
    else:
        st.plotly_chart(fig, use_container_width=True)
        return fig


def render_market_data_page():
    """Render the market data page."""
    st.header("📊 Market Data")
    
    # Symbol input
    symbol = st.text_input("Enter Stock Symbol", value="AAPL", max_chars=10).upper()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📈 Fetch Option Chain", type="primary", use_container_width=True):
            try:
                with st.spinner(f"Fetching option chain for {symbol}..."):
                    chain = fetch_option_chain(symbol)
                    
                    st.success(f"✓ Retrieved option chain for {symbol}")
                    
                    # Display summary
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Underlying Price", f"${chain.underlyingPrice:.2f}")
                    with col_b:
                        st.metric("Implied Volatility", f"{chain.volatility:.2%}")
                    with col_c:
                        st.metric("Interest Rate", f"{chain.interestRate:.2%}")
                    
                    # Display options table
                    st.subheader("Available Options")
                    df = chain.to_dataframe()
                    
                    # Filter and display key columns
                    display_cols = ['putCall', 'strikePrice', 'bid', 'ask', 'last', 
                                  'volatility', 'delta', 'gamma', 'theta', 'vega']
                    available_cols = [col for col in display_cols if col in df.columns]
                    
                    st.dataframe(
                        df[available_cols].head(20),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Full Chain (CSV)",
                        csv,
                        f"{symbol}_option_chain.csv",
                        "text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error fetching option chain: {str(e)}")
                logger.error(f"Option chain fetch error: {e}", exc_info=True)
    
    with col2:
        if st.button("📉 Fetch Historical Data", type="primary", use_container_width=True):
            try:
                with st.spinner(f"Fetching historical data for {symbol}..."):
                    history = fetch_historical_data(symbol)
                    
                    st.success(f"✓ Retrieved historical data for {symbol}")
                    
                    # Convert to dataframe
                    df = history.to_dataframe()
                    
                    # Display chart
                    st.subheader("Price History")
                    fig = go.Figure(data=[go.Candlestick(
                        x=df['datetime'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close']
                    )])
                    
                    fig.update_layout(
                        title=f"{symbol} Price History",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=config.CHART_HEIGHT,
                        template=config.CHART_TEMPLATE
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display statistics
                    st.subheader("Statistics")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Latest Close", f"${df['close'].iloc[-1]:.2f}")
                    with col_b:
                        st.metric("High", f"${df['high'].max():.2f}")
                    with col_c:
                        st.metric("Low", f"${df['low'].min():.2f}")
                    with col_d:
                        avg_volume = df['volume'].mean()
                        st.metric("Avg Volume", f"{avg_volume:,.0f}")
                    
            except Exception as e:
                st.error(f"❌ Error fetching historical data: {str(e)}")
                logger.error(f"Historical data fetch error: {e}", exc_info=True)


def render_implied_volatility_page():
    """Render the implied volatility calculator page."""
    st.header("🔍 Implied Volatility Calculator")
    
    st.info("Calculate the implied volatility from an observed market price.")
    
    # Model selection
    models_with_iv = {k: v for k, v in MODEL_CONFIGS.items() if v['supports_implied_vol']}
    
    selected_model = st.selectbox(
        "Select Pricing Model",
        list(models_with_iv.keys()),
        format_func=lambda x: models_with_iv[x]['display_name']
    )
    
    # Option type
    option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
    
    # Parameters
    st.subheader("Input Parameters")
    
    params = {}
    params['option_type'] = 'c' if option_type == "Call" else 'p'
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        params['fs'] = st.number_input("Underlying Price", min_value=0.01, value=100.0, step=1.0)
        params['x'] = st.number_input("Strike Price", min_value=0.01, value=100.0, step=1.0)
    
    with col2:
        params['t'] = st.number_input("Time to Expiration (years)", min_value=0.001, value=1.0, step=0.01)
        params['r'] = st.number_input("Risk-Free Rate", min_value=0.0, value=0.05, step=0.01, format="%.4f")
    
    with col3:
        if 'q' in MODEL_CONFIGS[selected_model]['parameters']:
            params['q'] = st.number_input("Dividend Yield", min_value=0.0, value=0.02, step=0.01, format="%.4f")
        elif 'rf' in MODEL_CONFIGS[selected_model]['parameters']:
            params['rf'] = st.number_input("Foreign Rate", min_value=0.0, value=0.03, step=0.01, format="%.4f")
        
        observed_price = st.number_input("Observed Market Price", min_value=0.01, value=10.0, step=0.1)
    
    # Calculate button
    if st.button("🔍 Calculate Implied Volatility", type="primary", use_container_width=True):
        try:
            with st.spinner("Calculating implied volatility..."):
                iv = calculate_implied_volatility(selected_model, params, observed_price)
                
                st.success("✓ Calculation Complete")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Implied Volatility",
                        format_percentage(iv),
                        help="The volatility implied by the market price"
                    )
                
                with col2:
                    st.metric(
                        "Annualized (%)",
                        f"{iv * 100:.2f}%"
                    )
                
                # Verify by calculating price with this IV
                verify_params = params.copy()
                verify_params['v'] = iv
                result = calculate_option_price(selected_model, verify_params)
                
                st.info(f"**Verification**: Using IV of {iv:.4f}, the calculated price is ${result.value:.4f} "
                       f"(market price: ${observed_price:.4f})")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Implied volatility calculation error: {e}", exc_info=True)


def render_about_page():
    """Render the about page."""
    st.header("ℹ️ About")
    
    st.markdown("""
    ## Quantitative Trading System
    
    This application provides advanced option pricing and analysis capabilities using industry-standard models.
    
    ### Supported Models
    
    - **Black-Scholes**: Classic European option pricing for non-dividend paying stocks
    - **Merton**: Extension of Black-Scholes for stocks with continuous dividend yields
    - **Black-76**: Options on futures and commodities
    - **Garman-Kohlhagen**: Foreign exchange (FX) options
    - **Asian-76**: Average price options on commodities
    - **Kirk's Approximation**: Spread options
    - **American Options**: Early exercise capability using Bjerksund-Stensland approximation
    
    ### Features
    
    - ✓ Real-time option pricing
    - ✓ Complete Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
    - ✓ Implied volatility calculation
    - ✓ Market data integration (with mock data support)
    - ✓ Interactive visualizations
    - ✓ Export capabilities
    
    ### Technology Stack
    
    - **Framework**: Streamlit
    - **Pricing Engine**: Custom implementation based on industry-standard formulas
    - **Data Visualization**: Plotly
    - **Data Processing**: Pandas, NumPy, SciPy
    
    ### Demo Mode
    
    Currently running in **demo mode** with mock data. To use real market data, 
    configure a TDAmeritrade API key in the `.env` file.
    
    ---
    
    **Version**: 0.5.1  
    **Author**: Davis Edwards & Daniel Rojas
    """)


def main():
    """Main application entry point."""
    render_header()
    page = render_sidebar()
    
    st.markdown("---")
    
    # Route to appropriate page
    if page == "Option Pricing":
        render_option_pricing_page()
    elif page == "Market Data":
        render_market_data_page()
    elif page == "Implied Volatility":
        render_implied_volatility_page()
    elif page == "About":
        render_about_page()


if __name__ == "__main__":
    main()
