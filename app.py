"""Main Streamlit application for Quantitative Trading System."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging

import config
from services.pricing_service import (
    calculate_option_price,
    calculate_implied_volatility,
    MODEL_CONFIGS,
    PricingResult
)
from services.api_service import fetch_option_chain, fetch_historical_data
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


def render_option_pricing_page():
    """Render the main option pricing page."""
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
                render_greeks_chart(result)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Pricing calculation error: {e}", exc_info=True)


def render_greeks_chart(result: PricingResult):
    """Render a chart showing the Greeks."""
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
    
    st.plotly_chart(fig, use_container_width=True)


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
