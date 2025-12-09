"""Visualization service module for creating interactive charts.

This module provides functions for creating various types of charts for option analysis,
including sensitivity analysis, Greeks visualization, payoff diagrams, and historical data charts.
All charts are created using Plotly for interactivity.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Tuple
from services.pricing_service import calculate_option_price, MODEL_CONFIGS


def create_sensitivity_chart(param_name: str, param_range: Tuple[float, float, float],
                            base_params: Dict, model: str) -> go.Figure:
    """Create interactive sensitivity analysis chart.
    
    Shows how option price changes as one parameter varies while holding others constant.
    Uses vectorized NumPy calculations for improved performance.
    
    Args:
        param_name: Parameter to vary (e.g., 'fs', 'v', 't', 'x', 'r')
        param_range: (min, max, step) for parameter variation
        base_params: Base parameters for calculation
        model: Pricing model to use
        
    Returns:
        Plotly figure object with sensitivity chart
    """
    min_val, max_val, step = param_range
    
    # Generate parameter values using NumPy for efficiency
    param_values = np.arange(min_val, max_val + step, step)
    
    # Get the pricing function and model config
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model}")
    
    config = MODEL_CONFIGS[model]
    pricing_function = config['function']
    
    # Prepare base arguments as arrays
    # Extract all parameters except the one we're varying
    param_names = config['parameters']
    
    # Build argument arrays for vectorized calculation
    args_arrays = []
    for param in param_names:
        if param == param_name:
            # This is the parameter we're varying - use the array
            args_arrays.append(param_values)
        else:
            # This parameter stays constant - broadcast to array
            args_arrays.append(np.full_like(param_values, base_params[param]))
    
    try:
        # Call pricing function with vectorized arrays
        # The optlib functions should handle numpy arrays
        results = pricing_function(*args_arrays)
        
        # Extract just the values (first element of tuple)
        if isinstance(results, tuple):
            prices = results[0]
        else:
            prices = results
            
        # Convert to list for Plotly
        prices = np.array(prices)
        
    except Exception:
        # If vectorization fails, fall back to loop-based calculation
        prices = []
        for val in param_values:
            params = base_params.copy()
            params[param_name] = val
            
            try:
                result = calculate_option_price(model, params)
                prices.append(result.value)
            except Exception:
                prices.append(None)
        prices = np.array(prices)
    
    # Create the chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=param_values,
        y=prices,
        mode='lines+markers',
        name='Option Price',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Format parameter name for display
    param_display_names = {
        'fs': 'Underlying Price',
        'x': 'Strike Price',
        't': 'Time to Expiration (years)',
        'r': 'Risk-Free Rate',
        'v': 'Volatility',
        'q': 'Dividend Yield',
        'rf': 'Foreign Risk-Free Rate',
        'corr': 'Correlation'
    }
    
    param_display = param_display_names.get(param_name, param_name)
    
    fig.update_layout(
        title=f'Sensitivity Analysis: {param_display}',
        xaxis_title=param_display,
        yaxis_title='Option Price',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def create_greeks_chart(params: Dict, model: str) -> go.Figure:
    """Create chart showing all Greeks for given parameters.
    
    Args:
        params: Pricing parameters
        model: Pricing model to use
        
    Returns:
        Plotly figure with subplots for each Greek
    """
    # Calculate option price and Greeks
    result = calculate_option_price(model, params)
    
    # Create subplots for each Greek
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'Value'),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
    # Add indicators for each Greek
    greeks_data = [
        ('Delta', result.delta, 1, 1),
        ('Gamma', result.gamma, 1, 2),
        ('Theta', result.theta, 1, 3),
        ('Vega', result.vega, 2, 1),
        ('Rho', result.rho, 2, 2),
        ('Value', result.value, 2, 3)
    ]
    
    for name, value, row, col in greeks_data:
        fig.add_trace(
            go.Indicator(
                mode='number',
                value=value,
                title={'text': name},
                number={'valueformat': '.4f'}
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Option Greeks',
        height=400,
        template='plotly_white'
    )
    
    return fig



def create_payoff_diagram(positions: List[Dict]) -> go.Figure:
    """Create payoff diagram for option strategy.
    
    Shows the profit/loss at expiration for a combination of option positions.
    
    Args:
        positions: List of option positions, each containing:
            - option_type: 'c' or 'p'
            - strike: Strike price
            - premium: Option premium paid/received
            - quantity: Number of contracts (positive for long, negative for short)
            
    Returns:
        Plotly figure showing payoff at expiration
    """
    if not positions:
        # Return empty figure if no positions
        fig = go.Figure()
        fig.update_layout(
            title='Payoff Diagram',
            xaxis_title='Underlying Price at Expiration',
            yaxis_title='Profit/Loss',
            template='plotly_white',
            height=500
        )
        return fig
    
    # Determine price range for x-axis
    strikes = [pos['strike'] for pos in positions]
    min_strike = min(strikes)
    max_strike = max(strikes)
    
    # Create price range (50% below min strike to 50% above max strike)
    price_range = np.linspace(min_strike * 0.5, max_strike * 1.5, 200)
    
    # Calculate payoff for each position
    total_payoff = np.zeros_like(price_range)
    
    fig = go.Figure()
    
    for i, pos in enumerate(positions):
        option_type = pos['option_type']
        strike = pos['strike']
        premium = pos['premium']
        quantity = pos['quantity']
        
        # Calculate payoff at expiration for this position
        if option_type == 'c':
            # Call option payoff
            intrinsic_value = np.maximum(price_range - strike, 0)
        else:
            # Put option payoff
            intrinsic_value = np.maximum(strike - price_range, 0)
        
        # Account for premium and quantity
        # Long positions (quantity > 0): pay premium, receive intrinsic value
        # Short positions (quantity < 0): receive premium, pay intrinsic value
        position_payoff = quantity * (intrinsic_value - premium)
        
        total_payoff += position_payoff
        
        # Add trace for individual position
        position_label = f"{'Long' if quantity > 0 else 'Short'} {abs(quantity)} {option_type.upper()} @ {strike}"
        fig.add_trace(go.Scatter(
            x=price_range,
            y=position_payoff,
            mode='lines',
            name=position_label,
            line=dict(dash='dash', width=1),
            opacity=0.5
        ))
    
    # Add trace for total payoff
    fig.add_trace(go.Scatter(
        x=price_range,
        y=total_payoff,
        mode='lines',
        name='Total Payoff',
        line=dict(color='black', width=3)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5)
    
    fig.update_layout(
        title='Strategy Payoff Diagram',
        xaxis_title='Underlying Price at Expiration',
        yaxis_title='Profit/Loss',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig


def create_historical_chart(price_history) -> go.Figure:
    """Create candlestick chart from historical data.
    
    Args:
        price_history: Pricehistory object from optlib.instruments
        
    Returns:
        Plotly candlestick chart
    """
    # Extract data from price history
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    for candle in price_history.candles:
        dates.append(candle['datetime'])
        opens.append(candle['open'])
        highs.append(candle['high'])
        lows.append(candle['low'])
        closes.append(candle['close'])
        volumes.append(candle['volume'])
    
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{price_history.symbol} Price', 'Volume')
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['green' if closes[i] >= opens[i] else 'red' for i in range(len(closes))]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=volumes,
            name='Volume',
            marker_color=colors,
            opacity=0.5
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{price_history.symbol} Historical Data',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    
    return fig
