"""Sensitivity Analysis Page

This page provides interactive sensitivity analysis for option pricing,
allowing users to see how option prices and Greeks change as parameters vary.
"""

import streamlit as st
import numpy as np
from typing import Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from services.pricing_service import (
    calculate_option_price,
    MODEL_CONFIGS,
    PricingResult
)
from services.visualization_service import (
    create_sensitivity_chart,
    create_greeks_chart
)
from utils.formatters import format_price, format_greek

# Page configuration
st.set_page_config(
    page_title="Sensitivity Analysis - Quantitative Trading System",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
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
    </style>
""", unsafe_allow_html=True)


def render_header():
    """Render the page header."""
    st.markdown('<div class="main-header">📊 Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyze how option prices change with parameter variations</div>', unsafe_allow_html=True)


def get_base_parameters(selected_model: str, option_type: str) -> Dict:
    """Render input fields for base parameters and return the parameter dictionary.
    
    Args:
        selected_model: The selected pricing model
        option_type: 'Call' or 'Put'
        
    Returns:
        Dictionary of base parameters
    """
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
                    help=param_info.get('description', ''),
                    key=f"base_{param}"
                )
            elif param in ['t', 'ta']:
                params[param] = st.number_input(
                    param_info.get('display_name', param),
                    min_value=param_info.get('min', 0.001),
                    max_value=param_info.get('max', 100.0),
                    value=param_info.get('default', 1.0),
                    step=0.01,
                    help=param_info.get('description', ''),
                    key=f"base_{param}"
                )
            elif param in ['r', 'q', 'rf', 'v', 'v1', 'v2']:
                params[param] = st.number_input(
                    param_info.get('display_name', param),
                    min_value=param_info.get('min', 0.0),
                    max_value=param_info.get('max', 2.0),
                    value=param_info.get('default', 0.05),
                    step=0.01,
                    format="%.4f",
                    help=param_info.get('description', ''),
                    key=f"base_{param}"
                )
            elif param == 'corr':
                params[param] = st.slider(
                    param_info.get('display_name', param),
                    min_value=-1.0,
                    max_value=1.0,
                    value=param_info.get('default', 0.5),
                    step=0.01,
                    help=param_info.get('description', ''),
                    key=f"base_{param}"
                )
        
        col_idx += 1
    
    return params


def get_sensitivity_parameter_range(param_name: str, base_value: float):
    """Get the range inputs for sensitivity analysis.
    
    Args:
        param_name: Name of the parameter to vary
        base_value: Base value of the parameter
        
    Returns:
        Tuple of (min_val, max_val, step)
    """
    param_info = config.PARAMETER_INFO.get(param_name, {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Default min: 50% of base value or parameter minimum
        default_min = max(base_value * 0.5, param_info.get('min', 0.01))
        min_val = st.number_input(
            "Minimum Value",
            min_value=param_info.get('min', 0.01),
            max_value=param_info.get('max', 10000.0),
            value=float(default_min),
            step=0.01 if param_name in ['r', 'q', 'rf', 'v', 'v1', 'v2', 't', 'ta'] else 1.0,
            format="%.4f" if param_name in ['r', 'q', 'rf', 'v', 'v1', 'v2'] else "%.2f",
            key=f"sens_min_{param_name}"
        )
    
    with col2:
        # Default max: 150% of base value or parameter maximum
        default_max = min(base_value * 1.5, param_info.get('max', 10000.0))
        max_val = st.number_input(
            "Maximum Value",
            min_value=param_info.get('min', 0.01),
            max_value=param_info.get('max', 10000.0),
            value=float(default_max),
            step=0.01 if param_name in ['r', 'q', 'rf', 'v', 'v1', 'v2', 't', 'ta'] else 1.0,
            format="%.4f" if param_name in ['r', 'q', 'rf', 'v', 'v1', 'v2'] else "%.2f",
            key=f"sens_max_{param_name}"
        )
    
    with col3:
        # Default step: 5% of range
        default_step = (default_max - default_min) / 20
        step = st.number_input(
            "Step Size",
            min_value=0.001,
            max_value=(max_val - min_val) / 2,
            value=float(default_step),
            step=0.001 if param_name in ['r', 'q', 'rf', 'v', 'v1', 'v2'] else 0.1,
            format="%.4f" if param_name in ['r', 'q', 'rf', 'v', 'v1', 'v2'] else "%.2f",
            key=f"sens_step_{param_name}"
        )
    
    return (min_val, max_val, step)


def main():
    """Main function for the sensitivity analysis page."""
    render_header()
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Select Pricing Model",
            list(MODEL_CONFIGS.keys()),
            format_func=lambda x: MODEL_CONFIGS[x]['display_name'],
            key="model_select"
        )
    
    with col2:
        option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True, key="option_type")
    
    # Display model description
    st.info(f"**{MODEL_CONFIGS[selected_model]['display_name']}**: {MODEL_CONFIGS[selected_model]['description']}")
    
    st.markdown("---")
    
    # Base parameters section
    st.subheader("📝 Base Parameters")
    st.markdown("Enter the base option parameters for analysis:")
    
    base_params = get_base_parameters(selected_model, option_type)
    
    st.markdown("---")
    
    # Sensitivity parameter selection
    st.subheader("🔍 Sensitivity Analysis Configuration")
    
    # Get parameters that can be varied (exclude option_type)
    available_params = [p for p in MODEL_CONFIGS[selected_model]['parameters'] if p != 'option_type']
    
    # Create display names for parameters
    param_display_map = {p: config.PARAMETER_INFO.get(p, {}).get('display_name', p) for p in available_params}
    
    selected_param = st.selectbox(
        "Select Parameter to Vary",
        available_params,
        format_func=lambda x: param_display_map[x],
        key="param_select"
    )
    
    st.markdown(f"**Analyzing**: {param_display_map[selected_param]}")
    st.markdown("Define the range over which to vary this parameter:")
    
    # Get the base value for the selected parameter
    base_value = base_params[selected_param]
    
    # Get range inputs
    param_range = get_sensitivity_parameter_range(selected_param, base_value)
    
    # Validate range
    if param_range[0] >= param_range[1]:
        st.error("⚠️ Minimum value must be less than maximum value")
        return
    
    if param_range[2] <= 0:
        st.error("⚠️ Step size must be positive")
        return
    
    st.markdown("---")
    
    # Calculate and display results
    if st.button("📊 Generate Sensitivity Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Calculating sensitivity analysis..."):
                # Create sensitivity chart
                sensitivity_fig = create_sensitivity_chart(
                    selected_param,
                    param_range,
                    base_params,
                    selected_model
                )
                
                st.success("✓ Analysis Complete")
                
                # Display sensitivity chart
                st.subheader("📈 Price Sensitivity Chart")
                st.plotly_chart(sensitivity_fig, use_container_width=True)
                
                # Calculate base option price and Greeks
                st.subheader("📊 Greeks at Base Parameters")
                base_result = calculate_option_price(selected_model, base_params)
                
                # Display base results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Option Price",
                        format_price(base_result.value),
                        help="Theoretical fair value at base parameters"
                    )
                
                with col2:
                    st.metric(
                        "Delta (Δ)",
                        format_greek(base_result.delta),
                        help="Rate of change w.r.t. underlying price"
                    )
                
                with col3:
                    st.metric(
                        "Gamma (Γ)",
                        format_greek(base_result.gamma),
                        help="Rate of change of delta"
                    )
                
                # Display all Greeks
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Theta (Θ)",
                        format_greek(base_result.theta),
                        help="Rate of change w.r.t. time"
                    )
                
                with col2:
                    st.metric(
                        "Vega (ν)",
                        format_greek(base_result.vega),
                        help="Rate of change w.r.t. volatility"
                    )
                
                with col3:
                    st.metric(
                        "Rho (ρ)",
                        format_greek(base_result.rho),
                        help="Rate of change w.r.t. interest rate"
                    )
                
                # Create and display Greeks chart
                st.subheader("📊 Greeks Visualization")
                greeks_fig = create_greeks_chart(base_params, selected_model)
                st.plotly_chart(greeks_fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import logging
            logging.error(f"Sensitivity analysis error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
