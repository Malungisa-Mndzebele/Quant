"""
AI Trading Agent - Analytics Page

This page displays AI model performance metrics, prediction accuracy,
feature importance, model confidence distribution, and sentiment analysis trends.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any, List
import os

from ai.inference import AIEngine
from services.sentiment_service import SentimentService
from services.market_data_service import MarketDataService
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Analytics - AI Trading Agent",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'ensemble'

if 'analytics_symbol' not in st.session_state:
    st.session_state.analytics_symbol = 'AAPL'

if 'analytics_timeframe' not in st.session_state:
    st.session_state.analytics_timeframe = '30D'


# Initialize services
@st.cache_resource
def get_services():
    """Initialize and cache services"""
    try:
        ai_engine = AIEngine()
        ai_engine.load_models()
        sentiment_service = SentimentService()
        market_service = MarketDataService()
        
        return ai_engine, sentiment_service, market_service
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        logger.error(f"Service initialization error: {e}")
        return None, None, None


def load_model_metrics() -> Dict[str, Any]:
    """Load model performance metrics from training history"""
    metrics = {
        'lstm': {},
        'rf': {},
        'ensemble': {}
    }
    
    # Try to load LSTM metrics
    lstm_metrics_path = 'data/models/lstm_model_metrics.json'
    if os.path.exists(lstm_metrics_path):
        import json
        with open(lstm_metrics_path, 'r') as f:
            metrics['lstm'] = json.load(f)
    
    # Try to load RF metrics
    rf_metrics_path = 'data/models/rf_model_metrics.json'
    if os.path.exists(rf_metrics_path):
        import json
        with open(rf_metrics_path, 'r') as f:
            metrics['rf'] = json.load(f)
    
    return metrics


def create_accuracy_over_time_chart(predictions_df: pd.DataFrame) -> go.Figure:
    """Create prediction accuracy over time chart"""
    if predictions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No prediction data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Calculate rolling accuracy
    predictions_df['correct'] = predictions_df['predicted'] == predictions_df['actual']
    predictions_df['rolling_accuracy'] = predictions_df['correct'].rolling(window=20, min_periods=1).mean() * 100
    
    fig = go.Figure()
    
    # Add rolling accuracy line
    fig.add_trace(go.Scatter(
        x=predictions_df['date'],
        y=predictions_df['rolling_accuracy'],
        mode='lines',
        name='Rolling Accuracy (20-day)',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    # Add overall accuracy line
    overall_accuracy = predictions_df['correct'].mean() * 100
    fig.add_hline(
        y=overall_accuracy,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Overall: {overall_accuracy:.1f}%",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Prediction Accuracy Over Time",
        xaxis_title="Date",
        yaxis_title="Accuracy (%)",
        hovermode='x unified',
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig


def create_feature_importance_chart(ai_engine: AIEngine) -> go.Figure:
    """Create feature importance chart from Random Forest model"""
    try:
        if ai_engine.rf_model is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Random Forest model not loaded",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Get feature importance
        importance_df = ai_engine.rf_model.get_feature_importance(top_n=15)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker=dict(
                color=importance_df['importance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=importance_df['importance'].round(4),
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Top 15 Feature Importances (Random Forest)",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500,
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating feature importance chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig


def create_confidence_distribution_chart(predictions_df: pd.DataFrame) -> go.Figure:
    """Create model confidence distribution chart"""
    if predictions_df.empty or 'confidence' not in predictions_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No confidence data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    fig = go.Figure()
    
    # Create histogram
    fig.add_trace(go.Histogram(
        x=predictions_df['confidence'],
        nbinsx=20,
        name='Confidence Distribution',
        marker=dict(
            color='rgba(31, 119, 180, 0.7)',
            line=dict(color='rgba(31, 119, 180, 1)', width=1)
        )
    ))
    
    # Add mean line
    mean_confidence = predictions_df['confidence'].mean()
    fig.add_vline(
        x=mean_confidence,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_confidence:.2f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Model Confidence Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Frequency",
        height=400,
        showlegend=False
    )
    
    return fig


def create_confusion_matrix_chart(predictions_df: pd.DataFrame) -> go.Figure:
    """Create confusion matrix heatmap"""
    if predictions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No prediction data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    
    classes = ['buy', 'hold', 'sell']
    cm = confusion_matrix(
        predictions_df['actual'],
        predictions_df['predicted'],
        labels=classes
    )
    
    # Normalize by row (actual class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=classes,
        y=classes,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        colorbar=dict(title="Proportion")
    ))
    
    fig.update_layout(
        title="Confusion Matrix (Normalized by Actual Class)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def create_sentiment_trend_chart(sentiment_data: List[Dict[str, Any]]) -> go.Figure:
    """Create sentiment analysis trend chart"""
    if not sentiment_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No sentiment data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    df = pd.DataFrame(sentiment_data)
    
    fig = go.Figure()
    
    # Add sentiment score line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sentiment_score'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=6)
    ))
    
    # Add neutral line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Neutral",
        annotation_position="right"
    )
    
    # Color background regions
    fig.add_hrect(
        y0=0, y1=1,
        fillcolor="green", opacity=0.1,
        layer="below", line_width=0,
    )
    fig.add_hrect(
        y0=-1, y1=0,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
    )
    
    fig.update_layout(
        title="News Sentiment Trend",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        hovermode='x unified',
        height=400,
        yaxis=dict(range=[-1, 1])
    )
    
    return fig


def create_model_comparison_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Create model comparison chart"""
    models = []
    accuracies = []
    
    if 'lstm' in metrics and 'direction_accuracy' in metrics['lstm']:
        models.append('LSTM')
        accuracies.append(metrics['lstm']['direction_accuracy'] * 100)
    
    if 'rf' in metrics and 'train_accuracy' in metrics['rf']:
        models.append('Random Forest')
        accuracies.append(metrics['rf']['train_accuracy'] * 100)
    
    if not models:
        fig = go.Figure()
        fig.add_annotation(
            text="No model metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=models,
        y=accuracies,
        marker=dict(
            color=accuracies,
            colorscale='RdYlGn',
            cmin=0,
            cmax=100,
            showscale=True,
            colorbar=dict(title="Accuracy (%)")
        ),
        text=[f"{acc:.1f}%" for acc in accuracies],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Accuracy (%)",
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig


def generate_sample_predictions(days: int = 30) -> pd.DataFrame:
    """Generate sample prediction data for demonstration"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Simulate predictions with some accuracy
    np.random.seed(42)
    predictions = np.random.choice(['buy', 'hold', 'sell'], size=days, p=[0.3, 0.4, 0.3])
    
    # Actual outcomes (80% match predictions for demo)
    actual = predictions.copy()
    mismatch_indices = np.random.choice(days, size=int(days * 0.2), replace=False)
    for idx in mismatch_indices:
        choices = ['buy', 'hold', 'sell']
        choices.remove(predictions[idx])
        actual[idx] = np.random.choice(choices)
    
    # Confidence scores
    confidence = np.random.beta(8, 2, size=days)  # Skewed towards higher confidence
    
    df = pd.DataFrame({
        'date': dates,
        'predicted': predictions,
        'actual': actual,
        'confidence': confidence
    })
    
    return df


def generate_sample_sentiment_data(days: int = 30) -> List[Dict[str, Any]]:
    """Generate sample sentiment data for demonstration"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(42)
    sentiment_scores = np.random.normal(0.1, 0.3, size=days)  # Slightly positive bias
    sentiment_scores = np.clip(sentiment_scores, -1, 1)
    
    data = []
    for date, score in zip(dates, sentiment_scores):
        data.append({
            'date': date,
            'sentiment_score': score
        })
    
    return data


# Main page layout
st.title("📊 AI Model Analytics")
st.markdown("Monitor AI model performance, prediction accuracy, and sentiment analysis trends")

# Initialize services
ai_engine, sentiment_service, market_service = get_services()

if ai_engine is None:
    st.error("Failed to initialize AI engine. Please check your configuration.")
    st.stop()

# Sidebar controls
st.sidebar.header("Analytics Controls")

# Model selection
model_options = ['Ensemble', 'LSTM', 'Random Forest']
selected_model = st.sidebar.selectbox(
    "Select Model",
    model_options,
    index=0
)
st.session_state.selected_model = selected_model.lower()

# Symbol selection for analysis
symbol = st.sidebar.text_input(
    "Stock Symbol",
    value=st.session_state.analytics_symbol,
    max_chars=10
).upper()
st.session_state.analytics_symbol = symbol

# Timeframe selection
timeframe_options = ['7D', '30D', '90D', '1Y']
timeframe = st.sidebar.selectbox(
    "Timeframe",
    timeframe_options,
    index=1
)
st.session_state.analytics_timeframe = timeframe

# Refresh button
if st.sidebar.button("🔄 Refresh Analytics", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

# Load model metrics
metrics = load_model_metrics()

# Model Information Section
st.header("🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    if ai_engine is not None:
        st.metric(
            "Models Loaded",
            len(ai_engine.available_models),
            delta=None
        )
    else:
        st.metric("Models Loaded", "0")

with col2:
    if ai_engine is not None and ai_engine.lstm_model is not None:
        st.metric(
            "LSTM Sequence Length",
            ai_engine.lstm_model.sequence_length,
            delta=None
        )
    else:
        st.metric("LSTM Status", "Not Loaded")

with col3:
    if ai_engine is not None and ai_engine.rf_model is not None:
        st.metric(
            "RF Estimators",
            ai_engine.rf_model.n_estimators,
            delta=None
        )
    else:
        st.metric("RF Status", "Not Loaded")

# Model Performance Metrics Section
st.header("📈 Model Performance Metrics")

col1, col2 = st.columns(2)

with col1:
    # Model comparison chart
    st.plotly_chart(
        create_model_comparison_chart(metrics),
        use_container_width=True
    )

with col2:
    # Display detailed metrics
    st.subheader("Detailed Metrics")
    
    if st.session_state.selected_model == 'lstm' and 'lstm' in metrics:
        lstm_metrics = metrics['lstm']
        if lstm_metrics:
            st.write("**LSTM Model:**")
            if 'mse' in lstm_metrics:
                st.write(f"- MSE: {lstm_metrics['mse']:.6f}")
            if 'mae' in lstm_metrics:
                st.write(f"- MAE: {lstm_metrics['mae']:.6f}")
            if 'direction_accuracy' in lstm_metrics:
                st.write(f"- Direction Accuracy: {lstm_metrics['direction_accuracy']*100:.2f}%")
        else:
            st.info("No LSTM metrics available. Train the model to see metrics.")
    
    elif st.session_state.selected_model == 'random forest' and 'rf' in metrics:
        rf_metrics = metrics['rf']
        if rf_metrics:
            st.write("**Random Forest Model:**")
            if 'train_accuracy' in rf_metrics:
                st.write(f"- Training Accuracy: {rf_metrics['train_accuracy']*100:.2f}%")
            if 'cv_mean' in rf_metrics:
                st.write(f"- CV Mean: {rf_metrics['cv_mean']*100:.2f}%")
            if 'cv_std' in rf_metrics:
                st.write(f"- CV Std: {rf_metrics['cv_std']*100:.2f}%")
            if 'oob_score' in rf_metrics and rf_metrics['oob_score']:
                st.write(f"- OOB Score: {rf_metrics['oob_score']*100:.2f}%")
        else:
            st.info("No Random Forest metrics available. Train the model to see metrics.")
    
    else:
        st.write("**Ensemble Model:**")
        if ai_engine is not None:
            st.write(f"- LSTM Weight: {ai_engine.ensemble_weights.get('lstm', 0.4):.1%}")
            st.write(f"- RF Weight: {ai_engine.ensemble_weights.get('rf', 0.6):.1%}")
            st.write(f"- Min Confidence: {ai_engine.min_confidence_threshold:.1%}")
        else:
            st.info("AI engine not initialized")

# Prediction Accuracy Over Time Section
st.header("🎯 Prediction Accuracy Over Time")

# Generate sample data (in production, this would come from a database)
predictions_df = generate_sample_predictions(days=60)

col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(
        create_accuracy_over_time_chart(predictions_df),
        use_container_width=True
    )

with col2:
    st.subheader("Accuracy Statistics")
    
    overall_accuracy = (predictions_df['predicted'] == predictions_df['actual']).mean() * 100
    st.metric("Overall Accuracy", f"{overall_accuracy:.1f}%")
    
    # Per-class accuracy
    for action in ['buy', 'hold', 'sell']:
        mask = predictions_df['actual'] == action
        if mask.sum() > 0:
            class_accuracy = (
                predictions_df[mask]['predicted'] == predictions_df[mask]['actual']
            ).mean() * 100
            st.metric(f"{action.capitalize()} Accuracy", f"{class_accuracy:.1f}%")

# Confusion Matrix Section
st.header("🔀 Confusion Matrix")

st.plotly_chart(
    create_confusion_matrix_chart(predictions_df),
    use_container_width=True
)

# Feature Importance Section
st.header("⭐ Feature Importance")

if ai_engine is not None and ai_engine.rf_model is not None:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(
            create_feature_importance_chart(ai_engine),
            use_container_width=True
        )
    
    with col2:
        st.subheader("Top Features")
        try:
            importance_df = ai_engine.rf_model.get_feature_importance(top_n=5)
            for idx, row in importance_df.iterrows():
                st.write(f"**{idx+1}. {row['feature']}**")
                st.progress(float(row['importance']))
        except Exception as e:
            st.error(f"Error displaying feature importance: {e}")
else:
    st.info("Random Forest model not loaded. Feature importance is not available.")

# Model Confidence Distribution Section
st.header("📊 Model Confidence Distribution")

col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(
        create_confidence_distribution_chart(predictions_df),
        use_container_width=True
    )

with col2:
    st.subheader("Confidence Statistics")
    
    mean_conf = predictions_df['confidence'].mean()
    median_conf = predictions_df['confidence'].median()
    std_conf = predictions_df['confidence'].std()
    
    st.metric("Mean Confidence", f"{mean_conf:.2f}")
    st.metric("Median Confidence", f"{median_conf:.2f}")
    st.metric("Std Deviation", f"{std_conf:.2f}")
    
    # High confidence predictions
    high_conf_pct = (predictions_df['confidence'] > 0.8).mean() * 100
    st.metric("High Confidence (>0.8)", f"{high_conf_pct:.1f}%")

# Recent Predictions Explanation Section
st.header("💡 Recent Prediction Explanations")

if ai_engine is not None and market_service is not None:
    try:
        # Get recent data for the selected symbol
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        data = market_service.get_bars(
            symbol=symbol,
            timeframe='1Day',
            start=start_date,
            end=end_date,
            use_cache=True
        )
        
        if data is not None and len(data) >= 60:
            # Get explanation for the symbol
            explanation = ai_engine.explain_prediction(symbol, data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Prediction for {symbol}")
                st.write(f"**Action:** {explanation['action'].upper()}")
                st.write(f"**Confidence:** {explanation['confidence']:.2%}")
                st.write(f"**Timestamp:** {explanation['timestamp']}")
                
                st.subheader("Model Contributions")
                for model, contrib in explanation['model_contributions'].items():
                    st.write(f"**{model.upper()}:**")
                    st.write(f"  - Score: {contrib['score']:.3f}")
                    st.write(f"  - Weight: {contrib['weight']:.3f}")
                    st.write(f"  - Contribution: {contrib['contribution']:.3f}")
            
            with col2:
                st.subheader("Technical Analysis")
                tech_analysis = explanation['technical_analysis']
                
                if 'rsi' in tech_analysis:
                    rsi_data = tech_analysis['rsi']
                    st.write(f"**RSI:** {rsi_data['value']:.2f} - {rsi_data['interpretation']}")
                
                if 'macd' in tech_analysis:
                    macd_data = tech_analysis['macd']
                    st.write(f"**MACD:** {macd_data['interpretation']}")
                
                if 'moving_averages' in tech_analysis:
                    ma_data = tech_analysis['moving_averages']
                    st.write(f"**SMA 20:** ${ma_data['sma_20']:.2f}")
                    st.write(f"**SMA 50:** ${ma_data['sma_50']:.2f}")
            
            # Reasoning
            st.subheader("Reasoning")
            for reason in explanation['reasoning']:
                st.write(f"- {reason}")
        else:
            st.warning(f"Insufficient data for {symbol}. Need at least 60 days of historical data.")

    except Exception as e:
        st.error(f"Error generating prediction explanation: {e}")
        logger.error(f"Prediction explanation error: {e}")
else:
    st.info("AI engine or market service not initialized. Cannot generate predictions.")

# Sentiment Analysis Trends Section
st.header("📰 Sentiment Analysis Trends")

# Generate sample sentiment data (in production, this would come from sentiment service)
sentiment_data = generate_sample_sentiment_data(days=30)

col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(
        create_sentiment_trend_chart(sentiment_data),
        use_container_width=True
    )

with col2:
    st.subheader("Sentiment Statistics")
    
    sentiment_df = pd.DataFrame(sentiment_data)
    mean_sentiment = sentiment_df['sentiment_score'].mean()
    
    st.metric("Average Sentiment", f"{mean_sentiment:.3f}")
    
    # Sentiment classification
    positive_pct = (sentiment_df['sentiment_score'] > 0.2).mean() * 100
    negative_pct = (sentiment_df['sentiment_score'] < -0.2).mean() * 100
    neutral_pct = 100 - positive_pct - negative_pct
    
    st.write("**Sentiment Distribution:**")
    st.write(f"- Positive: {positive_pct:.1f}%")
    st.write(f"- Neutral: {neutral_pct:.1f}%")
    st.write(f"- Negative: {negative_pct:.1f}%")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>AI Model Analytics Dashboard | Last Updated: {}</p>
        <p>Note: Some metrics are based on simulated data for demonstration purposes.</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    unsafe_allow_html=True
)
