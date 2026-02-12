# Alert Service Implementation

## Overview

The Alert Service provides comprehensive notification and alerting capabilities for the AI Trading Agent. It supports multiple alert types, priority levels, and delivery channels including in-app notifications and email.

## Features

### Alert Types

- **Price Movement**: Alerts for significant price changes
- **AI Signal**: High-confidence trading signals from AI models
- **Risk Limit**: Warnings when approaching or exceeding risk limits
- **Order Execution**: Notifications for order status changes
- **Stop Loss**: Alerts when stop-loss is triggered
- **Take Profit**: Alerts when take-profit is triggered
- **Daily Loss Limit**: Critical alerts when daily loss limit is reached
- **Position Opened/Closed**: Notifications for position changes
- **Market Status**: Market open/close notifications

### Priority Levels

- **LOW**: Informational alerts
- **MEDIUM**: Standard notifications
- **HIGH**: Important warnings
- **CRITICAL**: Urgent alerts requiring immediate attention

### Notification Channels

- **In-App**: Stored alerts viewable in the application
- **Email**: Email notifications (requires SMTP configuration)
- **Push**: Reserved for future implementation

## Usage

### Basic Setup

```python
from services.alert_service import AlertService, AlertConfig, NotificationChannel

# Create configuration
config = AlertConfig(
    enabled_channels=[NotificationChannel.IN_APP],
    price_movement_threshold=0.05,  # 5% price change
    high_confidence_threshold=0.8,   # 80% confidence for AI signals
)

# Initialize service
alert_service = AlertService(config=config)
```

### Email Configuration

```python
from services.alert_service import AlertConfig, NotificationChannel

config = AlertConfig(
    enabled_channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
    email_address="your-email@example.com",
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    smtp_username="your-email@example.com",
    smtp_password="your-app-password"
)

alert_service = AlertService(config=config)
```

### Sending Alerts

#### Price Movement Alert

```python
alert = alert_service.alert_price_movement(
    symbol="AAPL",
    old_price=150.0,
    new_price=157.5  # 5% increase
)
```

#### AI Signal Alert

```python
alert = alert_service.alert_ai_signal(
    symbol="AAPL",
    action="buy",
    confidence=0.85,
    reasoning="Strong bullish momentum with positive sentiment",
    target_price=165.0
)
```

#### Risk Limit Alert

```python
alert = alert_service.alert_risk_limit(
    limit_type="daily_loss",
    current_value=900.0,
    limit_value=1000.0,
    message="Approaching daily loss limit"
)
```

#### Order Execution Alert

```python
alert = alert_service.alert_order_execution(
    symbol="AAPL",
    action="buy",
    quantity=100,
    price=150.0,
    order_id="order_12345",
    status="filled"
)
```

#### Stop-Loss Alert

```python
alert = alert_service.alert_stop_loss_triggered(
    symbol="AAPL",
    entry_price=150.0,
    exit_price=142.5,
    quantity=100,
    loss_amount=-750.0
)
```

### Retrieving Alerts

```python
# Get all alerts
all_alerts = alert_service.get_alerts()

# Get unread alerts only
unread_alerts = alert_service.get_alerts(unread_only=True)

# Get alerts for specific symbol
aapl_alerts = alert_service.get_alerts(symbol="AAPL")

# Get alerts of specific type
price_alerts = alert_service.get_alerts(alert_type=AlertType.PRICE_MOVEMENT)

# Get limited number of alerts
recent_alerts = alert_service.get_alerts(limit=10)
```

### Managing Alerts

```python
# Mark alert as read
alert_service.mark_as_read(alert_id)

# Mark all alerts as read
alert_service.mark_all_as_read()

# Get unread count
unread_count = alert_service.get_unread_count()

# Clear old alerts
alert_service.clear_alerts(older_than_days=7)

# Clear all alerts
alert_service.clear_alerts()
```

### Alert Statistics

```python
stats = alert_service.get_alert_stats()
print(f"Total alerts: {stats['total_alerts']}")
print(f"Unread alerts: {stats['unread_alerts']}")
print(f"By type: {stats['by_type']}")
print(f"By priority: {stats['by_priority']}")
```

### Callback Registration

Register callbacks to receive real-time notifications:

```python
def on_price_alert(alert):
    print(f"Price alert: {alert.title}")
    print(f"Message: {alert.message}")

# Register callback
alert_service.register_callback(AlertType.PRICE_MOVEMENT, on_price_alert)

# Unregister callback
alert_service.unregister_callback(AlertType.PRICE_MOVEMENT, on_price_alert)
```

## Integration Examples

### With Market Data Service

```python
from services.market_data_service import MarketDataService
from services.alert_service import AlertService

market_service = MarketDataService()
alert_service = AlertService()

# Track price changes
last_price = None
symbol = "AAPL"

while True:
    quote = market_service.get_latest_quote(symbol)
    current_price = quote.mid_price
    
    if last_price is not None:
        # Check for significant price movement
        alert_service.alert_price_movement(
            symbol=symbol,
            old_price=last_price,
            new_price=current_price
        )
    
    last_price = current_price
    time.sleep(60)  # Check every minute
```

### With AI Engine

```python
from ai.inference import AIEngine
from services.alert_service import AlertService

ai_engine = AIEngine()
alert_service = AlertService()

# Analyze stock and send alert if high confidence
symbol = "AAPL"
recommendation = ai_engine.get_recommendation(symbol)

if recommendation.confidence >= 0.8:
    alert_service.alert_ai_signal(
        symbol=symbol,
        action=recommendation.action,
        confidence=recommendation.confidence,
        reasoning=recommendation.reasoning,
        target_price=recommendation.target_price
    )
```

### With Risk Service

```python
from services.risk_service import RiskService
from services.alert_service import AlertService

risk_service = RiskService()
alert_service = AlertService()

# Monitor daily loss limit
daily_loss = risk_service._get_daily_loss()
limit = risk_service.config.daily_loss_limit

if daily_loss >= limit * 0.9:  # 90% of limit
    alert_service.alert_risk_limit(
        limit_type="daily_loss",
        current_value=daily_loss,
        limit_value=limit,
        message=f"Daily loss at {daily_loss/limit:.0%} of limit"
    )
```

### With Trading Service

```python
from services.trading_service import TradingService
from services.alert_service import AlertService

trading_service = TradingService()
alert_service = AlertService()

# Place order and send alert
order = trading_service.place_order(
    symbol="AAPL",
    qty=100,
    side="buy",
    order_type="market"
)

alert_service.alert_order_execution(
    symbol=order.symbol,
    action=order.side,
    quantity=order.quantity,
    price=order.filled_avg_price or 0.0,
    order_id=order.order_id,
    status=order.status.value
)
```

## Alert Configuration Options

### Alert Filtering

```python
config = AlertConfig(
    # Only send medium priority and above
    min_priority=AlertPriority.MEDIUM,
    
    # Only enable specific alert types
    enabled_alert_types=[
        AlertType.AI_SIGNAL,
        AlertType.RISK_LIMIT,
        AlertType.ORDER_EXECUTION
    ]
)
```

### Threshold Configuration

```python
config = AlertConfig(
    # Alert on 3% price movements
    price_movement_threshold=0.03,
    
    # Only alert on 90%+ confidence signals
    high_confidence_threshold=0.90
)
```

## Best Practices

1. **Configure Appropriate Thresholds**: Set thresholds that match your trading style and risk tolerance

2. **Use Priority Levels Wisely**: Reserve CRITICAL priority for truly urgent situations

3. **Filter Alerts**: Enable only the alert types you need to avoid notification fatigue

4. **Regular Cleanup**: Periodically clear old alerts to maintain performance

5. **Test Email Configuration**: Verify email settings work before relying on them for critical alerts

6. **Monitor Unread Count**: Keep track of unread alerts to ensure you don't miss important notifications

7. **Use Callbacks for Real-Time**: Register callbacks for time-sensitive alerts that need immediate action

## Testing

Run the test suite:

```bash
pytest tests/test_alert_service.py -v
```

## Requirements Validation

The alert service implements the following requirements:

- **12.1**: Price movement alerts with configurable thresholds
- **12.2**: AI signal alerts for high-confidence recommendations
- **12.3**: Risk limit alerts with warning levels
- **12.4**: Order execution alerts for all order status changes
- **12.5**: Multiple notification channels (in-app and email)

## Future Enhancements

- Push notifications for mobile devices
- SMS notifications
- Webhook support for custom integrations
- Alert templates and customization
- Alert scheduling (quiet hours)
- Alert aggregation and batching
- Advanced filtering rules
