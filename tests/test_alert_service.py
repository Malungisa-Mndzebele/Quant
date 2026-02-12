"""Tests for alert service."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings

from services.alert_service import (
    AlertService,
    AlertConfig,
    AlertType,
    AlertPriority,
    NotificationChannel,
    Alert
)


@pytest.fixture
def alert_config():
    """Create test alert configuration"""
    return AlertConfig(
        enabled_channels=[NotificationChannel.IN_APP],
        price_movement_threshold=0.05,
        high_confidence_threshold=0.8,
        min_priority=AlertPriority.LOW,
        enabled_alert_types=list(AlertType)
    )


@pytest.fixture
def alert_service(alert_config):
    """Create alert service instance"""
    return AlertService(config=alert_config)


def test_initialization(alert_service):
    """Test alert service initialization"""
    assert alert_service is not None
    assert len(alert_service._alerts) == 0
    assert alert_service._alert_counter == 0


def test_send_alert_basic(alert_service):
    """Test sending a basic alert"""
    alert = alert_service.send_alert(
        alert_type=AlertType.PRICE_MOVEMENT,
        title="Test Alert",
        message="This is a test alert",
        priority=AlertPriority.MEDIUM,
        symbol="AAPL"
    )
    
    assert alert is not None
    assert alert.alert_type == AlertType.PRICE_MOVEMENT
    assert alert.title == "Test Alert"
    assert alert.message == "This is a test alert"
    assert alert.priority == AlertPriority.MEDIUM
    assert alert.symbol == "AAPL"
    assert not alert.read
    
    # Check alert is stored
    alerts = alert_service.get_alerts()
    assert len(alerts) == 1
    assert alerts[0].alert_id == alert.alert_id


def test_alert_filtering_by_priority(alert_config):
    """Test alert filtering by minimum priority"""
    alert_config.min_priority = AlertPriority.HIGH
    service = AlertService(config=alert_config)
    
    # Low priority alert should be filtered
    alert_low = service.send_alert(
        alert_type=AlertType.PRICE_MOVEMENT,
        title="Low Priority",
        message="Should be filtered",
        priority=AlertPriority.LOW
    )
    assert alert_low is None
    
    # High priority alert should pass
    alert_high = service.send_alert(
        alert_type=AlertType.PRICE_MOVEMENT,
        title="High Priority",
        message="Should pass",
        priority=AlertPriority.HIGH
    )
    assert alert_high is not None
    
    alerts = service.get_alerts()
    assert len(alerts) == 1


def test_alert_filtering_by_type(alert_config):
    """Test alert filtering by enabled types"""
    alert_config.enabled_alert_types = [AlertType.PRICE_MOVEMENT]
    service = AlertService(config=alert_config)
    
    # Enabled type should pass
    alert_enabled = service.send_alert(
        alert_type=AlertType.PRICE_MOVEMENT,
        title="Enabled Type",
        message="Should pass",
        priority=AlertPriority.MEDIUM
    )
    assert alert_enabled is not None
    
    # Disabled type should be filtered
    alert_disabled = service.send_alert(
        alert_type=AlertType.AI_SIGNAL,
        title="Disabled Type",
        message="Should be filtered",
        priority=AlertPriority.MEDIUM
    )
    assert alert_disabled is None
    
    alerts = service.get_alerts()
    assert len(alerts) == 1


def test_alert_price_movement(alert_service):
    """Test price movement alert"""
    # Small movement (below threshold)
    alert_small = alert_service.alert_price_movement(
        symbol="AAPL",
        old_price=100.0,
        new_price=102.0  # 2% change
    )
    assert alert_small is None
    
    # Large movement (above threshold)
    alert_large = alert_service.alert_price_movement(
        symbol="AAPL",
        old_price=100.0,
        new_price=107.0  # 7% change
    )
    assert alert_large is not None
    assert alert_large.symbol == "AAPL"
    assert alert_large.priority == AlertPriority.HIGH
    assert "7.0%" in alert_large.message
    
    # Critical movement (10%+)
    alert_critical = alert_service.alert_price_movement(
        symbol="TSLA",
        old_price=200.0,
        new_price=180.0  # -10% change
    )
    assert alert_critical is not None
    assert alert_critical.priority == AlertPriority.CRITICAL


def test_alert_ai_signal(alert_service):
    """Test AI signal alert"""
    # Low confidence (below threshold)
    alert_low = alert_service.alert_ai_signal(
        symbol="AAPL",
        action="buy",
        confidence=0.7,
        reasoning="Test reasoning"
    )
    assert alert_low is None
    
    # High confidence (above threshold)
    alert_high = alert_service.alert_ai_signal(
        symbol="AAPL",
        action="buy",
        confidence=0.85,
        reasoning="Strong buy signal",
        target_price=150.0
    )
    assert alert_high is not None
    assert alert_high.symbol == "AAPL"
    assert "buy" in alert_high.message.lower()
    assert "85.0%" in alert_high.message
    assert "$150.00" in alert_high.message


def test_alert_risk_limit(alert_service):
    """Test risk limit alert"""
    # Warning level (75%)
    alert_warning = alert_service.alert_risk_limit(
        limit_type="daily_loss",
        current_value=750.0,
        limit_value=1000.0,
        message="Approaching daily loss limit"
    )
    assert alert_warning is not None
    assert alert_warning.priority == AlertPriority.MEDIUM
    
    # Critical level (100%)
    alert_critical = alert_service.alert_risk_limit(
        limit_type="daily_loss",
        current_value=1000.0,
        limit_value=1000.0,
        message="Daily loss limit reached"
    )
    assert alert_critical is not None
    assert alert_critical.priority == AlertPriority.CRITICAL


def test_alert_order_execution(alert_service):
    """Test order execution alert"""
    alert = alert_service.alert_order_execution(
        symbol="AAPL",
        action="buy",
        quantity=100,
        price=150.0,
        order_id="order_123",
        status="filled"
    )
    
    assert alert is not None
    assert alert.symbol == "AAPL"
    assert alert.priority == AlertPriority.MEDIUM
    assert "order_123" in alert.message
    assert alert.data['total_value'] == 15000.0


def test_alert_stop_loss_triggered(alert_service):
    """Test stop-loss alert"""
    alert = alert_service.alert_stop_loss_triggered(
        symbol="AAPL",
        entry_price=150.0,
        exit_price=140.0,
        quantity=100,
        loss_amount=-1000.0
    )
    
    assert alert is not None
    assert alert.symbol == "AAPL"
    assert alert.priority == AlertPriority.HIGH
    assert alert.alert_type == AlertType.STOP_LOSS
    assert "$1000.00" in alert.message


def test_alert_take_profit_triggered(alert_service):
    """Test take-profit alert"""
    alert = alert_service.alert_take_profit_triggered(
        symbol="AAPL",
        entry_price=140.0,
        exit_price=150.0,
        quantity=100,
        profit_amount=1000.0
    )
    
    assert alert is not None
    assert alert.symbol == "AAPL"
    assert alert.priority == AlertPriority.MEDIUM
    assert alert.alert_type == AlertType.TAKE_PROFIT
    assert "$1000.00" in alert.message


def test_alert_daily_loss_limit(alert_service):
    """Test daily loss limit alert"""
    alert = alert_service.alert_daily_loss_limit(
        daily_loss=1000.0,
        limit=1000.0
    )
    
    assert alert is not None
    assert alert.priority == AlertPriority.CRITICAL
    assert alert.alert_type == AlertType.DAILY_LOSS_LIMIT
    assert "disabled" in alert.message.lower()


def test_get_alerts_filtering(alert_service):
    """Test getting alerts with filters"""
    # Create multiple alerts
    alert_service.send_alert(
        alert_type=AlertType.PRICE_MOVEMENT,
        title="Alert 1",
        message="Message 1",
        priority=AlertPriority.LOW,
        symbol="AAPL"
    )
    
    alert_service.send_alert(
        alert_type=AlertType.AI_SIGNAL,
        title="Alert 2",
        message="Message 2",
        priority=AlertPriority.HIGH,
        symbol="GOOGL"
    )
    
    alert_service.send_alert(
        alert_type=AlertType.PRICE_MOVEMENT,
        title="Alert 3",
        message="Message 3",
        priority=AlertPriority.MEDIUM,
        symbol="AAPL"
    )
    
    # Get all alerts
    all_alerts = alert_service.get_alerts()
    assert len(all_alerts) == 3
    
    # Filter by type
    price_alerts = alert_service.get_alerts(alert_type=AlertType.PRICE_MOVEMENT)
    assert len(price_alerts) == 2
    
    # Filter by symbol
    aapl_alerts = alert_service.get_alerts(symbol="AAPL")
    assert len(aapl_alerts) == 2
    
    # Filter with limit
    limited_alerts = alert_service.get_alerts(limit=2)
    assert len(limited_alerts) == 2


def test_mark_as_read(alert_service):
    """Test marking alerts as read"""
    alert = alert_service.send_alert(
        alert_type=AlertType.PRICE_MOVEMENT,
        title="Test Alert",
        message="Test message",
        priority=AlertPriority.MEDIUM
    )
    
    assert not alert.read
    assert alert_service.get_unread_count() == 1
    
    # Mark as read
    result = alert_service.mark_as_read(alert.alert_id)
    assert result is True
    assert alert.read
    assert alert_service.get_unread_count() == 0
    
    # Try to mark non-existent alert
    result = alert_service.mark_as_read("non_existent")
    assert result is False


def test_mark_all_as_read(alert_service):
    """Test marking all alerts as read"""
    # Create multiple alerts
    for i in range(3):
        alert_service.send_alert(
            alert_type=AlertType.PRICE_MOVEMENT,
            title=f"Alert {i}",
            message=f"Message {i}",
            priority=AlertPriority.MEDIUM
        )
    
    assert alert_service.get_unread_count() == 3
    
    alert_service.mark_all_as_read()
    assert alert_service.get_unread_count() == 0


def test_clear_alerts(alert_service):
    """Test clearing alerts"""
    # Create alerts
    for i in range(5):
        alert_service.send_alert(
            alert_type=AlertType.PRICE_MOVEMENT,
            title=f"Alert {i}",
            message=f"Message {i}",
            priority=AlertPriority.MEDIUM
        )
    
    assert len(alert_service.get_alerts()) == 5
    
    # Clear all
    alert_service.clear_alerts()
    assert len(alert_service.get_alerts()) == 0


def test_callback_registration(alert_service):
    """Test callback registration and triggering"""
    callback_called = []
    
    def test_callback(alert: Alert):
        callback_called.append(alert.alert_id)
    
    # Register callback
    alert_service.register_callback(AlertType.PRICE_MOVEMENT, test_callback)
    
    # Send alert
    alert = alert_service.send_alert(
        alert_type=AlertType.PRICE_MOVEMENT,
        title="Test Alert",
        message="Test message",
        priority=AlertPriority.MEDIUM
    )
    
    # Check callback was called
    assert len(callback_called) == 1
    assert callback_called[0] == alert.alert_id
    
    # Unregister callback
    alert_service.unregister_callback(AlertType.PRICE_MOVEMENT, test_callback)
    
    # Send another alert
    alert_service.send_alert(
        alert_type=AlertType.PRICE_MOVEMENT,
        title="Test Alert 2",
        message="Test message 2",
        priority=AlertPriority.MEDIUM
    )
    
    # Callback should not be called again
    assert len(callback_called) == 1


def test_get_alert_stats(alert_service):
    """Test getting alert statistics"""
    # Create various alerts
    alert_service.send_alert(
        alert_type=AlertType.PRICE_MOVEMENT,
        title="Alert 1",
        message="Message 1",
        priority=AlertPriority.LOW
    )
    
    alert_service.send_alert(
        alert_type=AlertType.AI_SIGNAL,
        title="Alert 2",
        message="Message 2",
        priority=AlertPriority.HIGH
    )
    
    alert_service.send_alert(
        alert_type=AlertType.PRICE_MOVEMENT,
        title="Alert 3",
        message="Message 3",
        priority=AlertPriority.MEDIUM
    )
    
    # Mark one as read
    alerts = alert_service.get_alerts()
    alert_service.mark_as_read(alerts[0].alert_id)
    
    # Get stats
    stats = alert_service.get_alert_stats()
    
    assert stats['total_alerts'] == 3
    assert stats['unread_alerts'] == 2
    assert stats['read_alerts'] == 1
    assert stats['by_type']['price_movement'] == 2
    assert stats['by_type']['ai_signal'] == 1
    assert stats['by_priority']['low'] == 1
    assert stats['by_priority']['medium'] == 1
    assert stats['by_priority']['high'] == 1


def test_alert_storage_limit(alert_service):
    """Test that alert storage respects maximum limit"""
    # Create more than 100 alerts
    for i in range(150):
        alert_service.send_alert(
            alert_type=AlertType.PRICE_MOVEMENT,
            title=f"Alert {i}",
            message=f"Message {i}",
            priority=AlertPriority.MEDIUM
        )
    
    # Should only keep last 100 (deque maxlen=100)
    # The deque will automatically drop the oldest alerts
    alerts = alert_service.get_alerts()
    assert len(alerts) == 100
    
    # Check that we have alerts from the later range (50-149)
    # Since deque drops from the left when full, we should have alerts 50-149
    alert_titles = [a.title for a in alerts]
    
    # Should NOT have early alerts
    assert "Alert 0" not in alert_titles
    assert "Alert 10" not in alert_titles
    
    # Should have later alerts
    assert any("Alert 149" in title for title in alert_titles)
    assert any("Alert 140" in title for title in alert_titles)


def test_alert_to_dict(alert_service):
    """Test converting alert to dictionary"""
    alert = alert_service.send_alert(
        alert_type=AlertType.PRICE_MOVEMENT,
        title="Test Alert",
        message="Test message",
        priority=AlertPriority.MEDIUM,
        symbol="AAPL",
        data={'price': 150.0}
    )
    
    alert_dict = alert.to_dict()
    
    assert alert_dict['alert_id'] == alert.alert_id
    assert alert_dict['alert_type'] == 'price_movement'
    assert alert_dict['priority'] == 'medium'
    assert alert_dict['title'] == "Test Alert"
    assert alert_dict['message'] == "Test message"
    assert alert_dict['symbol'] == "AAPL"
    assert alert_dict['data']['price'] == 150.0
    assert alert_dict['read'] is False


@patch('smtplib.SMTP')
def test_email_notification(mock_smtp, alert_config):
    """Test email notification sending"""
    # Configure email
    alert_config.enabled_channels = [NotificationChannel.EMAIL]
    alert_config.email_address = "test@example.com"
    alert_config.smtp_server = "smtp.example.com"
    alert_config.smtp_port = 587
    alert_config.smtp_username = "user"
    alert_config.smtp_password = "pass"
    
    service = AlertService(config=alert_config)
    
    # Mock SMTP
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_server
    
    # Send alert
    alert = service.send_alert(
        alert_type=AlertType.PRICE_MOVEMENT,
        title="Test Alert",
        message="Test message",
        priority=AlertPriority.HIGH,
        symbol="AAPL"
    )
    
    # Verify email was sent
    assert mock_smtp.called
    assert mock_server.starttls.called
    assert mock_server.login.called
    assert mock_server.send_message.called


def test_update_config(alert_service):
    """Test updating alert configuration"""
    new_config = AlertConfig(
        enabled_channels=[NotificationChannel.EMAIL],
        price_movement_threshold=0.10,
        high_confidence_threshold=0.9
    )
    
    alert_service.update_config(new_config)
    
    assert alert_service.config.price_movement_threshold == 0.10
    assert alert_service.config.high_confidence_threshold == 0.9
    assert NotificationChannel.EMAIL in alert_service.config.enabled_channels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



# Property-Based Tests

@given(
    alert_type=st.sampled_from(list(AlertType)),
    priority=st.sampled_from(list(AlertPriority)),
    title=st.text(min_size=1, max_size=100),
    message=st.text(min_size=1, max_size=500),
    symbol=st.one_of(st.none(), st.text(min_size=1, max_size=10)),
    enabled_channels=st.lists(
        st.sampled_from(list(NotificationChannel)),
        min_size=1,
        max_size=len(NotificationChannel),
        unique=True
    )
)
@settings(max_examples=100)
def test_property_alert_delivery_guarantee(
    alert_type,
    priority,
    title,
    message,
    symbol,
    enabled_channels
):
    """
    Property 12: Alert delivery guarantee
    
    For any configured alert condition that is triggered, a notification 
    should be delivered through at least one enabled channel.
    
    Validates: Requirements 12.1, 12.2
    
    Feature: ai-trading-agent, Property 12: Alert delivery guarantee
    """
    # Create config with enabled channels
    config = AlertConfig(
        enabled_channels=enabled_channels,
        min_priority=AlertPriority.LOW,
        enabled_alert_types=list(AlertType)
    )
    
    # For email channel, we need to configure email settings
    if NotificationChannel.EMAIL in enabled_channels:
        config.email_address = "test@example.com"
        config.smtp_server = "smtp.example.com"
        config.smtp_username = "user"
        config.smtp_password = "pass"
    
    service = AlertService(config=config)
    
    # Track delivery through callbacks
    delivered_channels = []
    
    def in_app_callback(alert: Alert):
        delivered_channels.append(NotificationChannel.IN_APP)
    
    # Register callback to track in-app delivery
    service.register_callback(alert_type, in_app_callback)
    
    # Mock email sending if email is enabled
    with patch('services.alert_service.smtplib.SMTP') as mock_smtp:
        if NotificationChannel.EMAIL in enabled_channels:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Send alert
        alert = service.send_alert(
            alert_type=alert_type,
            title=title,
            message=message,
            priority=priority,
            symbol=symbol
        )
        
        # Alert should be created (not filtered out)
        assert alert is not None, (
            f"Alert was filtered out when it should have been delivered. "
            f"Type: {alert_type}, Priority: {priority}"
        )
        
        # Verify delivery through at least one channel
        delivery_confirmed = False
        
        # Check in-app delivery
        if NotificationChannel.IN_APP in enabled_channels:
            # Alert should be in storage
            stored_alerts = service.get_alerts()
            assert any(a.alert_id == alert.alert_id for a in stored_alerts), (
                "Alert not found in in-app storage despite IN_APP channel being enabled"
            )
            delivery_confirmed = True
        
        # Check email delivery
        if NotificationChannel.EMAIL in enabled_channels:
            # Email should have been attempted
            # Note: We're mocking SMTP, so we check if it was called
            if mock_smtp.called:
                delivery_confirmed = True
        
        # At least one channel must have delivered the alert
        assert delivery_confirmed, (
            f"Alert {alert.alert_id} was not delivered through any enabled channel. "
            f"Enabled channels: {[c.value for c in enabled_channels]}"
        )
        
        # Additional verification: callback should have been triggered for in-app
        if NotificationChannel.IN_APP in enabled_channels:
            assert NotificationChannel.IN_APP in delivered_channels, (
                "In-app callback was not triggered despite IN_APP channel being enabled"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
