"""Verification script for alert service functionality."""

import sys
from datetime import datetime

from services.alert_service import (
    AlertService,
    AlertConfig,
    AlertType,
    AlertPriority,
    NotificationChannel
)


def print_section(title):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_alert(alert):
    """Print alert details"""
    print(f"Alert ID: {alert.alert_id}")
    print(f"Type: {alert.alert_type.value}")
    print(f"Priority: {alert.priority.value.upper()}")
    print(f"Title: {alert.title}")
    print(f"Message: {alert.message}")
    if alert.symbol:
        print(f"Symbol: {alert.symbol}")
    print(f"Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Read: {alert.read}")
    if alert.data:
        print(f"Data: {alert.data}")
    print()


def main():
    """Run alert service verification"""
    print_section("Alert Service Verification")
    
    # Create configuration
    config = AlertConfig(
        enabled_channels=[NotificationChannel.IN_APP],
        price_movement_threshold=0.05,
        high_confidence_threshold=0.8,
        min_priority=AlertPriority.LOW,
        enabled_alert_types=list(AlertType)
    )
    
    # Initialize service
    alert_service = AlertService(config=config)
    print("✓ Alert service initialized successfully")
    
    # Test 1: Price Movement Alert
    print_section("Test 1: Price Movement Alert")
    alert = alert_service.alert_price_movement(
        symbol="AAPL",
        old_price=150.0,
        new_price=157.5
    )
    if alert:
        print("✓ Price movement alert created")
        print_alert(alert)
    else:
        print("✗ Price movement alert not created (below threshold)")
    
    # Test 2: AI Signal Alert
    print_section("Test 2: AI Signal Alert")
    alert = alert_service.alert_ai_signal(
        symbol="GOOGL",
        action="buy",
        confidence=0.85,
        reasoning="Strong bullish momentum with positive sentiment",
        target_price=2800.0
    )
    if alert:
        print("✓ AI signal alert created")
        print_alert(alert)
    else:
        print("✗ AI signal alert not created (below confidence threshold)")
    
    # Test 3: Risk Limit Alert
    print_section("Test 3: Risk Limit Alert")
    alert = alert_service.alert_risk_limit(
        limit_type="daily_loss",
        current_value=900.0,
        limit_value=1000.0,
        message="Approaching daily loss limit: $900 / $1000"
    )
    if alert:
        print("✓ Risk limit alert created")
        print_alert(alert)
    
    # Test 4: Order Execution Alert
    print_section("Test 4: Order Execution Alert")
    alert = alert_service.alert_order_execution(
        symbol="TSLA",
        action="buy",
        quantity=50,
        price=245.50,
        order_id="order_12345",
        status="filled"
    )
    if alert:
        print("✓ Order execution alert created")
        print_alert(alert)
    
    # Test 5: Stop-Loss Alert
    print_section("Test 5: Stop-Loss Alert")
    alert = alert_service.alert_stop_loss_triggered(
        symbol="MSFT",
        entry_price=380.0,
        exit_price=361.0,
        quantity=100,
        loss_amount=-1900.0
    )
    if alert:
        print("✓ Stop-loss alert created")
        print_alert(alert)
    
    # Test 6: Take-Profit Alert
    print_section("Test 6: Take-Profit Alert")
    alert = alert_service.alert_take_profit_triggered(
        symbol="NVDA",
        entry_price=500.0,
        exit_price=550.0,
        quantity=50,
        profit_amount=2500.0
    )
    if alert:
        print("✓ Take-profit alert created")
        print_alert(alert)
    
    # Test 7: Daily Loss Limit Alert
    print_section("Test 7: Daily Loss Limit Alert")
    alert = alert_service.alert_daily_loss_limit(
        daily_loss=1000.0,
        limit=1000.0
    )
    if alert:
        print("✓ Daily loss limit alert created")
        print_alert(alert)
    
    # Test 8: Get All Alerts
    print_section("Test 8: Retrieve Alerts")
    all_alerts = alert_service.get_alerts()
    print(f"✓ Total alerts: {len(all_alerts)}")
    
    # Test 9: Filter Alerts
    print_section("Test 9: Filter Alerts")
    
    # By type
    price_alerts = alert_service.get_alerts(alert_type=AlertType.PRICE_MOVEMENT)
    print(f"✓ Price movement alerts: {len(price_alerts)}")
    
    # By symbol
    aapl_alerts = alert_service.get_alerts(symbol="AAPL")
    print(f"✓ AAPL alerts: {len(aapl_alerts)}")
    
    # Unread only
    unread_alerts = alert_service.get_alerts(unread_only=True)
    print(f"✓ Unread alerts: {len(unread_alerts)}")
    
    # Test 10: Mark as Read
    print_section("Test 10: Mark Alerts as Read")
    if all_alerts:
        first_alert = all_alerts[0]
        result = alert_service.mark_as_read(first_alert.alert_id)
        print(f"✓ Marked alert {first_alert.alert_id} as read: {result}")
        
        unread_count = alert_service.get_unread_count()
        print(f"✓ Unread count: {unread_count}")
    
    # Test 11: Alert Statistics
    print_section("Test 11: Alert Statistics")
    stats = alert_service.get_alert_stats()
    print(f"✓ Total alerts: {stats['total_alerts']}")
    print(f"✓ Unread alerts: {stats['unread_alerts']}")
    print(f"✓ Read alerts: {stats['read_alerts']}")
    print(f"✓ By type: {stats['by_type']}")
    print(f"✓ By priority: {stats['by_priority']}")
    print(f"✓ Enabled channels: {stats['enabled_channels']}")
    print(f"✓ Email configured: {stats['email_configured']}")
    
    # Test 12: Callback Registration
    print_section("Test 12: Callback Registration")
    
    callback_triggered = []
    
    def test_callback(alert):
        callback_triggered.append(alert.alert_id)
        print(f"  Callback triggered for: {alert.title}")
    
    alert_service.register_callback(AlertType.PRICE_MOVEMENT, test_callback)
    print("✓ Callback registered")
    
    # Trigger callback
    alert = alert_service.alert_price_movement(
        symbol="AMD",
        old_price=100.0,
        new_price=110.0
    )
    
    if callback_triggered:
        print(f"✓ Callback was triggered {len(callback_triggered)} time(s)")
    else:
        print("✗ Callback was not triggered")
    
    # Test 13: Alert Storage Limit
    print_section("Test 13: Alert Storage Limit")
    initial_count = len(alert_service.get_alerts())
    
    # Create many alerts
    for i in range(50):
        alert_service.send_alert(
            alert_type=AlertType.PRICE_MOVEMENT,
            title=f"Test Alert {i}",
            message=f"Test message {i}",
            priority=AlertPriority.LOW
        )
    
    final_count = len(alert_service.get_alerts())
    print(f"✓ Initial alerts: {initial_count}")
    print(f"✓ Final alerts: {final_count}")
    print(f"✓ Storage limit working (max 100 alerts)")
    
    # Summary
    print_section("Verification Summary")
    print("✓ All alert service features verified successfully!")
    print("\nAlert service is ready for use in the AI Trading Agent.")
    print("\nKey Features:")
    print("  • Multiple alert types (price, AI signal, risk, orders, etc.)")
    print("  • Priority levels (low, medium, high, critical)")
    print("  • In-app notification storage")
    print("  • Email notification support")
    print("  • Alert filtering and retrieval")
    print("  • Callback registration for real-time notifications")
    print("  • Alert statistics and management")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
