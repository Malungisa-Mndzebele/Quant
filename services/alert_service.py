"""Alert service for managing and delivering trading alerts and notifications."""

import logging
import smtplib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import deque

from config.settings import settings

logger = logging.getLogger(__name__)


class AlertType(str, Enum):
    """Types of alerts"""
    PRICE_MOVEMENT = "price_movement"
    AI_SIGNAL = "ai_signal"
    RISK_LIMIT = "risk_limit"
    ORDER_EXECUTION = "order_execution"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    MARKET_STATUS = "market_status"


class AlertPriority(str, Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(str, Enum):
    """Notification delivery channels"""
    IN_APP = "in_app"
    EMAIL = "email"
    PUSH = "push"  # Reserved for future implementation


@dataclass
class Alert:
    """Alert information"""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    symbol: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    read: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'priority': self.priority.value,
            'title': self.title,
            'message': self.message,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'read': self.read
        }


@dataclass
class AlertConfig:
    """Alert configuration"""
    enabled_channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.IN_APP]
    )
    email_address: Optional[str] = None
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    
    # Alert thresholds
    price_movement_threshold: float = 0.05  # 5% price change
    high_confidence_threshold: float = 0.8  # 80% confidence for AI signals
    
    # Alert filtering
    min_priority: AlertPriority = AlertPriority.LOW
    enabled_alert_types: List[AlertType] = field(
        default_factory=lambda: list(AlertType)
    )


class AlertService:
    """
    Service for managing alerts and notifications.
    
    Provides price movement alerts, AI signal alerts, risk limit alerts,
    order execution alerts, and supports multiple notification channels.
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Initialize alert service.
        
        Args:
            config: Alert configuration (defaults to basic config)
        """
        self.config = config or AlertConfig()
        
        # In-app alert storage (limited to last 100 alerts)
        self._alerts: deque = deque(maxlen=100)
        self._alert_counter = 0
        
        # Callback registry for real-time notifications
        self._callbacks: Dict[AlertType, List[Callable]] = {}
        
        # Price tracking for movement alerts
        self._last_prices: Dict[str, float] = {}
        
        logger.info(
            f"Initialized Alert Service with channels: "
            f"{[c.value for c in self.config.enabled_channels]}"
        )
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        self._alert_counter += 1
        return f"alert_{datetime.now().strftime('%Y%m%d')}_{self._alert_counter:06d}"
    
    def _should_send_alert(self, alert_type: AlertType, priority: AlertPriority) -> bool:
        """
        Check if alert should be sent based on configuration.
        
        Args:
            alert_type: Type of alert
            priority: Alert priority
            
        Returns:
            True if alert should be sent
        """
        # Check if alert type is enabled
        if alert_type not in self.config.enabled_alert_types:
            return False
        
        # Check if priority meets minimum threshold
        priority_order = {
            AlertPriority.LOW: 0,
            AlertPriority.MEDIUM: 1,
            AlertPriority.HIGH: 2,
            AlertPriority.CRITICAL: 3
        }
        
        if priority_order[priority] < priority_order[self.config.min_priority]:
            return False
        
        return True
    
    def _send_email(self, alert: Alert) -> bool:
        """
        Send alert via email.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if email sent successfully
        """
        if not self.config.email_address:
            logger.warning("Email address not configured")
            return False
        
        if not self.config.smtp_server:
            logger.warning("SMTP server not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.priority.value.upper()}] {alert.title}"
            msg['From'] = self.config.smtp_username or "AI Trading Agent"
            msg['To'] = self.config.email_address
            
            # Create HTML body
            html = f"""
            <html>
              <body>
                <h2>{alert.title}</h2>
                <p><strong>Priority:</strong> {alert.priority.value.upper()}</p>
                <p><strong>Type:</strong> {alert.alert_type.value}</p>
                {f'<p><strong>Symbol:</strong> {alert.symbol}</p>' if alert.symbol else ''}
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
                <p>{alert.message}</p>
                {self._format_alert_data_html(alert.data) if alert.data else ''}
              </body>
            </html>
            """
            
            # Attach HTML
            msg.attach(MIMEText(html, 'html'))
            
            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                if self.config.smtp_username and self.config.smtp_password:
                    server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email for alert {alert.alert_id}: {e}")
            return False
    
    def _format_alert_data_html(self, data: Dict[str, Any]) -> str:
        """Format alert data as HTML"""
        if not data:
            return ""
        
        html = "<h3>Details:</h3><ul>"
        for key, value in data.items():
            # Format key (convert snake_case to Title Case)
            formatted_key = key.replace('_', ' ').title()
            
            # Format value
            if isinstance(value, float):
                if abs(value) < 1:
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"${value:,.2f}" if 'price' in key.lower() or 'value' in key.lower() else f"{value:.2f}"
            else:
                formatted_value = str(value)
            
            html += f"<li><strong>{formatted_key}:</strong> {formatted_value}</li>"
        
        html += "</ul>"
        return html
    
    def _store_alert(self, alert: Alert):
        """Store alert in in-app storage"""
        self._alerts.append(alert)
        logger.debug(f"Stored alert {alert.alert_id} in in-app storage")
    
    def _trigger_callbacks(self, alert: Alert):
        """Trigger registered callbacks for alert type"""
        if alert.alert_type in self._callbacks:
            for callback in self._callbacks[alert.alert_type]:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def send_alert(
        self,
        alert_type: AlertType,
        title: str,
        message: str,
        priority: AlertPriority = AlertPriority.MEDIUM,
        symbol: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Optional[Alert]:
        """
        Send an alert through configured channels.
        
        Args:
            alert_type: Type of alert
            title: Alert title
            message: Alert message
            priority: Alert priority
            symbol: Related stock symbol (optional)
            data: Additional alert data (optional)
            
        Returns:
            Alert object if sent, None if filtered out
        """
        # Check if alert should be sent
        if not self._should_send_alert(alert_type, priority):
            logger.debug(f"Alert filtered out: {alert_type.value} (priority: {priority.value})")
            return None
        
        # Create alert
        alert = Alert(
            alert_id=self._generate_alert_id(),
            alert_type=alert_type,
            priority=priority,
            title=title,
            message=message,
            symbol=symbol,
            data=data or {}
        )
        
        logger.info(
            f"Sending alert: {alert.alert_type.value} - {alert.title} "
            f"(priority: {alert.priority.value})"
        )
        
        # Send through enabled channels
        delivery_success = False
        
        if NotificationChannel.IN_APP in self.config.enabled_channels:
            self._store_alert(alert)
            delivery_success = True
        
        if NotificationChannel.EMAIL in self.config.enabled_channels:
            if self._send_email(alert):
                delivery_success = True
        
        # Trigger callbacks
        self._trigger_callbacks(alert)
        
        if not delivery_success:
            logger.warning(f"Alert {alert.alert_id} was not delivered through any channel")
        
        return alert
    
    def alert_price_movement(
        self,
        symbol: str,
        old_price: float,
        new_price: float,
        threshold: Optional[float] = None
    ) -> Optional[Alert]:
        """
        Send alert for significant price movement.
        
        Args:
            symbol: Stock symbol
            old_price: Previous price
            new_price: Current price
            threshold: Movement threshold (defaults to config)
            
        Returns:
            Alert object if sent, None otherwise
        """
        threshold = threshold or self.config.price_movement_threshold
        
        # Calculate price change
        price_change = new_price - old_price
        price_change_pct = price_change / old_price if old_price > 0 else 0
        
        # Check if movement exceeds threshold
        if abs(price_change_pct) < threshold:
            return None
        
        # Determine priority based on magnitude
        if abs(price_change_pct) >= 0.10:  # 10%
            priority = AlertPriority.CRITICAL
        elif abs(price_change_pct) >= 0.07:  # 7%
            priority = AlertPriority.HIGH
        else:
            priority = AlertPriority.MEDIUM
        
        # Create alert message
        direction = "up" if price_change > 0 else "down"
        title = f"{symbol} Price Alert"
        message = (
            f"{symbol} moved {direction} {abs(price_change_pct):.1%} "
            f"from ${old_price:.2f} to ${new_price:.2f}"
        )
        
        return self.send_alert(
            alert_type=AlertType.PRICE_MOVEMENT,
            title=title,
            message=message,
            priority=priority,
            symbol=symbol,
            data={
                'old_price': old_price,
                'new_price': new_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct
            }
        )
    
    def alert_ai_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        reasoning: str,
        target_price: Optional[float] = None
    ) -> Optional[Alert]:
        """
        Send alert for AI trading signal.
        
        Args:
            symbol: Stock symbol
            action: Trading action ('buy', 'sell', 'hold')
            confidence: Signal confidence (0.0 to 1.0)
            reasoning: Signal reasoning
            target_price: Target price (optional)
            
        Returns:
            Alert object if sent, None otherwise
        """
        # Only alert on high-confidence signals
        if confidence < self.config.high_confidence_threshold:
            return None
        
        # Determine priority based on confidence
        if confidence >= 0.95:
            priority = AlertPriority.CRITICAL
        elif confidence >= 0.90:
            priority = AlertPriority.HIGH
        else:
            priority = AlertPriority.MEDIUM
        
        # Create alert message
        title = f"AI Signal: {action.upper()} {symbol}"
        message = (
            f"High-confidence {action} signal for {symbol} "
            f"(confidence: {confidence:.1%}). {reasoning}"
        )
        
        if target_price:
            message += f" Target: ${target_price:.2f}"
        
        return self.send_alert(
            alert_type=AlertType.AI_SIGNAL,
            title=title,
            message=message,
            priority=priority,
            symbol=symbol,
            data={
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'target_price': target_price
            }
        )
    
    def alert_risk_limit(
        self,
        limit_type: str,
        current_value: float,
        limit_value: float,
        message: str
    ) -> Optional[Alert]:
        """
        Send alert for risk limit warning or breach.
        
        Args:
            limit_type: Type of limit ('position_size', 'daily_loss', etc.)
            current_value: Current value
            limit_value: Limit value
            message: Alert message
            
        Returns:
            Alert object if sent
        """
        # Determine priority based on how close to limit
        ratio = current_value / limit_value if limit_value > 0 else 0
        
        if ratio >= 1.0:
            priority = AlertPriority.CRITICAL
        elif ratio >= 0.9:
            priority = AlertPriority.HIGH
        elif ratio >= 0.75:
            priority = AlertPriority.MEDIUM
        else:
            priority = AlertPriority.LOW
        
        title = f"Risk Limit Warning: {limit_type.replace('_', ' ').title()}"
        
        return self.send_alert(
            alert_type=AlertType.RISK_LIMIT,
            title=title,
            message=message,
            priority=priority,
            data={
                'limit_type': limit_type,
                'current_value': current_value,
                'limit_value': limit_value,
                'ratio': ratio
            }
        )
    
    def alert_order_execution(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        order_id: str,
        status: str
    ) -> Optional[Alert]:
        """
        Send alert for order execution.
        
        Args:
            symbol: Stock symbol
            action: Order action ('buy' or 'sell')
            quantity: Number of shares
            price: Execution price
            order_id: Order ID
            status: Order status
            
        Returns:
            Alert object if sent
        """
        # Determine priority based on status
        if status in ['filled', 'partially_filled']:
            priority = AlertPriority.MEDIUM
        elif status in ['rejected', 'cancelled']:
            priority = AlertPriority.HIGH
        else:
            priority = AlertPriority.LOW
        
        # Create alert message
        title = f"Order {status.title()}: {action.upper()} {symbol}"
        message = (
            f"Order {order_id}: {action} {quantity} shares of {symbol} "
            f"@ ${price:.2f} - Status: {status}"
        )
        
        return self.send_alert(
            alert_type=AlertType.ORDER_EXECUTION,
            title=title,
            message=message,
            priority=priority,
            symbol=symbol,
            data={
                'order_id': order_id,
                'action': action,
                'quantity': quantity,
                'price': price,
                'status': status,
                'total_value': quantity * price
            }
        )
    
    def alert_stop_loss_triggered(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        loss_amount: float
    ) -> Optional[Alert]:
        """
        Send alert when stop-loss is triggered.
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            exit_price: Exit price
            quantity: Number of shares
            loss_amount: Total loss amount
            
        Returns:
            Alert object if sent
        """
        loss_pct = (entry_price - exit_price) / entry_price if entry_price > 0 else 0
        
        title = f"Stop-Loss Triggered: {symbol}"
        message = (
            f"Stop-loss triggered for {symbol}. "
            f"Closed {quantity} shares at ${exit_price:.2f} "
            f"(entry: ${entry_price:.2f}). "
            f"Loss: ${abs(loss_amount):.2f} ({abs(loss_pct):.1%})"
        )
        
        return self.send_alert(
            alert_type=AlertType.STOP_LOSS,
            title=title,
            message=message,
            priority=AlertPriority.HIGH,
            symbol=symbol,
            data={
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'loss_amount': loss_amount,
                'loss_pct': loss_pct
            }
        )
    
    def alert_take_profit_triggered(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        profit_amount: float
    ) -> Optional[Alert]:
        """
        Send alert when take-profit is triggered.
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            exit_price: Exit price
            quantity: Number of shares
            profit_amount: Total profit amount
            
        Returns:
            Alert object if sent
        """
        profit_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        
        title = f"Take-Profit Triggered: {symbol}"
        message = (
            f"Take-profit triggered for {symbol}. "
            f"Closed {quantity} shares at ${exit_price:.2f} "
            f"(entry: ${entry_price:.2f}). "
            f"Profit: ${profit_amount:.2f} ({profit_pct:.1%})"
        )
        
        return self.send_alert(
            alert_type=AlertType.TAKE_PROFIT,
            title=title,
            message=message,
            priority=AlertPriority.MEDIUM,
            symbol=symbol,
            data={
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'profit_amount': profit_amount,
                'profit_pct': profit_pct
            }
        )
    
    def alert_daily_loss_limit(
        self,
        daily_loss: float,
        limit: float
    ) -> Optional[Alert]:
        """
        Send alert when daily loss limit is reached.
        
        Args:
            daily_loss: Current daily loss
            limit: Daily loss limit
            
        Returns:
            Alert object if sent
        """
        title = "Daily Loss Limit Reached"
        message = (
            f"Daily loss limit has been reached: ${daily_loss:.2f} / ${limit:.2f}. "
            f"Trading has been disabled for the remainder of the day."
        )
        
        return self.send_alert(
            alert_type=AlertType.DAILY_LOSS_LIMIT,
            title=title,
            message=message,
            priority=AlertPriority.CRITICAL,
            data={
                'daily_loss': daily_loss,
                'limit': limit
            }
        )
    
    def get_alerts(
        self,
        limit: Optional[int] = None,
        unread_only: bool = False,
        alert_type: Optional[AlertType] = None,
        symbol: Optional[str] = None
    ) -> List[Alert]:
        """
        Get stored alerts with optional filtering.
        
        Args:
            limit: Maximum number of alerts to return
            unread_only: Only return unread alerts
            alert_type: Filter by alert type
            symbol: Filter by symbol
            
        Returns:
            List of Alert objects
        """
        alerts = list(self._alerts)
        
        # Apply filters
        if unread_only:
            alerts = [a for a in alerts if not a.read]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if symbol:
            symbol = symbol.upper()
            alerts = [a for a in alerts if a.symbol == symbol]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            alerts = alerts[:limit]
        
        return alerts
    
    def mark_as_read(self, alert_id: str) -> bool:
        """
        Mark an alert as read.
        
        Args:
            alert_id: Alert ID to mark as read
            
        Returns:
            True if alert was found and marked
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.read = True
                logger.debug(f"Marked alert {alert_id} as read")
                return True
        
        logger.warning(f"Alert {alert_id} not found")
        return False
    
    def mark_all_as_read(self):
        """Mark all alerts as read"""
        for alert in self._alerts:
            alert.read = True
        logger.info("Marked all alerts as read")
    
    def clear_alerts(self, older_than_days: Optional[int] = None):
        """
        Clear alerts from storage.
        
        Args:
            older_than_days: Only clear alerts older than this many days
        """
        if older_than_days is None:
            self._alerts.clear()
            logger.info("Cleared all alerts")
        else:
            cutoff_time = datetime.now() - timedelta(days=older_than_days)
            original_count = len(self._alerts)
            
            # Filter out old alerts
            self._alerts = deque(
                (a for a in self._alerts if a.timestamp > cutoff_time),
                maxlen=100
            )
            
            cleared_count = original_count - len(self._alerts)
            logger.info(f"Cleared {cleared_count} alerts older than {older_than_days} days")
    
    def register_callback(self, alert_type: AlertType, callback: Callable[[Alert], None]):
        """
        Register a callback for specific alert type.
        
        Args:
            alert_type: Alert type to listen for
            callback: Callback function that receives Alert object
        """
        if alert_type not in self._callbacks:
            self._callbacks[alert_type] = []
        
        self._callbacks[alert_type].append(callback)
        logger.info(f"Registered callback for {alert_type.value} alerts")
    
    def unregister_callback(self, alert_type: AlertType, callback: Callable[[Alert], None]):
        """
        Unregister a callback.
        
        Args:
            alert_type: Alert type
            callback: Callback function to remove
        """
        if alert_type in self._callbacks:
            try:
                self._callbacks[alert_type].remove(callback)
                logger.info(f"Unregistered callback for {alert_type.value} alerts")
            except ValueError:
                logger.warning(f"Callback not found for {alert_type.value}")
    
    def update_config(self, new_config: AlertConfig):
        """
        Update alert configuration.
        
        Args:
            new_config: New alert configuration
        """
        self.config = new_config
        logger.info(f"Updated alert configuration")
    
    def get_unread_count(self) -> int:
        """
        Get count of unread alerts.
        
        Returns:
            Number of unread alerts
        """
        return sum(1 for alert in self._alerts if not alert.read)
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Returns:
            Dictionary with alert statistics
        """
        total_alerts = len(self._alerts)
        unread_alerts = self.get_unread_count()
        
        # Count by type
        type_counts = {}
        for alert_type in AlertType:
            count = sum(1 for a in self._alerts if a.alert_type == alert_type)
            if count > 0:
                type_counts[alert_type.value] = count
        
        # Count by priority
        priority_counts = {}
        for priority in AlertPriority:
            count = sum(1 for a in self._alerts if a.priority == priority)
            if count > 0:
                priority_counts[priority.value] = count
        
        return {
            'total_alerts': total_alerts,
            'unread_alerts': unread_alerts,
            'read_alerts': total_alerts - unread_alerts,
            'by_type': type_counts,
            'by_priority': priority_counts,
            'enabled_channels': [c.value for c in self.config.enabled_channels],
            'email_configured': self.config.email_address is not None
        }
