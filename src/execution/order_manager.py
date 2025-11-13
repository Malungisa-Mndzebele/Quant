"""Order management system for creating, submitting, and tracking orders."""

import logging
import time
import uuid
from typing import Optional, Dict, List
from datetime import datetime

from src.models.signal import Signal, SignalAction
from src.models.order import Order, OrderAction, OrderType, OrderStatusEnum, OrderStatus
from src.brokers.base import BrokerageAdapter


logger = logging.getLogger(__name__)


class OrderError(Exception):
    """Exception raised for order-related errors."""
    pass


class OrderManager:
    """Manages order creation, submission, and tracking."""
    
    def __init__(self, broker: BrokerageAdapter, max_retries: int = 3):
        """
        Initialize OrderManager.
        
        Args:
            broker: BrokerageAdapter instance for order submission
            max_retries: Maximum number of retry attempts for failed submissions
        """
        self.broker = broker
        self.max_retries = max_retries
        self.orders: Dict[str, Order] = {}
        logger.info(f"OrderManager initialized with max_retries={max_retries}")
    
    def create_order(self, signal: Signal) -> Order:
        """
        Create an order from a trading signal.
        
        Args:
            signal: Trading signal to convert to order
            
        Returns:
            Order object
            
        Raises:
            OrderError: If signal is invalid or cannot be converted
        """
        # Validate signal
        if signal.action == SignalAction.HOLD:
            raise OrderError("Cannot create order from HOLD signal")
        
        # Generate unique order ID
        order_id = self._generate_order_id()
        
        # Convert signal action to order action
        if signal.action == SignalAction.BUY:
            action = OrderAction.BUY
        elif signal.action == SignalAction.SELL:
            action = OrderAction.SELL
        else:
            raise OrderError(f"Invalid signal action: {signal.action}")
        
        # Convert order type
        order_type = OrderType(signal.order_type.value)
        
        # Create order
        try:
            order = Order(
                order_id=order_id,
                symbol=signal.symbol,
                action=action,
                quantity=signal.quantity,
                order_type=order_type,
                limit_price=signal.limit_price,
                status=OrderStatusEnum.PENDING,
                timestamp=datetime.now()
            )
            
            # Validate order
            self._validate_order(order)
            
            # Store order
            self.orders[order_id] = order
            
            logger.info(f"Created order {order_id}: {action.value} {signal.quantity} {signal.symbol} @ {order_type.value}")
            return order
            
        except ValueError as e:
            raise OrderError(f"Failed to create order: {str(e)}")
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit an order to the brokerage with retry logic.
        
        Args:
            order: Order to submit
            
        Returns:
            True if submission successful
            
        Raises:
            OrderError: If submission fails after all retries
        """
        # Validate order before submission
        self._validate_order(order)
        
        # Attempt submission with retries
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Submitting order {order.order_id} (attempt {attempt}/{self.max_retries})")
                
                # Submit to broker
                broker_order_id = self.broker.submit_order(order)
                
                # Update order status
                order.status = OrderStatusEnum.SUBMITTED
                order.order_id = broker_order_id  # Use broker's order ID
                self.orders[broker_order_id] = order
                
                logger.info(f"Order {broker_order_id} submitted successfully")
                return True
                
            except Exception as e:
                last_error = e
                logger.warning(f"Order submission attempt {attempt} failed: {str(e)}")
                
                if attempt < self.max_retries:
                    # Wait before retry (exponential backoff)
                    wait_time = 2 ** (attempt - 1)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # All retries failed
        order.status = OrderStatusEnum.REJECTED
        error_msg = f"Order submission failed after {self.max_retries} attempts: {str(last_error)}"
        logger.error(error_msg)
        raise OrderError(error_msg)
    
    def track_order(self, order_id: str) -> OrderStatus:
        """
        Track the status of an order.
        
        Args:
            order_id: ID of the order to track
            
        Returns:
            OrderStatus object with current status
            
        Raises:
            OrderError: If order not found or status cannot be retrieved
        """
        if order_id not in self.orders:
            raise OrderError(f"Order {order_id} not found")
        
        try:
            # Get status from broker
            broker_status = self.broker.get_order_status(order_id)
            
            # Update local order
            order = self.orders[order_id]
            order.status = OrderStatusEnum(broker_status.status)
            order.filled_quantity = broker_status.filled_quantity
            order.filled_price = broker_status.average_fill_price
            
            logger.debug(f"Order {order_id} status: {broker_status.status}")
            
            # Convert broker status to our OrderStatus
            return OrderStatus(
                order_id=broker_status.order_id,
                status=OrderStatusEnum(broker_status.status),
                filled_quantity=broker_status.filled_quantity,
                average_fill_price=broker_status.average_fill_price,
                message=broker_status.message
            )
            
        except Exception as e:
            error_msg = f"Failed to track order {order_id}: {str(e)}"
            logger.error(error_msg)
            raise OrderError(error_msg)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get an order by ID.
        
        Args:
            order_id: ID of the order
            
        Returns:
            Order object or None if not found
        """
        return self.orders.get(order_id)
    
    def get_all_orders(self) -> List[Order]:
        """
        Get all orders.
        
        Returns:
            List of all Order objects
        """
        return list(self.orders.values())
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if cancellation successful
            
        Raises:
            OrderError: If order cannot be cancelled
        """
        if order_id not in self.orders:
            raise OrderError(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatusEnum.PENDING, OrderStatusEnum.SUBMITTED]:
            raise OrderError(f"Cannot cancel order in status {order.status.value}")
        
        try:
            # Note: Actual cancellation would call broker.cancel_order()
            # For now, just update status
            order.status = OrderStatusEnum.CANCELLED
            logger.info(f"Order {order_id} cancelled")
            return True
            
        except Exception as e:
            error_msg = f"Failed to cancel order {order_id}: {str(e)}"
            logger.error(error_msg)
            raise OrderError(error_msg)
    
    def _validate_order(self, order: Order) -> None:
        """
        Validate an order before submission.
        
        Args:
            order: Order to validate
            
        Raises:
            OrderError: If order is invalid
        """
        # Validate symbol (basic check)
        if not order.symbol or not order.symbol.strip():
            raise OrderError("Order symbol cannot be empty")
        
        # Symbol should be uppercase and alphanumeric
        if not order.symbol.replace('.', '').replace('-', '').isalnum():
            raise OrderError(f"Invalid symbol format: {order.symbol}")
        
        # Validate quantity
        if order.quantity <= 0:
            raise OrderError(f"Order quantity must be positive, got {order.quantity}")
        
        # Validate order type specific requirements
        self._validate_order_type(order)
    
    def _validate_order_type(self, order: Order) -> None:
        """
        Validate order type specific requirements.
        
        Args:
            order: Order to validate
            
        Raises:
            OrderError: If order type requirements are not met
        """
        if order.order_type == OrderType.MARKET:
            # Market orders execute immediately at current market price
            # No price specification needed
            if order.limit_price is not None:
                logger.warning(f"Market order {order.order_id} has limit_price set, will be ignored")
            logger.debug(f"Validated MARKET order for {order.symbol}")
            
        elif order.order_type == OrderType.LIMIT:
            # Limit orders require a limit price
            if order.limit_price is None:
                raise OrderError("Limit orders require a limit_price")
            if order.limit_price <= 0:
                raise OrderError(f"Limit price must be positive, got {order.limit_price}")
            logger.debug(f"Validated LIMIT order for {order.symbol} @ ${order.limit_price}")
            
        elif order.order_type == OrderType.STOP_LOSS:
            # Stop-loss orders require a stop price (stored in limit_price field)
            if order.limit_price is None:
                raise OrderError("Stop-loss orders require a limit_price (stop price)")
            if order.limit_price <= 0:
                raise OrderError(f"Stop price must be positive, got {order.limit_price}")
            logger.debug(f"Validated STOP_LOSS order for {order.symbol} @ ${order.limit_price}")
            
        else:
            raise OrderError(f"Unsupported order type: {order.order_type}")
    
    def _generate_order_id(self) -> str:
        """
        Generate a unique order ID.
        
        Returns:
            Unique order ID string
        """
        return f"ORD-{uuid.uuid4().hex[:12].upper()}"
