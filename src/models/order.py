"""Order and trade models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class OrderAction(Enum):
    """Order actions."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"


class OrderStatusEnum(Enum):
    """Order status values."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Represents a trade order."""
    order_id: str
    symbol: str
    action: OrderAction
    quantity: int
    order_type: OrderType
    status: OrderStatusEnum = OrderStatusEnum.PENDING
    limit_price: Optional[float] = None
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    timestamp: datetime = None

    def __post_init__(self):
        """Validate order and set defaults."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # Convert string to enum if needed
        if isinstance(self.action, str):
            self.action = OrderAction(self.action)
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type)
        if isinstance(self.status, str):
            self.status = OrderStatusEnum(self.status)
        
        # Validate quantity
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")
        
        # Validate limit price for limit orders
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders require a limit_price")
        
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError(f"Limit price must be positive, got {self.limit_price}")


@dataclass
class OrderStatus:
    """Represents the status of an order."""
    order_id: str
    status: OrderStatusEnum
    filled_quantity: int
    average_fill_price: Optional[float]
    message: str = ""

    def __post_init__(self):
        """Validate order status."""
        if isinstance(self.status, str):
            self.status = OrderStatusEnum(self.status)
        
        if self.filled_quantity < 0:
            raise ValueError(f"Filled quantity cannot be negative, got {self.filled_quantity}")
        
        if self.average_fill_price is not None and self.average_fill_price <= 0:
            raise ValueError(f"Average fill price must be positive, got {self.average_fill_price}")


@dataclass
class Trade:
    """Represents a completed trade."""
    trade_id: str
    symbol: str
    action: OrderAction
    quantity: int
    price: float
    timestamp: datetime
    pnl: Optional[float] = None

    def __post_init__(self):
        """Validate trade."""
        if isinstance(self.action, str):
            self.action = OrderAction(self.action)
        
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")
        
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
