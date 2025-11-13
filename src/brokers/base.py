"""Abstract base class for brokerage adapters."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class OrderStatus:
    """Represents the status of an order."""
    order_id: str
    status: str  # 'PENDING', 'FILLED', 'CANCELLED', 'REJECTED'
    filled_quantity: int
    average_fill_price: Optional[float]
    message: str


@dataclass
class AccountInfo:
    """Represents brokerage account information."""
    account_id: str
    cash_balance: float
    buying_power: float
    portfolio_value: float


@dataclass
class Position:
    """Represents a stock position."""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float = 0.0
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        return (self.current_price - self.entry_price) * self.quantity
    
    def update_price(self, price: float) -> None:
        """Update current price of the position."""
        self.current_price = price


class BrokerageAdapter(ABC):
    """Abstract base class for brokerage adapters."""
    
    @abstractmethod
    def authenticate(self, credentials: Dict) -> bool:
        """
        Authenticate with the brokerage API.
        
        Args:
            credentials: Dictionary containing API keys/tokens
            
        Returns:
            True if authentication successful
            
        Raises:
            BrokerageError: If authentication fails
        """
        pass
    
    @abstractmethod
    def submit_order(self, order) -> str:
        """
        Submit an order to the brokerage.
        
        Args:
            order: Order object to submit
            
        Returns:
            Order ID assigned by the brokerage
            
        Raises:
            OrderError: If order submission fails
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get the status of an order.
        
        Args:
            order_id: ID of the order to check
            
        Returns:
            OrderStatus object
            
        Raises:
            BrokerageError: If status cannot be retrieved
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """
        Get account information.
        
        Returns:
            AccountInfo object with balance and portfolio value
            
        Raises:
            BrokerageError: If account info cannot be retrieved
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get current positions.
        
        Returns:
            List of Position objects
            
        Raises:
            BrokerageError: If positions cannot be retrieved
        """
        pass
