"""Simulated brokerage adapter for paper trading."""

import logging
from typing import Dict, List
from datetime import datetime
import uuid

from .base import BrokerageAdapter, AccountInfo, OrderStatus as BrokerOrderStatus, Position
from .errors import BrokerageError, OrderError, AuthenticationError
from .credentials import ConnectionStatus
from ..models.order import Order, OrderStatusEnum, OrderAction


logger = logging.getLogger(__name__)


class SimulatedBrokerageAdapter(BrokerageAdapter):
    """
    Simulated brokerage adapter for paper trading.
    
    This adapter simulates order execution using current market prices
    without connecting to a real brokerage. It maintains a simulated
    account with cash balance and positions.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize simulated brokerage adapter.
        
        Args:
            initial_capital: Starting cash balance for simulated account
        """
        self.mode = "simulation"
        self.connection_status = ConnectionStatus()
        self.account_id = f"SIM-{uuid.uuid4().hex[:8]}"
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.current_prices: Dict[str, float] = {}
        
        logger.info(f"[{self.mode.upper()}] Initialized SimulatedBrokerageAdapter with ${initial_capital:,.2f}")
    
    def authenticate(self, credentials: Dict) -> bool:
        """
        Authenticate with simulated brokerage (always succeeds).
        
        Args:
            credentials: Dictionary containing credentials (not used in simulation)
            
        Returns:
            True (always succeeds for simulation)
        """
        try:
            logger.info(f"[{self.mode.upper()}] Authenticating simulated brokerage")
            self.connection_status.set_connected(authenticated=True)
            
            # Retrieve and store account info at startup
            account_info = self.get_account_info()
            self.connection_status.set_account_info({
                'account_id': account_info.account_id,
                'cash_balance': account_info.cash_balance,
                'portfolio_value': account_info.portfolio_value
            })
            
            logger.info(f"[{self.mode.upper()}] Authentication successful - Account ID: {self.account_id}")
            return True
        except Exception as e:
            error_msg = f"Authentication failed: {str(e)}"
            logger.error(f"[{self.mode.upper()}] {error_msg}")
            self.connection_status.set_disconnected(error_msg)
            raise AuthenticationError(error_msg)
    
    def submit_order(self, order: Order) -> str:
        """
        Submit an order to the simulated brokerage.
        
        Args:
            order: Order object to submit
            
        Returns:
            Order ID
            
        Raises:
            AuthenticationError: If not authenticated
            OrderError: If order validation fails or insufficient funds
        """
        if not self.connection_status.is_authenticated:
            raise AuthenticationError("Must authenticate before submitting orders")
        
        logger.info(f"[{self.mode.upper()}] Submitting order: {order.action.value} {order.quantity} {order.symbol} @ {order.order_type.value}")
        
        # Validate order
        if order.quantity <= 0:
            raise OrderError(f"Invalid quantity: {order.quantity}")
        
        # Get current price for the symbol
        if order.symbol not in self.current_prices:
            raise OrderError(f"No price data available for {order.symbol}")
        
        current_price = self.current_prices[order.symbol]
        
        # Simulate order fill based on order type
        fill_price = self._calculate_fill_price(order, current_price)
        
        # Check if we have sufficient funds/shares
        if order.action == OrderAction.BUY:
            required_cash = fill_price * order.quantity
            if required_cash > self.cash_balance:
                raise OrderError(
                    f"Insufficient funds: need ${required_cash:,.2f}, have ${self.cash_balance:,.2f}"
                )
        elif order.action == OrderAction.SELL:
            position = self.positions.get(order.symbol)
            if not position or position.quantity < order.quantity:
                available = position.quantity if position else 0
                raise OrderError(
                    f"Insufficient shares: need {order.quantity}, have {available}"
                )
        
        # Execute the simulated fill
        self._execute_fill(order, fill_price)
        
        # Store order
        self.orders[order.order_id] = order
        
        logger.info(
            f"[{self.mode.upper()}] Order filled: {order.order_id} - "
            f"{order.action.value} {order.quantity} {order.symbol} @ ${fill_price:.2f}"
        )
        
        return order.order_id
    
    def _calculate_fill_price(self, order: Order, current_price: float) -> float:
        """
        Calculate the fill price based on order type.
        
        Args:
            order: Order to calculate fill price for
            current_price: Current market price
            
        Returns:
            Fill price
        """
        if order.order_type.value == "MARKET":
            return current_price
        elif order.order_type.value == "LIMIT":
            # For simulation, assume limit orders fill at limit price if favorable
            if order.action == OrderAction.BUY and order.limit_price >= current_price:
                return order.limit_price
            elif order.action == OrderAction.SELL and order.limit_price <= current_price:
                return order.limit_price
            else:
                raise OrderError(f"Limit order not fillable at current price")
        elif order.order_type.value == "STOP_LOSS":
            # For simulation, stop loss fills at current price
            return current_price
        else:
            return current_price
    
    def _execute_fill(self, order: Order, fill_price: float) -> None:
        """
        Execute the simulated order fill and update account state.
        
        Args:
            order: Order to fill
            fill_price: Price at which to fill the order
        """
        order.status = OrderStatusEnum.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        
        if order.action == OrderAction.BUY:
            # Deduct cash
            cost = fill_price * order.quantity
            self.cash_balance -= cost
            
            # Add or update position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                # Calculate weighted average entry price
                total_quantity = pos.quantity + order.quantity
                total_cost = (pos.entry_price * pos.quantity) + (fill_price * order.quantity)
                new_entry_price = total_cost / total_quantity
                pos.quantity = total_quantity
                pos.entry_price = new_entry_price
                pos.current_price = fill_price
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=fill_price,
                    current_price=fill_price
                )
        
        elif order.action == OrderAction.SELL:
            # Add cash
            proceeds = fill_price * order.quantity
            self.cash_balance += proceeds
            
            # Update or remove position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                pos.quantity -= order.quantity
                if pos.quantity == 0:
                    del self.positions[order.symbol]
    
    def get_order_status(self, order_id: str) -> BrokerOrderStatus:
        """
        Get the status of an order.
        
        Args:
            order_id: ID of the order to check
            
        Returns:
            OrderStatus object
            
        Raises:
            BrokerageError: If order not found
        """
        if not self.connection_status.is_authenticated:
            raise AuthenticationError("Must authenticate before checking order status")
        
        if order_id not in self.orders:
            raise BrokerageError(f"Order not found: {order_id}")
        
        order = self.orders[order_id]
        
        return BrokerOrderStatus(
            order_id=order.order_id,
            status=order.status.value,
            filled_quantity=order.filled_quantity,
            average_fill_price=order.filled_price,
            message=f"Simulated order {order.status.value.lower()}"
        )
    
    def get_account_info(self) -> AccountInfo:
        """
        Get simulated account information.
        
        Returns:
            AccountInfo object with balance and portfolio value
            
        Raises:
            AuthenticationError: If not authenticated
        """
        if not self.connection_status.is_authenticated:
            raise AuthenticationError("Must authenticate before retrieving account info")
        
        # Calculate portfolio value
        positions_value = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        portfolio_value = self.cash_balance + positions_value
        
        logger.debug(
            f"[{self.mode.upper()}] Account info: "
            f"Cash=${self.cash_balance:,.2f}, "
            f"Positions=${positions_value:,.2f}, "
            f"Total=${portfolio_value:,.2f}"
        )
        
        return AccountInfo(
            account_id=self.account_id,
            cash_balance=self.cash_balance,
            buying_power=self.cash_balance,  # In simulation, buying power = cash
            portfolio_value=portfolio_value
        )
    
    def get_positions(self) -> List[Position]:
        """
        Get current simulated positions.
        
        Returns:
            List of Position objects
            
        Raises:
            AuthenticationError: If not authenticated
        """
        if not self.connection_status.is_authenticated:
            raise AuthenticationError("Must authenticate before retrieving positions")
        
        logger.debug(f"[{self.mode.upper()}] Retrieved {len(self.positions)} positions")
        return list(self.positions.values())
    
    def update_market_price(self, symbol: str, price: float) -> None:
        """
        Update the current market price for a symbol.
        
        This method is used to provide price data for order execution.
        
        Args:
            symbol: Stock symbol
            price: Current market price
        """
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        
        self.current_prices[symbol] = price
        
        # Update position prices
        if symbol in self.positions:
            self.positions[symbol].update_price(price)
    
    def update_market_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current market prices for multiple symbols.
        
        Args:
            prices: Dictionary mapping symbols to prices
        """
        for symbol, price in prices.items():
            self.update_market_price(symbol, price)
    
    def reset(self) -> None:
        """Reset the simulated account to initial state."""
        logger.info(f"[{self.mode.upper()}] Resetting simulated account")
        self.cash_balance = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.current_prices.clear()
