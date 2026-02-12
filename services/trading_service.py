"""Trading service for executing trades through Alpaca Trading API."""

import logging
import time
from datetime import datetime, time as dt_time
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
    ClosePositionRequest
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderType,
    OrderStatus as AlpacaOrderStatus,
    QueryOrderStatus
)

from config.settings import settings

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    NEW = "new"
    ACCEPTED = "accepted"


class AssetClass(str, Enum):
    """Asset class enumeration"""
    STOCKS = "stocks"
    CRYPTO = "crypto"
    FOREX = "forex"
    OPTIONS = "options"


@dataclass
class TradingSchedule:
    """
    Trading schedule configuration for automated trading.
    
    Defines when automated trading is allowed to execute trades.
    """
    # Days of week (0=Monday, 6=Sunday)
    active_days: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4})  # Mon-Fri
    
    # Trading hours (24-hour format)
    start_time: dt_time = field(default_factory=lambda: dt_time(9, 30))  # 9:30 AM
    end_time: dt_time = field(default_factory=lambda: dt_time(16, 0))    # 4:00 PM
    
    # Asset class this schedule applies to
    asset_class: AssetClass = AssetClass.STOCKS
    
    # Timezone (default: US/Eastern for stock market)
    timezone: str = "US/Eastern"
    
    # Whether schedule is enabled
    enabled: bool = True
    
    def is_trading_allowed(self, check_time: Optional[datetime] = None) -> bool:
        """
        Check if trading is allowed at the given time.
        
        Args:
            check_time: Time to check (defaults to current time)
            
        Returns:
            True if trading is allowed, False otherwise
        """
        if not self.enabled:
            return False
        
        if check_time is None:
            check_time = datetime.now()
        
        # Check day of week
        if check_time.weekday() not in self.active_days:
            return False
        
        # Check time of day
        current_time = check_time.time()
        
        # Handle schedules that cross midnight
        if self.start_time <= self.end_time:
            # Normal schedule (e.g., 9:30 AM - 4:00 PM)
            return self.start_time <= current_time <= self.end_time
        else:
            # Schedule crosses midnight (e.g., 11:00 PM - 2:00 AM)
            return current_time >= self.start_time or current_time <= self.end_time
    
    def get_next_trading_window(self, from_time: Optional[datetime] = None) -> Optional[datetime]:
        """
        Get the next time when trading will be allowed.
        
        Args:
            from_time: Starting time (defaults to current time)
            
        Returns:
            Next trading window start time, or None if schedule is disabled
        """
        if not self.enabled:
            return None
        
        if from_time is None:
            from_time = datetime.now()
        
        # If currently in trading window, return current time
        if self.is_trading_allowed(from_time):
            return from_time
        
        # Find next trading day
        current_date = from_time.date()
        for days_ahead in range(1, 8):  # Check next 7 days
            next_date = current_date + timedelta(days=days_ahead)
            next_datetime = datetime.combine(next_date, self.start_time)
            
            if next_datetime.weekday() in self.active_days:
                return next_datetime
        
        return None
    
    def validate(self) -> List[str]:
        """
        Validate schedule configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate days
        if not self.active_days:
            errors.append("At least one active day must be specified")
        
        for day in self.active_days:
            if not 0 <= day <= 6:
                errors.append(f"Invalid day of week: {day} (must be 0-6)")
        
        # Validate times
        if self.start_time == self.end_time:
            errors.append("Start time and end time cannot be the same")
        
        # Validate asset class
        if self.asset_class not in AssetClass:
            errors.append(f"Invalid asset class: {self.asset_class}")
        
        return errors
    
    def get_schedule_description(self) -> str:
        """
        Get human-readable description of the schedule.
        
        Returns:
            Schedule description string
        """
        if not self.enabled:
            return "Schedule disabled"
        
        # Day names
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        active_day_names = [day_names[d] for d in sorted(self.active_days)]
        days_str = ", ".join(active_day_names)
        
        # Time range
        time_str = f"{self.start_time.strftime('%I:%M %p')} - {self.end_time.strftime('%I:%M %p')}"
        
        return f"{self.asset_class.value.title()}: {days_str}, {time_str} ({self.timezone})"


# Import timedelta for schedule calculations
from datetime import timedelta


@dataclass
class Order:
    """Trading order information"""
    order_id: str
    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market' or 'limit'
    status: OrderStatus
    filled_qty: int
    filled_avg_price: Optional[float]
    limit_price: Optional[float]
    submitted_at: datetime
    filled_at: Optional[datetime]
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active (not terminal state)"""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.NEW,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED
        ]


@dataclass
class Position:
    """Open position information"""
    symbol: str
    quantity: int
    side: str  # 'long' or 'short'
    entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_pl_pct: float
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.side == 'long'
    
    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.side == 'short'


class TradingService:
    """
    Service for executing trades through Alpaca Trading API.
    
    Provides order placement, status tracking, position management,
    support for both paper and live trading modes, and trading schedule
    management for automated trading.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True
    ):
        """
        Initialize trading service.
        
        Args:
            api_key: Alpaca API key (defaults to settings)
            api_secret: Alpaca API secret (defaults to settings)
            paper: Whether to use paper trading mode (default: True)
        """
        self.api_key = api_key or settings.alpaca.api_key
        self.api_secret = api_secret or settings.alpaca.secret_key
        self.paper = paper
        
        # Validate credentials
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials are required")
        
        # Initialize trading client
        try:
            self.client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.paper
            )
            logger.info(
                f"Initialized Alpaca Trading API "
                f"({'PAPER' if self.paper else 'LIVE'} trading mode)"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca Trading API: {e}")
            raise
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.2  # 200ms between requests
        
        # Retry configuration
        self._max_retries = 3
        self._retry_delay = 1.0  # seconds
        
        # Trading schedules by asset class
        self._schedules: Dict[AssetClass, TradingSchedule] = {
            AssetClass.STOCKS: TradingSchedule(
                active_days={0, 1, 2, 3, 4},  # Mon-Fri
                start_time=dt_time(9, 30),
                end_time=dt_time(16, 0),
                asset_class=AssetClass.STOCKS,
                timezone="US/Eastern",
                enabled=True
            ),
            AssetClass.CRYPTO: TradingSchedule(
                active_days={0, 1, 2, 3, 4, 5, 6},  # 7 days
                start_time=dt_time(0, 0),
                end_time=dt_time(23, 59),
                asset_class=AssetClass.CRYPTO,
                timezone="UTC",
                enabled=True
            ),
            AssetClass.FOREX: TradingSchedule(
                active_days={0, 1, 2, 3, 4},  # Mon-Fri
                start_time=dt_time(0, 0),
                end_time=dt_time(23, 59),
                asset_class=AssetClass.FOREX,
                timezone="UTC",
                enabled=True
            )
        }
        
        # Automated trading mode
        self._automated_trading_enabled = False
    
    def _rate_limit(self):
        """Enforce rate limiting between API requests"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _retry_on_failure(self, func, *args, **kwargs):
        """
        Retry a function call with exponential backoff on failure.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self._max_retries):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"API request failed (attempt {attempt + 1}/{self._max_retries}): {e}"
                )
                
                if attempt < self._max_retries - 1:
                    # Exponential backoff
                    delay = self._retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
        
        # All retries failed
        logger.error(f"All retry attempts failed: {last_exception}")
        raise last_exception
    
    def _map_alpaca_status(self, alpaca_status: str) -> OrderStatus:
        """
        Map Alpaca order status to our OrderStatus enum.
        
        Args:
            alpaca_status: Alpaca order status string
            
        Returns:
            OrderStatus enum value
        """
        status_map = {
            'new': OrderStatus.NEW,
            'accepted': OrderStatus.ACCEPTED,
            'pending_new': OrderStatus.PENDING,
            'accepted_for_bidding': OrderStatus.ACCEPTED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.EXPIRED,
            'replaced': OrderStatus.CANCELLED,
            'pending_cancel': OrderStatus.PENDING,
            'pending_replace': OrderStatus.PENDING,
            'rejected': OrderStatus.REJECTED,
            'suspended': OrderStatus.REJECTED,
            'calculated': OrderStatus.ACCEPTED,
        }
        
        return status_map.get(alpaca_status.lower(), OrderStatus.PENDING)
    
    def _create_order_from_alpaca(self, alpaca_order) -> Order:
        """
        Convert Alpaca order object to our Order dataclass.
        
        Args:
            alpaca_order: Alpaca order object
            
        Returns:
            Order dataclass instance
        """
        return Order(
            order_id=alpaca_order.id,
            symbol=alpaca_order.symbol,
            quantity=int(alpaca_order.qty),
            side=alpaca_order.side.value,
            order_type=alpaca_order.type.value,
            status=self._map_alpaca_status(alpaca_order.status),
            filled_qty=int(alpaca_order.filled_qty or 0),
            filled_avg_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
            limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
            submitted_at=alpaca_order.submitted_at,
            filled_at=alpaca_order.filled_at
        )
    
    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        time_in_force: str = 'day'
    ) -> Order:
        """
        Place a trade order.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            qty: Number of shares to trade
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market' or 'limit')
            limit_price: Limit price (required for limit orders)
            time_in_force: Time in force ('day', 'gtc', 'ioc', 'fok')
            
        Returns:
            Order object with order details
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If order submission fails
        """
        # Validate inputs
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        if qty <= 0:
            raise ValueError("Quantity must be positive")
        
        side = side.lower()
        if side not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
        
        order_type = order_type.lower()
        if order_type not in ['market', 'limit']:
            raise ValueError("Order type must be 'market' or 'limit'")
        
        if order_type == 'limit' and limit_price is None:
            raise ValueError("Limit price is required for limit orders")
        
        if order_type == 'limit' and limit_price <= 0:
            raise ValueError("Limit price must be positive")
        
        time_in_force = time_in_force.lower()
        tif_map = {
            'day': TimeInForce.DAY,
            'gtc': TimeInForce.GTC,
            'ioc': TimeInForce.IOC,
            'fok': TimeInForce.FOK
        }
        
        if time_in_force not in tif_map:
            raise ValueError(f"Invalid time_in_force: {time_in_force}")
        
        symbol = symbol.upper()
        
        # Log order details
        logger.info(
            f"Placing {order_type} {side} order: "
            f"{qty} shares of {symbol}"
            f"{f' @ ${limit_price}' if limit_price else ''} "
            f"({'PAPER' if self.paper else 'LIVE'})"
        )
        
        # Create order request
        def submit_order():
            order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
            tif = tif_map[time_in_force]
            
            if order_type == 'market':
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif
                )
            else:  # limit order
                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=limit_price
                )
            
            # Submit order
            alpaca_order = self.client.submit_order(request)
            
            return self._create_order_from_alpaca(alpaca_order)
        
        try:
            order = self._retry_on_failure(submit_order)
            
            logger.info(
                f"Order submitted successfully: {order.order_id} "
                f"(status: {order.status.value})"
            )
            
            return order
        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {e}")
            raise
    
    def get_order_status(self, order_id: str) -> Order:
        """
        Get the status of an order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order object with current status
            
        Raises:
            ValueError: If order_id is invalid
            Exception: If API request fails
        """
        if not order_id or not isinstance(order_id, str):
            raise ValueError("Order ID must be a non-empty string")
        
        logger.info(f"Fetching status for order {order_id}")
        
        def fetch_order():
            alpaca_order = self.client.get_order_by_id(order_id)
            return self._create_order_from_alpaca(alpaca_order)
        
        try:
            order = self._retry_on_failure(fetch_order)
            
            logger.info(
                f"Order {order_id} status: {order.status.value} "
                f"(filled: {order.filled_qty}/{order.quantity})"
            )
            
            return order
        except Exception as e:
            logger.error(f"Failed to fetch order status for {order_id}: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation was successful
            
        Raises:
            ValueError: If order_id is invalid
            Exception: If cancellation fails
        """
        if not order_id or not isinstance(order_id, str):
            raise ValueError("Order ID must be a non-empty string")
        
        logger.info(f"Cancelling order {order_id}")
        
        def cancel():
            self.client.cancel_order_by_id(order_id)
            return True
        
        try:
            result = self._retry_on_failure(cancel)
            logger.info(f"Order {order_id} cancelled successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise
    
    def get_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of Position objects
            
        Raises:
            Exception: If API request fails
        """
        logger.info("Fetching open positions")
        
        def fetch_positions():
            alpaca_positions = self.client.get_all_positions()
            
            positions = []
            for pos in alpaca_positions:
                position = Position(
                    symbol=pos.symbol,
                    quantity=int(pos.qty),
                    side='long' if int(pos.qty) > 0 else 'short',
                    entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    cost_basis=float(pos.cost_basis),
                    unrealized_pl=float(pos.unrealized_pl),
                    unrealized_pl_pct=float(pos.unrealized_plpc)
                )
                positions.append(position)
            
            return positions
        
        try:
            positions = self._retry_on_failure(fetch_positions)
            
            logger.info(f"Fetched {len(positions)} open positions")
            
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            raise
    
    def close_position(
        self,
        symbol: str,
        qty: Optional[int] = None,
        percentage: Optional[float] = None
    ) -> Order:
        """
        Close an open position.
        
        Args:
            symbol: Stock symbol to close
            qty: Number of shares to close (optional, closes all if not specified)
            percentage: Percentage of position to close (optional, 0-100)
            
        Returns:
            Order object for the closing order
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If position close fails
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        if qty is not None and qty <= 0:
            raise ValueError("Quantity must be positive")
        
        if percentage is not None and (percentage <= 0 or percentage > 100):
            raise ValueError("Percentage must be between 0 and 100")
        
        if qty is not None and percentage is not None:
            raise ValueError("Cannot specify both qty and percentage")
        
        symbol = symbol.upper()
        
        logger.info(
            f"Closing position for {symbol} "
            f"({f'{qty} shares' if qty else f'{percentage}%' if percentage else 'all shares'})"
        )
        
        def close():
            # Build close request
            if qty is not None:
                request = ClosePositionRequest(qty=str(qty))
                alpaca_order = self.client.close_position(symbol, request)
            elif percentage is not None:
                request = ClosePositionRequest(percentage=str(percentage))
                alpaca_order = self.client.close_position(symbol, request)
            else:
                # Close entire position
                alpaca_order = self.client.close_position(symbol)
            
            return self._create_order_from_alpaca(alpaca_order)
        
        try:
            order = self._retry_on_failure(close)
            
            logger.info(
                f"Position close order submitted: {order.order_id} "
                f"(status: {order.status.value})"
            )
            
            return order
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            raise
    
    def get_account(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account details including:
            - cash: Available cash
            - portfolio_value: Total portfolio value
            - buying_power: Available buying power
            - equity: Total equity
            - last_equity: Previous day's equity
            
        Raises:
            Exception: If API request fails
        """
        logger.info("Fetching account information")
        
        def fetch_account():
            account = self.client.get_account()
            
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'daytrade_count': int(account.daytrade_count),
                'daytrading_buying_power': float(account.daytrading_buying_power),
                'regt_buying_power': float(account.regt_buying_power),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'account_number': account.account_number,
                'status': account.status.value
            }
        
        try:
            account_info = self._retry_on_failure(fetch_account)
            
            logger.info(
                f"Account info: Portfolio value = ${account_info['portfolio_value']:.2f}, "
                f"Cash = ${account_info['cash']:.2f}"
            )
            
            return account_info
        except Exception as e:
            logger.error(f"Failed to fetch account information: {e}")
            raise
    
    def get_orders(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        symbols: Optional[List[str]] = None
    ) -> List[Order]:
        """
        Get orders with optional filtering.
        
        Args:
            status: Filter by status ('open', 'closed', 'all')
            limit: Maximum number of orders to return
            symbols: Filter by symbols
            
        Returns:
            List of Order objects
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If API request fails
        """
        if limit <= 0:
            raise ValueError("Limit must be positive")
        
        if status and status not in ['open', 'closed', 'all']:
            raise ValueError("Status must be 'open', 'closed', or 'all'")
        
        logger.info(f"Fetching orders (status={status}, limit={limit})")
        
        def fetch_orders():
            # Map status to Alpaca enum
            query_status = None
            if status == 'open':
                query_status = QueryOrderStatus.OPEN
            elif status == 'closed':
                query_status = QueryOrderStatus.CLOSED
            elif status == 'all':
                query_status = QueryOrderStatus.ALL
            
            # Build request
            request_params = {
                'limit': limit
            }
            
            if query_status:
                request_params['status'] = query_status
            
            if symbols:
                request_params['symbols'] = [s.upper() for s in symbols]
            
            request = GetOrdersRequest(**request_params)
            
            alpaca_orders = self.client.get_orders(request)
            
            orders = [
                self._create_order_from_alpaca(order)
                for order in alpaca_orders
            ]
            
            return orders
        
        try:
            orders = self._retry_on_failure(fetch_orders)
            
            logger.info(f"Fetched {len(orders)} orders")
            
            return orders
        except Exception as e:
            logger.error(f"Failed to fetch orders: {e}")
            raise
    
    def is_paper_trading(self) -> bool:
        """
        Check if currently in paper trading mode.
        
        Returns:
            True if paper trading, False if live trading
        """
        return self.paper
    
    def get_trading_mode(self) -> str:
        """
        Get current trading mode as string.
        
        Returns:
            'paper' or 'live'
        """
        return 'paper' if self.paper else 'live'
    
    # ========== Trading Schedule Management ==========
    
    def set_trading_schedule(
        self,
        asset_class: AssetClass,
        schedule: TradingSchedule
    ) -> None:
        """
        Set trading schedule for an asset class.
        
        Args:
            asset_class: Asset class to configure
            schedule: Trading schedule configuration
            
        Raises:
            ValueError: If schedule is invalid
        """
        # Validate schedule
        errors = schedule.validate()
        if errors:
            error_msg = "; ".join(errors)
            raise ValueError(f"Invalid trading schedule: {error_msg}")
        
        # Ensure asset class matches
        if schedule.asset_class != asset_class:
            raise ValueError(
                f"Schedule asset class ({schedule.asset_class}) "
                f"does not match specified asset class ({asset_class})"
            )
        
        self._schedules[asset_class] = schedule
        
        logger.info(
            f"Updated trading schedule for {asset_class.value}: "
            f"{schedule.get_schedule_description()}"
        )
    
    def get_trading_schedule(self, asset_class: AssetClass) -> TradingSchedule:
        """
        Get trading schedule for an asset class.
        
        Args:
            asset_class: Asset class to query
            
        Returns:
            Trading schedule configuration
        """
        return self._schedules.get(asset_class, TradingSchedule(asset_class=asset_class))
    
    def get_all_schedules(self) -> Dict[AssetClass, TradingSchedule]:
        """
        Get all trading schedules.
        
        Returns:
            Dictionary mapping asset classes to schedules
        """
        return self._schedules.copy()
    
    def is_trading_allowed(
        self,
        asset_class: AssetClass = AssetClass.STOCKS,
        check_time: Optional[datetime] = None
    ) -> bool:
        """
        Check if trading is allowed for an asset class at the given time.
        
        Args:
            asset_class: Asset class to check
            check_time: Time to check (defaults to current time)
            
        Returns:
            True if trading is allowed, False otherwise
        """
        schedule = self._schedules.get(asset_class)
        
        if schedule is None:
            # No schedule configured, allow trading
            logger.warning(
                f"No schedule configured for {asset_class.value}, "
                f"allowing trading by default"
            )
            return True
        
        return schedule.is_trading_allowed(check_time)
    
    def enable_automated_trading(self) -> None:
        """
        Enable automated trading mode.
        
        When enabled, schedule enforcement is active and trades
        will only be executed during configured trading hours.
        """
        self._automated_trading_enabled = True
        logger.info("Automated trading enabled - schedule enforcement active")
    
    def disable_automated_trading(self) -> None:
        """
        Disable automated trading mode.
        
        When disabled, schedule enforcement is not active.
        """
        self._automated_trading_enabled = False
        logger.info("Automated trading disabled")
    
    def is_automated_trading_enabled(self) -> bool:
        """
        Check if automated trading is enabled.
        
        Returns:
            True if automated trading is enabled
        """
        return self._automated_trading_enabled
    
    def validate_schedule_against_market_hours(
        self,
        asset_class: AssetClass
    ) -> List[str]:
        """
        Validate trading schedule against typical market hours.
        
        Provides warnings if schedule is outside normal market hours.
        
        Args:
            asset_class: Asset class to validate
            
        Returns:
            List of warning messages (empty if no issues)
        """
        schedule = self._schedules.get(asset_class)
        
        if schedule is None:
            return [f"No schedule configured for {asset_class.value}"]
        
        warnings = []
        
        # Check against typical market hours
        if asset_class == AssetClass.STOCKS:
            # US stock market: Mon-Fri, 9:30 AM - 4:00 PM ET
            market_start = dt_time(9, 30)
            market_end = dt_time(16, 0)
            market_days = {0, 1, 2, 3, 4}
            
            # Check if schedule extends beyond market hours
            if schedule.start_time < market_start:
                warnings.append(
                    f"Schedule starts before market open "
                    f"({schedule.start_time.strftime('%I:%M %p')} < "
                    f"{market_start.strftime('%I:%M %p')})"
                )
            
            if schedule.end_time > market_end:
                warnings.append(
                    f"Schedule ends after market close "
                    f"({schedule.end_time.strftime('%I:%M %p')} > "
                    f"{market_end.strftime('%I:%M %p')})"
                )
            
            # Check if trading on weekends
            weekend_days = schedule.active_days - market_days
            if weekend_days:
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                weekend_names = [day_names[d] for d in sorted(weekend_days)]
                warnings.append(
                    f"Schedule includes non-trading days: {', '.join(weekend_names)}"
                )
        
        elif asset_class == AssetClass.CRYPTO:
            # Crypto markets are 24/7, no warnings needed
            pass
        
        elif asset_class == AssetClass.FOREX:
            # Forex markets are 24/5 (Mon-Fri)
            if 5 in schedule.active_days or 6 in schedule.active_days:
                warnings.append(
                    "Forex markets are typically closed on weekends"
                )
        
        return warnings
    
    def place_order_with_schedule_check(
        self,
        symbol: str,
        qty: int,
        side: str,
        asset_class: AssetClass = AssetClass.STOCKS,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        time_in_force: str = 'day',
        override_schedule: bool = False
    ) -> Order:
        """
        Place a trade order with schedule enforcement.
        
        This method checks trading schedules before placing orders
        when automated trading is enabled.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            qty: Number of shares to trade
            side: Order side ('buy' or 'sell')
            asset_class: Asset class of the symbol
            order_type: Order type ('market' or 'limit')
            limit_price: Limit price (required for limit orders)
            time_in_force: Time in force ('day', 'gtc', 'ioc', 'fok')
            override_schedule: If True, bypass schedule check (for manual trades)
            
        Returns:
            Order object with order details
            
        Raises:
            ValueError: If parameters are invalid or trading not allowed
            Exception: If order submission fails
        """
        # Check schedule if automated trading is enabled and not overridden
        if self._automated_trading_enabled and not override_schedule:
            if not self.is_trading_allowed(asset_class):
                schedule = self._schedules.get(asset_class)
                next_window = schedule.get_next_trading_window() if schedule else None
                
                error_msg = (
                    f"Trading not allowed for {asset_class.value} at this time. "
                    f"Current schedule: {schedule.get_schedule_description() if schedule else 'None'}"
                )
                
                if next_window:
                    error_msg += f" Next trading window: {next_window.strftime('%Y-%m-%d %I:%M %p')}"
                
                logger.warning(error_msg)
                raise ValueError(error_msg)
        
        # Place order using existing method
        return self.place_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force
        )
    
    def get_schedule_status(self) -> Dict[str, Any]:
        """
        Get current schedule status for all asset classes.
        
        Returns:
            Dictionary with schedule status information
        """
        status = {
            'automated_trading_enabled': self._automated_trading_enabled,
            'current_time': datetime.now().isoformat(),
            'schedules': {}
        }
        
        for asset_class, schedule in self._schedules.items():
            is_allowed = schedule.is_trading_allowed()
            next_window = schedule.get_next_trading_window()
            
            status['schedules'][asset_class.value] = {
                'enabled': schedule.enabled,
                'description': schedule.get_schedule_description(),
                'trading_allowed_now': is_allowed,
                'next_trading_window': next_window.isoformat() if next_window else None,
                'active_days': sorted(list(schedule.active_days)),
                'start_time': schedule.start_time.strftime('%H:%M'),
                'end_time': schedule.end_time.strftime('%H:%M'),
                'timezone': schedule.timezone
            }
        
        return status
