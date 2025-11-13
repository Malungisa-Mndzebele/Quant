"""Robinhood brokerage adapter implementation."""
import logging
from typing import Dict, List, Optional
from datetime import datetime

from src.brokers.base import BrokerageAdapter, AccountInfo, Position, OrderStatus
from src.models.order import Order
from src.brokers.errors import BrokerageError, AuthenticationError, OrderError

logger = logging.getLogger(__name__)


class RobinhoodAdapter(BrokerageAdapter):
    """
    Robinhood brokerage integration adapter.
    
    Uses robin_stocks library for Robinhood API access.
    Supports stock trading with market and limit orders.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize Robinhood adapter.
        
        Args:
            initial_capital: Initial capital (not used in live mode)
        """
        super().__init__()
        self.initial_capital = initial_capital
        self.authenticated = False
        self.account_number = None
        self.rh = None  # robin_stocks instance
        
        logger.info("Initialized RobinhoodAdapter")
    
    def authenticate(self, credentials: Dict) -> bool:
        """
        Authenticate with Robinhood.
        
        Args:
            credentials: Dictionary containing:
                - username: Robinhood username/email
                - password: Robinhood password
                - mfa_code: (Optional) MFA code if 2FA is enabled
        
        Returns:
            True if authentication successful
            
        Raises:
            AuthenticationError: If authentication fails
        """
        try:
            import robin_stocks.robinhood as rh
            self.rh = rh
            
            username = credentials.get('username') or credentials.get('api_key')
            password = credentials.get('password') or credentials.get('api_secret')
            mfa_code = credentials.get('mfa_code')
            
            if not username or not password:
                raise AuthenticationError("Username and password are required")
            
            logger.info(f"Authenticating with Robinhood as {username}")
            
            # Login to Robinhood
            if mfa_code:
                login_result = rh.login(username, password, mfa_code=mfa_code)
            else:
                login_result = rh.login(username, password)
            
            if not login_result:
                raise AuthenticationError("Robinhood authentication failed")
            
            # Get account number
            account_info = rh.profiles.load_account_profile()
            self.account_number = account_info.get('account_number')
            
            self.authenticated = True
            logger.info(f"Successfully authenticated with Robinhood - Account: {self.account_number}")
            
            return True
            
        except ImportError:
            error_msg = "robin_stocks library not installed. Install with: pip install robin-stocks"
            logger.error(error_msg)
            raise BrokerageError(error_msg)
        except Exception as e:
            error_msg = f"Robinhood authentication failed: {str(e)}"
            logger.error(error_msg)
            raise AuthenticationError(error_msg) from e
    
    def submit_order(self, order: Order) -> str:
        """
        Submit order to Robinhood.
        
        Args:
            order: Order object to submit
            
        Returns:
            Order ID from Robinhood
            
        Raises:
            OrderError: If order submission fails
        """
        if not self.authenticated:
            raise BrokerageError("Not authenticated with Robinhood")
        
        try:
            logger.info(f"Submitting order to Robinhood: {order}")
            
            # Prepare order parameters
            symbol = order.symbol
            quantity = order.quantity
            side = order.action.lower()  # 'buy' or 'sell'
            
            # Submit based on order type
            if order.order_type == "MARKET":
                if side == 'buy':
                    result = self.rh.orders.order_buy_market(symbol, quantity)
                else:
                    result = self.rh.orders.order_sell_market(symbol, quantity)
            
            elif order.order_type == "LIMIT":
                if not order.limit_price:
                    raise OrderError("Limit price required for limit orders")
                
                if side == 'buy':
                    result = self.rh.orders.order_buy_limit(
                        symbol, 
                        quantity, 
                        order.limit_price
                    )
                else:
                    result = self.rh.orders.order_sell_limit(
                        symbol, 
                        quantity, 
                        order.limit_price
                    )
            
            elif order.order_type == "STOP_LOSS":
                if not order.limit_price:
                    raise OrderError("Stop price required for stop loss orders")
                
                result = self.rh.orders.order_sell_stop_loss(
                    symbol,
                    quantity,
                    order.limit_price
                )
            
            else:
                raise OrderError(f"Unsupported order type: {order.order_type}")
            
            # Extract order ID
            if result and 'id' in result:
                order_id = result['id']
                logger.info(f"Order submitted successfully - Robinhood Order ID: {order_id}")
                return order_id
            else:
                raise OrderError(f"Failed to submit order: {result}")
                
        except Exception as e:
            error_msg = f"Failed to submit order to Robinhood: {str(e)}"
            logger.error(error_msg)
            raise OrderError(error_msg) from e
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get order status from Robinhood.
        
        Args:
            order_id: Robinhood order ID
            
        Returns:
            OrderStatus object
        """
        if not self.authenticated:
            raise BrokerageError("Not authenticated with Robinhood")
        
        try:
            order_info = self.rh.orders.get_stock_order_info(order_id)
            
            if not order_info:
                return OrderStatus(
                    order_id=order_id,
                    status="UNKNOWN",
                    filled_quantity=0,
                    average_fill_price=None,
                    message="Order not found"
                )
            
            # Map Robinhood status to our status
            rh_state = order_info.get('state', '').lower()
            status_map = {
                'filled': 'FILLED',
                'confirmed': 'PENDING',
                'queued': 'PENDING',
                'partially_filled': 'PARTIALLY_FILLED',
                'cancelled': 'CANCELLED',
                'rejected': 'REJECTED',
                'failed': 'REJECTED'
            }
            
            status = status_map.get(rh_state, 'UNKNOWN')
            filled_qty = float(order_info.get('cumulative_quantity', 0))
            avg_price = float(order_info.get('average_price', 0)) if order_info.get('average_price') else None
            
            return OrderStatus(
                order_id=order_id,
                status=status,
                filled_quantity=int(filled_qty),
                average_fill_price=avg_price,
                message=rh_state
            )
            
        except Exception as e:
            logger.error(f"Failed to get order status from Robinhood: {e}")
            return OrderStatus(
                order_id=order_id,
                status="UNKNOWN",
                filled_quantity=0,
                average_fill_price=None,
                message=str(e)
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order on Robinhood.
        
        Args:
            order_id: Robinhood order ID
            
        Returns:
            True if cancellation successful
        """
        if not self.authenticated:
            raise BrokerageError("Not authenticated with Robinhood")
        
        try:
            result = self.rh.orders.cancel_stock_order(order_id)
            success = result is not None
            
            if success:
                logger.info(f"Successfully cancelled order {order_id}")
            else:
                logger.warning(f"Failed to cancel order {order_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_account_info(self) -> AccountInfo:
        """
        Get account information from Robinhood.
        
        Returns:
            AccountInfo object with account details
        """
        if not self.authenticated:
            raise BrokerageError("Not authenticated with Robinhood")
        
        try:
            # Get portfolio data
            portfolio = self.rh.profiles.load_portfolio_profile()
            
            # Extract values
            equity = float(portfolio.get('equity', 0))
            cash = float(portfolio.get('withdrawable_amount', 0))
            buying_power = float(portfolio.get('buying_power', 0))
            
            return AccountInfo(
                account_id=self.account_number or "RH-Account",
                cash_balance=cash,
                buying_power=buying_power,
                portfolio_value=equity
            )
            
        except Exception as e:
            logger.error(f"Failed to get account info from Robinhood: {e}")
            raise BrokerageError(f"Failed to get account info: {str(e)}") from e
    
    def get_positions(self) -> List[Position]:
        """
        Get current positions from Robinhood.
        
        Returns:
            List of Position objects
        """
        if not self.authenticated:
            raise BrokerageError("Not authenticated with Robinhood")
        
        try:
            positions = []
            rh_positions = self.rh.account.get_open_stock_positions()
            
            for pos in rh_positions:
                quantity = float(pos.get('quantity', 0))
                
                if quantity > 0:
                    # Get instrument details for symbol
                    instrument_url = pos.get('instrument')
                    instrument = self.rh.stocks.get_instrument_by_url(instrument_url)
                    symbol = instrument.get('symbol', 'UNKNOWN')
                    
                    avg_price = float(pos.get('average_buy_price', 0))
                    
                    # Get current price
                    quote = self.rh.stocks.get_latest_price(symbol)
                    current_price = float(quote[0]) if quote else avg_price
                    
                    position = Position(
                        symbol=symbol,
                        quantity=int(quantity),
                        average_price=avg_price,
                        current_price=current_price,
                        market_value=current_price * quantity,
                        unrealized_pnl=(current_price - avg_price) * quantity,
                        unrealized_pnl_pct=((current_price - avg_price) / avg_price) * 100 if avg_price > 0 else 0
                    )
                    
                    positions.append(position)
            
            logger.debug(f"Retrieved {len(positions)} positions from Robinhood")
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions from Robinhood: {e}")
            return []
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote from Robinhood.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with quote data
        """
        if not self.authenticated:
            raise BrokerageError("Not authenticated with Robinhood")
        
        try:
            quote = self.rh.stocks.get_quotes(symbol)
            
            if quote and len(quote) > 0:
                q = quote[0]
                return {
                    'symbol': symbol,
                    'price': float(q.get('last_trade_price', 0)),
                    'bid': float(q.get('bid_price', 0)),
                    'ask': float(q.get('ask_price', 0)),
                    'volume': int(float(q.get('volume', 0))),
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol} from Robinhood: {e}")
            return None
    
    def disconnect(self):
        """Disconnect from Robinhood."""
        try:
            if self.authenticated and self.rh:
                self.rh.authentication.logout()
                logger.info("Logged out from Robinhood")
        except Exception as e:
            logger.error(f"Error during Robinhood logout: {e}")
        finally:
            self.authenticated = False
            self.account_number = None
