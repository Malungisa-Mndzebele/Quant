"""Paper trading service for simulated trading with virtual money."""

import logging
import json
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from services.market_data_service import MarketDataService
from services.trading_service import Order, Position, OrderStatus

logger = logging.getLogger(__name__)


@dataclass
class PaperAccount:
    """Paper trading account information"""
    account_id: str
    initial_capital: float
    cash: float
    portfolio_value: float
    equity: float
    buying_power: float
    created_at: datetime
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        return data


@dataclass
class PaperPosition:
    """Paper trading position"""
    symbol: str
    quantity: int
    side: str  # 'long' or 'short'
    entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_pl_pct: float
    entry_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat()
        return data
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.side == 'long'
    
    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.side == 'short'


class PaperTradingService:
    """
    Service for paper trading with virtual money.
    
    Provides a complete simulation of trading using real-time market prices
    without risking actual capital. Maintains separate virtual portfolio,
    tracks performance, and provides clear visual distinction from live trading.
    """
    
    def __init__(
        self,
        db_path: str = "data/database/paper_trading.db",
        market_data_service: Optional[MarketDataService] = None,
        initial_capital: float = 100000.0
    ):
        """
        Initialize paper trading service.
        
        Args:
            db_path: Path to SQLite database for paper trading
            market_data_service: Market data service for real-time prices
            initial_capital: Starting capital for paper trading account
        """
        self.db_path = db_path
        self.market_data_service = market_data_service
        self.initial_capital = initial_capital
        
        # Ensure database directory exists
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Load or create account
        self.account = self._load_or_create_account()
        
        logger.info(
            f"Initialized paper trading service: "
            f"Account ID={self.account.account_id}, "
            f"Capital=${self.account.cash:.2f}"
        )
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create account table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_account (
                account_id TEXT PRIMARY KEY,
                initial_capital REAL NOT NULL,
                cash REAL NOT NULL,
                portfolio_value REAL NOT NULL,
                equity REAL NOT NULL,
                buying_power REAL NOT NULL,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)
        
        # Create positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                UNIQUE(symbol)
            )
        """)
        
        # Create orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                status TEXT NOT NULL,
                filled_qty INTEGER NOT NULL,
                filled_avg_price REAL,
                limit_price REAL,
                submitted_at TEXT NOT NULL,
                filled_at TEXT
            )
        """)
        
        # Create transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_transactions (
                transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                total_value REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        
        # Create performance snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                portfolio_value REAL NOT NULL,
                cash REAL NOT NULL,
                equity REAL NOT NULL,
                positions_value REAL NOT NULL,
                num_positions INTEGER NOT NULL,
                total_return REAL NOT NULL,
                total_return_pct REAL NOT NULL
            )
        """)
        
        # Create indices
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_orders_symbol 
            ON paper_orders(symbol)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_orders_status 
            ON paper_orders(status)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_transactions_symbol 
            ON paper_transactions(symbol)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_snapshots_timestamp 
            ON paper_snapshots(timestamp)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("Paper trading database initialized")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _load_or_create_account(self) -> PaperAccount:
        """Load existing account or create new one"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM paper_account LIMIT 1")
        row = cursor.fetchone()
        
        if row:
            # Load existing account
            account = PaperAccount(
                account_id=row['account_id'],
                initial_capital=row['initial_capital'],
                cash=row['cash'],
                portfolio_value=row['portfolio_value'],
                equity=row['equity'],
                buying_power=row['buying_power'],
                created_at=datetime.fromisoformat(row['created_at']),
                last_updated=datetime.fromisoformat(row['last_updated'])
            )
            logger.info(f"Loaded existing paper trading account: {account.account_id}")
        else:
            # Create new account
            account_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            now = datetime.now()
            
            account = PaperAccount(
                account_id=account_id,
                initial_capital=self.initial_capital,
                cash=self.initial_capital,
                portfolio_value=self.initial_capital,
                equity=self.initial_capital,
                buying_power=self.initial_capital * 2,  # 2x leverage
                created_at=now,
                last_updated=now
            )
            
            cursor.execute("""
                INSERT INTO paper_account 
                (account_id, initial_capital, cash, portfolio_value, equity, 
                 buying_power, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                account.account_id,
                account.initial_capital,
                account.cash,
                account.portfolio_value,
                account.equity,
                account.buying_power,
                account.created_at.isoformat(),
                account.last_updated.isoformat()
            ))
            
            conn.commit()
            logger.info(f"Created new paper trading account: {account_id}")
        
        conn.close()
        return account
    
    def _update_account(self):
        """Update account in database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        self.account.last_updated = datetime.now()
        
        cursor.execute("""
            UPDATE paper_account 
            SET cash = ?, portfolio_value = ?, equity = ?, 
                buying_power = ?, last_updated = ?
            WHERE account_id = ?
        """, (
            self.account.cash,
            self.account.portfolio_value,
            self.account.equity,
            self.account.buying_power,
            self.account.last_updated.isoformat(),
            self.account.account_id
        ))
        
        conn.commit()
        conn.close()
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price
            
        Raises:
            ValueError: If market data service is not configured
            Exception: If price fetch fails
        """
        if not self.market_data_service:
            raise ValueError("Market data service is required for paper trading")
        
        try:
            quote = self.market_data_service.get_latest_quote(symbol)
            # Use mid price between bid and ask
            price = (quote.bid_price + quote.ask_price) / 2
            return price
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            raise
    
    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = 'market',
        limit_price: Optional[float] = None
    ) -> Order:
        """
        Place a paper trading order.
        
        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            limit_price: Limit price (required for limit orders)
            
        Returns:
            Order object
            
        Raises:
            ValueError: If parameters are invalid or insufficient funds
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
        
        symbol = symbol.upper()
        
        # Get current price
        current_price = self._get_current_price(symbol)
        
        # For market orders, use current price
        # For limit orders, check if limit price is acceptable
        execution_price = current_price
        
        if order_type == 'limit':
            if side == 'buy' and limit_price < current_price:
                # Limit buy below market - order pending
                status = OrderStatus.PENDING
                execution_price = None
            elif side == 'sell' and limit_price > current_price:
                # Limit sell above market - order pending
                status = OrderStatus.PENDING
                execution_price = None
            else:
                # Limit price is acceptable - fill immediately
                status = OrderStatus.FILLED
                execution_price = limit_price
        else:
            # Market order - fill immediately
            status = OrderStatus.FILLED
        
        # Check if we have sufficient funds/shares
        if status == OrderStatus.FILLED:
            if side == 'buy':
                required_cash = qty * execution_price
                if required_cash > self.account.cash:
                    raise ValueError(
                        f"Insufficient funds: Required ${required_cash:.2f}, "
                        f"Available ${self.account.cash:.2f}"
                    )
            else:  # sell
                # Check if we have the position
                position = self._get_position(symbol)
                if not position or position.quantity < qty:
                    available = position.quantity if position else 0
                    raise ValueError(
                        f"Insufficient shares: Required {qty}, Available {available}"
                    )
        
        # Create order
        order_id = f"paper_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        now = datetime.now()
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            quantity=qty,
            side=side,
            order_type=order_type,
            status=status,
            filled_qty=qty if status == OrderStatus.FILLED else 0,
            filled_avg_price=execution_price,
            limit_price=limit_price,
            submitted_at=now,
            filled_at=now if status == OrderStatus.FILLED else None
        )
        
        # Save order to database
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO paper_orders 
            (order_id, symbol, quantity, side, order_type, status, 
             filled_qty, filled_avg_price, limit_price, submitted_at, filled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order.order_id,
            order.symbol,
            order.quantity,
            order.side,
            order.order_type,
            order.status.value,
            order.filled_qty,
            order.filled_avg_price,
            order.limit_price,
            order.submitted_at.isoformat(),
            order.filled_at.isoformat() if order.filled_at else None
        ))
        
        conn.commit()
        conn.close()
        
        # If filled, execute the trade
        if status == OrderStatus.FILLED:
            self._execute_trade(order)
        
        price_str = f"${execution_price:.2f}" if execution_price is not None else "pending"
        logger.info(
            f"Paper order placed: {order_id} - {side} {qty} {symbol} "
            f"@ {price_str} (status: {status.value})"
        )
        
        return order
    
    def _execute_trade(self, order: Order):
        """
        Execute a filled order and update positions/account.
        
        Args:
            order: Filled order to execute
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        symbol = order.symbol
        qty = order.filled_qty
        price = order.filled_avg_price
        side = order.side
        
        # Record transaction
        total_value = qty * price
        
        cursor.execute("""
            INSERT INTO paper_transactions 
            (order_id, symbol, quantity, side, price, total_value, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            order.order_id,
            symbol,
            qty,
            side,
            price,
            total_value,
            datetime.now().isoformat()
        ))
        
        # Update position
        if side == 'buy':
            # Add to position
            cursor.execute("""
                SELECT * FROM paper_positions WHERE symbol = ?
            """, (symbol,))
            
            row = cursor.fetchone()
            
            if row:
                # Update existing position
                existing_qty = row['quantity']
                existing_price = row['entry_price']
                
                # Calculate new average price
                new_qty = existing_qty + qty
                new_avg_price = ((existing_qty * existing_price) + (qty * price)) / new_qty
                
                cursor.execute("""
                    UPDATE paper_positions 
                    SET quantity = ?, entry_price = ?
                    WHERE symbol = ?
                """, (new_qty, new_avg_price, symbol))
            else:
                # Create new position
                cursor.execute("""
                    INSERT INTO paper_positions 
                    (symbol, quantity, side, entry_price, entry_time)
                    VALUES (?, ?, ?, ?, ?)
                """, (symbol, qty, 'long', price, datetime.now().isoformat()))
            
            # Decrease cash
            self.account.cash -= total_value
        
        else:  # sell
            # Reduce position
            cursor.execute("""
                SELECT * FROM paper_positions WHERE symbol = ?
            """, (symbol,))
            
            row = cursor.fetchone()
            
            if row:
                existing_qty = row['quantity']
                new_qty = existing_qty - qty
                
                if new_qty > 0:
                    # Update position
                    cursor.execute("""
                        UPDATE paper_positions 
                        SET quantity = ?
                        WHERE symbol = ?
                    """, (new_qty, symbol))
                else:
                    # Close position
                    cursor.execute("""
                        DELETE FROM paper_positions WHERE symbol = ?
                    """, (symbol,))
            
            # Increase cash
            self.account.cash += total_value
        
        conn.commit()
        conn.close()
        
        # Update account values
        self._recalculate_account()
        
        logger.info(
            f"Trade executed: {side} {qty} {symbol} @ ${price:.2f}, "
            f"Cash: ${self.account.cash:.2f}"
        )
    
    def _get_position(self, symbol: str) -> Optional[PaperPosition]:
        """Get position for a symbol"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM paper_positions WHERE symbol = ?
        """, (symbol.upper(),))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Get current price
        try:
            current_price = self._get_current_price(symbol)
        except:
            current_price = row['entry_price']
        
        quantity = row['quantity']
        entry_price = row['entry_price']
        cost_basis = quantity * entry_price
        market_value = quantity * current_price
        unrealized_pl = market_value - cost_basis
        unrealized_pl_pct = (unrealized_pl / cost_basis * 100) if cost_basis > 0 else 0.0
        
        return PaperPosition(
            symbol=row['symbol'],
            quantity=quantity,
            side=row['side'],
            entry_price=entry_price,
            current_price=current_price,
            market_value=market_value,
            cost_basis=cost_basis,
            unrealized_pl=unrealized_pl,
            unrealized_pl_pct=unrealized_pl_pct,
            entry_time=datetime.fromisoformat(row['entry_time'])
        )
    
    def _recalculate_account(self):
        """Recalculate account values based on current positions"""
        positions = self.get_positions()
        
        positions_value = sum(pos.market_value for pos in positions)
        self.account.portfolio_value = self.account.cash + positions_value
        self.account.equity = self.account.portfolio_value
        self.account.buying_power = self.account.cash * 2  # 2x leverage
        
        self._update_account()
    
    def get_positions(self) -> List[PaperPosition]:
        """
        Get all open paper trading positions.
        
        Returns:
            List of PaperPosition objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT symbol FROM paper_positions")
        rows = cursor.fetchall()
        conn.close()
        
        positions = []
        for row in rows:
            position = self._get_position(row['symbol'])
            if position:
                positions.append(position)
        
        return positions
    
    def get_account(self) -> Dict[str, Any]:
        """
        Get paper trading account information.
        
        Returns:
            Dictionary with account details
        """
        # Recalculate to ensure current values
        self._recalculate_account()
        
        return {
            'account_id': self.account.account_id,
            'cash': self.account.cash,
            'portfolio_value': self.account.portfolio_value,
            'buying_power': self.account.buying_power,
            'equity': self.account.equity,
            'initial_capital': self.account.initial_capital,
            'total_return': self.account.portfolio_value - self.account.initial_capital,
            'total_return_pct': ((self.account.portfolio_value - self.account.initial_capital) / 
                                self.account.initial_capital * 100),
            'created_at': self.account.created_at.isoformat(),
            'last_updated': self.account.last_updated.isoformat(),
            'mode': 'PAPER TRADING'
        }
    
    def get_order_status(self, order_id: str) -> Order:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order object
            
        Raises:
            ValueError: If order not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM paper_orders WHERE order_id = ?
        """, (order_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise ValueError(f"Order not found: {order_id}")
        
        return Order(
            order_id=row['order_id'],
            symbol=row['symbol'],
            quantity=row['quantity'],
            side=row['side'],
            order_type=row['order_type'],
            status=OrderStatus(row['status']),
            filled_qty=row['filled_qty'],
            filled_avg_price=row['filled_avg_price'],
            limit_price=row['limit_price'],
            submitted_at=datetime.fromisoformat(row['submitted_at']),
            filled_at=datetime.fromisoformat(row['filled_at']) if row['filled_at'] else None
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful
            
        Raises:
            ValueError: If order not found or cannot be cancelled
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT status FROM paper_orders WHERE order_id = ?
        """, (order_id,))
        
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            raise ValueError(f"Order not found: {order_id}")
        
        status = OrderStatus(row['status'])
        
        if status not in [OrderStatus.PENDING, OrderStatus.NEW]:
            conn.close()
            raise ValueError(f"Cannot cancel order in status: {status.value}")
        
        cursor.execute("""
            UPDATE paper_orders 
            SET status = ?
            WHERE order_id = ?
        """, (OrderStatus.CANCELLED.value, order_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Paper order cancelled: {order_id}")
        
        return True
    
    def close_position(self, symbol: str) -> Order:
        """
        Close an open position.
        
        Args:
            symbol: Symbol to close
            
        Returns:
            Order object for closing order
            
        Raises:
            ValueError: If position not found
        """
        position = self._get_position(symbol)
        
        if not position:
            raise ValueError(f"No position found for {symbol}")
        
        # Place sell order for entire position
        return self.place_order(
            symbol=symbol,
            qty=position.quantity,
            side='sell',
            order_type='market'
        )
    
    def save_session(self, session_name: Optional[str] = None) -> str:
        """
        Save current paper trading session.
        
        Args:
            session_name: Optional name for the session
            
        Returns:
            Session file path
        """
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_dir = Path('data/paper_trading_sessions')
        session_dir.mkdir(parents=True, exist_ok=True)
        
        session_file = session_dir / f"{session_name}.json"
        
        # Gather session data
        session_data = {
            'session_name': session_name,
            'saved_at': datetime.now().isoformat(),
            'account': self.get_account(),
            'positions': [pos.to_dict() for pos in self.get_positions()],
            'database_path': str(self.db_path)
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Paper trading session saved: {session_file}")
        
        return str(session_file)
    
    def reset_account(self, initial_capital: Optional[float] = None):
        """
        Reset paper trading account to initial state.
        
        Args:
            initial_capital: New initial capital (optional, uses current if not specified)
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Clear all positions
        cursor.execute("DELETE FROM paper_positions")
        
        # Clear all orders
        cursor.execute("DELETE FROM paper_orders")
        
        # Clear all transactions
        cursor.execute("DELETE FROM paper_transactions")
        
        # Clear all snapshots
        cursor.execute("DELETE FROM paper_snapshots")
        
        # Reset account
        now = datetime.now()
        self.account = PaperAccount(
            account_id=self.account.account_id,
            initial_capital=self.initial_capital,
            cash=self.initial_capital,
            portfolio_value=self.initial_capital,
            equity=self.initial_capital,
            buying_power=self.initial_capital * 2,
            created_at=self.account.created_at,
            last_updated=now
        )
        
        cursor.execute("""
            UPDATE paper_account 
            SET initial_capital = ?, cash = ?, portfolio_value = ?, 
                equity = ?, buying_power = ?, last_updated = ?
            WHERE account_id = ?
        """, (
            self.account.initial_capital,
            self.account.cash,
            self.account.portfolio_value,
            self.account.equity,
            self.account.buying_power,
            self.account.last_updated.isoformat(),
            self.account.account_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Paper trading account reset: ${self.initial_capital:.2f}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for paper trading session.
        
        Returns:
            Dictionary with performance metrics
        """
        account = self.get_account()
        positions = self.get_positions()
        
        total_return = account['total_return']
        total_return_pct = account['total_return_pct']
        
        # Calculate position metrics
        num_positions = len(positions)
        total_unrealized_pl = sum(pos.unrealized_pl for pos in positions)
        
        winning_positions = [pos for pos in positions if pos.unrealized_pl > 0]
        losing_positions = [pos for pos in positions if pos.unrealized_pl < 0]
        
        return {
            'account_id': account['account_id'],
            'initial_capital': account['initial_capital'],
            'current_value': account['portfolio_value'],
            'cash': account['cash'],
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'num_positions': num_positions,
            'total_unrealized_pl': total_unrealized_pl,
            'winning_positions': len(winning_positions),
            'losing_positions': len(losing_positions),
            'mode': 'PAPER TRADING',
            'created_at': account['created_at'],
            'last_updated': account['last_updated']
        }
    
    def is_paper_trading(self) -> bool:
        """
        Check if this is paper trading (always True).
        
        Returns:
            True
        """
        return True
    
    def get_trading_mode(self) -> str:
        """
        Get trading mode.
        
        Returns:
            'paper'
        """
        return 'paper'
