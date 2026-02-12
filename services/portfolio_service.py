"""Portfolio service for managing portfolio state and performance tracking."""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np

from services.trading_service import TradingService, Position

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """Transaction record"""
    transaction_id: int
    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    price: float
    total_value: float
    commission: float
    timestamp: datetime
    order_id: str
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PerformanceMetrics:
    """Portfolio performance statistics"""
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    current_streak: int
    longest_win_streak: int
    longest_loss_streak: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class PortfolioService:
    """
    Service for managing portfolio state and performance tracking.
    
    Provides portfolio value calculation, position tracking, performance metrics,
    transaction history, and tax reporting functionality.
    """
    
    def __init__(
        self,
        db_path: str = "data/database/portfolio.db",
        trading_service: Optional[TradingService] = None
    ):
        """
        Initialize portfolio service.
        
        Args:
            db_path: Path to SQLite database file
            trading_service: Optional TradingService instance for fetching positions
        """
        self.db_path = db_path
        self.trading_service = trading_service
        
        # Ensure database directory exists
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Initialized portfolio service with database: {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                total_value REAL NOT NULL,
                commission REAL DEFAULT 0.0,
                timestamp TEXT NOT NULL,
                order_id TEXT NOT NULL,
                notes TEXT
            )
        """)
        
        # Create portfolio_snapshots table for historical tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                portfolio_value REAL NOT NULL,
                cash REAL NOT NULL,
                equity REAL NOT NULL,
                positions_value REAL NOT NULL,
                num_positions INTEGER NOT NULL
            )
        """)
        
        # Create indices for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_symbol 
            ON transactions(symbol)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_timestamp 
            ON transactions(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp 
            ON portfolio_snapshots(timestamp)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("Database initialized successfully")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def record_transaction(
        self,
        symbol: str,
        quantity: int,
        side: str,
        price: float,
        order_id: str,
        commission: float = 0.0,
        notes: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> int:
        """
        Record a transaction in the database.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
            price: Execution price per share
            order_id: Order ID from broker
            commission: Commission paid
            notes: Optional notes
            timestamp: Transaction timestamp (defaults to now)
            
        Returns:
            Transaction ID
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not symbol:
            raise ValueError("Symbol is required")
        
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if side not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
        
        if price <= 0:
            raise ValueError("Price must be positive")
        
        if commission < 0:
            raise ValueError("Commission cannot be negative")
        
        if not order_id:
            raise ValueError("Order ID is required")
        
        timestamp = timestamp or datetime.now()
        total_value = quantity * price
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO transactions 
            (symbol, quantity, side, price, total_value, commission, timestamp, order_id, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol.upper(),
            quantity,
            side,
            price,
            total_value,
            commission,
            timestamp.isoformat(),
            order_id,
            notes
        ))
        
        transaction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(
            f"Recorded transaction {transaction_id}: "
            f"{side} {quantity} {symbol} @ ${price:.2f}"
        )
        
        return transaction_id
    
    def get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value.
        
        Returns:
            Total portfolio value (cash + positions)
            
        Raises:
            ValueError: If trading service is not configured
            Exception: If calculation fails
        """
        if not self.trading_service:
            raise ValueError("Trading service is required to calculate portfolio value")
        
        try:
            # Get account info for cash and equity
            account = self.trading_service.get_account()
            portfolio_value = account['portfolio_value']
            
            logger.info(f"Portfolio value: ${portfolio_value:.2f}")
            
            return portfolio_value
        except Exception as e:
            logger.error(f"Failed to calculate portfolio value: {e}")
            raise
    
    def get_positions(self) -> List[Position]:
        """
        Get all current positions from broker.
        
        Returns:
            List of Position objects
            
        Raises:
            ValueError: If trading service is not configured
            Exception: If fetching positions fails
        """
        if not self.trading_service:
            raise ValueError("Trading service is required to fetch positions")
        
        try:
            positions = self.trading_service.get_positions()
            
            logger.info(f"Fetched {len(positions)} positions")
            
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            raise
    
    def get_performance_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PerformanceMetrics:
        """
        Calculate portfolio performance metrics.
        
        Args:
            start_date: Start date for calculation (optional)
            end_date: End date for calculation (optional)
            
        Returns:
            PerformanceMetrics object
            
        Raises:
            Exception: If calculation fails
        """
        try:
            # Get transactions for the period
            transactions = self.get_transaction_history(start_date, end_date)
            
            if not transactions:
                # No transactions, return zero metrics
                return PerformanceMetrics(
                    total_return=0.0,
                    total_return_pct=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    max_drawdown_pct=0.0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    largest_win=0.0,
                    largest_loss=0.0,
                    current_streak=0,
                    longest_win_streak=0,
                    longest_loss_streak=0
                )
            
            # Calculate trade-level P&L
            trades = self._calculate_trade_pnl(transactions)
            
            if not trades:
                # No completed trades
                return PerformanceMetrics(
                    total_return=0.0,
                    total_return_pct=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    max_drawdown_pct=0.0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    largest_win=0.0,
                    largest_loss=0.0,
                    current_streak=0,
                    longest_win_streak=0,
                    longest_loss_streak=0
                )
            
            # Calculate metrics
            total_return = sum(trade['pnl'] for trade in trades)
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            total_trades = len(trades)
            num_winning = len(winning_trades)
            num_losing = len(losing_trades)
            
            win_rate = num_winning / total_trades if total_trades > 0 else 0.0
            
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0.0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0.0
            
            largest_win = max([t['pnl'] for t in winning_trades]) if winning_trades else 0.0
            largest_loss = min([t['pnl'] for t in losing_trades]) if losing_trades else 0.0
            
            # Profit factor
            gross_profit = sum(t['pnl'] for t in winning_trades)
            gross_loss = abs(sum(t['pnl'] for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            
            # Calculate equity curve for drawdown and Sharpe
            equity_curve = self._calculate_equity_curve(trades)
            
            # Max drawdown
            max_drawdown, max_drawdown_pct = self._calculate_max_drawdown(equity_curve)
            
            # Sharpe ratio (assuming daily returns)
            sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
            
            # Calculate streaks
            current_streak, longest_win_streak, longest_loss_streak = self._calculate_streaks(trades)
            
            # Calculate return percentage (relative to initial capital)
            initial_capital = equity_curve[0] if len(equity_curve) > 0 else 0.0
            total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0.0
            
            metrics = PerformanceMetrics(
                total_return=total_return,
                total_return_pct=total_return_pct,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_pct=max_drawdown_pct,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                winning_trades=num_winning,
                losing_trades=num_losing,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                current_streak=current_streak,
                longest_win_streak=longest_win_streak,
                longest_loss_streak=longest_loss_streak
            )
            
            logger.info(
                f"Performance metrics: Return={total_return:.2f} ({total_return_pct:.2f}%), "
                f"Sharpe={sharpe_ratio:.2f}, Win Rate={win_rate:.2%}"
            )
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            raise
    
    def _calculate_trade_pnl(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """
        Calculate P&L for completed trades using FIFO matching.
        
        Args:
            transactions: List of transactions
            
        Returns:
            List of trade dictionaries with P&L
        """
        # Group by symbol
        by_symbol = {}
        for txn in transactions:
            if txn.symbol not in by_symbol:
                by_symbol[txn.symbol] = []
            by_symbol[txn.symbol].append(txn)
        
        trades = []
        
        # Match buys and sells for each symbol
        for symbol, txns in by_symbol.items():
            # Sort by timestamp
            txns.sort(key=lambda t: t.timestamp)
            
            # FIFO queue of open positions
            open_positions = []
            
            for txn in txns:
                if txn.side == 'buy':
                    # Add to open positions
                    open_positions.append({
                        'quantity': txn.quantity,
                        'price': txn.price,
                        'timestamp': txn.timestamp,
                        'commission': txn.commission
                    })
                else:  # sell
                    # Match against open positions
                    remaining_qty = txn.quantity
                    sell_price = txn.price
                    sell_commission = txn.commission
                    
                    while remaining_qty > 0 and open_positions:
                        pos = open_positions[0]
                        
                        # Determine quantity to close
                        close_qty = min(remaining_qty, pos['quantity'])
                        
                        # Calculate P&L
                        buy_cost = close_qty * pos['price']
                        sell_proceeds = close_qty * sell_price
                        commission = pos['commission'] * (close_qty / pos['quantity']) + \
                                   sell_commission * (close_qty / txn.quantity)
                        
                        pnl = sell_proceeds - buy_cost - commission
                        
                        # Record trade
                        trades.append({
                            'symbol': symbol,
                            'quantity': close_qty,
                            'entry_price': pos['price'],
                            'exit_price': sell_price,
                            'entry_time': pos['timestamp'],
                            'exit_time': txn.timestamp,
                            'pnl': pnl,
                            'pnl_pct': (pnl / buy_cost * 100) if buy_cost > 0 else 0.0
                        })
                        
                        # Update quantities
                        remaining_qty -= close_qty
                        pos['quantity'] -= close_qty
                        
                        # Remove position if fully closed
                        if pos['quantity'] == 0:
                            open_positions.pop(0)
        
        return trades
    
    def _calculate_equity_curve(self, trades: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calculate equity curve from trades.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Numpy array of cumulative equity
        """
        if not trades:
            return np.array([0.0])
        
        # Sort by exit time
        sorted_trades = sorted(trades, key=lambda t: t['exit_time'])
        
        # Calculate cumulative P&L
        pnls = [t['pnl'] for t in sorted_trades]
        equity_curve = np.cumsum([0.0] + pnls)
        
        return equity_curve
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, float]:
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve: Array of cumulative equity
            
        Returns:
            Tuple of (max_drawdown_dollars, max_drawdown_percent)
        """
        if len(equity_curve) == 0:
            return 0.0, 0.0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdown
        drawdown = equity_curve - running_max
        
        # Max drawdown in dollars
        max_dd = abs(np.min(drawdown))
        
        # Max drawdown in percent
        max_dd_pct = 0.0
        if len(running_max) > 0:
            max_equity = np.max(running_max)
            if max_equity > 0:
                max_dd_pct = (max_dd / max_equity) * 100
        
        return max_dd, max_dd_pct
    
    def _calculate_sharpe_ratio(
        self,
        equity_curve: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe ratio from equity curve.
        
        Args:
            equity_curve: Array of cumulative equity
            risk_free_rate: Annual risk-free rate (default: 2%)
            
        Returns:
            Sharpe ratio
        """
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate returns
        returns = np.diff(equity_curve)
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate mean and std of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming daily returns)
        # Sharpe = (mean_return - risk_free_rate) / std_return * sqrt(252)
        daily_rf = risk_free_rate / 252
        sharpe = (mean_return - daily_rf) / std_return * np.sqrt(252)
        
        return sharpe
    
    def _calculate_streaks(self, trades: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        """
        Calculate win/loss streaks.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Tuple of (current_streak, longest_win_streak, longest_loss_streak)
        """
        if not trades:
            return 0, 0, 0
        
        # Sort by exit time
        sorted_trades = sorted(trades, key=lambda t: t['exit_time'])
        
        current_streak = 0
        longest_win_streak = 0
        longest_loss_streak = 0
        
        current_win_streak = 0
        current_loss_streak = 0
        
        for trade in sorted_trades:
            if trade['pnl'] > 0:
                current_win_streak += 1
                current_loss_streak = 0
                longest_win_streak = max(longest_win_streak, current_win_streak)
            elif trade['pnl'] < 0:
                current_loss_streak += 1
                current_win_streak = 0
                longest_loss_streak = max(longest_loss_streak, current_loss_streak)
        
        # Current streak is the last active streak
        if current_win_streak > 0:
            current_streak = current_win_streak
        elif current_loss_streak > 0:
            current_streak = -current_loss_streak
        
        return current_streak, longest_win_streak, longest_loss_streak
    
    def get_transaction_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None
    ) -> List[Transaction]:
        """
        Retrieve transaction history with optional filtering.
        
        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            symbol: Filter by symbol (optional)
            
        Returns:
            List of Transaction objects
            
        Raises:
            Exception: If query fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM transactions WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol.upper())
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            transactions = []
            for row in rows:
                txn = Transaction(
                    transaction_id=row['transaction_id'],
                    symbol=row['symbol'],
                    quantity=row['quantity'],
                    side=row['side'],
                    price=row['price'],
                    total_value=row['total_value'],
                    commission=row['commission'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    order_id=row['order_id'],
                    notes=row['notes']
                )
                transactions.append(txn)
            
            conn.close()
            
            logger.info(
                f"Retrieved {len(transactions)} transactions "
                f"(start={start_date}, end={end_date}, symbol={symbol})"
            )
            
            return transactions
        except Exception as e:
            logger.error(f"Failed to retrieve transaction history: {e}")
            raise
    
    def export_for_taxes(self, year: int) -> pd.DataFrame:
        """
        Export trades for tax reporting.
        
        Args:
            year: Tax year to export
            
        Returns:
            DataFrame with tax-relevant trade information
            
        Raises:
            ValueError: If year is invalid
            Exception: If export fails
        """
        if year < 2000 or year > datetime.now().year:
            raise ValueError(f"Invalid year: {year}")
        
        try:
            # Get transactions for the year
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31, 23, 59, 59)
            
            transactions = self.get_transaction_history(start_date, end_date)
            
            if not transactions:
                logger.info(f"No transactions found for year {year}")
                return pd.DataFrame()
            
            # Calculate trade-level P&L
            trades = self._calculate_trade_pnl(transactions)
            
            if not trades:
                logger.info(f"No completed trades found for year {year}")
                return pd.DataFrame()
            
            # Filter trades that closed in the tax year
            year_trades = [
                t for t in trades
                if t['exit_time'].year == year
            ]
            
            if not year_trades:
                logger.info(f"No trades closed in year {year}")
                return pd.DataFrame()
            
            # Create DataFrame with tax-relevant columns
            df = pd.DataFrame(year_trades)
            
            # Format for tax reporting
            tax_df = pd.DataFrame({
                'Symbol': df['symbol'],
                'Quantity': df['quantity'],
                'Date Acquired': df['entry_time'].apply(lambda x: x.strftime('%m/%d/%Y')),
                'Date Sold': df['exit_time'].apply(lambda x: x.strftime('%m/%d/%Y')),
                'Purchase Price': df['entry_price'].apply(lambda x: f"${x:.2f}"),
                'Sale Price': df['exit_price'].apply(lambda x: f"${x:.2f}"),
                'Cost Basis': (df['quantity'] * df['entry_price']).apply(lambda x: f"${x:.2f}"),
                'Proceeds': (df['quantity'] * df['exit_price']).apply(lambda x: f"${x:.2f}"),
                'Gain/Loss': df['pnl'].apply(lambda x: f"${x:.2f}"),
                'Gain/Loss %': df['pnl_pct'].apply(lambda x: f"{x:.2f}%"),
                'Holding Period (Days)': (df['exit_time'] - df['entry_time']).apply(lambda x: x.days),
                'Term': (df['exit_time'] - df['entry_time']).apply(
                    lambda x: 'Long-term' if x.days > 365 else 'Short-term'
                )
            })
            
            logger.info(f"Exported {len(tax_df)} trades for tax year {year}")
            
            return tax_df
        except Exception as e:
            logger.error(f"Failed to export tax data for year {year}: {e}")
            raise
    
    def save_portfolio_snapshot(self) -> int:
        """
        Save current portfolio state as a snapshot.
        
        Returns:
            Snapshot ID
            
        Raises:
            ValueError: If trading service is not configured
            Exception: If save fails
        """
        if not self.trading_service:
            raise ValueError("Trading service is required to save snapshot")
        
        try:
            # Get current portfolio state
            account = self.trading_service.get_account()
            positions = self.trading_service.get_positions()
            
            portfolio_value = account['portfolio_value']
            cash = account['cash']
            equity = account['equity']
            positions_value = sum(pos.market_value for pos in positions)
            num_positions = len(positions)
            
            # Save to database
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO portfolio_snapshots
                (timestamp, portfolio_value, cash, equity, positions_value, num_positions)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                portfolio_value,
                cash,
                equity,
                positions_value,
                num_positions
            ))
            
            snapshot_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(
                f"Saved portfolio snapshot {snapshot_id}: "
                f"Value=${portfolio_value:.2f}, Positions={num_positions}"
            )
            
            return snapshot_id
        except Exception as e:
            logger.error(f"Failed to save portfolio snapshot: {e}")
            raise
    
    def get_portfolio_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical portfolio snapshots.
        
        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            
        Returns:
            DataFrame with portfolio history
            
        Raises:
            Exception: If query fails
        """
        try:
            conn = self._get_connection()
            
            # Build query
            query = "SELECT * FROM portfolio_snapshots WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp ASC"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Retrieved {len(df)} portfolio snapshots")
            
            return df
        except Exception as e:
            logger.error(f"Failed to retrieve portfolio history: {e}")
            raise

    def get_asset_class_allocation(self) -> Dict[str, float]:
        """
        Get portfolio allocation by asset class.
        
        Returns:
            Dictionary mapping asset class names to allocation percentages
            
        Raises:
            ValueError: If trading service is not configured
            Exception: If calculation fails
        """
        if not self.trading_service:
            raise ValueError("Trading service is required to calculate allocation")
        
        try:
            from services.market_data_service import AssetClass, MarketDataService
            from utils.asset_analysis import calculate_portfolio_allocation
            
            # Get current positions
            positions = self.trading_service.get_positions()
            account = self.trading_service.get_account()
            
            if not positions:
                logger.info("No positions found, returning empty allocation")
                return {
                    AssetClass.STOCK.value: 0.0,
                    AssetClass.CRYPTO.value: 0.0,
                    AssetClass.FOREX.value: 0.0
                }
            
            # Initialize market data service for asset class detection
            mds = MarketDataService()
            
            # Build positions dict with asset classes
            positions_dict = {}
            for pos in positions:
                asset_class = mds.detect_asset_class(pos.symbol)
                positions_dict[pos.symbol] = {
                    'value': abs(pos.market_value),
                    'asset_class': asset_class
                }
            
            # Calculate allocation
            total_value = account['portfolio_value']
            allocation = calculate_portfolio_allocation(positions_dict, total_value)
            
            # Convert to percentages and string keys
            allocation_pct = {
                asset_class.value: alloc * 100
                for asset_class, alloc in allocation.items()
            }
            
            logger.info(f"Asset class allocation: {allocation_pct}")
            
            return allocation_pct
        except Exception as e:
            logger.error(f"Failed to calculate asset class allocation: {e}")
            raise
    
    def get_correlation_matrix(
        self,
        symbols: Optional[List[str]] = None,
        window: int = 30
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for portfolio positions.
        
        Args:
            symbols: List of symbols to include (defaults to all positions)
            window: Rolling window for correlation calculation (days)
            
        Returns:
            Correlation matrix DataFrame
            
        Raises:
            ValueError: If trading service is not configured
            Exception: If calculation fails
        """
        if not self.trading_service:
            raise ValueError("Trading service is required to calculate correlation")
        
        try:
            from services.market_data_service import MarketDataService
            from utils.asset_analysis import calculate_correlation_matrix
            from datetime import timedelta
            
            # Get symbols from positions if not provided
            if symbols is None:
                positions = self.trading_service.get_positions()
                symbols = [pos.symbol for pos in positions]
            
            if not symbols or len(symbols) < 2:
                logger.warning("Need at least 2 symbols for correlation matrix")
                return pd.DataFrame()
            
            # Fetch historical data for all symbols
            mds = MarketDataService()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=window + 10)  # Extra buffer
            
            price_data = {}
            for symbol in symbols:
                try:
                    asset_class = mds.detect_asset_class(symbol)
                    bars = mds.get_bars(
                        symbol=symbol,
                        timeframe='1Day',
                        start=start_date,
                        end=end_date,
                        asset_class=asset_class
                    )
                    
                    if not bars.empty:
                        price_data[symbol] = bars['close']
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
                    continue
            
            if len(price_data) < 2:
                logger.warning("Insufficient data for correlation matrix")
                return pd.DataFrame()
            
            # Calculate correlation matrix
            corr_matrix = calculate_correlation_matrix(price_data, window)
            
            logger.info(f"Calculated correlation matrix for {len(price_data)} symbols")
            
            return corr_matrix
        except Exception as e:
            logger.error(f"Failed to calculate correlation matrix: {e}")
            raise
    
    def get_diversification_metrics(self) -> Dict[str, Any]:
        """
        Calculate portfolio diversification metrics.
        
        Returns:
            Dictionary with diversification metrics including:
            - asset_class_allocation: Allocation by asset class
            - correlation_matrix: Correlation between positions
            - diversification_ratio: Overall diversification score
            - concentration_risk: Largest position as % of portfolio
            
        Raises:
            ValueError: If trading service is not configured
            Exception: If calculation fails
        """
        if not self.trading_service:
            raise ValueError("Trading service is required to calculate diversification")
        
        try:
            from utils.asset_analysis import calculate_diversification_ratio
            
            # Get positions
            positions = self.trading_service.get_positions()
            account = self.trading_service.get_account()
            
            if not positions:
                logger.info("No positions found")
                return {
                    'asset_class_allocation': {},
                    'correlation_matrix': pd.DataFrame(),
                    'diversification_ratio': 0.0,
                    'concentration_risk': 0.0,
                    'num_positions': 0
                }
            
            # Asset class allocation
            allocation = self.get_asset_class_allocation()
            
            # Correlation matrix
            corr_matrix = self.get_correlation_matrix()
            
            # Calculate concentration risk (largest position %)
            total_value = account['portfolio_value']
            if total_value > 0:
                position_values = [abs(pos.market_value) for pos in positions]
                concentration_risk = max(position_values) / total_value * 100
            else:
                concentration_risk = 0.0
            
            # Calculate diversification ratio
            if len(positions) >= 2 and not corr_matrix.empty:
                # Build positions dict for diversification calculation
                positions_dict = {}
                for pos in positions:
                    weight = abs(pos.market_value) / total_value if total_value > 0 else 0
                    # Estimate volatility from unrealized P&L
                    volatility = abs(pos.unrealized_pl_pct) if pos.unrealized_pl_pct else 0.01
                    positions_dict[pos.symbol] = {
                        'weight': weight,
                        'volatility': volatility
                    }
                
                diversification_ratio = calculate_diversification_ratio(
                    positions_dict,
                    corr_matrix
                )
            else:
                diversification_ratio = 1.0  # Single position or no correlation data
            
            metrics = {
                'asset_class_allocation': allocation,
                'correlation_matrix': corr_matrix.to_dict() if not corr_matrix.empty else {},
                'diversification_ratio': float(diversification_ratio),
                'concentration_risk': float(concentration_risk),
                'num_positions': len(positions)
            }
            
            logger.info(
                f"Diversification metrics: "
                f"Ratio={diversification_ratio:.2f}, "
                f"Concentration={concentration_risk:.1f}%"
            )
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to calculate diversification metrics: {e}")
            raise
    
    def export_trades_csv(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> str:
        """
        Export trades to CSV format.
        
        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            
        Returns:
            CSV string with trade data
            
        Raises:
            Exception: If export fails
        """
        try:
            transactions = self.get_transaction_history(start_date, end_date)
            
            if not transactions:
                logger.info("No transactions to export")
                return ""
            
            # Create DataFrame
            txn_data = []
            for txn in transactions:
                txn_data.append({
                    'Date': txn.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'Symbol': txn.symbol,
                    'Side': txn.side.upper(),
                    'Quantity': txn.quantity,
                    'Price': txn.price,
                    'Total Value': txn.total_value,
                    'Commission': txn.commission,
                    'Order ID': txn.order_id,
                    'Notes': txn.notes or ''
                })
            
            df = pd.DataFrame(txn_data)
            csv_string = df.to_csv(index=False)
            
            logger.info(f"Exported {len(transactions)} trades to CSV")
            
            return csv_string
        except Exception as e:
            logger.error(f"Failed to export trades to CSV: {e}")
            raise
    
    def export_portfolio_history_csv(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> str:
        """
        Export portfolio history to CSV format.
        
        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            
        Returns:
            CSV string with portfolio history data
            
        Raises:
            Exception: If export fails
        """
        try:
            history = self.get_portfolio_history(start_date, end_date)
            
            if history.empty:
                logger.info("No portfolio history to export")
                return ""
            
            # Format the DataFrame for export
            export_df = history.copy()
            export_df['timestamp'] = export_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            export_df = export_df.rename(columns={
                'timestamp': 'Date',
                'portfolio_value': 'Portfolio Value',
                'cash': 'Cash',
                'equity': 'Equity',
                'positions_value': 'Positions Value',
                'num_positions': 'Number of Positions'
            })
            
            csv_string = export_df.to_csv(index=False)
            
            logger.info(f"Exported {len(history)} portfolio snapshots to CSV")
            
            return csv_string
        except Exception as e:
            logger.error(f"Failed to export portfolio history to CSV: {e}")
            raise
    
    def export_tax_report_csv(self, year: int) -> str:
        """
        Export tax report to CSV format.
        
        Args:
            year: Tax year to export
            
        Returns:
            CSV string with tax report data
            
        Raises:
            ValueError: If year is invalid
            Exception: If export fails
        """
        try:
            tax_df = self.export_for_taxes(year)
            
            if tax_df.empty:
                logger.info(f"No tax data to export for year {year}")
                return ""
            
            csv_string = tax_df.to_csv(index=False)
            
            logger.info(f"Exported tax report for year {year} to CSV")
            
            return csv_string
        except Exception as e:
            logger.error(f"Failed to export tax report to CSV: {e}")
            raise
