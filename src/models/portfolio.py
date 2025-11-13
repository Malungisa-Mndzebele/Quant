"""Portfolio model."""

from typing import Dict, List, Optional
from datetime import datetime
from .position import Position
from .order import Trade
from .performance import PerformanceMetrics


class Portfolio:
    """Manages portfolio positions, cash balance, and performance tracking."""
    
    def __init__(self, initial_capital: float):
        """Initialize portfolio with starting capital."""
        if initial_capital <= 0:
            raise ValueError(f"Initial capital must be positive, got {initial_capital}")
        
        self.cash_balance: float = initial_capital
        self.positions: Dict[str, Position] = {}
        self.initial_value: float = initial_capital
        self.trade_history: List[Trade] = []
        self._daily_values: List[tuple[datetime, float]] = []
        self._peak_value: float = initial_capital

    def add_position(self, symbol: str, quantity: int, entry_price: float) -> None:
        """Add a new position or update existing position."""
        if quantity == 0:
            raise ValueError("Cannot add position with zero quantity")
        
        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {entry_price}")
        
        if symbol in self.positions:
            # Update existing position - calculate weighted average entry price
            existing = self.positions[symbol]
            total_quantity = existing.quantity + quantity
            
            if total_quantity == 0:
                # Position closed
                del self.positions[symbol]
            else:
                # Calculate weighted average entry price
                total_cost = (existing.entry_price * existing.quantity) + (entry_price * quantity)
                avg_entry_price = total_cost / total_quantity
                existing.quantity = total_quantity
                existing.entry_price = avg_entry_price
        else:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price
            )

    def remove_position(self, symbol: str) -> None:
        """Remove a position from the portfolio."""
        if symbol in self.positions:
            del self.positions[symbol]

    def update_prices(self, current_prices: Dict[str, float]) -> None:
        """Update current prices for all positions."""
        for symbol, price in current_prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get a specific position."""
        return self.positions.get(symbol)

    def get_total_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """Calculate total portfolio value (cash + positions)."""
        if current_prices:
            self.update_prices(current_prices)
        
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash_balance + positions_value

    def get_positions_value(self) -> float:
        """Get total value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())

    def get_unrealized_pnl(self) -> float:
        """Calculate total unrealized profit/loss."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def record_trade(self, trade: Trade) -> None:
        """Record a trade in the history."""
        self.trade_history.append(trade)

    def record_daily_value(self, date: datetime, value: float) -> None:
        """Record portfolio value for a specific date."""
        self._daily_values.append((date, value))

    def get_total_return(self, current_value: Optional[float] = None) -> float:
        """Calculate total return as a percentage."""
        if current_value is None:
            current_value = self.get_total_value()
        return ((current_value - self.initial_value) / self.initial_value) * 100

    def get_daily_return(self) -> float:
        """Calculate daily return based on last two recorded values."""
        if len(self._daily_values) < 2:
            return 0.0
        
        prev_value = self._daily_values[-2][1]
        current_value = self._daily_values[-1][1]
        return ((current_value - prev_value) / prev_value) * 100

    def update_peak_value(self, current_value: float) -> None:
        """Update peak portfolio value for drawdown calculation."""
        if current_value > self._peak_value:
            self._peak_value = current_value

    def get_max_drawdown(self, current_value: Optional[float] = None) -> float:
        """Calculate maximum drawdown as a percentage."""
        if current_value is None:
            current_value = self.get_total_value()
        
        self.update_peak_value(current_value)
        
        if self._peak_value == 0:
            return 0.0
        
        drawdown = ((self._peak_value - current_value) / self._peak_value) * 100
        return max(0.0, drawdown)

    def get_win_rate(self) -> float:
        """Calculate win rate from trade history."""
        if not self.trade_history:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trade_history if trade.pnl and trade.pnl > 0)
        return (winning_trades / len(self.trade_history)) * 100

    def calculate_returns(self, current_prices: Optional[Dict[str, float]] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics for the portfolio.
        
        Args:
            current_prices: Optional dict of current prices to update positions
        
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        # Get current portfolio value
        current_value = self.get_total_value(current_prices)
        
        # Calculate total return
        total_return = self.get_total_return(current_value)
        
        # Calculate daily return
        daily_return = self.get_daily_return()
        
        # Calculate Sharpe ratio from daily values
        if len(self._daily_values) >= 2:
            daily_returns = []
            for i in range(1, len(self._daily_values)):
                prev_val = self._daily_values[i-1][1]
                curr_val = self._daily_values[i][1]
                if prev_val > 0:
                    ret = (curr_val - prev_val) / prev_val
                    daily_returns.append(ret)
            
            sharpe_ratio = PerformanceMetrics.calculate_sharpe_ratio(daily_returns)
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        if self._daily_values:
            equity_curve = [val for _, val in self._daily_values]
            max_drawdown = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        else:
            max_drawdown = self.get_max_drawdown(current_value)
        
        # Calculate win rate
        win_rate = self.get_win_rate()
        
        # Calculate total PnL
        total_pnl = current_value - self.initial_value
        
        return PerformanceMetrics(
            total_return=total_return,
            daily_return=daily_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            num_trades=len(self.trade_history),
            total_pnl=total_pnl
        )

    def __repr__(self) -> str:
        """String representation of portfolio."""
        total_value = self.get_total_value()
        return (f"Portfolio(cash={self.cash_balance:.2f}, "
                f"positions={len(self.positions)}, "
                f"total_value={total_value:.2f})")
