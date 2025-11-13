"""Risk management system."""

from typing import Tuple, Optional, Dict
from datetime import datetime, date
import logging
from ..models.order import Order, OrderAction
from ..models.portfolio import Portfolio
from ..models.config import RiskConfig


logger = logging.getLogger(__name__)


class RiskViolation(Exception):
    """Exception raised when a risk limit is violated."""
    pass


class RiskManager:
    """Manages risk limits and validates orders before execution."""
    
    def __init__(self, config: RiskConfig):
        """Initialize risk manager with configuration.
        
        Args:
            config: Risk configuration with limits and constraints
        """
        self.config = config
        self._daily_start_value: Optional[float] = None
        self._daily_pnl: float = 0.0
        self._last_reset_date: Optional[date] = None
        logger.info(f"RiskManager initialized with config: {config}")
    
    def validate_order(self, order: Order, portfolio: Portfolio, current_price: float) -> Tuple[bool, str]:
        """Validate an order against all risk rules.
        
        Args:
            order: The order to validate
            portfolio: Current portfolio state
            current_price: Current market price for the symbol
            
        Returns:
            Tuple of (is_valid, reason). If valid, reason is empty string.
            If invalid, reason contains the violation message.
        """
        try:
            # Check symbol whitelist
            if not self._check_allowed_symbol(order.symbol):
                return False, f"Symbol {order.symbol} is not in allowed symbols list"
            
            # Check sufficient cash for buy orders (check early)
            if order.action == OrderAction.BUY:
                order_value = order.quantity * current_price
                if portfolio.cash_balance < order_value:
                    return False, f"Insufficient cash: need ${order_value:.2f}, have ${portfolio.cash_balance:.2f}"
            
            # Check sufficient position for sell orders (check early)
            if order.action == OrderAction.SELL:
                position = portfolio.get_position(order.symbol)
                if position is None or position.quantity < order.quantity:
                    available = position.quantity if position else 0
                    return False, f"Insufficient position: need {order.quantity} shares, have {available}"
            
            # Check maximum order value
            if not self._check_max_order_value(order, current_price):
                return False, f"Order value ${order.quantity * current_price:.2f} exceeds maximum order value ${self.config.max_order_value:.2f}"
            
            # Check position size limit
            if not self._check_position_size_limit(order, portfolio, current_price):
                max_value = portfolio.get_total_value() * (self.config.max_position_size_pct / 100)
                return False, f"Order would exceed position size limit of {self.config.max_position_size_pct}% (${max_value:.2f})"
            
            # Check daily loss limit
            if not self._check_daily_loss_limit(portfolio):
                max_loss = portfolio.initial_value * (self.config.max_daily_loss_pct / 100)
                return False, f"Daily loss limit of {self.config.max_daily_loss_pct}% (${max_loss:.2f}) has been breached"
            
            # Check maximum positions limit
            if not self._check_max_positions(order, portfolio):
                return False, f"Maximum number of positions ({self.config.max_positions}) reached"
            
            logger.info(f"Order validation passed for {order.symbol} {order.action.value} {order.quantity}")
            return True, ""
            
        except Exception as e:
            logger.error(f"Error during order validation: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _check_allowed_symbol(self, symbol: str) -> bool:
        """Check if symbol is in allowed list (if configured)."""
        if self.config.allowed_symbols is None:
            return True
        return symbol in self.config.allowed_symbols
    
    def _check_position_size_limit(self, order: Order, portfolio: Portfolio, current_price: float) -> bool:
        """Check if order would exceed position size limit.
        
        The position size limit is calculated as a percentage of total portfolio value.
        For buy orders, we check if the new position value would exceed the limit.
        For sell orders, we always allow them (reducing position is allowed).
        """
        if order.action == OrderAction.SELL:
            # Always allow selling (reducing position)
            return True
        
        portfolio_value = portfolio.get_total_value()
        max_position_value = portfolio_value * (self.config.max_position_size_pct / 100)
        
        # Calculate what the position value would be after this order
        existing_position = portfolio.get_position(order.symbol)
        existing_value = existing_position.market_value if existing_position else 0.0
        order_value = order.quantity * current_price
        new_position_value = existing_value + order_value
        
        return new_position_value <= max_position_value
    
    def _check_daily_loss_limit(self, portfolio: Portfolio) -> bool:
        """Check if daily loss limit has been breached.
        
        Resets daily tracking at the start of each new trading day.
        """
        today = date.today()
        
        # Reset daily tracking if it's a new day
        if self._last_reset_date != today:
            self._reset_daily_tracking(portfolio)
        
        # Calculate current daily P&L
        current_value = portfolio.get_total_value()
        daily_pnl = current_value - self._daily_start_value
        daily_loss_pct = (daily_pnl / self._daily_start_value) * 100
        
        # Check if loss exceeds limit (negative percentage)
        max_loss_pct = -self.config.max_daily_loss_pct
        
        if daily_loss_pct < max_loss_pct:
            logger.warning(f"Daily loss limit breached: {daily_loss_pct:.2f}% < {max_loss_pct:.2f}%")
            return False
        
        return True
    
    def _check_max_positions(self, order: Order, portfolio: Portfolio) -> bool:
        """Check if adding a new position would exceed max positions limit."""
        if self.config.max_positions is None:
            return True
        
        # Only check for buy orders that would create a new position
        if order.action == OrderAction.BUY:
            if order.symbol not in portfolio.positions:
                # This would be a new position
                if len(portfolio.positions) >= self.config.max_positions:
                    return False
        
        return True
    
    def _check_max_order_value(self, order: Order, current_price: float) -> bool:
        """Check if order value exceeds maximum order value limit."""
        if self.config.max_order_value is None:
            return True
        
        order_value = order.quantity * current_price
        return order_value <= self.config.max_order_value
    
    def _reset_daily_tracking(self, portfolio: Portfolio) -> None:
        """Reset daily tracking for a new trading day."""
        self._daily_start_value = portfolio.get_total_value()
        self._daily_pnl = 0.0
        self._last_reset_date = date.today()
        logger.info(f"Daily tracking reset. Starting value: ${self._daily_start_value:.2f}")
    
    def check_daily_loss_limit(self, portfolio: Portfolio) -> bool:
        """Public method to check daily loss limit.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            True if within limit, False if limit breached
        """
        return self._check_daily_loss_limit(portfolio)
    
    def get_daily_pnl(self, portfolio: Portfolio) -> float:
        """Get current daily profit/loss in dollars.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            Daily P&L in dollars
        """
        today = date.today()
        if self._last_reset_date != today:
            self._reset_daily_tracking(portfolio)
        
        current_value = portfolio.get_total_value()
        return current_value - self._daily_start_value
    
    def get_daily_pnl_pct(self, portfolio: Portfolio) -> float:
        """Get current daily profit/loss as a percentage.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            Daily P&L as percentage
        """
        daily_pnl = self.get_daily_pnl(portfolio)
        if self._daily_start_value == 0:
            return 0.0
        return (daily_pnl / self._daily_start_value) * 100
    
    def calculate_exposure(self, portfolio: Portfolio) -> Dict[str, float]:
        """Calculate current portfolio risk exposure metrics.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            Dictionary containing exposure metrics:
            - total_exposure: Total value of all positions
            - exposure_pct: Exposure as percentage of portfolio value
            - cash_pct: Cash as percentage of portfolio value
            - leverage: Current leverage ratio
        """
        portfolio_value = portfolio.get_total_value()
        positions_value = portfolio.get_positions_value()
        
        if portfolio_value == 0:
            return {
                'total_exposure': 0.0,
                'exposure_pct': 0.0,
                'cash_pct': 100.0,
                'leverage': 0.0
            }
        
        exposure_pct = (positions_value / portfolio_value) * 100
        cash_pct = (portfolio.cash_balance / portfolio_value) * 100
        leverage = positions_value / portfolio_value
        
        return {
            'total_exposure': positions_value,
            'exposure_pct': exposure_pct,
            'cash_pct': cash_pct,
            'leverage': leverage
        }
    
    def calculate_position_concentration(self, portfolio: Portfolio) -> Dict[str, float]:
        """Calculate position concentration metrics.
        
        Shows how concentrated the portfolio is in individual positions.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            Dictionary mapping symbol to its percentage of total portfolio value
        """
        portfolio_value = portfolio.get_total_value()
        
        if portfolio_value == 0 or not portfolio.positions:
            return {}
        
        concentration = {}
        for symbol, position in portfolio.positions.items():
            position_pct = (position.market_value / portfolio_value) * 100
            concentration[symbol] = position_pct
        
        return concentration
    
    def get_largest_position_pct(self, portfolio: Portfolio) -> Tuple[Optional[str], float]:
        """Get the largest position as a percentage of portfolio value.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            Tuple of (symbol, percentage). Returns (None, 0.0) if no positions.
        """
        concentration = self.calculate_position_concentration(portfolio)
        
        if not concentration:
            return None, 0.0
        
        largest_symbol = max(concentration, key=concentration.get)
        largest_pct = concentration[largest_symbol]
        
        return largest_symbol, largest_pct
    
    def get_risk_metrics_report(self, portfolio: Portfolio) -> Dict[str, any]:
        """Generate a comprehensive risk metrics report.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            Dictionary containing all risk metrics
        """
        exposure = self.calculate_exposure(portfolio)
        concentration = self.calculate_position_concentration(portfolio)
        largest_position = self.get_largest_position_pct(portfolio)
        daily_pnl = self.get_daily_pnl(portfolio)
        daily_pnl_pct = self.get_daily_pnl_pct(portfolio)
        
        # Calculate risk limit utilization
        portfolio_value = portfolio.get_total_value()
        max_daily_loss_dollars = portfolio.initial_value * (self.config.max_daily_loss_pct / 100)
        daily_loss_utilization = 0.0
        if daily_pnl < 0:
            daily_loss_utilization = (abs(daily_pnl) / max_daily_loss_dollars) * 100
        
        return {
            'portfolio_value': portfolio_value,
            'cash_balance': portfolio.cash_balance,
            'positions_count': len(portfolio.positions),
            'exposure': exposure,
            'position_concentration': concentration,
            'largest_position': {
                'symbol': largest_position[0],
                'percentage': largest_position[1]
            },
            'daily_pnl': daily_pnl,
            'daily_pnl_pct': daily_pnl_pct,
            'daily_loss_limit_pct': self.config.max_daily_loss_pct,
            'daily_loss_utilization_pct': daily_loss_utilization,
            'risk_limits': {
                'max_position_size_pct': self.config.max_position_size_pct,
                'max_daily_loss_pct': self.config.max_daily_loss_pct,
                'max_positions': self.config.max_positions,
                'max_order_value': self.config.max_order_value,
                'allowed_symbols': self.config.allowed_symbols
            }
        }
    
    def format_risk_report(self, portfolio: Portfolio) -> str:
        """Format risk metrics as a human-readable string.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            Formatted risk report string
        """
        metrics = self.get_risk_metrics_report(portfolio)
        
        lines = [
            "=" * 60,
            "RISK METRICS REPORT",
            "=" * 60,
            f"Portfolio Value: ${metrics['portfolio_value']:,.2f}",
            f"Cash Balance: ${metrics['cash_balance']:,.2f}",
            f"Number of Positions: {metrics['positions_count']}",
            "",
            "EXPOSURE:",
            f"  Total Exposure: ${metrics['exposure']['total_exposure']:,.2f}",
            f"  Exposure %: {metrics['exposure']['exposure_pct']:.2f}%",
            f"  Cash %: {metrics['exposure']['cash_pct']:.2f}%",
            f"  Leverage: {metrics['exposure']['leverage']:.2f}x",
            "",
            "DAILY P&L:",
            f"  Daily P&L: ${metrics['daily_pnl']:,.2f}",
            f"  Daily P&L %: {metrics['daily_pnl_pct']:.2f}%",
            f"  Daily Loss Limit: {metrics['daily_loss_limit_pct']:.2f}%",
            f"  Loss Limit Utilization: {metrics['daily_loss_utilization_pct']:.2f}%",
            "",
        ]
        
        if metrics['largest_position']['symbol']:
            lines.extend([
                "POSITION CONCENTRATION:",
                f"  Largest Position: {metrics['largest_position']['symbol']} ({metrics['largest_position']['percentage']:.2f}%)",
                ""
            ])
            
            if metrics['position_concentration']:
                lines.append("  All Positions:")
                for symbol, pct in sorted(metrics['position_concentration'].items(), 
                                         key=lambda x: x[1], reverse=True):
                    lines.append(f"    {symbol}: {pct:.2f}%")
                lines.append("")
        
        lines.extend([
            "RISK LIMITS:",
            f"  Max Position Size: {metrics['risk_limits']['max_position_size_pct']:.2f}%",
            f"  Max Daily Loss: {metrics['risk_limits']['max_daily_loss_pct']:.2f}%",
            f"  Max Positions: {metrics['risk_limits']['max_positions'] or 'Unlimited'}",
            f"  Max Order Value: ${metrics['risk_limits']['max_order_value']:,.2f}" if metrics['risk_limits']['max_order_value'] else "  Max Order Value: Unlimited",
            "=" * 60
        ])
        
        return "\n".join(lines)
