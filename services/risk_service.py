"""Risk management service for enforcing trading limits and position sizing."""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from config.settings import settings, RiskConfig

logger = logging.getLogger(__name__)


class RiskViolationType(str, Enum):
    """Types of risk violations"""
    POSITION_SIZE = "position_size"
    PORTFOLIO_RISK = "portfolio_risk"
    DAILY_LOSS = "daily_loss"
    MAX_POSITIONS = "max_positions"
    STOP_LOSS = "stop_loss"
    INSUFFICIENT_FUNDS = "insufficient_funds"


@dataclass
class ValidationResult:
    """Result of trade validation"""
    is_valid: bool
    violations: List[RiskViolationType]
    messages: List[str]
    suggested_quantity: Optional[int] = None
    
    @property
    def has_violations(self) -> bool:
        """Check if there are any violations"""
        return len(self.violations) > 0
    
    def get_error_message(self) -> str:
        """Get formatted error message"""
        if not self.has_violations:
            return "Trade is valid"
        return "; ".join(self.messages)


@dataclass
class TradeRequest:
    """Trade request for validation"""
    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    price: float
    order_type: str = 'market'


@dataclass
class Position:
    """Position information for risk calculations"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pl: float
    unrealized_pl_pct: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    portfolio_value: float
    total_exposure: float
    exposure_pct: float
    var_95: float  # Value at Risk (95% confidence)
    max_drawdown: float
    daily_pl: float
    daily_pl_pct: float
    open_positions: int
    risk_score: float  # 0-100, higher is riskier
    
    def is_high_risk(self, threshold: float = 70.0) -> bool:
        """Check if portfolio is in high risk territory"""
        return self.risk_score >= threshold


class RiskService:
    """
    Service for risk management and position sizing.
    
    Enforces trading limits, calculates position sizes,
    monitors portfolio risk, and triggers stop-loss orders.
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        """
        Initialize risk service.
        
        Args:
            config: Risk configuration (defaults to settings.risk)
        """
        self.config = config or settings.risk
        
        # Daily loss tracking
        self._daily_losses: Dict[str, float] = {}  # date -> loss amount
        self._last_reset_date = datetime.now().date()
        
        logger.info(
            f"Initialized Risk Service with config: "
            f"max_position_size={self.config.max_position_size:.1%}, "
            f"daily_loss_limit=${self.config.daily_loss_limit:.2f}"
        )
    
    def _reset_daily_tracking_if_needed(self):
        """Reset daily tracking at start of new trading day"""
        current_date = datetime.now().date()
        
        if current_date > self._last_reset_date:
            logger.info(f"Resetting daily tracking for new trading day: {current_date}")
            self._daily_losses.clear()
            self._last_reset_date = current_date
    
    def _get_daily_loss(self) -> float:
        """
        Get total loss for current trading day.
        
        Returns:
            Total loss amount (positive number)
        """
        self._reset_daily_tracking_if_needed()
        date_key = str(self._last_reset_date)
        return self._daily_losses.get(date_key, 0.0)
    
    def _record_loss(self, loss_amount: float):
        """
        Record a loss for daily tracking.
        
        Args:
            loss_amount: Loss amount (positive number)
        """
        if loss_amount <= 0:
            return
        
        self._reset_daily_tracking_if_needed()
        date_key = str(self._last_reset_date)
        
        current_loss = self._daily_losses.get(date_key, 0.0)
        self._daily_losses[date_key] = current_loss + loss_amount
        
        logger.info(f"Recorded loss: ${loss_amount:.2f}, Daily total: ${self._daily_losses[date_key]:.2f}")
    
    def validate_trade(
        self,
        trade: TradeRequest,
        portfolio_value: float,
        current_positions: List[Position],
        available_cash: float
    ) -> ValidationResult:
        """
        Validate a trade request against risk limits.
        
        Args:
            trade: Trade request to validate
            portfolio_value: Current portfolio value
            current_positions: List of current positions
            available_cash: Available cash for trading
            
        Returns:
            ValidationResult with validation status and any violations
        """
        violations = []
        messages = []
        suggested_quantity = None
        
        # Calculate trade value
        trade_value = trade.quantity * trade.price
        
        # Check 1: Position size limit
        position_size_pct = trade_value / portfolio_value if portfolio_value > 0 else 0
        
        if position_size_pct > self.config.max_position_size:
            violations.append(RiskViolationType.POSITION_SIZE)
            messages.append(
                f"Position size {position_size_pct:.1%} exceeds limit "
                f"{self.config.max_position_size:.1%}"
            )
            
            # Suggest maximum allowed quantity
            max_value = portfolio_value * self.config.max_position_size
            suggested_quantity = int(max_value / trade.price)
            
            if suggested_quantity > 0:
                messages[-1] += f" (suggested quantity: {suggested_quantity})"
        
        # Check 2: Sufficient funds for buy orders
        if trade.side.lower() == 'buy':
            if trade_value > available_cash:
                violations.append(RiskViolationType.INSUFFICIENT_FUNDS)
                messages.append(
                    f"Insufficient funds: ${trade_value:.2f} required, "
                    f"${available_cash:.2f} available"
                )
                
                # Suggest quantity based on available cash
                if suggested_quantity is None:
                    suggested_quantity = int(available_cash / trade.price)
                    if suggested_quantity > 0:
                        messages[-1] += f" (suggested quantity: {suggested_quantity})"
        
        # Check 3: Maximum open positions
        if trade.side.lower() == 'buy':
            # Check if this is a new position (not adding to existing)
            is_new_position = not any(pos.symbol == trade.symbol for pos in current_positions)
            
            if is_new_position and len(current_positions) >= self.config.max_open_positions:
                violations.append(RiskViolationType.MAX_POSITIONS)
                messages.append(
                    f"Maximum open positions ({self.config.max_open_positions}) reached"
                )
        
        # Check 4: Daily loss limit
        daily_loss = self._get_daily_loss()
        
        if daily_loss >= self.config.daily_loss_limit:
            violations.append(RiskViolationType.DAILY_LOSS)
            messages.append(
                f"Daily loss limit reached: ${daily_loss:.2f} / "
                f"${self.config.daily_loss_limit:.2f}"
            )
        
        # Check 5: Portfolio risk limit
        total_exposure = sum(abs(pos.quantity * pos.current_price) for pos in current_positions)
        
        if trade.side.lower() == 'buy':
            total_exposure += trade_value
        
        exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Portfolio risk is considered violated if exposure is too high
        # We use a threshold of 1.0 - max_portfolio_risk as the maximum allowed exposure
        max_exposure_pct = 1.0 - self.config.max_portfolio_risk
        
        if exposure_pct > max_exposure_pct:
            violations.append(RiskViolationType.PORTFOLIO_RISK)
            messages.append(
                f"Portfolio exposure {exposure_pct:.1%} exceeds safe limit "
                f"{max_exposure_pct:.1%}"
            )
        
        # Create result
        is_valid = len(violations) == 0
        
        if is_valid:
            logger.info(f"Trade validation passed for {trade.symbol}: {trade.quantity} shares @ ${trade.price}")
        else:
            logger.warning(
                f"Trade validation failed for {trade.symbol}: "
                f"{', '.join(v.value for v in violations)}"
            )
        
        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            messages=messages,
            suggested_quantity=suggested_quantity
        )
    
    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        current_price: float,
        portfolio_value: float,
        available_cash: float
    ) -> int:
        """
        Calculate appropriate position size based on risk parameters and signal strength.
        
        Uses Kelly Criterion-inspired sizing with signal strength as confidence.
        
        Args:
            symbol: Stock symbol
            signal_strength: Signal confidence (0.0 to 1.0)
            current_price: Current stock price
            portfolio_value: Total portfolio value
            available_cash: Available cash for trading
            
        Returns:
            Number of shares to trade (0 if signal too weak or insufficient funds)
        """
        # Validate inputs
        if signal_strength <= 0 or signal_strength > 1.0:
            logger.warning(f"Invalid signal strength {signal_strength} for {symbol}")
            return 0
        
        if current_price <= 0:
            logger.warning(f"Invalid price ${current_price} for {symbol}")
            return 0
        
        if portfolio_value <= 0:
            logger.warning(f"Invalid portfolio value ${portfolio_value}")
            return 0
        
        # Calculate base position size (percentage of portfolio)
        # Scale position size with signal strength
        # Strong signals (0.8-1.0) get max position size
        # Weak signals (0.6-0.8) get proportionally less
        min_confidence = 0.6
        
        if signal_strength < min_confidence:
            logger.info(
                f"Signal strength {signal_strength:.2f} below minimum "
                f"{min_confidence:.2f} for {symbol}"
            )
            return 0
        
        # Scale from 0 to max_position_size based on signal strength
        # At min_confidence: 50% of max size
        # At 1.0: 100% of max size
        confidence_factor = (signal_strength - min_confidence) / (1.0 - min_confidence)
        position_size_pct = self.config.max_position_size * (0.5 + 0.5 * confidence_factor)
        
        # Calculate position value
        position_value = portfolio_value * position_size_pct
        
        # Limit to available cash
        position_value = min(position_value, available_cash)
        
        # Calculate number of shares
        quantity = int(position_value / current_price)
        
        logger.info(
            f"Calculated position size for {symbol}: {quantity} shares "
            f"(signal: {signal_strength:.2f}, size: {position_size_pct:.1%})"
        )
        
        return quantity
    
    def check_stop_loss(self, position: Position, current_price: float) -> bool:
        """
        Check if stop-loss should trigger for a position.
        
        Args:
            position: Position to check
            current_price: Current market price
            
        Returns:
            True if stop-loss should trigger, False otherwise
        """
        # If explicit stop-loss is set, use it
        if position.stop_loss is not None:
            if current_price <= position.stop_loss:
                logger.warning(
                    f"Stop-loss triggered for {position.symbol}: "
                    f"${current_price:.2f} <= ${position.stop_loss:.2f}"
                )
                return True
            return False
        
        # Otherwise, use percentage-based stop-loss
        loss_pct = (position.entry_price - current_price) / position.entry_price
        
        if loss_pct >= self.config.stop_loss_pct:
            logger.warning(
                f"Stop-loss triggered for {position.symbol}: "
                f"loss {loss_pct:.1%} >= {self.config.stop_loss_pct:.1%}"
            )
            
            # Record the loss
            loss_amount = abs(position.unrealized_pl)
            self._record_loss(loss_amount)
            
            return True
        
        return False
    
    def check_take_profit(self, position: Position, current_price: float) -> bool:
        """
        Check if take-profit should trigger for a position.
        
        Args:
            position: Position to check
            current_price: Current market price
            
        Returns:
            True if take-profit should trigger, False otherwise
        """
        # If explicit take-profit is set, use it
        if position.take_profit is not None:
            if current_price >= position.take_profit:
                logger.info(
                    f"Take-profit triggered for {position.symbol}: "
                    f"${current_price:.2f} >= ${position.take_profit:.2f}"
                )
                return True
            return False
        
        # Otherwise, use percentage-based take-profit
        profit_pct = (current_price - position.entry_price) / position.entry_price
        
        if profit_pct >= self.config.take_profit_pct:
            logger.info(
                f"Take-profit triggered for {position.symbol}: "
                f"profit {profit_pct:.1%} >= {self.config.take_profit_pct:.1%}"
            )
            return True
        
        return False
    
    def get_portfolio_risk(
        self,
        portfolio_value: float,
        positions: List[Position],
        daily_pl: float
    ) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            portfolio_value: Current portfolio value
            positions: List of current positions
            daily_pl: Profit/loss for current day
            
        Returns:
            RiskMetrics with risk analysis
        """
        # Calculate total exposure
        total_exposure = sum(abs(pos.quantity * pos.current_price) for pos in positions)
        exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate Value at Risk (simplified - using position volatility)
        # In production, this would use historical volatility and correlations
        total_unrealized_pl = sum(pos.unrealized_pl for pos in positions)
        var_95 = abs(total_unrealized_pl) * 1.65  # Approximate 95% VaR
        
        # Calculate maximum drawdown (from unrealized losses)
        losses = [pos.unrealized_pl for pos in positions if pos.unrealized_pl < 0]
        max_drawdown = abs(sum(losses)) if losses else 0.0
        
        # Calculate daily P&L percentage
        daily_pl_pct = (daily_pl / portfolio_value * 100) if portfolio_value > 0 else 0
        
        # Calculate risk score (0-100)
        risk_score = self._calculate_risk_score(
            exposure_pct=exposure_pct,
            daily_loss=self._get_daily_loss(),
            max_drawdown=max_drawdown,
            num_positions=len(positions),
            portfolio_value=portfolio_value
        )
        
        metrics = RiskMetrics(
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            exposure_pct=exposure_pct,
            var_95=var_95,
            max_drawdown=max_drawdown,
            daily_pl=daily_pl,
            daily_pl_pct=daily_pl_pct,
            open_positions=len(positions),
            risk_score=risk_score
        )
        
        logger.info(
            f"Portfolio risk metrics: exposure={exposure_pct:.1%}, "
            f"risk_score={risk_score:.1f}, positions={len(positions)}"
        )
        
        return metrics
    
    def _calculate_risk_score(
        self,
        exposure_pct: float,
        daily_loss: float,
        max_drawdown: float,
        num_positions: int,
        portfolio_value: float
    ) -> float:
        """
        Calculate overall risk score (0-100).
        
        Higher score indicates higher risk.
        
        Args:
            exposure_pct: Portfolio exposure percentage
            daily_loss: Daily loss amount
            max_drawdown: Maximum drawdown
            num_positions: Number of open positions
            portfolio_value: Total portfolio value
            
        Returns:
            Risk score (0-100)
        """
        score = 0.0
        
        # Factor 1: Exposure (0-30 points)
        # High exposure increases risk
        max_safe_exposure = 1.0 - self.config.max_portfolio_risk
        if exposure_pct > max_safe_exposure:
            score += 30 * (exposure_pct - max_safe_exposure) / (1.0 - max_safe_exposure)
        else:
            score += 30 * (exposure_pct / max_safe_exposure) * 0.5  # Lower score for safe exposure
        
        # Factor 2: Daily loss (0-30 points)
        # Approaching daily loss limit increases risk
        if daily_loss > 0:
            loss_ratio = daily_loss / self.config.daily_loss_limit
            score += 30 * min(loss_ratio, 1.0)
        
        # Factor 3: Drawdown (0-25 points)
        # Large drawdowns increase risk
        if portfolio_value > 0:
            drawdown_pct = max_drawdown / portfolio_value
            score += 25 * min(drawdown_pct / 0.2, 1.0)  # 20% drawdown = max points
        
        # Factor 4: Concentration (0-15 points)
        # Too few or too many positions increases risk
        if num_positions == 0:
            score += 15
        elif num_positions < 3:
            score += 15 * (1 - num_positions / 3)  # Under-diversified
        elif num_positions > self.config.max_open_positions * 0.8:
            score += 15 * ((num_positions - self.config.max_open_positions * 0.8) / 
                          (self.config.max_open_positions * 0.2))  # Over-concentrated
        
        return min(score, 100.0)
    
    def suggest_risk_reduction(
        self,
        positions: List[Position],
        risk_metrics: RiskMetrics
    ) -> List[str]:
        """
        Suggest actions to reduce portfolio risk.
        
        Args:
            positions: Current positions
            risk_metrics: Current risk metrics
            
        Returns:
            List of risk reduction suggestions
        """
        suggestions = []
        
        # Check if risk is high
        if not risk_metrics.is_high_risk():
            return ["Portfolio risk is within acceptable limits"]
        
        # Suggestion 1: Reduce exposure if too high
        max_safe_exposure = 1.0 - self.config.max_portfolio_risk
        if risk_metrics.exposure_pct > max_safe_exposure:
            reduction_needed = risk_metrics.total_exposure * (
                risk_metrics.exposure_pct - max_safe_exposure
            )
            suggestions.append(
                f"Reduce total exposure by ${reduction_needed:.2f} "
                f"({risk_metrics.exposure_pct:.1%} -> {max_safe_exposure:.1%})"
            )
        
        # Suggestion 2: Close losing positions if daily loss limit approached
        daily_loss = self._get_daily_loss()
        if daily_loss > self.config.daily_loss_limit * 0.7:  # 70% of limit
            losing_positions = [p for p in positions if p.unrealized_pl < 0]
            if losing_positions:
                # Sort by loss amount (worst first)
                losing_positions.sort(key=lambda p: p.unrealized_pl)
                worst_position = losing_positions[0]
                suggestions.append(
                    f"Consider closing {worst_position.symbol} "
                    f"(loss: ${abs(worst_position.unrealized_pl):.2f}) "
                    f"to limit daily losses"
                )
        
        # Suggestion 3: Reduce position concentration
        if len(positions) < 3 and len(positions) > 0:
            suggestions.append(
                f"Diversify portfolio: currently only {len(positions)} position(s). "
                f"Consider adding 2-3 more positions to reduce concentration risk"
            )
        
        # Suggestion 4: Set stop-losses if not set
        positions_without_stops = [p for p in positions if p.stop_loss is None]
        if positions_without_stops:
            suggestions.append(
                f"Set stop-loss orders for {len(positions_without_stops)} position(s) "
                f"without protection"
            )
        
        # Suggestion 5: Take profits on winning positions
        winning_positions = [
            p for p in positions 
            if p.unrealized_pl_pct >= self.config.take_profit_pct
        ]
        if winning_positions:
            suggestions.append(
                f"Consider taking profits on {len(winning_positions)} position(s) "
                f"that have reached profit targets"
            )
        
        # Suggestion 6: Reduce position sizes if approaching max positions
        if len(positions) >= self.config.max_open_positions * 0.8:
            suggestions.append(
                f"Approaching maximum positions ({len(positions)}/{self.config.max_open_positions}). "
                f"Consider consolidating or closing smaller positions"
            )
        
        logger.info(f"Generated {len(suggestions)} risk reduction suggestions")
        
        return suggestions
    
    def update_config(self, new_config: RiskConfig):
        """
        Update risk configuration.
        
        Args:
            new_config: New risk configuration
        """
        self.config = new_config
        logger.info(f"Updated risk configuration: {new_config}")
    
    def reset_daily_limits(self):
        """Reset daily loss tracking (for testing or manual reset)"""
        self._daily_losses.clear()
        self._last_reset_date = datetime.now().date()
        logger.info("Daily limits reset")
    
    def get_daily_loss_remaining(self) -> float:
        """
        Get remaining daily loss allowance.
        
        Returns:
            Remaining loss allowance (positive number)
        """
        daily_loss = self._get_daily_loss()
        remaining = max(0, self.config.daily_loss_limit - daily_loss)
        return remaining
    
    def is_trading_allowed(self) -> Tuple[bool, str]:
        """
        Check if trading is currently allowed based on risk limits.
        
        Returns:
            Tuple of (is_allowed, reason)
        """
        # Check daily loss limit
        daily_loss = self._get_daily_loss()
        
        if daily_loss >= self.config.daily_loss_limit:
            return False, f"Daily loss limit reached: ${daily_loss:.2f}"
        
        return True, "Trading allowed"
