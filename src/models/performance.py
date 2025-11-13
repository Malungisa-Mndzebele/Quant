"""Performance metrics model."""

from dataclasses import dataclass
from typing import List
import math


@dataclass
class PerformanceMetrics:
    """Performance metrics for portfolio or strategy evaluation."""
    total_return: float
    daily_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int = 0
    total_pnl: float = 0.0

    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio from a list of returns.
        
        Args:
            returns: List of period returns (as percentages or decimals)
            risk_free_rate: Risk-free rate (annualized)
        
        Returns:
            Sharpe ratio
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        # Calculate mean return
        mean_return = sum(returns) / len(returns)
        
        # Calculate standard deviation
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return 0.0
        
        # Calculate Sharpe ratio
        excess_return = mean_return - risk_free_rate
        sharpe = excess_return / std_dev
        
        # Annualize assuming daily returns (252 trading days)
        sharpe_annualized = sharpe * math.sqrt(252)
        
        return sharpe_annualized

    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """
        Calculate maximum drawdown from an equity curve.
        
        Args:
            equity_curve: List of portfolio values over time
        
        Returns:
            Maximum drawdown as a percentage
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            drawdown = ((peak - value) / peak) * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)
        
        return max_dd

    def __repr__(self) -> str:
        """String representation of performance metrics."""
        return (f"PerformanceMetrics(total_return={self.total_return:.2f}%, "
                f"sharpe_ratio={self.sharpe_ratio:.2f}, "
                f"max_drawdown={self.max_drawdown:.2f}%, "
                f"win_rate={self.win_rate:.2f}%)")
