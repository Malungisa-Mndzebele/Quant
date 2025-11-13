"""Backtest result and reporting."""

from dataclasses import dataclass
from datetime import date
from typing import List
import json
from pathlib import Path

from ..models.order import Trade


@dataclass
class BacktestResult:
    """Contains backtest results and performance metrics."""
    
    strategy_name: str
    symbols: List[str]
    start_date: date
    end_date: date
    initial_capital: float
    final_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    trade_history: List[Trade]
    
    def generate_report(self) -> str:
        """
        Generate a formatted text report of backtest results.
        
        Returns:
            Formatted string report
        """
        # Calculate additional metrics
        total_pnl = self.final_value - self.initial_capital
        
        # Count winning and losing trades
        winning_trades = sum(1 for trade in self.trade_history if trade.pnl and trade.pnl > 0)
        losing_trades = sum(1 for trade in self.trade_history if trade.pnl and trade.pnl < 0)
        
        # Calculate average win and loss
        wins = [trade.pnl for trade in self.trade_history if trade.pnl and trade.pnl > 0]
        losses = [trade.pnl for trade in self.trade_history if trade.pnl and trade.pnl < 0]
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        # Build report
        report_lines = [
            "=" * 70,
            "BACKTEST RESULTS",
            "=" * 70,
            "",
            f"Strategy: {self.strategy_name}",
            f"Symbols: {', '.join(self.symbols)}",
            f"Period: {self.start_date} to {self.end_date}",
            "",
            "-" * 70,
            "PERFORMANCE SUMMARY",
            "-" * 70,
            "",
            f"Initial Capital:    ${self.initial_capital:,.2f}",
            f"Final Value:        ${self.final_value:,.2f}",
            f"Total P&L:          ${total_pnl:,.2f}",
            f"Total Return:       {self.total_return:.2f}%",
            "",
            f"Sharpe Ratio:       {self.sharpe_ratio:.2f}",
            f"Max Drawdown:       {self.max_drawdown:.2f}%",
            "",
            "-" * 70,
            "TRADING STATISTICS",
            "-" * 70,
            "",
            f"Total Trades:       {self.num_trades}",
            f"Winning Trades:     {winning_trades}",
            f"Losing Trades:      {losing_trades}",
            f"Win Rate:           {self.win_rate:.2f}%",
            "",
            f"Average Win:        ${avg_win:,.2f}",
            f"Average Loss:       ${avg_loss:,.2f}",
        ]
        
        # Add profit factor if applicable
        if avg_loss != 0:
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
            report_lines.append(f"Profit Factor:      {profit_factor:.2f}")
        
        report_lines.extend([
            "",
            "=" * 70,
        ])
        
        return "\n".join(report_lines)
    
    def to_dict(self) -> dict:
        """
        Convert backtest result to dictionary.
        
        Returns:
            Dictionary representation of backtest result
        """
        return {
            'strategy_name': self.strategy_name,
            'symbols': self.symbols,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'final_value': self.final_value,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'trade_history': [
                {
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'action': trade.action.value,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'timestamp': trade.timestamp.isoformat(),
                    'pnl': trade.pnl
                }
                for trade in self.trade_history
            ]
        }
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save backtest result to a JSON file.
        
        Args:
            filepath: Path to save the result file
        """
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def export_trades_csv(self, filepath: str) -> None:
        """
        Export trade history to CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Write CSV
        with open(filepath, 'w') as f:
            # Write header
            f.write("trade_id,symbol,action,quantity,price,timestamp,pnl\n")
            
            # Write trades
            for trade in self.trade_history:
                f.write(
                    f"{trade.trade_id},"
                    f"{trade.symbol},"
                    f"{trade.action.value},"
                    f"{trade.quantity},"
                    f"{trade.price:.2f},"
                    f"{trade.timestamp.isoformat()},"
                    f"{trade.pnl if trade.pnl is not None else ''}\n"
                )
    
    def __repr__(self) -> str:
        """String representation of backtest result."""
        return (
            f"BacktestResult(strategy={self.strategy_name}, "
            f"return={self.total_return:.2f}%, "
            f"sharpe={self.sharpe_ratio:.2f}, "
            f"trades={self.num_trades})"
        )
