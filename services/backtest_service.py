"""
Backtesting engine for testing trading strategies against historical data.

This module provides comprehensive backtesting functionality including
strategy simulation, performance metrics calculation, equity curve generation,
and strategy comparison.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for backtesting"""
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Trade:
    """Represents a completed trade in backtest"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    commission: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat()
        data['exit_time'] = self.exit_time.isoformat()
        return data


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
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
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['start_date'] = self.start_date.isoformat()
        data['end_date'] = self.end_date.isoformat()
        data['trades'] = [t.to_dict() for t in self.trades]
        return data


@dataclass
class Strategy:
    """Trading strategy configuration"""
    name: str
    signal_func: Callable[[pd.DataFrame], str]  # Returns 'buy', 'sell', or 'hold'
    position_size: float = 1.0  # Fraction of capital to use per trade
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    commission_pct: float = 0.001  # 0.1% commission



class BacktestEngine:
    """
    Backtesting engine for testing trading strategies.
    
    Simulates trading strategies against historical data and calculates
    performance metrics.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital for backtest
        """
        self.initial_capital = initial_capital
        logger.info(f"Initialized backtest engine with ${initial_capital:,.2f}")
    
    def run_backtest(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """
        Run backtest for a strategy over a time period.
        
        Args:
            strategy: Strategy to test
            data: Historical price data (must have 'close', 'high', 'low' columns)
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResult with performance metrics
            
        Raises:
            ValueError: If data is invalid or date range is invalid
        """
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        required_columns = ['close', 'high', 'low']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Filter data to date range
        if isinstance(data.index, pd.DatetimeIndex):
            mask = (data.index >= start_date) & (data.index <= end_date)
            data = data[mask]
        
        if data.empty:
            raise ValueError("No data in specified date range")
        
        logger.info(
            f"Running backtest for {strategy.name} from {start_date} to {end_date}"
        )
        
        # Initialize state
        capital = self.initial_capital
        position = None  # Current open position
        trades = []
        equity_curve = [capital]
        
        # Simulate trading
        for i in range(len(data)):
            current_date = data.index[i] if isinstance(data.index, pd.DatetimeIndex) else datetime.now()
            current_price = data.iloc[i]['close']
            high_price = data.iloc[i]['high']
            low_price = data.iloc[i]['low']
            
            # Get signal from strategy
            signal = strategy.signal_func(data.iloc[:i+1])
            
            # Check stop-loss and take-profit if position is open
            if position:
                # Check stop-loss
                if strategy.stop_loss_pct and low_price <= position['stop_loss']:
                    # Exit at stop-loss
                    exit_price = position['stop_loss']
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                    commission = exit_price * position['quantity'] * strategy.commission_pct
                    pnl -= commission + position['commission']
                    
                    capital += position['entry_price'] * position['quantity'] + pnl
                    
                    trade = Trade(
                        symbol=position['symbol'],
                        entry_time=position['entry_time'],
                        exit_time=current_date,
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        quantity=position['quantity'],
                        side='long',
                        pnl=pnl,
                        pnl_pct=(pnl / (position['entry_price'] * position['quantity'])) * 100,
                        commission=commission + position['commission']
                    )
                    trades.append(trade)
                    position = None
                    equity_curve.append(capital)
                    continue
                
                # Check take-profit
                if strategy.take_profit_pct and high_price >= position['take_profit']:
                    # Exit at take-profit
                    exit_price = position['take_profit']
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                    commission = exit_price * position['quantity'] * strategy.commission_pct
                    pnl -= commission + position['commission']
                    
                    capital += position['entry_price'] * position['quantity'] + pnl
                    
                    trade = Trade(
                        symbol=position['symbol'],
                        entry_time=position['entry_time'],
                        exit_time=current_date,
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        quantity=position['quantity'],
                        side='long',
                        pnl=pnl,
                        pnl_pct=(pnl / (position['entry_price'] * position['quantity'])) * 100,
                        commission=commission + position['commission']
                    )
                    trades.append(trade)
                    position = None
                    equity_curve.append(capital)
                    continue
            
            # Process signal
            if signal == 'buy' and not position:
                # Open long position
                position_value = capital * strategy.position_size
                quantity = int(position_value / current_price)
                
                if quantity > 0:
                    commission = current_price * quantity * strategy.commission_pct
                    capital -= current_price * quantity
                    
                    # Calculate stop-loss and take-profit levels
                    stop_loss = None
                    if strategy.stop_loss_pct:
                        stop_loss = current_price * (1 - strategy.stop_loss_pct)
                    
                    take_profit = None
                    if strategy.take_profit_pct:
                        take_profit = current_price * (1 + strategy.take_profit_pct)
                    
                    position = {
                        'symbol': 'BACKTEST',
                        'entry_time': current_date,
                        'entry_price': current_price,
                        'quantity': quantity,
                        'commission': commission,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
            
            elif signal == 'sell' and position:
                # Close long position
                exit_price = current_price
                pnl = (exit_price - position['entry_price']) * position['quantity']
                commission = exit_price * position['quantity'] * strategy.commission_pct
                pnl -= commission + position['commission']
                
                capital += position['entry_price'] * position['quantity'] + pnl
                
                trade = Trade(
                    symbol=position['symbol'],
                    entry_time=position['entry_time'],
                    exit_time=current_date,
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    quantity=position['quantity'],
                    side='long',
                    pnl=pnl,
                    pnl_pct=(pnl / (position['entry_price'] * position['quantity'])) * 100,
                    commission=commission + position['commission']
                )
                trades.append(trade)
                position = None
                equity_curve.append(capital)
        
        # Close any remaining position at end
        if position:
            exit_price = data.iloc[-1]['close']
            pnl = (exit_price - position['entry_price']) * position['quantity']
            commission = exit_price * position['quantity'] * strategy.commission_pct
            pnl -= commission + position['commission']
            
            capital += position['entry_price'] * position['quantity'] + pnl
            
            trade = Trade(
                symbol=position['symbol'],
                entry_time=position['entry_time'],
                exit_time=data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now(),
                entry_price=position['entry_price'],
                exit_price=exit_price,
                quantity=position['quantity'],
                side='long',
                pnl=pnl,
                pnl_pct=(pnl / (position['entry_price'] * position['quantity'])) * 100,
                commission=commission + position['commission']
            )
            trades.append(trade)
            equity_curve.append(capital)
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades, equity_curve)
        
        result = BacktestResult(
            strategy_name=strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=capital,
            total_return=metrics['total_return'],
            total_return_pct=metrics['total_return_pct'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            max_drawdown_pct=metrics['max_drawdown_pct'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            total_trades=metrics['total_trades'],
            winning_trades=metrics['winning_trades'],
            losing_trades=metrics['losing_trades'],
            avg_win=metrics['avg_win'],
            avg_loss=metrics['avg_loss'],
            largest_win=metrics['largest_win'],
            largest_loss=metrics['largest_loss'],
            trades=trades,
            equity_curve=equity_curve
        )
        
        logger.info(
            f"Backtest complete: Return={result.total_return_pct:.2f}%, "
            f"Trades={result.total_trades}, Win Rate={result.win_rate:.2%}"
        )
        
        return result

    
    def calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: List[float]
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from trades.
        
        Args:
            trades: List of completed trades
            equity_curve: List of equity values over time
            
        Returns:
            Dictionary of performance metrics
        """
        if not trades:
            return {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        # Calculate total return
        total_return = sum(t.pnl for t in trades)
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Win/loss statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        total_trades = len(trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        
        win_rate = num_winning / total_trades if total_trades > 0 else 0.0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0.0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Max drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = equity_array - running_max
        max_drawdown = abs(np.min(drawdown))
        max_drawdown_pct = (max_drawdown / np.max(running_max)) * 100 if np.max(running_max) > 0 else 0.0
        
        # Sharpe ratio
        if len(equity_curve) > 1:
            returns = np.diff(equity_array)
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': num_winning,
            'losing_trades': num_losing,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
    
    def generate_equity_curve(self, trades: List[Trade]) -> pd.Series:
        """
        Generate equity curve from trades.
        
        Args:
            trades: List of completed trades
            
        Returns:
            Pandas Series with equity over time
        """
        if not trades:
            return pd.Series([self.initial_capital])
        
        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda t: t.exit_time)
        
        # Calculate cumulative equity
        equity = [self.initial_capital]
        for trade in sorted_trades:
            equity.append(equity[-1] + trade.pnl)
        
        # Create series with timestamps
        timestamps = [sorted_trades[0].entry_time] + [t.exit_time for t in sorted_trades]
        
        return pd.Series(equity, index=timestamps)
    
    def compare_strategies(
        self,
        strategies: List[Strategy],
        data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Compare performance of multiple strategies.
        
        Args:
            strategies: List of strategies to compare
            data: Historical price data
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for strategy in strategies:
            result = self.run_backtest(strategy, data, start_date, end_date)
            
            results.append({
                'Strategy': result.strategy_name,
                'Total Return': result.total_return,
                'Return %': result.total_return_pct,
                'Sharpe Ratio': result.sharpe_ratio,
                'Max Drawdown': result.max_drawdown,
                'Max Drawdown %': result.max_drawdown_pct,
                'Win Rate': result.win_rate,
                'Profit Factor': result.profit_factor,
                'Total Trades': result.total_trades,
                'Winning Trades': result.winning_trades,
                'Losing Trades': result.losing_trades
            })
        
        return pd.DataFrame(results)
