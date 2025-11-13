"""Backtesting engine for strategy evaluation."""

from datetime import date, datetime
from typing import List, Dict, Optional
import pandas as pd

from ..models.market_data import MarketData
from ..models.order import Order, OrderAction, OrderType, OrderStatusEnum, Trade
from ..models.portfolio import Portfolio
from ..strategies.base import Strategy
from ..data.base import MarketDataProvider


class Backtester:
    """Backtesting engine that simulates strategy performance using historical data."""
    
    def __init__(self, data_provider: MarketDataProvider):
        """
        Initialize backtester with a market data provider.
        
        Args:
            data_provider: Market data provider for historical data
        """
        self.data_provider = data_provider
        self.portfolio: Optional[Portfolio] = None
        self.strategy: Optional[Strategy] = None
        self.trades: List[Trade] = []
        self._order_counter = 0
    
    def run_backtest(
        self,
        strategy: Strategy,
        symbols: List[str],
        start_date: date,
        end_date: date,
        initial_capital: float
    ) -> 'BacktestResult':
        """
        Execute a backtest for a strategy over a date range.
        
        Args:
            strategy: Trading strategy to test
            symbols: List of stock symbols to trade
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Starting capital for the backtest
        
        Returns:
            BacktestResult object with performance metrics and trade history
        """
        # Initialize portfolio and strategy
        self.portfolio = Portfolio(initial_capital)
        self.strategy = strategy
        self.trades = []
        self._order_counter = 0
        
        # Reset strategy state
        strategy.reset_state()
        
        # Fetch historical data for all symbols
        historical_data = {}
        for symbol in symbols:
            try:
                df = self.data_provider.get_historical_data(symbol, start_date, end_date)
                if df is not None and not df.empty:
                    historical_data[symbol] = df
            except Exception as e:
                print(f"Warning: Could not fetch data for {symbol}: {e}")
        
        if not historical_data:
            raise ValueError("No historical data available for any symbols")
        
        # Get all unique dates across all symbols
        all_dates = set()
        for df in historical_data.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)
        
        # Iterate through each date
        for current_date in all_dates:
            # Build current prices dict for this date
            current_prices = {}
            
            # Process market data for each symbol
            for symbol, df in historical_data.items():
                if current_date not in df.index:
                    continue
                
                row = df.loc[current_date]
                
                # Create MarketData object
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=current_date if isinstance(current_date, datetime) else datetime.combine(current_date, datetime.min.time()),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume'])
                )
                
                # Update current prices
                current_prices[symbol] = market_data.close
                
                # Feed data to strategy
                strategy.on_data(market_data)
            
            # Generate signals from strategy
            signals = strategy.generate_signals()
            
            # Process each signal
            for signal in signals:
                # Create order from signal
                order = self._create_order_from_signal(signal)
                
                # Skip if order creation failed (e.g., HOLD signal)
                if order is None:
                    continue
                
                # Simulate order fill
                if signal.symbol in current_prices:
                    filled_order = self._simulate_order_fill(order, current_prices[signal.symbol])
                    
                    # Update portfolio based on filled order
                    self._update_portfolio(filled_order, current_prices[signal.symbol])
                    
                    # Notify strategy of fill
                    strategy.on_order_filled(filled_order)
            
            # Update portfolio prices and record daily value
            self.portfolio.update_prices(current_prices)
            portfolio_value = self.portfolio.get_total_value()
            self.portfolio.record_daily_value(
                current_date if isinstance(current_date, datetime) else datetime.combine(current_date, datetime.min.time()),
                portfolio_value
            )
        
        # Calculate final performance metrics
        final_value = self.portfolio.get_total_value()
        performance = self.portfolio.calculate_returns()
        
        # Create and return backtest result
        from .result import BacktestResult
        return BacktestResult(
            strategy_name=strategy.name,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_value=final_value,
            total_return=performance.total_return,
            sharpe_ratio=performance.sharpe_ratio,
            max_drawdown=performance.max_drawdown,
            num_trades=len(self.trades),
            win_rate=performance.win_rate,
            trade_history=self.trades.copy()
        )
    
    def _create_order_from_signal(self, signal) -> Order:
        """
        Create an order from a trading signal.
        
        Args:
            signal: Trading signal
        
        Returns:
            Order object
        """
        self._order_counter += 1
        order_id = f"BACKTEST_{self._order_counter}"
        
        # Convert SignalAction to OrderAction
        from ..models.signal import SignalAction
        if signal.action == SignalAction.BUY:
            order_action = OrderAction.BUY
        elif signal.action == SignalAction.SELL:
            order_action = OrderAction.SELL
        else:
            # Handle HOLD or other actions - skip creating order
            return None
        
        return Order(
            order_id=order_id,
            symbol=signal.symbol,
            action=order_action,
            quantity=signal.quantity,
            order_type=OrderType(signal.order_type),
            limit_price=signal.limit_price,
            timestamp=signal.timestamp
        )
    
    def _simulate_order_fill(self, order: Order, current_price: float) -> Order:
        """
        Simulate order fill based on current market price.
        
        Args:
            order: Order to fill
            current_price: Current market price
        
        Returns:
            Filled order
        """
        # Determine fill price based on order type
        if order.order_type == OrderType.MARKET:
            fill_price = current_price
        elif order.order_type == OrderType.LIMIT:
            # For buy limit orders, fill if current price <= limit price
            # For sell limit orders, fill if current price >= limit price
            if order.action == OrderAction.BUY:
                if current_price <= order.limit_price:
                    fill_price = order.limit_price
                else:
                    # Order not filled
                    return order
            else:  # SELL
                if current_price >= order.limit_price:
                    fill_price = order.limit_price
                else:
                    # Order not filled
                    return order
        elif order.order_type == OrderType.STOP_LOSS:
            # Simplified stop loss: fill at current price
            fill_price = current_price
        else:
            fill_price = current_price
        
        # Update order with fill information
        order.status = OrderStatusEnum.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        
        return order
    
    def _update_portfolio(self, order: Order, fill_price: float) -> None:
        """
        Update portfolio based on filled order.
        
        Args:
            order: Filled order
            fill_price: Price at which order was filled
        """
        if order.status != OrderStatusEnum.FILLED:
            return
        
        # Calculate trade cost
        trade_value = fill_price * order.quantity
        
        if order.action == OrderAction.BUY:
            # Deduct cash for buy
            self.portfolio.cash_balance -= trade_value
            
            # Add or update position
            self.portfolio.add_position(order.symbol, order.quantity, fill_price)
        
        elif order.action == OrderAction.SELL:
            # Add cash for sell
            self.portfolio.cash_balance += trade_value
            
            # Calculate PnL if we have a position
            pnl = None
            if order.symbol in self.portfolio.positions:
                position = self.portfolio.positions[order.symbol]
                pnl = (fill_price - position.entry_price) * order.quantity
            
            # Remove or update position
            self.portfolio.add_position(order.symbol, -order.quantity, fill_price)
            
            # Record trade with PnL
            trade = Trade(
                trade_id=order.order_id,
                symbol=order.symbol,
                action=order.action,
                quantity=order.quantity,
                price=fill_price,
                timestamp=order.timestamp,
                pnl=pnl
            )
            self.trades.append(trade)
            self.portfolio.record_trade(trade)
        
        # For buy orders, also record trade (without PnL)
        if order.action == OrderAction.BUY:
            trade = Trade(
                trade_id=order.order_id,
                symbol=order.symbol,
                action=order.action,
                quantity=order.quantity,
                price=fill_price,
                timestamp=order.timestamp,
                pnl=None
            )
            self.trades.append(trade)
            self.portfolio.record_trade(trade)
