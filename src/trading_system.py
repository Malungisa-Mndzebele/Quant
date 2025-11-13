"""Main trading system orchestration."""

import logging
from typing import Optional, Dict, List
from datetime import datetime

from src.models.config import SystemConfig, TradingMode
from src.utils.config_loader import load_config, ConfigurationError
from src.data.base import MarketDataProvider
from src.data.yfinance_provider import YFinanceProvider
from src.data.cache import DataCache
from src.strategies.base import Strategy
from src.strategies.moving_average_crossover import MovingAverageCrossover
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.models.portfolio import Portfolio
from src.brokers.base import BrokerageAdapter
from src.brokers.factory import BrokerageFactory
from src.brokers.errors import BrokerageError


class TradingSystemError(Exception):
    """Base exception for trading system errors."""
    pass


class TradingSystem:
    """
    Main trading system that orchestrates all components.
    
    This class initializes and coordinates:
    - Configuration loading
    - Market data providers
    - Trading strategies
    - Order management
    - Risk management
    - Portfolio tracking
    - Brokerage connections
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the trading system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Optional[SystemConfig] = None
        self.logger: Optional[logging.Logger] = None
        
        # Components
        self.data_provider: Optional[MarketDataProvider] = None
        self.data_cache: Optional[DataCache] = None
        self.strategies: List[Strategy] = []
        self.order_manager: Optional[OrderManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.portfolio: Optional[Portfolio] = None
        self.brokerage: Optional[BrokerageAdapter] = None
        
        self._initialized = False
        self._running = False
    
    def initialize(self) -> None:
        """
        Initialize all system components in the correct order.
        
        Raises:
            TradingSystemError: If initialization fails
        """
        try:
            # Step 1: Load configuration
            self._load_configuration()
            
            # Step 2: Setup logging
            self._setup_logging()
            
            self.logger.info("=" * 60)
            self.logger.info("Initializing Trading System")
            self.logger.info(f"Mode: {self.config.trading.mode.value.upper()}")
            self.logger.info("=" * 60)
            
            # Step 3: Initialize data provider
            self._initialize_data_provider()
            
            # Step 4: Initialize brokerage connection
            self._initialize_brokerage()
            
            # Step 5: Initialize portfolio
            self._initialize_portfolio()
            
            # Step 6: Initialize risk manager
            self._initialize_risk_manager()
            
            # Step 7: Initialize order manager
            self._initialize_order_manager()
            
            # Step 8: Initialize strategies
            self._initialize_strategies()
            
            self._initialized = True
            self.logger.info("=" * 60)
            self.logger.info("Trading System Initialized Successfully")
            self.logger.info("=" * 60)
            
        except Exception as e:
            error_msg = f"Failed to initialize trading system: {e}"
            if self.logger:
                self.logger.error(error_msg, exc_info=True)
            raise TradingSystemError(error_msg) from e
    
    def _load_configuration(self) -> None:
        """Load and validate configuration."""
        try:
            self.config = load_config(self.config_path)
        except ConfigurationError as e:
            raise TradingSystemError(f"Configuration error: {e}") from e
    
    def _setup_logging(self) -> None:
        """Setup comprehensive logging configuration with file and console handlers."""
        import os
        from logging.handlers import RotatingFileHandler
        
        log_level = getattr(logging, self.config.trading.log_level.upper())
        
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Mode indicator for all log messages
        mode_indicator = f"[{self.config.trading.mode.value.upper()}]"
        
        # Create custom formatter with mode indicator and context
        log_format = f'%(asctime)s - {mode_indicator} - [%(levelname)s] - %(name)s - %(message)s'
        formatter = logging.Formatter(
            log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler - INFO and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler - All logs with rotation
        log_file = os.path.join(log_dir, f"trading_system_{self.config.trading.mode.value}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler - ERROR and above only
        error_log_file = os.path.join(log_dir, f"trading_system_{self.config.trading.mode.value}_errors.log")
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        
        # Enhanced formatter for error logs with stack traces
        error_formatter = logging.Formatter(
            f'%(asctime)s - {mode_indicator} - [%(levelname)s] - %(name)s - %(funcName)s:%(lineno)d\n'
            f'%(message)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        root_logger.addHandler(error_handler)
        
        # Get trading system logger
        self.logger = logging.getLogger('TradingSystem')
        
        # Log initial setup info
        self.logger.debug(f"Logging configured: level={self.config.trading.log_level.upper()}")
        self.logger.debug(f"Log file: {log_file}")
        self.logger.debug(f"Error log file: {error_log_file}")
    
    def _initialize_data_provider(self) -> None:
        """Initialize market data provider."""
        self.logger.info(f"Initializing data provider: {self.config.data.provider}")
        
        if self.config.data.provider.lower() == 'yfinance':
            self.data_provider = YFinanceProvider()
        else:
            raise TradingSystemError(
                f"Unsupported data provider: {self.config.data.provider}"
            )
        
        # Initialize data cache if enabled
        if self.config.data.cache_enabled:
            # Convert TTL from seconds to days for DataCache
            max_age_days = max(1, self.config.data.cache_ttl_seconds // 86400)
            self.data_cache = DataCache(
                cache_dir=self.config.data.cache_dir,
                max_age_days=max_age_days
            )
            self.logger.info(f"Data caching enabled: {self.config.data.cache_dir}")
    
    def _initialize_brokerage(self) -> None:
        """Initialize brokerage connection and authenticate."""
        self.logger.info(f"Initializing brokerage: {self.config.brokerage.provider}")
        
        try:
            # Create brokerage adapter using factory
            self.brokerage = BrokerageFactory.create_adapter(
                config=self.config.brokerage,
                trading_mode=self.config.trading.mode,
                initial_capital=self.config.trading.initial_capital
            )
            
            # Authenticate
            self.logger.info("Authenticating with brokerage...")
            auth_success = self.brokerage.authenticate({
                'api_key': self.config.brokerage.api_key,
                'api_secret': self.config.brokerage.api_secret,
                'account_id': self.config.brokerage.account_id
            })
            
            if not auth_success:
                raise TradingSystemError("Brokerage authentication failed")
            
            self.logger.info("Brokerage authentication successful")
            
            # Retrieve and validate account information
            self._retrieve_account_info()
            
        except BrokerageError as e:
            raise TradingSystemError(f"Brokerage initialization failed: {e}") from e
    
    def _retrieve_account_info(self) -> None:
        """Retrieve and log account information from brokerage."""
        try:
            account_info = self.brokerage.get_account_info()
            
            self.logger.info("Account Information:")
            self.logger.info(f"  Account ID: {account_info.account_id}")
            self.logger.info(f"  Cash Balance: ${account_info.cash_balance:,.2f}")
            self.logger.info(f"  Buying Power: ${account_info.buying_power:,.2f}")
            self.logger.info(f"  Portfolio Value: ${account_info.portfolio_value:,.2f}")
            
            # Validate account has sufficient funds
            if account_info.cash_balance < 0:
                self.logger.warning("Account has negative cash balance!")
            
            # Get current positions
            positions = self.brokerage.get_positions()
            if positions:
                self.logger.info(f"  Current Positions: {len(positions)}")
                for pos in positions:
                    self.logger.info(
                        f"    {pos.symbol}: {pos.quantity} shares @ ${pos.entry_price:.2f}"
                    )
            else:
                self.logger.info("  Current Positions: None")
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve account info: {e}")
            raise
    
    def _initialize_portfolio(self) -> None:
        """Initialize portfolio tracker."""
        self.logger.info("Initializing portfolio tracker")
        
        # Get initial capital from config or account
        if self.config.brokerage.is_simulated():
            initial_capital = self.config.trading.initial_capital
        else:
            # Use actual account balance for live trading
            account_info = self.brokerage.get_account_info()
            initial_capital = account_info.cash_balance
        
        self.portfolio = Portfolio(initial_capital=initial_capital)
        
        # Load existing positions from brokerage
        positions = self.brokerage.get_positions()
        for pos in positions:
            self.portfolio.add_position(
                symbol=pos.symbol,
                quantity=pos.quantity,
                entry_price=pos.entry_price
            )
        
        self.logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")
    
    def _initialize_risk_manager(self) -> None:
        """Initialize risk management system."""
        self.logger.info("Initializing risk manager")
        
        self.risk_manager = RiskManager(self.config.risk)
        
        self.logger.info(f"Risk limits configured:")
        self.logger.info(f"  Max position size: {self.config.risk.max_position_size_pct}%")
        self.logger.info(f"  Max daily loss: {self.config.risk.max_daily_loss_pct}%")
        self.logger.info(f"  Max leverage: {self.config.risk.max_portfolio_leverage}x")
    
    def _initialize_order_manager(self) -> None:
        """Initialize order management system."""
        self.logger.info("Initializing order manager")
        
        self.order_manager = OrderManager(
            broker=self.brokerage,
            max_retries=self.config.brokerage.max_retries
        )
    
    def _initialize_strategies(self) -> None:
        """Initialize trading strategies."""
        self.logger.info("Initializing strategies")
        
        for strategy_config in self.config.strategies:
            if not strategy_config.enabled:
                self.logger.info(f"  Skipping disabled strategy: {strategy_config.name}")
                continue
            
            strategy = self._create_strategy(strategy_config)
            if strategy:
                self.strategies.append(strategy)
                self.logger.info(
                    f"  Loaded strategy: {strategy_config.name} "
                    f"with params {strategy_config.params}"
                )
        
        if not self.strategies:
            raise TradingSystemError("No strategies enabled")
        
        self.logger.info(f"Total strategies loaded: {len(self.strategies)}")
    
    def _create_strategy(self, strategy_config) -> Optional[Strategy]:
        """
        Create a strategy instance from configuration.
        
        Args:
            strategy_config: Strategy configuration
            
        Returns:
            Strategy instance or None if strategy type not found
        """
        strategy_name = strategy_config.name.lower()
        
        if strategy_name == 'movingaveragecrossover':
            return MovingAverageCrossover(
                fast_period=strategy_config.params.get('fast_period', 10),
                slow_period=strategy_config.params.get('slow_period', 30),
                quantity=strategy_config.params.get('quantity', 100)
            )
        else:
            self.logger.warning(f"Unknown strategy type: {strategy_config.name}")
            return None
    
    def is_initialized(self) -> bool:
        """Check if system is initialized."""
        return self._initialized
    
    def is_running(self) -> bool:
        """Check if system is running."""
        return self._running
    
    def is_live_mode(self) -> bool:
        """Check if system is in live trading mode."""
        return self.config and self.config.is_live_mode()
    
    def get_status(self) -> Dict:
        """
        Get current system status.
        
        Returns:
            Dictionary with system status information
        """
        if not self._initialized:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'running' if self._running else 'initialized',
            'mode': self.config.trading.mode.value,
            'strategies': len(self.strategies),
            'portfolio_value': self.portfolio.get_total_value({}),
            'cash_balance': self.portfolio.cash_balance,
            'positions': len(self.portfolio.positions)
        }
    
    def run(self, symbols: List[str], interval_seconds: int = 60, health_check_interval: int = 300) -> None:
        """
        Run the main trading loop.
        
        Args:
            symbols: List of stock symbols to trade
            interval_seconds: Interval between market data fetches (default: 60 seconds)
            health_check_interval: Interval between health checks in seconds (default: 300 seconds / 5 minutes)
            
        Raises:
            TradingSystemError: If system is not initialized or encounters an error
        """
        if not self._initialized:
            raise TradingSystemError("System must be initialized before running")
        
        if not symbols:
            raise TradingSystemError("No symbols provided for trading")
        
        self.logger.info("=" * 60)
        self.logger.info("Starting Trading Loop")
        self.logger.info(f"Symbols: {', '.join(symbols)}")
        self.logger.info(f"Interval: {interval_seconds} seconds")
        self.logger.info(f"Health Check Interval: {health_check_interval} seconds")
        self.logger.info("=" * 60)
        
        self._running = True
        cycle_count = 0
        last_health_check = datetime.now()
        
        try:
            import time
            from src.models.market_data import MarketData
            
            while self._running:
                cycle_count += 1
                cycle_start = datetime.now()
                
                self.logger.info(f"--- Trading Cycle #{cycle_count} ---")
                
                try:
                    # Step 1: Fetch market data for all symbols
                    self.logger.info(f"Fetching market data for {len(symbols)} symbols...")
                    market_data_list = self._fetch_market_data(symbols)
                    
                    if not market_data_list:
                        self.logger.warning("No market data retrieved, skipping cycle")
                        time.sleep(interval_seconds)
                        continue
                    
                    # Step 2: Update portfolio prices
                    self.logger.debug("Updating portfolio prices...")
                    current_prices = {md.symbol: md.close for md in market_data_list}
                    self.portfolio.update_prices(current_prices)
                    
                    # Step 3: Execute strategies on new market data
                    self.logger.info("Executing strategies...")
                    all_signals = []
                    for market_data in market_data_list:
                        for strategy in self.strategies:
                            try:
                                strategy.on_data(market_data)
                                signals = strategy.generate_signals()
                                if signals:
                                    all_signals.extend(signals)
                                    self.logger.info(
                                        f"Strategy '{strategy.name}' generated {len(signals)} signal(s) for {market_data.symbol}"
                                    )
                            except Exception as e:
                                self.logger.error(
                                    f"Error in strategy '{strategy.name}' for {market_data.symbol}: {e}",
                                    exc_info=True
                                )
                    
                    # Step 4: Process signals and create orders
                    if all_signals:
                        self.logger.info(f"Processing {len(all_signals)} signal(s)...")
                        self._process_signals(all_signals, current_prices)
                    else:
                        self.logger.debug("No signals generated this cycle")
                    
                    # Step 5: Track pending orders
                    self._track_pending_orders()
                    
                    # Step 6: Log portfolio status
                    self._log_portfolio_status()
                    
                    # Step 7: Periodic health check
                    time_since_health_check = (datetime.now() - last_health_check).total_seconds()
                    if time_since_health_check >= health_check_interval:
                        self.log_system_health()
                        last_health_check = datetime.now()
                    
                    # Calculate cycle duration
                    cycle_duration = (datetime.now() - cycle_start).total_seconds()
                    self.logger.debug(f"Cycle #{cycle_count} completed in {cycle_duration:.2f} seconds")
                    
                    # Wait for next cycle
                    self.logger.info(f"Waiting {interval_seconds} seconds until next cycle...")
                    time.sleep(interval_seconds)
                    
                except KeyboardInterrupt:
                    self.logger.info("Received interrupt signal, shutting down...")
                    break
                except Exception as e:
                    self.logger.error(
                        f"Error in trading loop cycle #{cycle_count}: {e}",
                        exc_info=True
                    )
                    self.logger.warning("Continuing to next cycle after error...")
                    # Continue running unless it's a critical error
                    time.sleep(interval_seconds)
        
        except Exception as e:
            self.logger.critical(f"Critical error in trading loop: {e}", exc_info=True)
            raise
        
        finally:
            self._running = False
            self.logger.info(f"Trading loop stopped after {cycle_count} cycles")
            self.log_performance_summary()
    
    def _fetch_market_data(self, symbols: List[str]) -> List:
        """
        Fetch market data for all symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            List of MarketData objects
        """
        from src.models.market_data import MarketData
        
        market_data_list = []
        for symbol in symbols:
            try:
                quote = self.data_provider.get_quote(symbol)
                
                # Create MarketData object from quote
                market_data = MarketData(
                    symbol=quote.symbol,
                    timestamp=quote.timestamp,
                    open=quote.price,  # For real-time quotes, use current price
                    high=quote.price,
                    low=quote.price,
                    close=quote.price,
                    volume=quote.volume
                )
                market_data_list.append(market_data)
                
                self.logger.debug(f"Fetched data for {symbol}: ${quote.price:.2f}")
                
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}: {e}")
        
        return market_data_list
    
    def _process_signals(self, signals: List, current_prices: Dict[str, float]) -> None:
        """
        Process trading signals: create orders, validate risk, and submit.
        
        Args:
            signals: List of Signal objects
            current_prices: Dictionary of current prices by symbol
        """
        from src.models.signal import SignalAction
        
        for signal in signals:
            try:
                # Skip HOLD signals
                if signal.action == SignalAction.HOLD:
                    continue
                
                self.logger.info(
                    f"Processing signal: {signal.action.value} {signal.quantity} "
                    f"{signal.symbol} @ {signal.order_type.value}"
                )
                
                # Step 1: Create order from signal
                order = self.order_manager.create_order(signal)
                
                # Step 2: Get current price for risk validation
                current_price = current_prices.get(signal.symbol)
                if current_price is None:
                    self.logger.error(f"No current price available for {signal.symbol}")
                    continue
                
                # Step 3: Validate order against risk rules
                is_valid, reason = self.risk_manager.validate_order(
                    order, self.portfolio, current_price
                )
                
                if not is_valid:
                    self.logger.warning(f"Order rejected by risk manager: {reason}")
                    continue
                
                # Step 4: Submit order to brokerage
                self.logger.info(f"Submitting order {order.order_id}...")
                success = self.order_manager.submit_order(order)
                
                if success:
                    self.logger.info(f"Order {order.order_id} submitted successfully")
                else:
                    self.logger.error(f"Failed to submit order {order.order_id}")
                
            except Exception as e:
                self.logger.error(f"Error processing signal for {signal.symbol}: {e}", exc_info=True)
    
    def _track_pending_orders(self) -> None:
        """Track status of pending orders and update portfolio on fills."""
        from src.models.order import OrderStatusEnum, OrderAction
        
        pending_orders = [
            order for order in self.order_manager.get_all_orders()
            if order.status in [OrderStatusEnum.PENDING, OrderStatusEnum.SUBMITTED]
        ]
        
        if not pending_orders:
            return
        
        self.logger.debug(f"Tracking {len(pending_orders)} pending order(s)...")
        
        for order in pending_orders:
            try:
                # Get order status from broker
                status = self.order_manager.track_order(order.order_id)
                
                # Check if order was filled
                if status.status == OrderStatusEnum.FILLED.value:
                    self.logger.info(
                        f"Order {order.order_id} FILLED: {order.action.value} "
                        f"{status.filled_quantity} {order.symbol} @ ${status.average_fill_price:.2f}"
                    )
                    
                    # Update portfolio
                    self._update_portfolio_on_fill(order, status)
                    
                    # Notify strategies
                    for strategy in self.strategies:
                        try:
                            strategy.on_order_filled(order)
                        except Exception as e:
                            self.logger.error(
                                f"Error notifying strategy '{strategy.name}' of fill: {e}"
                            )
                
                elif status.status == OrderStatusEnum.REJECTED.value:
                    self.logger.warning(
                        f"Order {order.order_id} REJECTED: {status.message}"
                    )
                
                elif status.status == OrderStatusEnum.CANCELLED.value:
                    self.logger.info(f"Order {order.order_id} CANCELLED")
                
            except Exception as e:
                self.logger.error(f"Error tracking order {order.order_id}: {e}")
    
    def _update_portfolio_on_fill(self, order, status) -> None:
        """
        Update portfolio when an order is filled.
        
        Args:
            order: Order object
            status: OrderStatus object with fill details
        """
        from src.models.order import OrderAction, Trade
        import uuid
        
        fill_price = status.average_fill_price
        fill_quantity = status.filled_quantity
        
        if order.action == OrderAction.BUY:
            # Add position and deduct cash
            cost = fill_price * fill_quantity
            self.portfolio.cash_balance -= cost
            self.portfolio.add_position(order.symbol, fill_quantity, fill_price)
            
            self.logger.info(
                f"Portfolio updated: Added {fill_quantity} {order.symbol} @ ${fill_price:.2f}, "
                f"Cash: ${self.portfolio.cash_balance:.2f}"
            )
        
        elif order.action == OrderAction.SELL:
            # Remove position and add cash
            proceeds = fill_price * fill_quantity
            self.portfolio.cash_balance += proceeds
            
            # Calculate P&L
            position = self.portfolio.get_position(order.symbol)
            pnl = None
            if position:
                pnl = (fill_price - position.entry_price) * fill_quantity
            
            # Update position
            self.portfolio.add_position(order.symbol, -fill_quantity, fill_price)
            
            pnl_str = f"${pnl:.2f}" if pnl is not None else "$0.00"
            self.logger.info(
                f"Portfolio updated: Sold {fill_quantity} {order.symbol} @ ${fill_price:.2f}, "
                f"P&L: {pnl_str}, Cash: ${self.portfolio.cash_balance:.2f}"
            )
            
            # Record trade
            trade = Trade(
                trade_id=f"TRD-{uuid.uuid4().hex[:12].upper()}",
                symbol=order.symbol,
                action=order.action.value,
                quantity=fill_quantity,
                price=fill_price,
                timestamp=datetime.now(),
                pnl=pnl
            )
            self.portfolio.record_trade(trade)
    
    def _log_portfolio_status(self) -> None:
        """Log current portfolio status with comprehensive metrics."""
        total_value = self.portfolio.get_total_value()
        positions_value = self.portfolio.get_positions_value()
        unrealized_pnl = self.portfolio.get_unrealized_pnl()
        total_return = self.portfolio.get_total_return(total_value)
        
        # Get risk metrics
        exposure = self.risk_manager.calculate_exposure(self.portfolio)
        daily_pnl = self.risk_manager.get_daily_pnl(self.portfolio)
        daily_pnl_pct = self.risk_manager.get_daily_pnl_pct(self.portfolio)
        
        self.logger.info("=" * 60)
        self.logger.info("PORTFOLIO STATUS")
        self.logger.info(f"Total Value: ${total_value:,.2f}")
        self.logger.info(f"Cash Balance: ${self.portfolio.cash_balance:,.2f} ({exposure['cash_pct']:.1f}%)")
        self.logger.info(f"Positions Value: ${positions_value:,.2f} ({exposure['exposure_pct']:.1f}%)")
        self.logger.info(f"Unrealized P&L: ${unrealized_pnl:,.2f}")
        self.logger.info(f"Total Return: {total_return:.2f}%")
        self.logger.info(f"Daily P&L: ${daily_pnl:,.2f} ({daily_pnl_pct:.2f}%)")
        self.logger.info(f"Number of Positions: {len(self.portfolio.positions)}")
        self.logger.info(f"Leverage: {exposure['leverage']:.2f}x")
        
        if self.portfolio.positions:
            self.logger.info("Positions:")
            for symbol, position in self.portfolio.positions.items():
                position_pct = (position.market_value / total_value) * 100 if total_value > 0 else 0
                self.logger.info(
                    f"  {symbol}: {position.quantity} shares @ ${position.entry_price:.2f} "
                    f"(Current: ${position.current_price:.2f}, P&L: ${position.unrealized_pnl:.2f}, "
                    f"{position_pct:.1f}% of portfolio)"
                )
        
        # Log risk warnings if applicable
        if daily_pnl_pct < -self.config.risk.max_daily_loss_pct * 0.8:
            self.logger.warning(
                f"Daily loss approaching limit: {daily_pnl_pct:.2f}% "
                f"(limit: -{self.config.risk.max_daily_loss_pct:.2f}%)"
            )
        
        if exposure['leverage'] > self.config.risk.max_portfolio_leverage * 0.9:
            self.logger.warning(
                f"Leverage approaching limit: {exposure['leverage']:.2f}x "
                f"(limit: {self.config.risk.max_portfolio_leverage:.2f}x)"
            )
        
        self.logger.info("=" * 60)
    
    def log_system_health(self) -> None:
        """Log comprehensive system health and monitoring information."""
        self.logger.info("=" * 60)
        self.logger.info("SYSTEM HEALTH CHECK")
        self.logger.info("=" * 60)
        
        # System status
        self.logger.info(f"Status: {'RUNNING' if self._running else 'STOPPED'}")
        self.logger.info(f"Mode: {self.config.trading.mode.value.upper()}")
        self.logger.info(f"Initialized: {self._initialized}")
        
        # Component status
        self.logger.info(f"Data Provider: {self.data_provider.__class__.__name__ if self.data_provider else 'None'}")
        self.logger.info(f"Brokerage: {self.brokerage.__class__.__name__ if self.brokerage else 'None'}")
        self.logger.info(f"Active Strategies: {len(self.strategies)}")
        
        # Order statistics
        all_orders = self.order_manager.get_all_orders()
        from src.models.order import OrderStatusEnum
        pending_count = sum(1 for o in all_orders if o.status in [OrderStatusEnum.PENDING, OrderStatusEnum.SUBMITTED])
        filled_count = sum(1 for o in all_orders if o.status == OrderStatusEnum.FILLED)
        rejected_count = sum(1 for o in all_orders if o.status == OrderStatusEnum.REJECTED)
        
        self.logger.info(f"Total Orders: {len(all_orders)}")
        self.logger.info(f"  Pending: {pending_count}")
        self.logger.info(f"  Filled: {filled_count}")
        self.logger.info(f"  Rejected: {rejected_count}")
        
        # Portfolio metrics
        total_value = self.portfolio.get_total_value()
        self.logger.info(f"Portfolio Value: ${total_value:,.2f}")
        self.logger.info(f"Total Trades: {len(self.portfolio.trade_history)}")
        
        # Risk metrics
        exposure = self.risk_manager.calculate_exposure(self.portfolio)
        self.logger.info(f"Current Leverage: {exposure['leverage']:.2f}x / {self.config.risk.max_portfolio_leverage:.2f}x")
        
        daily_pnl_pct = self.risk_manager.get_daily_pnl_pct(self.portfolio)
        self.logger.info(f"Daily P&L: {daily_pnl_pct:.2f}% / -{self.config.risk.max_daily_loss_pct:.2f}% limit")
        
        self.logger.info("=" * 60)
    
    def log_performance_summary(self) -> None:
        """Log comprehensive performance summary."""
        self.logger.info("=" * 60)
        self.logger.info("PERFORMANCE SUMMARY")
        self.logger.info("=" * 60)
        
        # Calculate performance metrics
        metrics = self.portfolio.calculate_returns()
        
        self.logger.info(f"Initial Capital: ${self.portfolio.initial_value:,.2f}")
        self.logger.info(f"Current Value: ${self.portfolio.get_total_value():,.2f}")
        self.logger.info(f"Total P&L: ${metrics.total_pnl:,.2f}")
        self.logger.info(f"Total Return: {metrics.total_return:.2f}%")
        self.logger.info(f"Daily Return: {metrics.daily_return:.2f}%")
        self.logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        self.logger.info(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
        self.logger.info(f"Total Trades: {metrics.num_trades}")
        self.logger.info(f"Win Rate: {metrics.win_rate:.2f}%")
        
        # Trade statistics
        if self.portfolio.trade_history:
            winning_trades = [t for t in self.portfolio.trade_history if t.pnl and t.pnl > 0]
            losing_trades = [t for t in self.portfolio.trade_history if t.pnl and t.pnl < 0]
            
            if winning_trades:
                avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
                self.logger.info(f"Average Win: ${avg_win:.2f}")
            
            if losing_trades:
                avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)
                self.logger.info(f"Average Loss: ${avg_loss:.2f}")
            
            if winning_trades and losing_trades:
                profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades))
                self.logger.info(f"Profit Factor: {profit_factor:.2f}")
        
        self.logger.info("=" * 60)
    
    def shutdown(self) -> None:
        """Shutdown the trading system gracefully."""
        self.logger.info("Shutting down trading system...")
        
        self._running = False
        
        # Close any open connections
        if self.brokerage:
            self.logger.info("Closing brokerage connection")
        
        self.logger.info("Trading system shutdown complete")
