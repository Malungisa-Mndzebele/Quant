#!/usr/bin/env python3
"""
Main entry point for the Quantitative Trading System.

This CLI provides commands for running the trading system, backtesting strategies,
and validating configuration.
"""

import argparse
import sys
import signal
from datetime import datetime, date
from pathlib import Path

from src.trading_system import TradingSystem, TradingSystemError
from src.utils.config_loader import load_config, ConfigurationError


# Global reference for graceful shutdown
trading_system = None


def signal_handler(sig, frame):
    """Handle CTRL+C gracefully."""
    print("\n\nReceived interrupt signal (CTRL+C)")
    print("Shutting down gracefully...")
    
    if trading_system and trading_system.is_running():
        trading_system.shutdown()
    
    print("Shutdown complete. Goodbye!")
    sys.exit(0)


def run_command(args):
    """
    Run the trading system in live or simulation mode.
    
    Args:
        args: Parsed command-line arguments
    """
    global trading_system
    
    print("=" * 70)
    print("QUANTITATIVE TRADING SYSTEM")
    print("=" * 70)
    print()
    
    try:
        # Load configuration
        config_path = args.config
        print(f"Loading configuration from: {config_path}")
        
        # Initialize trading system
        trading_system = TradingSystem(config_path=config_path)
        trading_system.initialize()
        
        # Override mode if specified via CLI
        if args.mode:
            mode = args.mode.lower()
            if mode not in ['simulation', 'live']:
                print(f"Error: Invalid mode '{mode}'. Must be 'simulation' or 'live'")
                sys.exit(1)
            
            # Update config mode
            from src.models.config import TradingMode
            trading_system.config.trading.mode = TradingMode(mode)
            
            # Warn if switching to live mode
            if mode == 'live':
                print()
                print("!" * 70)
                print("WARNING: LIVE TRADING MODE ENABLED")
                print("Real money will be used for trading!")
                print("!" * 70)
                response = input("\nType 'YES' to confirm live trading: ")
                if response != 'YES':
                    print("Live trading cancelled.")
                    sys.exit(0)
                print()
        
        # Get symbols from config or use defaults
        symbols = trading_system.config.trading.symbols
        if not symbols:
            print("Error: No symbols configured for trading")
            sys.exit(1)
        
        # Get update interval
        interval = trading_system.config.trading.update_interval_seconds
        
        print()
        print(f"Starting trading system...")
        print(f"Mode: {trading_system.config.trading.mode.value.upper()}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Update Interval: {interval} seconds")
        print()
        print("Press CTRL+C to stop")
        print()
        
        # Run the trading loop
        trading_system.run(symbols=symbols, interval_seconds=interval)
        
    except ConfigurationError as e:
        print(f"\nConfiguration Error: {e}")
        sys.exit(1)
    except TradingSystemError as e:
        print(f"\nTrading System Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if trading_system:
            trading_system.shutdown()
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def backtest_command(args):
    """
    Run a backtest for a strategy.
    
    Args:
        args: Parsed command-line arguments
    """
    print("=" * 70)
    print("BACKTESTING")
    print("=" * 70)
    print()
    
    try:
        # Load configuration
        config_path = args.config
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        
        # Parse dates
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
        except ValueError as e:
            print(f"Error: Invalid date format. Use YYYY-MM-DD format.")
            print(f"Details: {e}")
            sys.exit(1)
        
        if start_date >= end_date:
            print("Error: Start date must be before end date")
            sys.exit(1)
        
        # Get symbols
        symbols = args.symbols if args.symbols else config.trading.symbols
        if not symbols:
            print("Error: No symbols specified. Use --symbols or configure in config file")
            sys.exit(1)
        
        # Get strategy
        strategy_name = args.strategy if args.strategy else None
        if not strategy_name and config.strategies:
            # Use first enabled strategy
            enabled_strategies = [s for s in config.strategies if s.enabled]
            if enabled_strategies:
                strategy_name = enabled_strategies[0].name
        
        if not strategy_name:
            print("Error: No strategy specified. Use --strategy or configure in config file")
            sys.exit(1)
        
        # Get initial capital
        initial_capital = args.capital if args.capital else config.trading.initial_capital
        
        print(f"Strategy: {strategy_name}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print()
        
        # Initialize data provider
        from src.data.yfinance_provider import YFinanceProvider
        data_provider = YFinanceProvider()
        
        # Create strategy instance
        strategy = create_strategy_instance(strategy_name, config)
        if not strategy:
            print(f"Error: Unknown strategy '{strategy_name}'")
            sys.exit(1)
        
        # Run backtest
        from src.backtesting.backtester import Backtester
        backtester = Backtester(data_provider)
        
        print("Running backtest...")
        result = backtester.run_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        print()
        print(result.generate_report())
        
        # Export results if requested
        if args.output:
            output_path = Path(args.output)
            
            # Save JSON result
            json_path = output_path.with_suffix('.json')
            result.save_to_file(str(json_path))
            print(f"\nResults saved to: {json_path}")
            
            # Export trades to CSV if requested
            if args.export_trades:
                csv_path = output_path.with_suffix('.csv')
                result.export_trades_csv(str(csv_path))
                print(f"Trades exported to: {csv_path}")
        
        # Generate equity curve visualization if requested
        if args.visualize:
            try:
                generate_equity_curve(result, args.output)
            except ImportError:
                print("\nWarning: matplotlib not installed. Cannot generate visualization.")
                print("Install with: pip install matplotlib")
        
    except ConfigurationError as e:
        print(f"\nConfiguration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def config_command(args):
    """
    Validate configuration file.
    
    Args:
        args: Parsed command-line arguments
    """
    print("=" * 70)
    print("CONFIGURATION VALIDATION")
    print("=" * 70)
    print()
    
    config_path = args.config
    print(f"Validating configuration: {config_path}")
    print()
    
    try:
        config = load_config(config_path)
        
        print("✓ Configuration is valid!")
        print()
        print("-" * 70)
        print("Configuration Summary:")
        print("-" * 70)
        print()
        print(f"Trading Mode: {config.trading.mode.value}")
        print(f"Initial Capital: ${config.trading.initial_capital:,.2f}")
        print(f"Symbols: {', '.join(config.trading.symbols) if config.trading.symbols else 'None configured'}")
        print(f"Update Interval: {config.trading.update_interval_seconds} seconds")
        print()
        print(f"Brokerage Provider: {config.brokerage.provider}")
        print(f"Data Provider: {config.data.provider}")
        print(f"Data Caching: {'Enabled' if config.data.cache_enabled else 'Disabled'}")
        print()
        print(f"Max Position Size: {config.risk.max_position_size_pct}%")
        print(f"Max Daily Loss: {config.risk.max_daily_loss_pct}%")
        print(f"Max Leverage: {config.risk.max_portfolio_leverage}x")
        print()
        
        enabled_strategies = [s for s in config.strategies if s.enabled]
        print(f"Strategies: {len(enabled_strategies)} enabled")
        for strategy in enabled_strategies:
            print(f"  - {strategy.name}: {strategy.params}")
        
        print()
        print("=" * 70)
        
    except ConfigurationError as e:
        print(f"✗ Configuration Error: {e}")
        print()
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_strategy_instance(strategy_name: str, config):
    """
    Create a strategy instance from name and configuration.
    
    Args:
        strategy_name: Name of the strategy
        config: System configuration
    
    Returns:
        Strategy instance or None
    """
    from src.strategies.moving_average_crossover import MovingAverageCrossover
    
    # Find strategy config
    strategy_config = None
    for s in config.strategies:
        if s.name.lower() == strategy_name.lower():
            strategy_config = s
            break
    
    # Create strategy instance
    strategy_name_lower = strategy_name.lower()
    
    if strategy_name_lower == 'movingaveragecrossover':
        params = strategy_config.params if strategy_config else {}
        return MovingAverageCrossover(
            fast_period=params.get('fast_period', 10),
            slow_period=params.get('slow_period', 30),
            quantity=params.get('quantity', 100)
        )
    
    return None


def generate_equity_curve(result, output_path):
    """
    Generate equity curve visualization.
    
    Args:
        result: BacktestResult object
        output_path: Base path for output file
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # Extract equity curve from trade history
    equity = [result.initial_capital]
    dates = [result.start_date]
    
    current_equity = result.initial_capital
    for trade in result.trade_history:
        if trade.pnl is not None:
            current_equity += trade.pnl
            equity.append(current_equity)
            dates.append(trade.timestamp.date() if hasattr(trade.timestamp, 'date') else trade.timestamp)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, equity, linewidth=2, color='#2E86AB')
    ax.fill_between(dates, equity, result.initial_capital, alpha=0.3, color='#2E86AB')
    
    # Add horizontal line at initial capital
    ax.axhline(y=result.initial_capital, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_title(f'Equity Curve - {result.strategy_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Add statistics box
    stats_text = (
        f"Return: {result.total_return:.2f}%\n"
        f"Sharpe: {result.sharpe_ratio:.2f}\n"
        f"Max DD: {result.max_drawdown:.2f}%\n"
        f"Trades: {result.num_trades}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, family='monospace')
    
    plt.tight_layout()
    
    # Save plot
    if output_path:
        plot_path = Path(output_path).with_suffix('.png')
    else:
        plot_path = Path(f'backtest_{result.strategy_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Equity curve saved to: {plot_path}")
    
    # Close plot to free memory
    plt.close()


def main():
    """Main entry point for the CLI."""
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create main parser
    parser = argparse.ArgumentParser(
        description='Quantitative Trading System - Algorithmic trading platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run trading system in simulation mode
  python main.py run
  
  # Run in live mode (requires confirmation)
  python main.py run --mode live
  
  # Run backtest
  python main.py backtest --start-date 2023-01-01 --end-date 2023-12-31 --symbols AAPL GOOGL
  
  # Validate configuration
  python main.py config
        """
    )
    
    # Add global arguments
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser(
        'run',
        help='Start the trading system'
    )
    run_parser.add_argument(
        '--mode',
        type=str,
        choices=['simulation', 'live'],
        help='Trading mode (overrides config file)'
    )
    run_parser.set_defaults(func=run_command)
    
    # Backtest command
    backtest_parser = subparsers.add_parser(
        'backtest',
        help='Run a strategy backtest'
    )
    backtest_parser.add_argument(
        '--strategy',
        type=str,
        help='Strategy name (default: first enabled strategy in config)'
    )
    backtest_parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Stock symbols to trade (default: from config)'
    )
    backtest_parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    backtest_parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date in YYYY-MM-DD format'
    )
    backtest_parser.add_argument(
        '--capital',
        type=float,
        help='Initial capital (default: from config)'
    )
    backtest_parser.add_argument(
        '--output',
        type=str,
        help='Output file path for results (JSON)'
    )
    backtest_parser.add_argument(
        '--export-trades',
        action='store_true',
        help='Export trade history to CSV'
    )
    backtest_parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate equity curve visualization (requires matplotlib)'
    )
    backtest_parser.set_defaults(func=backtest_command)
    
    # Config command
    config_parser = subparsers.add_parser(
        'config',
        help='Validate configuration file'
    )
    config_parser.set_defaults(func=config_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command specified
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
