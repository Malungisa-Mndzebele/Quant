"""Quick test script to verify all models work correctly."""

from datetime import datetime
from src.models import (
    MarketData, Quote, Signal, SignalAction, Order, OrderAction, OrderType,
    OrderStatusEnum, Trade, Position, Portfolio, PerformanceMetrics,
    RiskConfig, TradingConfig, BrokerageConfig, TradingMode
)


def test_market_data():
    """Test market data models."""
    print("Testing MarketData and Quote...")
    
    # Test MarketData
    md = MarketData(
        symbol="AAPL",
        timestamp=datetime.now(),
        open=150.0,
        high=155.0,
        low=149.0,
        close=154.0,
        volume=1000000
    )
    print(f"  ✓ MarketData: {md.symbol} @ ${md.close}")
    
    # Test Quote
    quote = Quote(
        symbol="AAPL",
        price=154.0,
        bid=153.9,
        ask=154.1,
        volume=1000,
        timestamp=datetime.now()
    )
    print(f"  ✓ Quote: {quote.symbol} @ ${quote.price}")


def test_signals_and_orders():
    """Test signal and order models."""
    print("\nTesting Signals and Orders...")
    
    # Test Signal
    signal = Signal(
        symbol="AAPL",
        action=SignalAction.BUY,
        quantity=10,
        order_type=OrderType.MARKET
    )
    print(f"  ✓ Signal: {signal.action.value} {signal.quantity} {signal.symbol}")
    
    # Test Order
    order = Order(
        order_id="ORD001",
        symbol="AAPL",
        action=OrderAction.BUY,
        quantity=10,
        order_type=OrderType.MARKET
    )
    print(f"  ✓ Order: {order.order_id} - {order.action.value} {order.quantity} {order.symbol}")
    
    # Test Trade
    trade = Trade(
        trade_id="TRD001",
        symbol="AAPL",
        action=OrderAction.BUY,
        quantity=10,
        price=150.0,
        timestamp=datetime.now(),
        pnl=50.0
    )
    print(f"  ✓ Trade: {trade.trade_id} - {trade.action.value} {trade.quantity} @ ${trade.price}")


def test_portfolio():
    """Test portfolio and position models."""
    print("\nTesting Portfolio and Positions...")
    
    # Create portfolio
    portfolio = Portfolio(initial_capital=100000.0)
    print(f"  ✓ Portfolio created with ${portfolio.initial_value:,.2f}")
    
    # Add position
    portfolio.add_position("AAPL", 10, 150.0)
    print(f"  ✓ Added position: 10 AAPL @ $150.00")
    
    # Update prices
    portfolio.update_prices({"AAPL": 155.0})
    
    # Get position
    position = portfolio.get_position("AAPL")
    print(f"  ✓ Position value: ${position.market_value:,.2f}")
    print(f"  ✓ Unrealized P&L: ${position.unrealized_pnl:,.2f}")
    
    # Portfolio metrics
    total_value = portfolio.get_total_value()
    total_return = portfolio.get_total_return(total_value)
    print(f"  ✓ Portfolio value: ${total_value:,.2f}")
    print(f"  ✓ Total return: {total_return:.2f}%")


def test_performance_metrics():
    """Test performance metrics."""
    print("\nTesting Performance Metrics...")
    
    # Test Sharpe ratio calculation
    returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
    print(f"  ✓ Sharpe ratio calculated: {sharpe:.2f}")
    
    # Test max drawdown calculation
    equity_curve = [100000, 105000, 103000, 108000, 102000]
    max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)
    print(f"  ✓ Max drawdown calculated: {max_dd:.2f}%")
    
    # Create performance metrics
    metrics = PerformanceMetrics(
        total_return=8.5,
        daily_return=0.5,
        sharpe_ratio=1.5,
        max_drawdown=5.2,
        win_rate=65.0,
        num_trades=20
    )
    print(f"  ✓ {metrics}")


def test_config_models():
    """Test configuration models."""
    print("\nTesting Configuration Models...")
    
    # Test RiskConfig
    risk_config = RiskConfig(
        max_position_size_pct=10.0,
        max_daily_loss_pct=5.0
    )
    print(f"  ✓ RiskConfig: max_position={risk_config.max_position_size_pct}%")
    
    # Test TradingConfig
    trading_config = TradingConfig(
        mode=TradingMode.SIMULATION,
        initial_capital=100000.0
    )
    print(f"  ✓ TradingConfig: mode={trading_config.mode.value}, capital=${trading_config.initial_capital:,.2f}")
    
    # Test BrokerageConfig
    brokerage_config = BrokerageConfig(
        provider="simulated"
    )
    print(f"  ✓ BrokerageConfig: provider={brokerage_config.provider}, simulated={brokerage_config.is_simulated()}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Core Data Models")
    print("=" * 60)
    
    test_market_data()
    test_signals_and_orders()
    test_portfolio()
    test_performance_metrics()
    test_config_models()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)
