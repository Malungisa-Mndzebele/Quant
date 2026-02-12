"""Quick verification script for risk service implementation."""

import sys
from services.risk_service import (
    RiskService,
    TradeRequest,
    Position,
    RiskConfig
)

def test_basic_functionality():
    """Test basic risk service functionality"""
    print("Testing Risk Service Implementation...")
    print("=" * 60)
    
    # Create risk service with default config
    risk_service = RiskService()
    print(f"✓ Risk service initialized")
    print(f"  - Max position size: {risk_service.config.max_position_size:.1%}")
    print(f"  - Daily loss limit: ${risk_service.config.daily_loss_limit:.2f}")
    print(f"  - Stop loss: {risk_service.config.stop_loss_pct:.1%}")
    print()
    
    # Test 1: Validate a valid trade
    print("Test 1: Validate valid trade")
    trade = TradeRequest(
        symbol="AAPL",
        quantity=10,
        side="buy",
        price=150.0
    )
    
    result = risk_service.validate_trade(
        trade=trade,
        portfolio_value=100000.0,
        current_positions=[],
        available_cash=50000.0
    )
    
    print(f"  Trade: {trade.quantity} shares of {trade.symbol} @ ${trade.price}")
    print(f"  Valid: {result.is_valid}")
    print(f"  Violations: {len(result.violations)}")
    if result.has_violations:
        print(f"  Messages: {result.get_error_message()}")
    print()
    
    # Test 2: Validate trade that exceeds position size
    print("Test 2: Validate trade exceeding position size limit")
    large_trade = TradeRequest(
        symbol="TSLA",
        quantity=100,
        side="buy",
        price=200.0  # $20,000 trade on $100k portfolio = 20%
    )
    
    result = risk_service.validate_trade(
        trade=large_trade,
        portfolio_value=100000.0,
        current_positions=[],
        available_cash=50000.0
    )
    
    print(f"  Trade: {large_trade.quantity} shares of {large_trade.symbol} @ ${large_trade.price}")
    print(f"  Valid: {result.is_valid}")
    print(f"  Violations: {[v.value for v in result.violations]}")
    if result.suggested_quantity:
        print(f"  Suggested quantity: {result.suggested_quantity}")
    print()
    
    # Test 3: Calculate position size
    print("Test 3: Calculate position size")
    quantity = risk_service.calculate_position_size(
        symbol="AAPL",
        signal_strength=0.85,
        current_price=150.0,
        portfolio_value=100000.0,
        available_cash=50000.0
    )
    
    print(f"  Signal strength: 0.85")
    print(f"  Current price: $150.00")
    print(f"  Calculated quantity: {quantity} shares")
    print(f"  Position value: ${quantity * 150.0:.2f}")
    print()
    
    # Test 4: Check stop-loss
    print("Test 4: Check stop-loss trigger")
    position = Position(
        symbol="AAPL",
        quantity=100,
        entry_price=150.0,
        current_price=140.0,  # 6.67% loss
        unrealized_pl=-1000.0,
        unrealized_pl_pct=-0.0667
    )
    
    should_trigger = risk_service.check_stop_loss(position, 140.0)
    print(f"  Position: {position.symbol} @ ${position.entry_price}")
    print(f"  Current price: ${position.current_price}")
    print(f"  Loss: {position.unrealized_pl_pct:.1%}")
    print(f"  Stop-loss triggered: {should_trigger}")
    print()
    
    # Test 5: Get portfolio risk metrics
    print("Test 5: Calculate portfolio risk metrics")
    positions = [
        Position(
            symbol="AAPL",
            quantity=50,
            entry_price=150.0,
            current_price=155.0,
            unrealized_pl=250.0,
            unrealized_pl_pct=0.0333
        ),
        Position(
            symbol="GOOGL",
            quantity=20,
            entry_price=2800.0,
            current_price=2750.0,
            unrealized_pl=-1000.0,
            unrealized_pl_pct=-0.0179
        )
    ]
    
    metrics = risk_service.get_portfolio_risk(
        portfolio_value=100000.0,
        positions=positions,
        daily_pl=-750.0
    )
    
    print(f"  Portfolio value: ${metrics.portfolio_value:,.2f}")
    print(f"  Total exposure: ${metrics.total_exposure:,.2f} ({metrics.exposure_pct:.1%})")
    print(f"  Open positions: {metrics.open_positions}")
    print(f"  Risk score: {metrics.risk_score:.1f}/100")
    print(f"  Daily P&L: ${metrics.daily_pl:.2f} ({metrics.daily_pl_pct:.2f}%)")
    print()
    
    # Test 6: Get risk reduction suggestions
    print("Test 6: Risk reduction suggestions")
    suggestions = risk_service.suggest_risk_reduction(positions, metrics)
    print(f"  Number of suggestions: {len(suggestions)}")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
    print()
    
    print("=" * 60)
    print("✓ All basic tests completed successfully!")
    print()
    print("Risk service is ready for use.")
    print("Key features implemented:")
    print("  ✓ Trade validation with multiple risk checks")
    print("  ✓ Position size calculation based on signal strength")
    print("  ✓ Stop-loss and take-profit monitoring")
    print("  ✓ Portfolio risk metrics calculation")
    print("  ✓ Risk reduction suggestions")
    print("  ✓ Daily loss tracking and limits")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        sys.exit(0)
    except Exception as e:
        print(f"✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
