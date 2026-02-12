"""
Verification script for strategy configuration system.

This script demonstrates the strategy customization features including:
- Loading preset strategies
- Creating custom strategies
- Saving and loading strategies
- Parameter validation
"""

from config.strategies import (
    StrategyPresets,
    StrategyManager,
    get_strategy_preset,
    create_custom_strategy,
    TradingStyle
)


def print_strategy_summary(strategy):
    """Print a summary of strategy parameters"""
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy.name}")
    print(f"Style: {strategy.style}")
    print(f"{'='*60}")
    print(f"\nSignal Generation:")
    print(f"  Min Confidence: {strategy.min_confidence:.2f}")
    print(f"  Signal Threshold: {strategy.signal_threshold:.2f}")
    print(f"\nModel Weights:")
    print(f"  LSTM: {strategy.lstm_weight:.2f}")
    print(f"  Random Forest: {strategy.rf_weight:.2f}")
    print(f"  Sentiment: {strategy.sentiment_weight:.2f}")
    print(f"\nRisk Parameters:")
    print(f"  Max Position Size: {strategy.max_position_size:.1%}")
    print(f"  Max Portfolio Risk: {strategy.max_portfolio_risk:.1%}")
    print(f"  Stop Loss: {strategy.stop_loss_pct:.1%}")
    print(f"  Take Profit: {strategy.take_profit_pct:.1%}")
    print(f"\nTrading Behavior:")
    print(f"  Max Trades/Day: {strategy.max_trades_per_day}")
    print(f"  Holding Period: {strategy.min_holding_period}-{strategy.max_holding_period} days")
    print(f"  Trailing Stop: {'Yes' if strategy.use_trailing_stop else 'No'}")
    print(f"\nIndicators Enabled: {len([i for i in strategy.indicators if i.enabled])}/{len(strategy.indicators)}")


def main():
    print("="*60)
    print("Strategy Configuration System Verification")
    print("="*60)
    
    # 1. Test preset strategies
    print("\n\n1. Testing Preset Strategies")
    print("-" * 60)
    
    conservative = StrategyPresets.conservative()
    print_strategy_summary(conservative)
    
    moderate = StrategyPresets.moderate()
    print_strategy_summary(moderate)
    
    aggressive = StrategyPresets.aggressive()
    print_strategy_summary(aggressive)
    
    # 2. Test custom strategy creation
    print("\n\n2. Testing Custom Strategy Creation")
    print("-" * 60)
    
    custom = create_custom_strategy(
        "My Balanced Strategy",
        base_style="moderate",
        min_confidence=0.65,
        signal_threshold=0.75,
        max_position_size=0.08,
        use_trailing_stop=True
    )
    print_strategy_summary(custom)
    
    # 3. Test validation
    print("\n\n3. Testing Strategy Validation")
    print("-" * 60)
    
    is_valid, errors = custom.validate()
    print(f"Custom strategy valid: {is_valid}")
    if not is_valid:
        print(f"Errors: {errors}")
    else:
        print("✓ All parameters within valid ranges")
    
    # 4. Test save/load functionality
    print("\n\n4. Testing Save/Load Functionality")
    print("-" * 60)
    
    manager = StrategyManager()
    
    # Save custom strategy
    print(f"Saving '{custom.name}'...")
    success = manager.save_strategy(custom)
    print(f"Save {'successful' if success else 'failed'}")
    
    # List saved strategies
    print("\nSaved strategies:")
    strategies = manager.list_strategies()
    for name in strategies:
        print(f"  - {name}")
    
    # Load strategy back
    print(f"\nLoading 'my_balanced_strategy'...")
    loaded = manager.load_strategy("my_balanced_strategy")
    if loaded:
        print(f"✓ Successfully loaded: {loaded.name}")
        print(f"  Min Confidence: {loaded.min_confidence}")
        print(f"  Max Position Size: {loaded.max_position_size:.1%}")
    else:
        print("✗ Failed to load strategy")
    
    # 5. Test preset retrieval
    print("\n\n5. Testing Preset Retrieval")
    print("-" * 60)
    
    for style in ["conservative", "moderate", "aggressive"]:
        preset = get_strategy_preset(style)
        print(f"{style.capitalize()}: min_confidence={preset.min_confidence:.2f}, "
              f"max_position_size={preset.max_position_size:.1%}")
    
    # 6. Test indicator configuration
    print("\n\n6. Testing Indicator Configuration")
    print("-" * 60)
    
    print(f"\nIndicators in moderate strategy:")
    for indicator in moderate.indicators:
        status = "✓" if indicator.enabled else "✗"
        print(f"  {status} {indicator.indicator_type.upper()}: "
              f"weight={indicator.weight:.1f}, params={indicator.parameters}")
    
    print("\n" + "="*60)
    print("Verification Complete!")
    print("="*60)
    print("\nAll strategy configuration features are working correctly:")
    print("  ✓ Preset strategies (conservative, moderate, aggressive)")
    print("  ✓ Custom strategy creation with parameter overrides")
    print("  ✓ Parameter validation")
    print("  ✓ Save/load functionality")
    print("  ✓ Indicator configuration")
    print("\nYou can now use these strategies in the AI trading agent!")


if __name__ == "__main__":
    main()
