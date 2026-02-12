"""Property-based and unit tests for scenario service."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

from services.scenario_service import (
    ScenarioService,
    MarketScenario,
    Position,
    ScenarioParameters,
    ScenarioResult
)


# Hypothesis strategies for generating test data
@st.composite
def position_strategy(draw):
    """Generate a valid Position for testing."""
    symbol = draw(st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=('Lu',))))
    quantity = draw(st.integers(min_value=1, max_value=1000))
    entry_price = draw(st.floats(min_value=1.0, max_value=1000.0))
    current_price = draw(st.floats(min_value=1.0, max_value=1000.0))
    market_value = current_price * quantity
    unrealized_pl = (current_price - entry_price) * quantity
    unrealized_pl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
    
    return Position(
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        current_price=current_price,
        market_value=market_value,
        unrealized_pl=unrealized_pl,
        unrealized_pl_pct=unrealized_pl_pct
    )


# Feature: ai-trading-agent, Property 19: Scenario simulation consistency
# Validates: Requirements 19.2
@given(
    scenario_type=st.sampled_from(list(MarketScenario)),
    positions=st.lists(position_strategy(), min_size=1, max_size=5),
    cash=st.floats(min_value=0.0, max_value=100000.0),
    random_seed=st.integers(min_value=0, max_value=1000000)
)
@settings(max_examples=100, deadline=None)
def test_scenario_simulation_consistency(scenario_type, positions, cash, random_seed):
    """
    Property 19: Scenario simulation consistency
    
    For any market scenario test, running the same scenario twice with identical 
    parameters should produce identical results.
    
    This property ensures that:
    1. Scenario simulations are deterministic when using the same random seed
    2. Results are reproducible for debugging and validation
    3. The simulation logic is consistent across runs
    
    Validates: Requirements 19.2
    """
    # Create historical volatility dict for all positions
    historical_volatility = {pos.symbol: 0.25 for pos in positions}
    
    # Run scenario first time with fixed random seed
    service1 = ScenarioService(random_seed=random_seed)
    result1 = service1.run_scenario(
        scenario_type=scenario_type,
        positions=positions,
        cash=cash,
        historical_volatility=historical_volatility
    )
    
    # Run scenario second time with same random seed
    service2 = ScenarioService(random_seed=random_seed)
    result2 = service2.run_scenario(
        scenario_type=scenario_type,
        positions=positions,
        cash=cash,
        historical_volatility=historical_volatility
    )
    
    # Verify all key metrics are identical
    assert result1.scenario_type == result2.scenario_type
    assert result1.duration_days == result2.duration_days
    assert result1.initial_portfolio_value == result2.initial_portfolio_value
    
    # For floating point comparisons, use small tolerance
    tolerance = 1e-6
    
    assert abs(result1.projected_portfolio_value - result2.projected_portfolio_value) < tolerance
    assert abs(result1.projected_return - result2.projected_return) < tolerance
    assert abs(result1.projected_return_pct - result2.projected_return_pct) < tolerance
    assert abs(result1.projected_max_drawdown - result2.projected_max_drawdown) < tolerance
    assert abs(result1.projected_max_drawdown_pct - result2.projected_max_drawdown_pct) < tolerance
    assert abs(result1.projected_volatility - result2.projected_volatility) < tolerance
    assert abs(result1.value_at_risk_95 - result2.value_at_risk_95) < tolerance
    assert abs(result1.probability_of_loss - result2.probability_of_loss) < tolerance
    assert abs(result1.risk_score - result2.risk_score) < tolerance
    
    # Verify position impacts are identical
    assert len(result1.position_impacts) == len(result2.position_impacts)
    
    for impact1, impact2 in zip(result1.position_impacts, result2.position_impacts):
        assert impact1['symbol'] == impact2['symbol']
        assert abs(impact1['projected_price'] - impact2['projected_price']) < tolerance
        assert abs(impact1['projected_pl'] - impact2['projected_pl']) < tolerance
        assert abs(impact1['probability_of_loss'] - impact2['probability_of_loss']) < tolerance


def test_basic():
    """Basic test."""
    service = ScenarioService()
    assert service is not None
