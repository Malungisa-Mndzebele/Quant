"""
Tests for trading strategy configuration.

This module tests strategy parameter validation, presets, and save/load functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from config.strategies import (
    StrategyParameters,
    StrategyPresets,
    StrategyManager,
    IndicatorConfig,
    IndicatorType,
    TradingStyle,
    get_strategy_preset,
    create_custom_strategy
)


class TestIndicatorConfig:
    """Tests for IndicatorConfig"""
    
    def test_valid_indicator_config(self):
        """Test creating valid indicator configuration"""
        config = IndicatorConfig(
            indicator_type=IndicatorType.RSI.value,
            enabled=True,
            weight=0.8,
            parameters={"period": 14}
        )
        assert config.indicator_type == "rsi"
        assert config.enabled is True
        assert config.weight == 0.8
        assert config.parameters["period"] == 14
    
    def test_invalid_indicator_type(self):
        """Test that invalid indicator type raises error"""
        with pytest.raises(ValueError, match="Invalid indicator type"):
            IndicatorConfig(
                indicator_type="invalid_indicator",
                enabled=True,
                weight=0.8
            )
    
    def test_invalid_weight(self):
        """Test that invalid weight raises error"""
        with pytest.raises(ValueError, match="weight must be between"):
            IndicatorConfig(
                indicator_type=IndicatorType.SMA.value,
                enabled=True,
                weight=1.5  # Invalid: > 1.0
            )
        
        with pytest.raises(ValueError, match="weight must be between"):
            IndicatorConfig(
                indicator_type=IndicatorType.SMA.value,
                enabled=True,
                weight=-0.1  # Invalid: < 0.0
            )


class TestStrategyParameters:
    """Tests for StrategyParameters"""
    
    def test_default_strategy(self):
        """Test creating strategy with default parameters"""
        strategy = StrategyParameters()
        assert strategy.name == "Default Strategy"
        assert strategy.style == TradingStyle.MODERATE.value
        assert len(strategy.indicators) > 0  # Should have default indicators
    
    def test_custom_strategy(self):
        """Test creating custom strategy"""
        strategy = StrategyParameters(
            name="Test Strategy",
            min_confidence=0.7,
            max_position_size=0.08
        )
        assert strategy.name == "Test Strategy"
        assert strategy.min_confidence == 0.7
        assert strategy.max_position_size == 0.08
    
    def test_validate_valid_strategy(self):
        """Test validation of valid strategy"""
        strategy = StrategyParameters()
        is_valid, errors = strategy.validate()
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_invalid_confidence(self):
        """Test validation catches invalid confidence values"""
        strategy = StrategyParameters(min_confidence=1.5)
        is_valid, errors = strategy.validate()
        assert is_valid is False
        assert any("min_confidence" in err for err in errors)
    
    def test_validate_invalid_model_weights(self):
        """Test validation catches invalid model weights"""
        strategy = StrategyParameters(
            lstm_weight=0.5,
            rf_weight=0.5,
            sentiment_weight=0.5  # Sum > 1.0
        )
        is_valid, errors = strategy.validate()
        assert is_valid is False
        assert any("weights must sum" in err for err in errors)
    
    def test_validate_invalid_risk_parameters(self):
        """Test validation catches invalid risk parameters"""
        strategy = StrategyParameters(max_position_size=1.5)
        is_valid, errors = strategy.validate()
        assert is_valid is False
        assert any("max_position_size" in err for err in errors)
    
    def test_validate_invalid_holding_period(self):
        """Test validation catches invalid holding periods"""
        strategy = StrategyParameters(
            min_holding_period=10,
            max_holding_period=5  # min > max
        )
        is_valid, errors = strategy.validate()
        assert is_valid is False
        assert any("holding_period" in err for err in errors)
    
    def test_to_dict(self):
        """Test converting strategy to dictionary"""
        strategy = StrategyParameters(name="Test")
        data = strategy.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "Test"
        assert "indicators" in data
        assert isinstance(data["indicators"], list)
    
    def test_from_dict(self):
        """Test creating strategy from dictionary"""
        data = {
            "name": "Test Strategy",
            "style": "moderate",
            "min_confidence": 0.7,
            "indicators": [
                {
                    "indicator_type": "rsi",
                    "enabled": True,
                    "weight": 0.8,
                    "parameters": {"period": 14}
                }
            ]
        }
        strategy = StrategyParameters.from_dict(data)
        assert strategy.name == "Test Strategy"
        assert strategy.min_confidence == 0.7
        assert len(strategy.indicators) == 1
        assert strategy.indicators[0].indicator_type == "rsi"


class TestStrategyPresets:
    """Tests for strategy presets"""
    
    def test_conservative_preset(self):
        """Test conservative strategy preset"""
        strategy = StrategyPresets.conservative()
        assert strategy.style == TradingStyle.CONSERVATIVE.value
        assert strategy.min_confidence >= 0.7  # High confidence
        assert strategy.max_position_size <= 0.1  # Small positions
        assert strategy.stop_loss_pct <= 0.05  # Tight stop loss
        
        # Validate preset
        is_valid, errors = strategy.validate()
        assert is_valid is True, f"Conservative preset validation failed: {errors}"
    
    def test_moderate_preset(self):
        """Test moderate strategy preset"""
        strategy = StrategyPresets.moderate()
        assert strategy.style == TradingStyle.MODERATE.value
        assert 0.5 <= strategy.min_confidence <= 0.7  # Moderate confidence
        assert strategy.max_position_size == 0.1  # Moderate positions
        
        # Validate preset
        is_valid, errors = strategy.validate()
        assert is_valid is True, f"Moderate preset validation failed: {errors}"
    
    def test_aggressive_preset(self):
        """Test aggressive strategy preset"""
        strategy = StrategyPresets.aggressive()
        assert strategy.style == TradingStyle.AGGRESSIVE.value
        assert strategy.min_confidence <= 0.6  # Lower confidence
        assert strategy.max_position_size >= 0.1  # Larger positions
        assert strategy.max_trades_per_day >= 15  # More trades
        
        # Validate preset
        is_valid, errors = strategy.validate()
        assert is_valid is True, f"Aggressive preset validation failed: {errors}"
    
    def test_get_preset(self):
        """Test getting preset by style"""
        conservative = StrategyPresets.get_preset(TradingStyle.CONSERVATIVE)
        assert conservative.style == TradingStyle.CONSERVATIVE.value
        
        moderate = StrategyPresets.get_preset(TradingStyle.MODERATE)
        assert moderate.style == TradingStyle.MODERATE.value
        
        aggressive = StrategyPresets.get_preset(TradingStyle.AGGRESSIVE)
        assert aggressive.style == TradingStyle.AGGRESSIVE.value
        
        # Test default (custom should return moderate)
        default = StrategyPresets.get_preset(TradingStyle.CUSTOM)
        assert default.style == TradingStyle.MODERATE.value


class TestStrategyManager:
    """Tests for StrategyManager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def manager(self, temp_dir):
        """Create strategy manager with temporary directory"""
        return StrategyManager(strategies_dir=temp_dir)
    
    def test_save_strategy(self, manager):
        """Test saving strategy to file"""
        strategy = StrategyParameters(name="Test Strategy")
        success = manager.save_strategy(strategy)
        assert success is True
        
        # Check file was created
        files = list(manager.strategies_dir.glob("*.json"))
        assert len(files) == 1
    
    def test_save_invalid_strategy(self, manager):
        """Test that invalid strategy cannot be saved"""
        strategy = StrategyParameters(min_confidence=1.5)  # Invalid
        success = manager.save_strategy(strategy)
        assert success is False
    
    def test_load_strategy(self, manager):
        """Test loading strategy from file"""
        # Save a strategy
        original = StrategyParameters(
            name="Test Strategy",
            min_confidence=0.75,
            signal_threshold=0.85  # Must be >= min_confidence
        )
        manager.save_strategy(original)
        
        # Load it back
        loaded = manager.load_strategy("test_strategy")
        assert loaded is not None
        assert loaded.name == "Test Strategy"
        assert loaded.min_confidence == 0.75
    
    def test_load_nonexistent_strategy(self, manager):
        """Test loading non-existent strategy returns None"""
        loaded = manager.load_strategy("nonexistent")
        assert loaded is None
    
    def test_list_strategies(self, manager):
        """Test listing saved strategies"""
        # Save multiple strategies
        manager.save_strategy(StrategyParameters(name="Strategy 1"))
        manager.save_strategy(StrategyParameters(name="Strategy 2"))
        manager.save_strategy(StrategyParameters(name="Strategy 3"))
        
        # List them
        strategies = manager.list_strategies()
        assert len(strategies) == 3
        assert "strategy_1" in strategies
        assert "strategy_2" in strategies
        assert "strategy_3" in strategies
    
    def test_delete_strategy(self, manager):
        """Test deleting a strategy"""
        # Save a strategy
        manager.save_strategy(StrategyParameters(name="Test Strategy"))
        
        # Delete it
        success = manager.delete_strategy("test_strategy")
        assert success is True
        
        # Verify it's gone
        strategies = manager.list_strategies()
        assert "test_strategy" not in strategies
    
    def test_delete_nonexistent_strategy(self, manager):
        """Test deleting non-existent strategy returns False"""
        success = manager.delete_strategy("nonexistent")
        assert success is False
    
    def test_get_preset_strategies(self, manager):
        """Test getting all preset strategies"""
        presets = manager.get_preset_strategies()
        assert "conservative" in presets
        assert "moderate" in presets
        assert "aggressive" in presets
        assert isinstance(presets["conservative"], StrategyParameters)


class TestConvenienceFunctions:
    """Tests for convenience functions"""
    
    def test_get_strategy_preset(self):
        """Test get_strategy_preset function"""
        conservative = get_strategy_preset("conservative")
        assert conservative.style == TradingStyle.CONSERVATIVE.value
        
        moderate = get_strategy_preset("moderate")
        assert moderate.style == TradingStyle.MODERATE.value
        
        aggressive = get_strategy_preset("aggressive")
        assert aggressive.style == TradingStyle.AGGRESSIVE.value
    
    def test_create_custom_strategy(self):
        """Test creating custom strategy with overrides"""
        strategy = create_custom_strategy(
            "My Custom Strategy",
            base_style="moderate",
            min_confidence=0.8,
            max_position_size=0.07
        )
        
        assert strategy.name == "My Custom Strategy"
        assert strategy.style == TradingStyle.CUSTOM.value
        assert strategy.min_confidence == 0.8
        assert strategy.max_position_size == 0.07
        
        # Should still have other moderate defaults
        assert strategy.max_trades_per_day == 10
    
    def test_create_custom_strategy_invalid(self):
        """Test that invalid custom strategy raises error"""
        with pytest.raises(ValueError, match="Invalid strategy parameters"):
            create_custom_strategy(
                "Invalid Strategy",
                base_style="moderate",
                min_confidence=1.5  # Invalid
            )


class TestStrategyParameterValidationProperty:
    """Property-based tests for strategy parameter validation"""
    
    def test_property_valid_parameters_pass_validation(self):
        """
        Property 11: Strategy parameter validation
        
        For any strategy with parameters within valid ranges, validation should pass.
        
        Validates: Requirements 11.2
        """
        from hypothesis import given, strategies as st, settings
        
        @given(
            min_confidence=st.floats(min_value=0.0, max_value=0.9),
            signal_threshold_offset=st.floats(min_value=0.0, max_value=0.1),
            lstm_weight=st.floats(min_value=0.0, max_value=1.0),
            rf_weight=st.floats(min_value=0.0, max_value=1.0),
            max_position_size=st.floats(min_value=0.01, max_value=1.0),
            max_portfolio_risk=st.floats(min_value=0.001, max_value=1.0),
            stop_loss_pct=st.floats(min_value=0.001, max_value=0.99),
            take_profit_pct=st.floats(min_value=0.001, max_value=0.99),
            max_trades_per_day=st.integers(min_value=1, max_value=100),
            min_holding_period=st.integers(min_value=1, max_value=30),
            max_holding_period_offset=st.integers(min_value=0, max_value=60),
            trailing_stop_pct=st.floats(min_value=0.001, max_value=0.99),
            rebalance_frequency=st.integers(min_value=1, max_value=365)
        )
        @settings(max_examples=100)
        def property_test(
            min_confidence,
            signal_threshold_offset,
            lstm_weight,
            rf_weight,
            max_position_size,
            max_portfolio_risk,
            stop_loss_pct,
            take_profit_pct,
            max_trades_per_day,
            min_holding_period,
            max_holding_period_offset,
            trailing_stop_pct,
            rebalance_frequency
        ):
            # Feature: ai-trading-agent, Property 11: Strategy parameter validation
            
            # Calculate signal_threshold to be >= min_confidence
            signal_threshold = min(min_confidence + signal_threshold_offset, 1.0)
            
            # Calculate sentiment_weight to make weights sum to 1.0
            sentiment_weight = max(0.0, 1.0 - lstm_weight - rf_weight)
            
            # Adjust weights if they exceed 1.0
            total = lstm_weight + rf_weight + sentiment_weight
            if total > 1.0:
                # Normalize weights
                lstm_weight = lstm_weight / total
                rf_weight = rf_weight / total
                sentiment_weight = sentiment_weight / total
            
            # Calculate max_holding_period to be >= min_holding_period
            max_holding_period = min_holding_period + max_holding_period_offset
            
            # Create strategy with valid parameters
            strategy = StrategyParameters(
                name="Property Test Strategy",
                min_confidence=min_confidence,
                signal_threshold=signal_threshold,
                lstm_weight=lstm_weight,
                rf_weight=rf_weight,
                sentiment_weight=sentiment_weight,
                max_position_size=max_position_size,
                max_portfolio_risk=max_portfolio_risk,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                max_trades_per_day=max_trades_per_day,
                min_holding_period=min_holding_period,
                max_holding_period=max_holding_period,
                use_trailing_stop=True,
                trailing_stop_pct=trailing_stop_pct,
                rebalance_frequency=rebalance_frequency
            )
            
            # Validate
            is_valid, errors = strategy.validate()
            
            # Property: All parameters within valid ranges should pass validation
            assert is_valid, f"Valid parameters failed validation: {errors}"
            assert len(errors) == 0, f"Valid parameters produced errors: {errors}"
        
        # Run the property test
        property_test()
    
    def test_property_invalid_confidence_fails_validation(self):
        """
        Property: Invalid confidence values should fail validation.
        
        For any strategy with min_confidence or signal_threshold outside [0.0, 1.0],
        or min_confidence > signal_threshold, validation should fail.
        
        Validates: Requirements 11.2
        """
        from hypothesis import given, strategies as st, settings, assume
        
        @given(
            min_confidence=st.floats(min_value=-1.0, max_value=2.0),
            signal_threshold=st.floats(min_value=-1.0, max_value=2.0)
        )
        @settings(max_examples=100)
        def property_test(min_confidence, signal_threshold):
            # Feature: ai-trading-agent, Property 11: Strategy parameter validation
            
            # Only test invalid cases
            is_invalid = (
                min_confidence < 0.0 or min_confidence > 1.0 or
                signal_threshold < 0.0 or signal_threshold > 1.0 or
                min_confidence > signal_threshold
            )
            assume(is_invalid)
            
            # Create strategy with invalid confidence parameters
            strategy = StrategyParameters(
                min_confidence=min_confidence,
                signal_threshold=signal_threshold
            )
            
            # Validate
            is_valid, errors = strategy.validate()
            
            # Property: Invalid confidence should fail validation
            assert not is_valid, f"Invalid confidence passed validation: min={min_confidence}, threshold={signal_threshold}"
            assert len(errors) > 0, "Invalid confidence produced no errors"
        
        # Run the property test
        property_test()
    
    def test_property_invalid_model_weights_fail_validation(self):
        """
        Property: Invalid model weights should fail validation.
        
        For any strategy where model weights don't sum to 1.0 (within tolerance),
        or any weight is outside [0.0, 1.0], validation should fail.
        
        Validates: Requirements 11.2
        """
        from hypothesis import given, strategies as st, settings, assume
        
        @given(
            lstm_weight=st.floats(min_value=-0.5, max_value=1.5),
            rf_weight=st.floats(min_value=-0.5, max_value=1.5),
            sentiment_weight=st.floats(min_value=-0.5, max_value=1.5)
        )
        @settings(max_examples=100)
        def property_test(lstm_weight, rf_weight, sentiment_weight):
            # Feature: ai-trading-agent, Property 11: Strategy parameter validation
            
            total = lstm_weight + rf_weight + sentiment_weight
            
            # Only test invalid cases
            is_invalid = (
                lstm_weight < 0.0 or lstm_weight > 1.0 or
                rf_weight < 0.0 or rf_weight > 1.0 or
                sentiment_weight < 0.0 or sentiment_weight > 1.0 or
                total < 0.99 or total > 1.01  # Outside tolerance
            )
            assume(is_invalid)
            
            # Create strategy with invalid weights
            strategy = StrategyParameters(
                lstm_weight=lstm_weight,
                rf_weight=rf_weight,
                sentiment_weight=sentiment_weight
            )
            
            # Validate
            is_valid, errors = strategy.validate()
            
            # Property: Invalid weights should fail validation
            assert not is_valid, f"Invalid weights passed validation: lstm={lstm_weight}, rf={rf_weight}, sentiment={sentiment_weight}"
            assert len(errors) > 0, "Invalid weights produced no errors"
        
        # Run the property test
        property_test()
    
    def test_property_invalid_risk_parameters_fail_validation(self):
        """
        Property: Invalid risk parameters should fail validation.
        
        For any strategy with risk parameters outside valid ranges,
        validation should fail.
        
        Validates: Requirements 11.2
        """
        from hypothesis import given, strategies as st, settings, assume
        
        @given(
            max_position_size=st.floats(min_value=-0.5, max_value=1.5),
            max_portfolio_risk=st.floats(min_value=-0.5, max_value=1.5),
            stop_loss_pct=st.floats(min_value=-0.5, max_value=1.5),
            take_profit_pct=st.floats(min_value=-0.5, max_value=1.5)
        )
        @settings(max_examples=100)
        def property_test(max_position_size, max_portfolio_risk, stop_loss_pct, take_profit_pct):
            # Feature: ai-trading-agent, Property 11: Strategy parameter validation
            
            # Only test invalid cases
            is_invalid = (
                max_position_size <= 0.0 or max_position_size > 1.0 or
                max_portfolio_risk <= 0.0 or max_portfolio_risk > 1.0 or
                stop_loss_pct <= 0.0 or stop_loss_pct >= 1.0 or
                take_profit_pct <= 0.0 or take_profit_pct >= 1.0
            )
            assume(is_invalid)
            
            # Create strategy with invalid risk parameters
            strategy = StrategyParameters(
                max_position_size=max_position_size,
                max_portfolio_risk=max_portfolio_risk,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct
            )
            
            # Validate
            is_valid, errors = strategy.validate()
            
            # Property: Invalid risk parameters should fail validation
            assert not is_valid, f"Invalid risk parameters passed validation"
            assert len(errors) > 0, "Invalid risk parameters produced no errors"
        
        # Run the property test
        property_test()
    
    def test_property_invalid_holding_periods_fail_validation(self):
        """
        Property: Invalid holding periods should fail validation.
        
        For any strategy where min_holding_period > max_holding_period,
        or either is non-positive, validation should fail.
        
        Validates: Requirements 11.2
        """
        from hypothesis import given, strategies as st, settings, assume
        
        @given(
            min_holding_period=st.integers(min_value=-10, max_value=100),
            max_holding_period=st.integers(min_value=-10, max_value=100)
        )
        @settings(max_examples=100)
        def property_test(min_holding_period, max_holding_period):
            # Feature: ai-trading-agent, Property 11: Strategy parameter validation
            
            # Only test invalid cases
            is_invalid = (
                min_holding_period <= 0 or
                max_holding_period <= 0 or
                min_holding_period > max_holding_period
            )
            assume(is_invalid)
            
            # Create strategy with invalid holding periods
            strategy = StrategyParameters(
                min_holding_period=min_holding_period,
                max_holding_period=max_holding_period
            )
            
            # Validate
            is_valid, errors = strategy.validate()
            
            # Property: Invalid holding periods should fail validation
            assert not is_valid, f"Invalid holding periods passed validation: min={min_holding_period}, max={max_holding_period}"
            assert len(errors) > 0, "Invalid holding periods produced no errors"
        
        # Run the property test
        property_test()
    
    def test_property_indicator_weight_bounds(self):
        """
        Property: Indicator weights must be within [0.0, 1.0].
        
        For any indicator with weight outside [0.0, 1.0],
        creation should fail.
        
        Validates: Requirements 11.2
        """
        from hypothesis import given, strategies as st, settings, assume
        
        @given(
            weight=st.floats(min_value=-1.0, max_value=2.0)
        )
        @settings(max_examples=100)
        def property_test(weight):
            # Feature: ai-trading-agent, Property 11: Strategy parameter validation
            
            # Only test invalid weights
            assume(weight < 0.0 or weight > 1.0)
            
            # Attempt to create indicator with invalid weight
            with pytest.raises(ValueError, match="weight must be between"):
                IndicatorConfig(
                    indicator_type=IndicatorType.RSI.value,
                    enabled=True,
                    weight=weight
                )
        
        # Run the property test
        property_test()


class TestStrategyIntegration:
    """Integration tests for strategy system"""
    
    def test_full_workflow(self):
        """Test complete workflow: create, save, load, modify, save"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StrategyManager(strategies_dir=Path(tmpdir))
            
            # 1. Create custom strategy
            strategy = create_custom_strategy(
                "My Strategy",
                base_style="moderate",
                min_confidence=0.75
            )
            
            # 2. Save it
            success = manager.save_strategy(strategy)
            assert success is True
            
            # 3. Load it back
            loaded = manager.load_strategy("my_strategy")
            assert loaded is not None
            assert loaded.name == "My Strategy"
            assert loaded.min_confidence == 0.75
            
            # 4. Modify it
            loaded.max_position_size = 0.08
            
            # 5. Save modified version
            success = manager.save_strategy(loaded, "my_strategy_v2")
            assert success is True
            
            # 6. Verify both versions exist
            strategies = manager.list_strategies()
            assert "my_strategy" in strategies
            assert "my_strategy_v2" in strategies
    
    def test_preset_to_custom_workflow(self):
        """Test workflow: load preset, customize, save"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StrategyManager(strategies_dir=Path(tmpdir))
            
            # 1. Get preset
            strategy = get_strategy_preset("conservative")
            
            # 2. Customize it
            strategy.name = "My Conservative Strategy"
            strategy.style = TradingStyle.CUSTOM.value
            strategy.min_confidence = 0.8
            
            # 3. Validate
            is_valid, errors = strategy.validate()
            assert is_valid is True
            
            # 4. Save
            success = manager.save_strategy(strategy)
            assert success is True
            
            # 5. Load and verify
            loaded = manager.load_strategy("my_conservative_strategy")
            assert loaded is not None
            assert loaded.min_confidence == 0.8
