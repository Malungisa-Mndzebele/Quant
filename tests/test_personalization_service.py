"""
Unit tests for personalization service.

Tests user preference learning, recommendation adjustment, and metrics tracking.
"""

import pytest
import os
import tempfile
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from services.personalization_service import (
    PersonalizationService,
    UserDecision,
    PersonalizationMetrics,
    create_personalization_service
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def service(temp_db):
    """Create personalization service with temporary database"""
    return PersonalizationService(
        db_path=temp_db,
        min_decisions_for_learning=5,  # Lower threshold for testing
        learning_rate=0.2
    )


class TestPersonalizationService:
    """Test suite for PersonalizationService"""
    
    def test_initialization(self, service):
        """Test service initialization"""
        assert service.db_path is not None
        assert service.min_decisions_for_learning == 5
        assert service.learning_rate == 0.2
        assert 'confidence_threshold' in service.preference_weights
        assert 'risk_preference' in service.preference_weights
        assert 'action_bias' in service.preference_weights
    
    def test_record_decision(self, service):
        """Test recording a user decision"""
        decision_id = service.record_decision(
            symbol='AAPL',
            recommendation_action='buy',
            recommendation_confidence=0.8,
            user_action='accepted',
            notes='Test decision'
        )
        
        assert decision_id > 0
        
        # Retrieve and verify
        decisions = service.get_decisions(symbol='AAPL')
        assert len(decisions) == 1
        assert decisions[0].symbol == 'AAPL'
        assert decisions[0].recommendation_action == 'buy'
        assert decisions[0].user_action == 'accepted'
    
    def test_record_multiple_decisions(self, service):
        """Test recording multiple decisions"""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        actions = ['buy', 'sell', 'hold']
        user_actions = ['accepted', 'rejected', 'accepted']
        
        for symbol, action, user_action in zip(symbols, actions, user_actions):
            service.record_decision(
                symbol=symbol,
                recommendation_action=action,
                recommendation_confidence=0.75,
                user_action=user_action
            )
        
        decisions = service.get_decisions()
        assert len(decisions) == 3
    
    def test_update_decision_outcome(self, service):
        """Test updating decision outcome"""
        decision_id = service.record_decision(
            symbol='AAPL',
            recommendation_action='buy',
            recommendation_confidence=0.8,
            user_action='accepted'
        )
        
        # Update outcome
        service.update_decision_outcome(
            decision_id=decision_id,
            outcome='profit',
            outcome_value=150.50
        )
        
        # Verify update
        decisions = service.get_decisions()
        assert decisions[0].outcome == 'profit'
        assert decisions[0].outcome_value == 150.50
    
    def test_get_decisions_with_filters(self, service):
        """Test filtering decisions"""
        # Record decisions for different symbols
        service.record_decision('AAPL', 'buy', 0.8, 'accepted')
        service.record_decision('GOOGL', 'sell', 0.7, 'rejected')
        service.record_decision('AAPL', 'hold', 0.6, 'accepted')
        
        # Filter by symbol
        aapl_decisions = service.get_decisions(symbol='AAPL')
        assert len(aapl_decisions) == 2
        assert all(d.symbol == 'AAPL' for d in aapl_decisions)
        
        # Filter by user action
        accepted_decisions = service.get_decisions(user_action='accepted')
        assert len(accepted_decisions) == 2
        assert all(d.user_action == 'accepted' for d in accepted_decisions)
    
    def test_preference_learning_insufficient_data(self, service):
        """Test that preferences don't update with insufficient data"""
        initial_threshold = service.preference_weights['confidence_threshold']
        
        # Record only 2 decisions (below threshold of 5)
        service.record_decision('AAPL', 'buy', 0.9, 'accepted')
        service.record_decision('GOOGL', 'sell', 0.5, 'rejected')
        
        # Preferences should not change
        assert service.preference_weights['confidence_threshold'] == initial_threshold
    
    def test_preference_learning_confidence_threshold(self, service):
        """Test learning confidence threshold from user behavior"""
        # Record decisions with high confidence that are accepted
        for i in range(6):
            service.record_decision(
                symbol=f'STOCK{i}',
                recommendation_action='buy',
                recommendation_confidence=0.85,
                user_action='accepted'
            )
        
        # Record decisions with low confidence that are rejected
        for i in range(4):
            service.record_decision(
                symbol=f'STOCK{i+6}',
                recommendation_action='buy',
                recommendation_confidence=0.55,
                user_action='rejected'
            )
        
        # Confidence threshold should increase toward accepted average
        # Should be around 0.85 * 0.9 = 0.765
        assert service.preference_weights['confidence_threshold'] > 0.7
    
    def test_preference_learning_action_bias(self, service):
        """Test learning action bias from user behavior"""
        # User accepts buy recommendations
        for i in range(5):
            service.record_decision(f'STOCK{i}', 'buy', 0.8, 'accepted')
        
        # User rejects sell recommendations
        for i in range(5):
            service.record_decision(f'STOCK{i+5}', 'sell', 0.8, 'rejected')
        
        # Buy bias should be high, sell bias should be low
        assert service.preference_weights['action_bias']['buy'] > 0.7
        assert service.preference_weights['action_bias']['sell'] < 0.3
    
    def test_get_personalization_metrics_no_data(self, service):
        """Test metrics with no decisions"""
        metrics = service.get_personalization_metrics()
        
        assert metrics.total_decisions == 0
        assert metrics.acceptance_rate == 0.0
        assert metrics.personalization_active is False
        assert metrics.personalization_confidence == 0.0
    
    def test_get_personalization_metrics_with_data(self, service):
        """Test metrics calculation with decisions"""
        # Record 10 decisions: 7 accepted, 2 rejected, 1 modified
        for i in range(7):
            service.record_decision(f'STOCK{i}', 'buy', 0.8, 'accepted')
        
        for i in range(2):
            service.record_decision(f'STOCK{i+7}', 'sell', 0.6, 'rejected')
        
        service.record_decision('STOCK9', 'hold', 0.7, 'modified')
        
        metrics = service.get_personalization_metrics()
        
        assert metrics.total_decisions == 10
        assert metrics.acceptance_rate == 0.7
        assert metrics.rejection_rate == 0.2
        assert metrics.modification_rate == 0.1
        assert metrics.personalization_active is True
        assert metrics.avg_confidence_accepted > 0.7
    
    def test_adjust_recommendation_insufficient_data(self, service):
        """Test that recommendations aren't adjusted with insufficient data"""
        recommendation = {
            'action': 'buy',
            'confidence': 0.75,
            'risk_level': 'medium'
        }
        
        # Record only 2 decisions (below threshold)
        service.record_decision('AAPL', 'buy', 0.8, 'accepted')
        service.record_decision('GOOGL', 'sell', 0.6, 'rejected')
        
        adjusted = service.adjust_recommendation(recommendation)
        
        # Should return unchanged
        assert adjusted['confidence'] == recommendation['confidence']
        assert 'personalized' not in adjusted
    
    def test_adjust_recommendation_with_learning(self, service):
        """Test recommendation adjustment after learning"""
        # Train: user accepts high-confidence buy recommendations
        for i in range(6):
            service.record_decision(f'STOCK{i}', 'buy', 0.85, 'accepted')
        
        recommendation = {
            'action': 'buy',
            'confidence': 0.75,
            'risk_level': 'medium'
        }
        
        adjusted = service.adjust_recommendation(recommendation)
        
        # Confidence should be boosted for buy actions
        assert adjusted['confidence'] >= recommendation['confidence']
        assert adjusted['personalized'] is True
        assert adjusted['personalization_confidence'] > 0
    
    def test_adjust_recommendation_action_bias(self, service):
        """Test adjustment based on action bias"""
        # User consistently rejects sell recommendations
        for i in range(6):
            service.record_decision(f'STOCK{i}', 'sell', 0.8, 'rejected')
        
        recommendation = {
            'action': 'sell',
            'confidence': 0.80,
            'risk_level': 'medium'
        }
        
        adjusted = service.adjust_recommendation(recommendation)
        
        # Confidence should be reduced for sell actions
        assert adjusted['confidence'] <= recommendation['confidence']
        # Should be marked as personalized
        assert adjusted['personalized'] is True
    
    def test_reset_personalization(self, service):
        """Test resetting personalization"""
        # Learn some preferences
        for i in range(6):
            service.record_decision(f'STOCK{i}', 'buy', 0.9, 'accepted')
        
        # Verify preferences changed
        assert service.preference_weights['action_bias']['buy'] > 0.5
        
        # Reset
        service.reset_personalization()
        
        # Preferences should be back to defaults
        assert service.preference_weights['confidence_threshold'] == 0.6
        assert service.preference_weights['risk_preference'] == 0.5
        assert service.preference_weights['action_bias']['buy'] == 0.5
        
        # But decisions should still exist
        decisions = service.get_decisions()
        assert len(decisions) == 6
    
    def test_improvement_score_calculation(self, service):
        """Test improvement score calculation"""
        # Record early decisions with low acceptance
        for i in range(10):
            action = 'accepted' if i < 3 else 'rejected'  # 30% acceptance
            service.record_decision(f'STOCK{i}', 'buy', 0.7, action)
        
        # Record recent decisions with high acceptance
        for i in range(10):
            action = 'accepted' if i < 8 else 'rejected'  # 80% acceptance
            service.record_decision(f'STOCK{i+10}', 'buy', 0.7, action)
        
        metrics = service.get_personalization_metrics()
        
        # Improvement score should be positive (acceptance increased)
        assert metrics.improvement_score >= 0.5
    
    def test_get_personalization_history(self, service):
        """Test retrieving personalization history"""
        # Record some decisions to trigger history recording
        for i in range(6):
            service.record_decision(f'STOCK{i}', 'buy', 0.8, 'accepted')
        
        history = service.get_personalization_history(days=30)
        
        # Should have at least one history entry
        assert len(history) > 0
        assert 'timestamp' in history.columns
        assert 'total_decisions' in history.columns
        assert 'acceptance_rate' in history.columns
    
    def test_export_decisions(self, service):
        """Test exporting decisions to DataFrame"""
        # Record some decisions
        service.record_decision('AAPL', 'buy', 0.8, 'accepted')
        service.record_decision('GOOGL', 'sell', 0.7, 'rejected')
        service.record_decision('MSFT', 'hold', 0.6, 'modified')
        
        df = service.export_decisions()
        
        assert len(df) == 3
        assert 'symbol' in df.columns
        assert 'recommendation_action' in df.columns
        assert 'user_action' in df.columns
        assert 'recommendation_confidence' in df.columns
    
    def test_export_decisions_with_date_filter(self, service):
        """Test exporting decisions with date filtering"""
        # Record decisions
        service.record_decision('AAPL', 'buy', 0.8, 'accepted')
        service.record_decision('GOOGL', 'sell', 0.7, 'rejected')
        
        # Export with date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        df = service.export_decisions(start_date=start_date, end_date=end_date)
        
        assert len(df) == 2
    
    def test_risk_preference_learning(self, service):
        """Test learning risk preference from outcomes"""
        # Record risky trades (low confidence) with good outcomes
        for i in range(6):
            decision_id = service.record_decision(
                symbol=f'STOCK{i}',
                recommendation_action='buy',
                recommendation_confidence=0.65,  # Risky
                user_action='accepted'
            )
            # Mark as profitable
            service.update_decision_outcome(decision_id, 'profit', 100.0)
        
        # Risk preference should increase
        assert service.preference_weights['risk_preference'] >= 0.5
    
    def test_concurrent_decisions(self, service):
        """Test handling multiple decisions for same symbol"""
        # Record multiple decisions for AAPL
        service.record_decision('AAPL', 'buy', 0.8, 'accepted')
        service.record_decision('AAPL', 'sell', 0.7, 'rejected')
        service.record_decision('AAPL', 'hold', 0.6, 'accepted')
        
        decisions = service.get_decisions(symbol='AAPL')
        assert len(decisions) == 3
    
    def test_create_personalization_service(self, temp_db):
        """Test factory function"""
        service = create_personalization_service(db_path=temp_db)
        
        assert isinstance(service, PersonalizationService)
        assert service.db_path == temp_db
        assert service.min_decisions_for_learning == 10
        assert service.learning_rate == 0.1


class TestUserDecision:
    """Test UserDecision dataclass"""
    
    def test_user_decision_creation(self):
        """Test creating UserDecision"""
        decision = UserDecision(
            decision_id=1,
            symbol='AAPL',
            recommendation_action='buy',
            recommendation_confidence=0.8,
            user_action='accepted',
            actual_trade_action='buy',
            timestamp=datetime.now()
        )
        
        assert decision.symbol == 'AAPL'
        assert decision.recommendation_action == 'buy'
        assert decision.user_action == 'accepted'
    
    def test_user_decision_to_dict(self):
        """Test converting UserDecision to dictionary"""
        decision = UserDecision(
            decision_id=1,
            symbol='AAPL',
            recommendation_action='buy',
            recommendation_confidence=0.8,
            user_action='accepted',
            actual_trade_action='buy',
            timestamp=datetime.now(),
            outcome='profit',
            outcome_value=150.0
        )
        
        data = decision.to_dict()
        
        assert data['symbol'] == 'AAPL'
        assert data['outcome'] == 'profit'
        assert data['outcome_value'] == 150.0
        assert isinstance(data['timestamp'], str)


class TestPersonalizationMetrics:
    """Test PersonalizationMetrics dataclass"""
    
    def test_metrics_creation(self):
        """Test creating PersonalizationMetrics"""
        metrics = PersonalizationMetrics(
            total_decisions=10,
            acceptance_rate=0.7,
            rejection_rate=0.2,
            modification_rate=0.1,
            avg_confidence_accepted=0.8,
            avg_confidence_rejected=0.6,
            personalization_active=True,
            personalization_confidence=0.9,
            improvement_score=0.75,
            learned_preferences={'test': 'value'}
        )
        
        assert metrics.total_decisions == 10
        assert metrics.acceptance_rate == 0.7
        assert metrics.personalization_active is True
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary"""
        metrics = PersonalizationMetrics(
            total_decisions=10,
            acceptance_rate=0.7,
            rejection_rate=0.2,
            modification_rate=0.1,
            avg_confidence_accepted=0.8,
            avg_confidence_rejected=0.6,
            personalization_active=True,
            personalization_confidence=0.9,
            improvement_score=0.75,
            learned_preferences={'test': 'value'}
        )
        
        data = metrics.to_dict()
        
        assert data['total_decisions'] == 10
        assert data['personalization_active'] is True
        assert 'learned_preferences' in data


class TestPersonalizationDataPersistence:
    """Property-based tests for personalization data persistence"""
    
    def test_property_personalization_data_persistence(self, service):
        """
        Property 13: Personalization data persistence
        
        For any user interaction with recommendations, the decision and outcome
        should be recorded for future personalization.
        
        Feature: ai-trading-agent, Property 13: Personalization data persistence
        Validates: Requirements 13.1
        """
        from hypothesis import given, strategies as st, settings
        
        @given(
            symbol=st.text(
                alphabet=st.characters(whitelist_categories=('Lu',), max_codepoint=90),
                min_size=1,
                max_size=5
            ),
            recommendation_action=st.sampled_from(['buy', 'sell', 'hold']),
            recommendation_confidence=st.floats(min_value=0.0, max_value=1.0),
            user_action=st.sampled_from(['accepted', 'rejected', 'modified', 'ignored']),
            actual_trade_action=st.one_of(
                st.none(),
                st.sampled_from(['buy', 'sell', 'hold'])
            ),
            outcome=st.one_of(
                st.none(),
                st.sampled_from(['profit', 'loss', 'neutral'])
            ),
            outcome_value=st.one_of(
                st.none(),
                st.floats(min_value=-10000.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
            ),
            notes=st.one_of(
                st.none(),
                st.text(max_size=100)
            )
        )
        @settings(max_examples=100)
        def property_test(
            symbol,
            recommendation_action,
            recommendation_confidence,
            user_action,
            actual_trade_action,
            outcome,
            outcome_value,
            notes
        ):
            # Record the decision
            decision_id = service.record_decision(
                symbol=symbol,
                recommendation_action=recommendation_action,
                recommendation_confidence=recommendation_confidence,
                user_action=user_action,
                actual_trade_action=actual_trade_action,
                notes=notes
            )
            
            # Verify decision was recorded with valid ID
            assert decision_id > 0, "Decision ID should be positive"
            
            # If outcome is provided, update it
            if outcome is not None:
                service.update_decision_outcome(
                    decision_id=decision_id,
                    outcome=outcome,
                    outcome_value=outcome_value
                )
            
            # Retrieve the decision by symbol
            decisions = service.get_decisions(symbol=symbol)
            
            # Verify decision exists
            assert len(decisions) > 0, "Decision should be retrievable"
            
            # Find the specific decision we just recorded
            recorded_decision = None
            for d in decisions:
                if d.decision_id == decision_id:
                    recorded_decision = d
                    break
            
            assert recorded_decision is not None, "Recorded decision should be found"
            
            # Verify all fields were persisted correctly
            assert recorded_decision.symbol == symbol, "Symbol should match"
            assert recorded_decision.recommendation_action == recommendation_action, "Action should match"
            assert abs(recorded_decision.recommendation_confidence - recommendation_confidence) < 0.0001, "Confidence should match"
            assert recorded_decision.user_action == user_action, "User action should match"
            assert recorded_decision.actual_trade_action == actual_trade_action, "Actual trade action should match"
            assert recorded_decision.notes == notes, "Notes should match"
            
            # Verify outcome was persisted if provided
            if outcome is not None:
                assert recorded_decision.outcome == outcome, "Outcome should match"
                if outcome_value is not None:
                    assert abs(recorded_decision.outcome_value - outcome_value) < 0.01, "Outcome value should match"
                else:
                    assert recorded_decision.outcome_value is None, "Outcome value should be None"
            
            # Verify timestamp was recorded
            assert recorded_decision.timestamp is not None, "Timestamp should be recorded"
            assert isinstance(recorded_decision.timestamp, datetime), "Timestamp should be datetime"
            
            # Verify decision can be exported (export_decisions doesn't take symbol parameter)
            df = service.export_decisions()
            assert len(df) > 0, "Decision should be exportable"
            
            # Verify the decision appears in the export
            exported_decision = df[df['decision_id'] == decision_id]
            assert len(exported_decision) == 1, "Decision should appear in export"
            
            # Verify decision appears in metrics
            metrics = service.get_personalization_metrics()
            assert metrics.total_decisions > 0, "Decision should be counted in metrics"
        
        # Run the property test
        property_test()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
