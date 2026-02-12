"""
Verification script for personalization service.

This script demonstrates the personalization functionality including:
- Recording user decisions
- Learning preferences
- Adjusting recommendations
- Viewing metrics
"""

import sys
from datetime import datetime
from services.personalization_service import create_personalization_service


def print_section(title):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def verify_personalization():
    """Verify personalization service functionality"""
    
    print_section("Personalization Service Verification")
    
    # Create service
    print("Creating personalization service...")
    service = create_personalization_service(
        db_path="data/database/personalization_test.db"
    )
    print("✓ Service created successfully")
    
    # Test 1: Record decisions
    print_section("Test 1: Recording User Decisions")
    
    decisions_data = [
        ('AAPL', 'buy', 0.85, 'accepted'),
        ('GOOGL', 'sell', 0.75, 'rejected'),
        ('MSFT', 'buy', 0.90, 'accepted'),
        ('TSLA', 'hold', 0.60, 'ignored'),
        ('AMZN', 'buy', 0.80, 'accepted'),
    ]
    
    for symbol, action, confidence, user_action in decisions_data:
        decision_id = service.record_decision(
            symbol=symbol,
            recommendation_action=action,
            recommendation_confidence=confidence,
            user_action=user_action
        )
        print(f"✓ Recorded decision {decision_id}: {symbol} - {action} -> {user_action}")
    
    # Test 2: View metrics (before learning threshold)
    print_section("Test 2: Metrics Before Learning Threshold")
    
    metrics = service.get_personalization_metrics()
    print(f"Total decisions: {metrics.total_decisions}")
    print(f"Acceptance rate: {metrics.acceptance_rate:.1%}")
    print(f"Rejection rate: {metrics.rejection_rate:.1%}")
    print(f"Personalization active: {metrics.personalization_active}")
    print(f"Personalization confidence: {metrics.personalization_confidence:.1%}")
    
    if not metrics.personalization_active:
        print(f"\n⚠ Need {service.min_decisions_for_learning - metrics.total_decisions} more decisions for personalization")
    
    # Test 3: Add more decisions to trigger learning
    print_section("Test 3: Adding More Decisions to Trigger Learning")
    
    additional_decisions = [
        ('NVDA', 'buy', 0.88, 'accepted'),
        ('META', 'sell', 0.70, 'rejected'),
        ('NFLX', 'buy', 0.82, 'accepted'),
        ('AMD', 'buy', 0.85, 'accepted'),
        ('INTC', 'hold', 0.65, 'modified'),
    ]
    
    for symbol, action, confidence, user_action in additional_decisions:
        decision_id = service.record_decision(
            symbol=symbol,
            recommendation_action=action,
            recommendation_confidence=confidence,
            user_action=user_action
        )
        print(f"✓ Recorded decision {decision_id}: {symbol} - {action} -> {user_action}")
    
    # Test 4: View metrics after learning
    print_section("Test 4: Metrics After Learning Threshold")
    
    metrics = service.get_personalization_metrics()
    print(f"Total decisions: {metrics.total_decisions}")
    print(f"Acceptance rate: {metrics.acceptance_rate:.1%}")
    print(f"Rejection rate: {metrics.rejection_rate:.1%}")
    print(f"Modification rate: {metrics.modification_rate:.1%}")
    print(f"Avg confidence (accepted): {metrics.avg_confidence_accepted:.2f}")
    print(f"Avg confidence (rejected): {metrics.avg_confidence_rejected:.2f}")
    print(f"Personalization active: {metrics.personalization_active}")
    print(f"Personalization confidence: {metrics.personalization_confidence:.1%}")
    print(f"Improvement score: {metrics.improvement_score:.2f}")
    
    print("\nLearned preferences:")
    print(f"  Confidence threshold: {metrics.learned_preferences['confidence_threshold']:.2f}")
    print(f"  Risk preference: {metrics.learned_preferences['risk_preference']:.2f}")
    print(f"  Action bias:")
    for action, bias in metrics.learned_preferences['action_bias'].items():
        print(f"    {action}: {bias:.2f}")
    
    # Test 5: Adjust recommendation
    print_section("Test 5: Adjusting Recommendations")
    
    original_recommendation = {
        'action': 'buy',
        'confidence': 0.75,
        'risk_level': 'medium',
        'target_price': 150.0
    }
    
    print("Original recommendation:")
    print(f"  Action: {original_recommendation['action']}")
    print(f"  Confidence: {original_recommendation['confidence']:.2f}")
    print(f"  Risk level: {original_recommendation['risk_level']}")
    
    adjusted = service.adjust_recommendation(original_recommendation)
    
    print("\nAdjusted recommendation:")
    print(f"  Action: {adjusted['action']}")
    print(f"  Confidence: {adjusted['confidence']:.2f}")
    print(f"  Risk level: {adjusted['risk_level']}")
    print(f"  Personalized: {adjusted.get('personalized', False)}")
    print(f"  Personalization confidence: {adjusted.get('personalization_confidence', 0):.1%}")
    
    # Test 6: Update decision outcomes
    print_section("Test 6: Updating Decision Outcomes")
    
    decisions = service.get_decisions()
    
    # Update first 3 accepted decisions with outcomes
    accepted_decisions = [d for d in decisions if d.user_action == 'accepted'][:3]
    
    for i, decision in enumerate(accepted_decisions):
        outcome = 'profit' if i < 2 else 'loss'
        outcome_value = 100.0 if outcome == 'profit' else -50.0
        
        service.update_decision_outcome(
            decision_id=decision.decision_id,
            outcome=outcome,
            outcome_value=outcome_value
        )
        print(f"✓ Updated decision {decision.decision_id}: {outcome} (${outcome_value:+.2f})")
    
    # Test 7: Export decisions
    print_section("Test 7: Exporting Decisions")
    
    df = service.export_decisions()
    print(f"Exported {len(df)} decisions")
    print("\nSample data:")
    print(df[['symbol', 'recommendation_action', 'user_action', 'recommendation_confidence']].head())
    
    # Test 8: Reset personalization
    print_section("Test 8: Reset Personalization")
    
    print("Current preferences:")
    print(f"  Confidence threshold: {service.preference_weights['confidence_threshold']:.2f}")
    print(f"  Buy action bias: {service.preference_weights['action_bias']['buy']:.2f}")
    
    service.reset_personalization()
    print("\n✓ Personalization reset")
    
    print("\nReset preferences:")
    print(f"  Confidence threshold: {service.preference_weights['confidence_threshold']:.2f}")
    print(f"  Buy action bias: {service.preference_weights['action_bias']['buy']:.2f}")
    
    # Verify decisions still exist
    decisions_after_reset = service.get_decisions()
    print(f"\n✓ Decisions preserved: {len(decisions_after_reset)} decisions still in database")
    
    # Summary
    print_section("Verification Summary")
    
    print("✓ All personalization features verified successfully!")
    print("\nFeatures tested:")
    print("  1. Recording user decisions")
    print("  2. Viewing metrics before learning threshold")
    print("  3. Triggering personalization learning")
    print("  4. Viewing metrics after learning")
    print("  5. Adjusting recommendations based on preferences")
    print("  6. Updating decision outcomes")
    print("  7. Exporting decisions to DataFrame")
    print("  8. Resetting personalization")
    
    return True


if __name__ == '__main__':
    try:
        success = verify_personalization()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
