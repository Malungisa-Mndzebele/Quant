# Personalization Engine - Quick Start Guide

## What is Personalization?

The personalization engine learns from your trading decisions and adapts AI recommendations to match your preferences. Over time, it understands which recommendations you accept or reject and adjusts future suggestions accordingly.

## Quick Start

### 1. Create the Service

```python
from services.personalization_service import create_personalization_service

# Create with defaults
service = create_personalization_service()
```

### 2. Record Your Decisions

Every time you interact with an AI recommendation, record your decision:

```python
# User accepted a buy recommendation
decision_id = service.record_decision(
    symbol='AAPL',
    recommendation_action='buy',
    recommendation_confidence=0.85,
    user_action='accepted'
)

# User rejected a sell recommendation
service.record_decision(
    symbol='GOOGL',
    recommendation_action='sell',
    recommendation_confidence=0.70,
    user_action='rejected'
)

# User modified a recommendation
service.record_decision(
    symbol='MSFT',
    recommendation_action='buy',
    recommendation_confidence=0.75,
    user_action='modified',
    actual_trade_action='hold'
)
```

### 3. Update Outcomes (Optional)

After a trade completes, update the outcome:

```python
service.update_decision_outcome(
    decision_id=decision_id,
    outcome='profit',
    outcome_value=150.50
)
```

### 4. Check Personalization Status

```python
metrics = service.get_personalization_metrics()

if metrics.personalization_active:
    print(f"✨ Personalization is active!")
    print(f"Confidence: {metrics.personalization_confidence:.0%}")
    print(f"Acceptance rate: {metrics.acceptance_rate:.1%}")
else:
    needed = 10 - metrics.total_decisions
    print(f"Need {needed} more decisions to activate personalization")
```

### 5. Adjust Recommendations

Once personalization is active, adjust recommendations:

```python
# Original AI recommendation
recommendation = {
    'action': 'buy',
    'confidence': 0.75,
    'risk_level': 'medium'
}

# Adjust based on your preferences
adjusted = service.adjust_recommendation(recommendation)

print(f"Original confidence: {recommendation['confidence']:.2f}")
print(f"Adjusted confidence: {adjusted['confidence']:.2f}")
```

## User Actions

When recording decisions, use these user actions:

- **`accepted`**: You followed the AI recommendation
- **`rejected`**: You ignored or did the opposite
- **`modified`**: You took a different action
- **`ignored`**: You saw it but took no action

## How It Learns

### Confidence Threshold
If you consistently accept recommendations with 0.85+ confidence but reject those below 0.70, the system learns your confidence threshold.

### Action Bias
If you accept most buy recommendations but reject sell recommendations, the system boosts confidence for buys and reduces it for sells.

### Risk Preference
If your risky trades (low confidence) are profitable, the system learns you're comfortable with higher risk.

## Example Workflow

```python
from services.personalization_service import create_personalization_service
from ai.inference import AIEngine

# Setup
ai_engine = AIEngine()
personalization = create_personalization_service()

# Get AI recommendation
recommendation = ai_engine.get_recommendation('AAPL', historical_data)

# Adjust based on your preferences
rec_dict = {
    'action': recommendation.action,
    'confidence': recommendation.confidence,
    'risk_level': recommendation.risk_level
}
adjusted = personalization.adjust_recommendation(rec_dict)

# Show to user
print(f"Action: {adjusted['action']}")
print(f"Confidence: {adjusted['confidence']:.1%}")

# User makes decision
user_decision = input("Accept? (y/n): ")
user_action = 'accepted' if user_decision == 'y' else 'rejected'

# Record decision
decision_id = personalization.record_decision(
    symbol='AAPL',
    recommendation_action=recommendation.action,
    recommendation_confidence=recommendation.confidence,
    user_action=user_action
)

# If accepted and trade executed, update outcome later
if user_action == 'accepted':
    # ... execute trade ...
    # ... wait for outcome ...
    personalization.update_decision_outcome(
        decision_id=decision_id,
        outcome='profit',
        outcome_value=100.0
    )
```

## Viewing Your Progress

```python
# Get metrics
metrics = service.get_personalization_metrics()

print(f"Total decisions: {metrics.total_decisions}")
print(f"Acceptance rate: {metrics.acceptance_rate:.1%}")
print(f"Improvement: {metrics.improvement_score:.1%}")

# View learned preferences
prefs = metrics.learned_preferences
print(f"\nYour preferences:")
print(f"  Confidence threshold: {prefs['confidence_threshold']:.2f}")
print(f"  Risk tolerance: {prefs['risk_preference']:.2f}")
print(f"  Buy bias: {prefs['action_bias']['buy']:.2f}")
print(f"  Sell bias: {prefs['action_bias']['sell']:.2f}")
```

## Resetting

If your preferences change or you want to start fresh:

```python
service.reset_personalization()
print("✓ Personalization reset to defaults")
```

This keeps your decision history but resets learned preferences.

## Tips

1. **Be Consistent**: Record every decision for best results
2. **Update Outcomes**: Helps the system learn from results
3. **Give It Time**: Need 10+ decisions before personalization activates
4. **Review Metrics**: Check your acceptance rate and improvement score
5. **Reset If Needed**: If your strategy changes, reset and start fresh

## Common Questions

**Q: How many decisions before it starts working?**
A: Minimum 10 decisions. More is better (20-30 for reliable patterns).

**Q: Can I see my decision history?**
A: Yes! Use `service.export_decisions()` to get a DataFrame.

**Q: What if I change my strategy?**
A: Use `service.reset_personalization()` to start fresh.

**Q: Does it affect the AI models?**
A: No, it only adjusts the confidence and presentation of recommendations.

**Q: Can I turn it off?**
A: Yes, just don't call `adjust_recommendation()`. The system still tracks decisions for future use.

## Next Steps

- Read [PERSONALIZATION_IMPLEMENTATION.md](PERSONALIZATION_IMPLEMENTATION.md) for detailed documentation
- Run `python verify_personalization.py` to see it in action
- Check `tests/test_personalization_service.py` for more examples
