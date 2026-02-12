# Personalization Engine Implementation

## Overview

The personalization engine learns from user behavior and adapts AI recommendations to match individual preferences. It tracks user decisions on recommendations, identifies patterns, and adjusts future recommendations accordingly.

## Features

### 1. User Decision Tracking
- Records every user interaction with AI recommendations
- Tracks acceptance, rejection, modification, or ignoring of recommendations
- Stores decision context (symbol, action, confidence, timestamp)
- Updates outcomes when trades complete (profit/loss)

### 2. Preference Learning
- Learns confidence threshold (minimum confidence user typically accepts)
- Identifies action bias (which actions user prefers: buy/sell/hold)
- Adapts to risk preference based on outcomes
- Requires minimum 10 decisions before personalization activates

### 3. Recommendation Adjustment
- Boosts confidence for actions user typically accepts
- Reduces confidence for actions user typically rejects
- Adjusts risk levels based on user's risk tolerance
- Indicates personalization status and confidence

### 4. Effectiveness Metrics
- Tracks acceptance/rejection/modification rates
- Calculates improvement score over time
- Shows average confidence for accepted vs rejected recommendations
- Displays learned preferences

### 5. Reset Functionality
- Resets preferences to defaults
- Preserves decision history for reference
- Allows starting fresh if preferences change

## Architecture

### Database Schema

```sql
-- User decisions table
CREATE TABLE user_decisions (
    decision_id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    recommendation_action TEXT NOT NULL,
    recommendation_confidence REAL NOT NULL,
    user_action TEXT NOT NULL,
    actual_trade_action TEXT,
    timestamp TEXT NOT NULL,
    outcome TEXT,
    outcome_value REAL,
    notes TEXT
);

-- Preferences table
CREATE TABLE preferences (
    preference_key TEXT PRIMARY KEY,
    preference_value TEXT NOT NULL,
    last_updated TEXT NOT NULL
);

-- History table
CREATE TABLE personalization_history (
    history_id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    total_decisions INTEGER NOT NULL,
    acceptance_rate REAL NOT NULL,
    confidence_threshold REAL NOT NULL,
    risk_preference REAL NOT NULL,
    improvement_score REAL NOT NULL
);
```

### Preference Weights

```python
preference_weights = {
    'confidence_threshold': 0.6,  # Minimum confidence user accepts
    'risk_preference': 0.5,       # 0=conservative, 1=aggressive
    'action_bias': {              # User's tendency per action
        'buy': 0.5,
        'sell': 0.5,
        'hold': 0.5
    },
    'indicator_weights': {        # Which indicators user values
        'rsi': 1.0,
        'macd': 1.0,
        'moving_averages': 1.0,
        'volatility': 1.0
    }
}
```

## Usage

### Basic Usage

```python
from services.personalization_service import create_personalization_service

# Create service
service = create_personalization_service()

# Record a user decision
decision_id = service.record_decision(
    symbol='AAPL',
    recommendation_action='buy',
    recommendation_confidence=0.85,
    user_action='accepted',
    notes='Strong technical signals'
)

# Update outcome after trade completes
service.update_decision_outcome(
    decision_id=decision_id,
    outcome='profit',
    outcome_value=150.50
)

# Get personalization metrics
metrics = service.get_personalization_metrics()
print(f"Acceptance rate: {metrics.acceptance_rate:.1%}")
print(f"Personalization active: {metrics.personalization_active}")

# Adjust a recommendation
recommendation = {
    'action': 'buy',
    'confidence': 0.75,
    'risk_level': 'medium'
}

adjusted = service.adjust_recommendation(recommendation)
print(f"Adjusted confidence: {adjusted['confidence']:.2f}")
```

### Integration with AI Engine

```python
from ai.inference import AIEngine
from services.personalization_service import create_personalization_service

# Create services
ai_engine = AIEngine()
personalization = create_personalization_service()

# Get AI recommendation
recommendation = ai_engine.get_recommendation('AAPL', data)

# Convert to dict for adjustment
rec_dict = {
    'action': recommendation.action,
    'confidence': recommendation.confidence,
    'risk_level': recommendation.risk_level,
    'target_price': recommendation.target_price
}

# Adjust based on user preferences
adjusted = personalization.adjust_recommendation(rec_dict)

# Display to user
print(f"Action: {adjusted['action']}")
print(f"Confidence: {adjusted['confidence']:.1%}")
if adjusted.get('personalized'):
    print(f"✨ Personalized (confidence: {adjusted['personalization_confidence']:.0%})")
```

### Viewing Metrics

```python
# Get comprehensive metrics
metrics = service.get_personalization_metrics()

print(f"Total decisions: {metrics.total_decisions}")
print(f"Acceptance rate: {metrics.acceptance_rate:.1%}")
print(f"Improvement score: {metrics.improvement_score:.2f}")

# Check if personalization is active
if metrics.personalization_active:
    print("✓ Personalization is active")
    print(f"Confidence: {metrics.personalization_confidence:.0%}")
else:
    needed = service.min_decisions_for_learning - metrics.total_decisions
    print(f"Need {needed} more decisions for personalization")

# View learned preferences
prefs = metrics.learned_preferences
print(f"Confidence threshold: {prefs['confidence_threshold']:.2f}")
print(f"Risk preference: {prefs['risk_preference']:.2f}")
print(f"Buy bias: {prefs['action_bias']['buy']:.2f}")
```

### Exporting Data

```python
from datetime import datetime, timedelta

# Export all decisions
df = service.export_decisions()
df.to_csv('user_decisions.csv', index=False)

# Export with date filter
start_date = datetime.now() - timedelta(days=30)
df = service.export_decisions(start_date=start_date)

# Get personalization history
history = service.get_personalization_history(days=30)
print(history[['timestamp', 'acceptance_rate', 'improvement_score']])
```

### Resetting Personalization

```python
# Reset to defaults (keeps decision history)
service.reset_personalization()

print("✓ Personalization reset to defaults")
print(f"Confidence threshold: {service.preference_weights['confidence_threshold']}")
```

## Learning Algorithm

### Confidence Threshold Learning

```python
# Calculate average confidence of accepted recommendations
avg_accepted_conf = mean([d.confidence for d in accepted_decisions])

# Set threshold slightly below average (90%)
new_threshold = avg_accepted_conf * 0.9

# Update with learning rate
current_threshold = (
    (1 - learning_rate) * current_threshold +
    learning_rate * new_threshold
)
```

### Action Bias Learning

```python
# Calculate acceptance rate per action
for action in ['buy', 'sell', 'hold']:
    acceptance_rate = accepted[action] / total[action]
    
    # Update bias with learning rate
    action_bias[action] = (
        (1 - learning_rate) * action_bias[action] +
        learning_rate * acceptance_rate
    )
```

### Risk Preference Learning

```python
# Analyze outcomes of risky trades (low confidence)
risky_decisions = [d for d in decisions if d.confidence < 0.7]
risky_profitable = [d for d in risky_decisions if d.outcome == 'profit']

risk_success_rate = len(risky_profitable) / len(risky_decisions)

# Adjust risk preference based on success
if risk_success_rate > 0.6:
    risk_preference += learning_rate * 0.1  # Increase
elif risk_success_rate < 0.4:
    risk_preference -= learning_rate * 0.1  # Decrease
```

## Configuration

### Service Parameters

```python
service = PersonalizationService(
    db_path="data/database/personalization.db",
    min_decisions_for_learning=10,  # Minimum decisions before learning
    learning_rate=0.1                # How fast to adapt (0.0 to 1.0)
)
```

### Tuning Learning Rate

- **Low (0.05-0.1)**: Slow, stable learning. Good for long-term patterns.
- **Medium (0.1-0.2)**: Balanced. Default recommendation.
- **High (0.2-0.5)**: Fast adaptation. Good for rapidly changing preferences.

### Minimum Decisions Threshold

- **Low (5-10)**: Quick personalization, may be noisy
- **Medium (10-20)**: Balanced. Default recommendation.
- **High (20-50)**: More reliable, takes longer to activate

## Performance Considerations

### Database Optimization

- Indices on `symbol`, `timestamp`, and `user_action` columns
- Efficient queries with proper filtering
- Periodic cleanup of old history records

### Memory Usage

- Preferences stored in memory for fast access
- Database queries only when needed
- Lazy loading of historical data

### Computation

- Learning updates triggered only after new decisions
- Cached metrics to avoid repeated calculations
- Efficient numpy operations for statistics

## Testing

### Unit Tests

```bash
# Run all personalization tests
pytest tests/test_personalization_service.py -v

# Run specific test
pytest tests/test_personalization_service.py::TestPersonalizationService::test_preference_learning -v
```

### Verification Script

```bash
# Run verification script
python verify_personalization.py
```

## Troubleshooting

### Personalization Not Activating

**Problem**: Metrics show `personalization_active: False`

**Solution**: 
- Check `total_decisions` in metrics
- Need at least `min_decisions_for_learning` decisions
- Record more user interactions

### Preferences Not Changing

**Problem**: Learned preferences remain at defaults

**Solution**:
- Ensure decisions have varied user actions (not all accepted/rejected)
- Check learning rate (may be too low)
- Verify decisions are being recorded correctly

### Confidence Not Adjusting

**Problem**: Adjusted recommendations have same confidence as original

**Solution**:
- Check if personalization is active
- Verify action bias is different from 0.5
- Ensure recommendation action matches recorded decisions

### Database Errors

**Problem**: SQLite errors when recording decisions

**Solution**:
- Check database path exists and is writable
- Verify database schema is initialized
- Check for file permissions issues

## Requirements Validation

This implementation satisfies the following requirements:

### Requirement 13.1
✓ Records user decisions on recommendations (accept/reject)
- `record_decision()` method tracks all user interactions
- Stores decision context and timestamp

### Requirement 13.2
✓ Adjusts recommendation weights based on user preferences
- `_update_preferences()` learns from user behavior
- Updates confidence threshold, action bias, and risk preference

### Requirement 13.3
✓ Indicates if personalization is active and confidence level
- `get_personalization_metrics()` returns status
- `adjust_recommendation()` adds personalization indicators

### Requirement 13.4
✓ Allows resetting personalization
- `reset_personalization()` clears learned preferences
- Preserves decision history for reference

### Requirement 13.5
✓ Shows how recommendations have improved over time
- `improvement_score` tracks acceptance rate changes
- `get_personalization_history()` shows trends
- Metrics compare recent vs early performance

## Future Enhancements

1. **Advanced Learning**
   - Neural network for preference modeling
   - Multi-armed bandit for exploration/exploitation
   - Collaborative filtering across users

2. **Additional Preferences**
   - Sector preferences
   - Time-of-day preferences
   - Market condition preferences

3. **Visualization**
   - Interactive preference dashboard
   - Learning progress charts
   - Decision outcome analysis

4. **Integration**
   - Real-time recommendation adjustment
   - Automated feedback from trade outcomes
   - Integration with portfolio performance

## Related Documentation

- [AI Inference Engine](AI_INFERENCE_ENGINE.md)
- [Trading Service](services/trading_service.py)
- [Requirements Document](.kiro/specs/ai-trading-agent/requirements.md)
- [Design Document](.kiro/specs/ai-trading-agent/design.md)
