# Error Recovery Implementation

## Overview

The Error Recovery Service provides comprehensive error handling capabilities for the AI Trading Agent, including:

- **State Persistence**: Automatic saving and restoration of system state
- **Automatic Retry**: Exponential backoff retry mechanism for transient failures
- **Graceful Degradation**: Fallback mechanisms when primary services fail
- **Error Logging**: Detailed error tracking and reporting
- **Circuit Breaker**: Prevents cascading failures by temporarily disabling failing services
- **Service Health Monitoring**: Track operational status of all services

## Architecture

### Core Components

1. **ErrorRecoveryService**: Main service class providing error handling capabilities
2. **ErrorRecord**: Dataclass for tracking error occurrences
3. **SystemState**: Dataclass for persisting system state
4. **ServiceStatus**: Enum for tracking service operational status
5. **ErrorSeverity**: Enum for categorizing error severity

### Key Features

#### 1. Automatic Retry with Exponential Backoff

```python
from services.error_recovery_service import ErrorRecoveryService

recovery = ErrorRecoveryService()

# Retry a function with automatic backoff
result = recovery.retry_with_backoff(
    fetch_market_data,
    symbol="AAPL",
    service_name="market_data",
    max_retries=3
)
```

**Configuration**:
- `max_retries`: Maximum number of retry attempts (default: 3)
- `base_delay`: Initial delay between retries (default: 1.0s)
- `max_delay`: Maximum delay between retries (default: 60.0s)
- `backoff_factor`: Exponential multiplier (default: 2.0)

**Retry Delays**:
- Attempt 1: base_delay (1s)
- Attempt 2: base_delay * backoff_factor (2s)
- Attempt 3: base_delay * backoff_factor² (4s)
- Capped at max_delay

#### 2. Graceful Degradation with Fallback

```python
# Use fallback when primary function fails
result = recovery.with_fallback(
    primary_func=fetch_live_data,
    fallback_func=fetch_cached_data,
    symbol="AAPL",
    service_name="market_data"
)
```

**Behavior**:
- Tries primary function with retry logic
- Falls back to secondary function if primary fails
- Updates service status to "degraded" when using fallback
- Marks service as "unavailable" if both fail

#### 3. State Persistence

```python
# Save system state
recovery.save_state(
    active_operations=[
        {'operation': 'fetch_data', 'symbol': 'AAPL', 'status': 'pending'}
    ],
    cached_data={'AAPL': {'price': 150.0}},
    configuration={'api_key': 'encrypted'},
    force=True  # Force immediate save
)

# Restore state on restart
state = recovery.restore_state()
if state:
    # Resume operations from saved state
    for op in state.active_operations:
        resume_operation(op)
```

**State Files**:
- `data/state/system_state.json`: Human-readable JSON format
- `data/state/system_state.pkl`: Python pickle format (more reliable)

**Rate Limiting**:
- Automatic rate limiting prevents excessive disk writes
- Default interval: 60 seconds between saves
- Use `force=True` to override rate limiting

#### 4. Circuit Breaker Pattern

```python
# Circuit breaker automatically opens after threshold failures
for i in range(10):
    try:
        result = recovery.retry_with_backoff(
            failing_function,
            service_name="external_api"
        )
    except Exception:
        pass

# Check if circuit is open
if recovery.is_circuit_open("external_api"):
    print("Service temporarily disabled")
    
# Manually reset circuit breaker
recovery.reset_circuit_breaker("external_api")
```

**Circuit States**:
- **Closed**: Normal operation, requests pass through
- **Open**: Service disabled after threshold failures
- **Half-Open**: Testing if service recovered after timeout

**Configuration**:
- Threshold: 5 failures before opening
- Timeout: 60 seconds before entering half-open state

#### 5. Error Logging and Reporting

```python
# Get error summary
summary = recovery.get_error_summary(
    since=datetime.now() - timedelta(hours=1),
    service_name="market_data",
    min_severity=ErrorSeverity.HIGH
)

print(f"Total errors: {summary['total_errors']}")
print(f"By service: {summary['errors_by_service']}")
print(f"By type: {summary['errors_by_type']}")
```

**Error Log Files**:
- Location: `data/errors/errors_YYYYMMDD.jsonl`
- Format: JSON Lines (one error per line)
- Rotation: Daily log files
- Retention: Configurable (default: keep all)

**Error Record Fields**:
- `timestamp`: When error occurred
- `error_type`: Exception class name
- `error_message`: Error message
- `severity`: LOW, MEDIUM, HIGH, CRITICAL
- `service_name`: Which service failed
- `function_name`: Which function failed
- `retry_count`: Number of retries attempted
- `stack_trace`: Full stack trace
- `context`: Additional context data

#### 6. Service Health Monitoring

```python
# Get overall system health
health = recovery.get_health_status()

print(f"Overall health: {health['overall_health']}")  # healthy, degraded, unhealthy
print(f"Service statuses: {health['service_statuses']}")
print(f"Recent errors: {health['recent_error_count']}")
print(f"Circuit breakers: {health['circuit_breakers']}")
```

**Service Status Levels**:
- **OPERATIONAL**: Service working normally
- **DEGRADED**: Service using fallback or experiencing issues
- **UNAVAILABLE**: Service completely failed

## Usage Examples

### Example 1: Protecting API Calls

```python
from services.error_recovery_service import get_error_recovery_service

recovery = get_error_recovery_service()

def fetch_quote(symbol):
    """Fetch quote with automatic retry"""
    return recovery.retry_with_backoff(
        market_data_service.get_latest_quote,
        symbol=symbol,
        service_name="market_data",
        max_retries=3
    )

# Use it
try:
    quote = fetch_quote("AAPL")
    print(f"Price: ${quote.mid_price}")
except Exception as e:
    print(f"Failed to fetch quote: {e}")
```

### Example 2: Using Decorator

```python
from services.error_recovery_service import with_error_recovery

@with_error_recovery("trading_service", max_retries=3)
def place_order(symbol, qty, side):
    """Place order with automatic retry"""
    return trading_service.place_order(symbol, qty, side)

# Use it
order = place_order("AAPL", 10, "buy")
```

### Example 3: Fallback to Cached Data

```python
def fetch_with_cache(symbol):
    """Fetch live data with cache fallback"""
    def fetch_live():
        return market_data_service.get_latest_quote(symbol)
    
    def fetch_cached():
        return cache.get(f"quote:{symbol}")
    
    return recovery.with_fallback(
        fetch_live,
        fetch_cached,
        service_name="market_data"
    )
```

### Example 4: State Persistence for Long Operations

```python
def process_large_dataset():
    """Process dataset with state persistence"""
    recovery = get_error_recovery_service()
    
    # Try to restore previous state
    state = recovery.restore_state()
    if state and state.active_operations:
        start_index = state.active_operations[0].get('index', 0)
        print(f"Resuming from index {start_index}")
    else:
        start_index = 0
    
    # Process data
    for i in range(start_index, len(dataset)):
        try:
            process_item(dataset[i])
            
            # Save progress periodically
            if i % 100 == 0:
                recovery.save_state(
                    active_operations=[{'operation': 'process', 'index': i}],
                    force=True
                )
        except Exception as e:
            # Save state on error
            recovery.save_state(
                active_operations=[{'operation': 'process', 'index': i, 'error': str(e)}],
                force=True
            )
            raise
```

## Integration with Existing Services

### Market Data Service

```python
# In services/market_data_service.py
from services.error_recovery_service import get_error_recovery_service

class MarketDataService:
    def __init__(self):
        self.recovery = get_error_recovery_service()
    
    def get_latest_quote(self, symbol):
        """Get quote with error recovery"""
        return self.recovery.retry_with_backoff(
            self._fetch_quote_from_api,
            symbol=symbol,
            service_name="market_data"
        )
    
    def _fetch_quote_from_api(self, symbol):
        # Actual API call
        pass
```

### Trading Service

```python
# In services/trading_service.py
from services.error_recovery_service import get_error_recovery_service

class TradingService:
    def __init__(self):
        self.recovery = get_error_recovery_service()
    
    def place_order(self, symbol, qty, side):
        """Place order with error recovery"""
        # Check circuit breaker
        if self.recovery.is_circuit_open("trading"):
            raise Exception("Trading service temporarily unavailable")
        
        return self.recovery.retry_with_backoff(
            self._submit_order_to_broker,
            symbol=symbol,
            qty=qty,
            side=side,
            service_name="trading",
            max_retries=2  # Fewer retries for trading
        )
```

## Configuration

### Environment Variables

```bash
# Error recovery configuration
ERROR_RECOVERY_MAX_RETRIES=3
ERROR_RECOVERY_BASE_DELAY=1.0
ERROR_RECOVERY_MAX_DELAY=60.0
ERROR_RECOVERY_BACKOFF_FACTOR=2.0
ERROR_RECOVERY_CIRCUIT_THRESHOLD=5
ERROR_RECOVERY_CIRCUIT_TIMEOUT=60.0
ERROR_RECOVERY_STATE_SAVE_INTERVAL=60.0
```

### Programmatic Configuration

```python
recovery = ErrorRecoveryService(
    state_dir="data/state",
    error_log_dir="data/errors",
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0
)

# Update circuit breaker settings
recovery._circuit_breaker_threshold = 10
recovery._circuit_breaker_timeout = 120.0

# Update state save interval
recovery._state_save_interval = 30.0  # Save every 30 seconds
```

## Monitoring and Debugging

### View Error History

```python
# Get recent errors
summary = recovery.get_error_summary(
    since=datetime.now() - timedelta(hours=1)
)

for error in summary['recent_errors']:
    print(f"{error['timestamp']}: {error['service_name']}.{error['function_name']}")
    print(f"  {error['error_type']}: {error['error_message']}")
    print(f"  Severity: {error['severity']}, Retries: {error['retry_count']}")
```

### Check Service Health

```python
health = recovery.get_health_status()

if health['overall_health'] != 'healthy':
    print("System health issues detected:")
    for service, status in health['service_statuses'].items():
        if status != 'operational':
            print(f"  {service}: {status}")
```

### Monitor Circuit Breakers

```python
for service, state in health['circuit_breakers'].items():
    if state != 'closed':
        print(f"Circuit breaker {service}: {state}")
```

## Best Practices

1. **Use Appropriate Retry Counts**:
   - Read operations: 3-5 retries
   - Write operations: 1-2 retries
   - Critical operations: 0-1 retries

2. **Set Meaningful Service Names**:
   - Use consistent naming across the application
   - Examples: "market_data", "trading", "sentiment", "portfolio"

3. **Implement Fallbacks for Critical Services**:
   - Always have a fallback for data fetching
   - Use cached data when live data unavailable
   - Provide degraded functionality rather than complete failure

4. **Save State for Long Operations**:
   - Save state periodically during long-running operations
   - Save state before critical operations
   - Always save state on errors

5. **Monitor Circuit Breakers**:
   - Alert when circuit breakers open
   - Investigate root cause before resetting
   - Consider implementing automatic health checks

6. **Review Error Logs Regularly**:
   - Check daily error summaries
   - Identify patterns in failures
   - Address recurring issues

## Testing

### Unit Tests

```bash
# Run all error recovery tests
python -m pytest tests/test_error_recovery_service.py -v

# Run specific test
python -m pytest tests/test_error_recovery_service.py::TestErrorRecoveryService::test_retry_with_backoff -v
```

### Integration Tests

```python
def test_market_data_with_recovery():
    """Test market data service with error recovery"""
    service = MarketDataService()
    
    # Should succeed with retry
    quote = service.get_latest_quote("AAPL")
    assert quote is not None
    
    # Check service status
    recovery = get_error_recovery_service()
    status = recovery.get_service_status("market_data")
    assert status == ServiceStatus.OPERATIONAL
```

## Troubleshooting

### Issue: Excessive Retries

**Symptom**: Operations taking too long due to many retries

**Solution**:
```python
# Reduce max retries for specific operations
recovery.retry_with_backoff(
    func,
    max_retries=1,  # Override default
    service_name="fast_operation"
)
```

### Issue: Circuit Breaker Stuck Open

**Symptom**: Service remains unavailable even after issue resolved

**Solution**:
```python
# Manually reset circuit breaker
recovery.reset_circuit_breaker("service_name")

# Or wait for automatic timeout (default: 60s)
```

### Issue: State Not Persisting

**Symptom**: State not saved between restarts

**Solution**:
```python
# Force immediate save
recovery.save_state(
    active_operations=ops,
    force=True  # Bypass rate limiting
)

# Check state directory permissions
import os
state_dir = Path("data/state")
print(f"State dir exists: {state_dir.exists()}")
print(f"State dir writable: {os.access(state_dir, os.W_OK)}")
```

### Issue: Error Logs Growing Too Large

**Symptom**: Error log files consuming too much disk space

**Solution**:
```python
# Implement log rotation
from pathlib import Path
from datetime import datetime, timedelta

error_dir = Path("data/errors")
cutoff_date = datetime.now() - timedelta(days=30)

for log_file in error_dir.glob("errors_*.jsonl"):
    # Parse date from filename
    date_str = log_file.stem.split('_')[1]
    file_date = datetime.strptime(date_str, '%Y%m%d')
    
    if file_date < cutoff_date:
        log_file.unlink()  # Delete old logs
```

## Performance Considerations

1. **State Save Frequency**: Default 60s interval balances persistence with performance
2. **Error History Size**: Limited to 1000 most recent errors to prevent memory issues
3. **Circuit Breaker Overhead**: Minimal - simple dictionary lookup
4. **Retry Delays**: Exponential backoff prevents overwhelming failed services

## Future Enhancements

1. **Distributed State**: Support for distributed state storage (Redis, etcd)
2. **Metrics Export**: Export metrics to Prometheus/Grafana
3. **Alerting Integration**: Send alerts to Slack/PagerDuty
4. **Adaptive Retry**: Adjust retry strategy based on error patterns
5. **Health Check Endpoints**: HTTP endpoints for monitoring tools

## References

- Requirements: 14.1, 14.2, 14.3, 14.4, 14.5
- Design Document: `.kiro/specs/ai-trading-agent/design.md`
- Test Suite: `tests/test_error_recovery_service.py`
