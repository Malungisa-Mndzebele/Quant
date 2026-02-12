# Risk Management Service Implementation

## Overview

The Risk Management Service (`services/risk_service.py`) has been successfully implemented to enforce trading limits, calculate position sizes, monitor portfolio risk, and trigger stop-loss orders.

## Implementation Status

✅ **COMPLETED** - All required functionality implemented and verified

## Features Implemented

### 1. Trade Validation (`validate_trade`)

Validates trade requests against multiple risk limits:

- **Position Size Limit**: Ensures no single position exceeds configured percentage of portfolio
- **Sufficient Funds**: Verifies available cash for buy orders
- **Maximum Positions**: Enforces limit on number of open positions
- **Daily Loss Limit**: Prevents trading when daily loss limit is reached
- **Portfolio Risk**: Monitors total portfolio exposure

Returns `ValidationResult` with:
- Validation status (pass/fail)
- List of violations
- Detailed error messages
- Suggested quantity adjustments

### 2. Position Size Calculation (`calculate_position_size`)

Calculates appropriate position size based on:

- **Signal Strength**: Scales position size with AI confidence (0.6-1.0)
- **Risk Parameters**: Respects max position size limits
- **Available Capital**: Limited by available cash
- **Kelly Criterion-Inspired**: Uses signal strength as confidence factor

Formula:
```
confidence_factor = (signal_strength - 0.6) / 0.4
position_size_pct = max_position_size * (0.5 + 0.5 * confidence_factor)
quantity = min(position_value, available_cash) / current_price
```

### 3. Stop-Loss Monitoring (`check_stop_loss`)

Monitors positions for stop-loss triggers:

- **Explicit Stop-Loss**: Uses position-specific stop-loss if set
- **Percentage-Based**: Falls back to configured stop-loss percentage
- **Loss Recording**: Tracks losses for daily limit enforcement
- **Automatic Triggering**: Returns true when stop-loss condition met

### 4. Take-Profit Monitoring (`check_take_profit`)

Monitors positions for take-profit triggers:

- **Explicit Take-Profit**: Uses position-specific take-profit if set
- **Percentage-Based**: Falls back to configured take-profit percentage
- **Profit Locking**: Helps secure gains at target levels

### 5. Portfolio Risk Metrics (`get_portfolio_risk`)

Calculates comprehensive risk metrics:

- **Total Exposure**: Sum of all position values
- **Exposure Percentage**: Exposure as percentage of portfolio
- **Value at Risk (VaR)**: 95% confidence VaR estimate
- **Maximum Drawdown**: Largest unrealized loss
- **Daily P&L**: Current day profit/loss
- **Risk Score**: 0-100 composite risk score

Risk Score Components:
- Exposure (30 points): Higher exposure = higher risk
- Daily Loss (30 points): Approaching limit = higher risk
- Drawdown (25 points): Large drawdowns = higher risk
- Concentration (15 points): Too few/many positions = higher risk

### 6. Risk Reduction Suggestions (`suggest_risk_reduction`)

Provides actionable recommendations when risk is high:

- Reduce exposure if too high
- Close losing positions approaching daily limit
- Diversify if under-concentrated
- Set stop-losses if missing
- Take profits on winning positions
- Consolidate if approaching max positions

## Configuration

Risk parameters are configured in `config/settings.py`:

```python
@dataclass
class RiskConfig:
    max_position_size: float = 0.1      # 10% of portfolio
    max_portfolio_risk: float = 0.02    # 2% of portfolio
    daily_loss_limit: float = 1000.0    # $1000
    stop_loss_pct: float = 0.05         # 5%
    take_profit_pct: float = 0.10       # 10%
    max_open_positions: int = 10
```

Environment variables:
- `MAX_POSITION_SIZE`: Maximum position size as decimal (default: 0.1)
- `MAX_PORTFOLIO_RISK`: Maximum portfolio risk as decimal (default: 0.02)
- `DAILY_LOSS_LIMIT`: Daily loss limit in dollars (default: 1000)
- `STOP_LOSS_PCT`: Stop-loss percentage as decimal (default: 0.05)
- `TAKE_PROFIT_PCT`: Take-profit percentage as decimal (default: 0.10)
- `MAX_OPEN_POSITIONS`: Maximum number of open positions (default: 10)

## Usage Examples

### Example 1: Validate a Trade

```python
from services.risk_service import RiskService, TradeRequest

risk_service = RiskService()

trade = TradeRequest(
    symbol="AAPL",
    quantity=50,
    side="buy",
    price=150.0
)

result = risk_service.validate_trade(
    trade=trade,
    portfolio_value=100000.0,
    current_positions=[],
    available_cash=50000.0
)

if result.is_valid:
    print("Trade approved")
else:
    print(f"Trade rejected: {result.get_error_message()}")
    if result.suggested_quantity:
        print(f"Suggested quantity: {result.suggested_quantity}")
```

### Example 2: Calculate Position Size

```python
quantity = risk_service.calculate_position_size(
    symbol="AAPL",
    signal_strength=0.85,  # 85% confidence
    current_price=150.0,
    portfolio_value=100000.0,
    available_cash=50000.0
)

print(f"Recommended position: {quantity} shares")
```

### Example 3: Check Stop-Loss

```python
from services.risk_service import Position

position = Position(
    symbol="AAPL",
    quantity=100,
    entry_price=150.0,
    current_price=140.0,
    unrealized_pl=-1000.0,
    unrealized_pl_pct=-0.0667
)

if risk_service.check_stop_loss(position, 140.0):
    print("Stop-loss triggered - close position")
```

### Example 4: Get Portfolio Risk

```python
metrics = risk_service.get_portfolio_risk(
    portfolio_value=100000.0,
    positions=current_positions,
    daily_pl=-500.0
)

print(f"Risk Score: {metrics.risk_score:.1f}/100")
print(f"Exposure: {metrics.exposure_pct:.1%}")

if metrics.is_high_risk():
    suggestions = risk_service.suggest_risk_reduction(
        positions=current_positions,
        metrics=metrics
    )
    for suggestion in suggestions:
        print(f"- {suggestion}")
```

## Data Models

### ValidationResult

```python
@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[RiskViolationType]
    messages: List[str]
    suggested_quantity: Optional[int] = None
```

### RiskMetrics

```python
@dataclass
class RiskMetrics:
    portfolio_value: float
    total_exposure: float
    exposure_pct: float
    var_95: float
    max_drawdown: float
    daily_pl: float
    daily_pl_pct: float
    open_positions: int
    risk_score: float
```

### TradeRequest

```python
@dataclass
class TradeRequest:
    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    price: float
    order_type: str = 'market'
```

### Position

```python
@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pl: float
    unrealized_pl_pct: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
```

## Risk Violation Types

```python
class RiskViolationType(str, Enum):
    POSITION_SIZE = "position_size"
    PORTFOLIO_RISK = "portfolio_risk"
    DAILY_LOSS = "daily_loss"
    MAX_POSITIONS = "max_positions"
    STOP_LOSS = "stop_loss"
    INSUFFICIENT_FUNDS = "insufficient_funds"
```

## Daily Loss Tracking

The service automatically tracks daily losses:

- Resets at start of each trading day
- Records losses when stop-losses trigger
- Prevents trading when daily limit reached
- Provides remaining loss allowance via `get_daily_loss_remaining()`

## Integration Points

The risk service integrates with:

1. **Trading Service**: Validates trades before execution
2. **AI Engine**: Receives signal strength for position sizing
3. **Portfolio Service**: Gets current positions and portfolio value
4. **Market Data Service**: Receives current prices for stop-loss checks

## Testing

Run verification script:

```bash
python verify_risk_service.py
```

Expected output:
- ✓ Trade validation tests
- ✓ Position sizing tests
- ✓ Stop-loss trigger tests
- ✓ Risk metrics calculation
- ✓ Risk reduction suggestions

## Requirements Satisfied

This implementation satisfies the following requirements:

- **Requirement 7.1**: Configure risk settings (position size, daily loss, allocation limits)
- **Requirement 7.2**: Reject trades exceeding risk limits
- **Requirement 7.3**: Disable trading when daily loss limit reached
- **Requirement 7.4**: Automatically close positions at stop-loss
- **Requirement 7.5**: Alert user and suggest risk reduction actions

## Next Steps

To complete the risk management feature:

1. **Property-Based Tests** (Task 12.2): Write property tests for risk limit enforcement
2. **Stop-Loss Tests** (Task 12.3): Write property tests for stop-loss accuracy
3. **Unit Tests** (Task 12.4): Write comprehensive unit tests
4. **Integration**: Integrate with trading service and AI engine
5. **UI Components**: Add risk management UI to dashboard

## Notes

- All monetary values use float for precision
- Daily tracking resets automatically at midnight
- Risk score is a composite metric (0-100 scale)
- Position sizing uses Kelly Criterion-inspired approach
- Stop-loss checks should be performed on every price update
- Configuration can be updated dynamically via `update_config()`

## Logging

The service logs important events:

- Trade validation results
- Position size calculations
- Stop-loss triggers
- Risk metric calculations
- Configuration changes

Log level can be configured via `LOG_LEVEL` environment variable.
