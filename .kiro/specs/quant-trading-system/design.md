# Quantitative Trading System - Design Document

## Overview

The Quantitative Trading System is a Python-based application that enables algorithmic trading of stocks through brokerage platforms like Public and Moomoo. The system follows a modular architecture with clear separation of concerns between strategy execution, data management, order handling, risk management, and brokerage integration.

The system supports two operational modes:
- **Simulation Mode**: Paper trading with live data for strategy testing
- **Live Mode**: Real trading with actual brokerage accounts

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Trading System Core                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │   Strategy   │─────▶│   Signal     │                     │
│  │   Engine     │      │   Generator  │                     │
│  └──────────────┘      └──────┬───────┘                     │
│                               │                              │
│                               ▼                              │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │     Risk     │◀─────│    Order     │                     │
│  │   Manager    │      │   Manager    │                     │
│  └──────────────┘      └──────┬───────┘                     │
│                               │                              │
│                               ▼                              │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │  Portfolio   │◀─────│  Brokerage   │                     │
│  │   Tracker    │      │  Connector   │                     │
│  └──────────────┘      └──────┬───────┘                     │
│                               │                              │
└───────────────────────────────┼───────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  External Services    │
                    ├───────────────────────┤
                    │ • Brokerage APIs      │
                    │ • Market Data APIs    │
                    └───────────────────────┘
```

### Component Architecture

The system is organized into the following layers:

1. **Strategy Layer**: Implements trading logic and signal generation
2. **Execution Layer**: Handles order creation, validation, and submission
3. **Data Layer**: Manages market data retrieval and caching
4. **Integration Layer**: Interfaces with external brokerage and data APIs
5. **Persistence Layer**: Stores configuration, backtest results, and trade history

## Components and Interfaces

### 1. Strategy Engine

**Purpose**: Executes trading strategies and generates trading signals

**Key Classes**:
- `Strategy` (Abstract Base Class)
  - `on_data(market_data)`: Called when new market data arrives
  - `generate_signals()`: Returns list of trading signals
  - `on_order_filled(order)`: Callback when order is executed

- `StrategyManager`
  - `register_strategy(strategy)`: Adds strategy to execution pipeline
  - `run_strategies(market_data)`: Executes all registered strategies
  - `get_signals()`: Collects signals from all strategies

**Interface**:
```python
class Strategy(ABC):
    @abstractmethod
    def on_data(self, market_data: MarketData) -> None:
        pass
    
    @abstractmethod
    def generate_signals(self) -> List[Signal]:
        pass
    
    def on_order_filled(self, order: Order) -> None:
        pass

class Signal:
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: int
    order_type: str  # 'MARKET', 'LIMIT', 'STOP_LOSS'
    limit_price: Optional[float]
    timestamp: datetime
```

### 2. Market Data Feed

**Purpose**: Retrieves and caches real-time and historical market data

**Key Classes**:
- `MarketDataProvider` (Abstract Base Class)
  - `get_quote(symbol)`: Returns current price quote
  - `get_historical_data(symbol, start_date, end_date)`: Returns OHLCV data
  
- `DataCache`
  - `cache_data(symbol, data)`: Stores data locally
  - `get_cached_data(symbol, date_range)`: Retrieves cached data
  
- `AlphaVantageProvider`, `YFinanceProvider`: Concrete implementations

**Interface**:
```python
class MarketDataProvider(ABC):
    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        pass
    
    @abstractmethod
    def get_historical_data(
        self, 
        symbol: str, 
        start_date: date, 
        end_date: date
    ) -> pd.DataFrame:
        pass

class Quote:
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: datetime
```

### 3. Order Manager

**Purpose**: Creates, validates, and tracks trade orders

**Key Classes**:
- `OrderManager`
  - `create_order(signal)`: Converts signal to order
  - `submit_order(order)`: Sends order to brokerage
  - `track_order(order_id)`: Monitors order status
  - `cancel_order(order_id)`: Cancels pending order

- `Order`
  - Represents a trade order with all necessary details

**Interface**:
```python
class Order:
    order_id: str
    symbol: str
    action: str  # 'BUY', 'SELL'
    quantity: int
    order_type: str
    limit_price: Optional[float]
    status: str  # 'PENDING', 'FILLED', 'CANCELLED', 'REJECTED'
    filled_price: Optional[float]
    filled_quantity: int
    timestamp: datetime

class OrderManager:
    def create_order(self, signal: Signal) -> Order:
        pass
    
    def submit_order(self, order: Order) -> bool:
        pass
    
    def get_order_status(self, order_id: str) -> str:
        pass
```

### 4. Brokerage Connector

**Purpose**: Interfaces with brokerage APIs for order execution and account management

**Key Classes**:
- `BrokerageAdapter` (Abstract Base Class)
  - `authenticate()`: Establishes connection with brokerage
  - `submit_order(order)`: Submits order to brokerage
  - `get_order_status(order_id)`: Checks order status
  - `get_account_info()`: Retrieves account balance and positions
  - `get_positions()`: Returns current positions

- `PublicBrokerageAdapter`, `MoomooBrokerageAdapter`: Platform-specific implementations

- `SimulatedBrokerageAdapter`: Paper trading implementation

**Interface**:
```python
class BrokerageAdapter(ABC):
    @abstractmethod
    def authenticate(self, credentials: Dict) -> bool:
        pass
    
    @abstractmethod
    def submit_order(self, order: Order) -> str:  # Returns order_id
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        pass
    
    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        pass

class AccountInfo:
    account_id: str
    cash_balance: float
    buying_power: float
    portfolio_value: float
```

### 5. Risk Manager

**Purpose**: Enforces risk limits and validates orders before submission

**Key Classes**:
- `RiskManager`
  - `validate_order(order, portfolio)`: Checks if order meets risk criteria
  - `check_position_limit(symbol, quantity, portfolio)`: Validates position size
  - `check_daily_loss_limit(portfolio)`: Checks daily loss threshold
  - `update_risk_metrics(portfolio)`: Calculates current exposure

- `RiskConfig`
  - Stores risk parameters (max position size, daily loss limit, etc.)

**Interface**:
```python
class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config
    
    def validate_order(self, order: Order, portfolio: Portfolio) -> Tuple[bool, str]:
        # Returns (is_valid, reason)
        pass
    
    def check_daily_loss_limit(self, portfolio: Portfolio) -> bool:
        pass

class RiskConfig:
    max_position_size_pct: float  # % of portfolio
    max_daily_loss_pct: float
    max_portfolio_leverage: float
    allowed_symbols: Optional[List[str]]
```

### 6. Portfolio Tracker

**Purpose**: Maintains current positions, cash balance, and performance metrics

**Key Classes**:
- `Portfolio`
  - `add_position(position)`: Adds or updates position
  - `remove_position(symbol)`: Closes position
  - `update_prices(quotes)`: Updates position values with current prices
  - `get_total_value()`: Returns portfolio value
  - `calculate_returns()`: Computes performance metrics

- `Position`
  - Represents a stock position with entry price and quantity

**Interface**:
```python
class Portfolio:
    cash_balance: float
    positions: Dict[str, Position]
    initial_value: float
    
    def add_position(self, symbol: str, quantity: int, entry_price: float):
        pass
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        pass
    
    def calculate_returns(self) -> PerformanceMetrics:
        pass

class Position:
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    
    def update_price(self, price: float):
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity

class PerformanceMetrics:
    total_return: float
    daily_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
```

### 7. Backtesting Module

**Purpose**: Simulates strategy performance using historical data

**Key Classes**:
- `Backtester`
  - `run_backtest(strategy, start_date, end_date)`: Executes backtest
  - `simulate_order_fill(order, market_data)`: Simulates order execution
  - `generate_report()`: Creates performance report

- `BacktestResult`
  - Stores backtest metrics and trade history

**Interface**:
```python
class Backtester:
    def __init__(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider
    
    def run_backtest(
        self, 
        strategy: Strategy, 
        symbols: List[str],
        start_date: date, 
        end_date: date,
        initial_capital: float
    ) -> BacktestResult:
        pass

class BacktestResult:
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: float
    final_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    trade_history: List[Trade]
```

## Data Models

### Core Data Structures

```python
@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class Trade:
    trade_id: str
    symbol: str
    action: str
    quantity: int
    price: float
    timestamp: datetime
    pnl: Optional[float]

@dataclass
class OrderStatus:
    order_id: str
    status: str
    filled_quantity: int
    average_fill_price: Optional[float]
    message: str
```

### Configuration Schema

```yaml
# config.yaml
trading:
  mode: "simulation"  # or "live"
  initial_capital: 100000
  
brokerage:
  provider: "public"  # or "moomoo"
  credentials:
    api_key: "${BROKERAGE_API_KEY}"
    api_secret: "${BROKERAGE_API_SECRET}"

data:
  provider: "yfinance"  # or "alphaavantage"
  cache_enabled: true
  cache_dir: "./data/cache"

risk:
  max_position_size_pct: 10.0
  max_daily_loss_pct: 5.0
  max_portfolio_leverage: 1.0

strategies:
  - name: "MovingAverageCrossover"
    enabled: true
    params:
      fast_period: 10
      slow_period: 30
```

## Error Handling

### Error Categories

1. **Data Errors**
   - Market data unavailable
   - Invalid symbol
   - API rate limit exceeded

2. **Order Errors**
   - Insufficient funds
   - Invalid order parameters
   - Order rejection by brokerage

3. **Connection Errors**
   - Brokerage API authentication failure
   - Network timeout
   - API service unavailable

4. **Risk Errors**
   - Position limit exceeded
   - Daily loss limit breached
   - Unauthorized symbol

### Error Handling Strategy

```python
class TradingSystemError(Exception):
    """Base exception for trading system"""
    pass

class DataError(TradingSystemError):
    """Market data related errors"""
    pass

class OrderError(TradingSystemError):
    """Order execution related errors"""
    pass

class RiskError(TradingSystemError):
    """Risk limit violations"""
    pass

class BrokerageError(TradingSystemError):
    """Brokerage connection/API errors"""
    pass
```

**Error Handling Patterns**:
- Retry logic with exponential backoff for transient errors (network, API limits)
- Circuit breaker pattern for brokerage connection failures
- Graceful degradation: continue with cached data if live data unavailable
- Comprehensive logging of all errors with context
- User notifications for critical errors (risk breaches, order failures)

## Testing Strategy

### Unit Tests

- Test each component in isolation with mocked dependencies
- Focus on:
  - Strategy signal generation logic
  - Risk validation rules
  - Order creation and validation
  - Portfolio calculations
  - Data caching mechanisms

### Integration Tests

- Test component interactions:
  - Strategy → Order Manager → Brokerage Connector flow
  - Market Data Feed → Strategy Engine integration
  - Risk Manager validation in order flow
  - Portfolio updates after order fills

### Backtesting Tests

- Validate backtesting engine with known strategies and historical data
- Compare results against manual calculations
- Test edge cases (market gaps, low volume, extreme volatility)

### End-to-End Tests

- Simulate complete trading cycles in simulation mode
- Test mode switching (simulation ↔ live)
- Verify configuration loading and validation
- Test error recovery scenarios

### Test Data

- Use synthetic market data for predictable test scenarios
- Maintain historical data snapshots for regression testing
- Mock brokerage API responses for integration tests

## Security Considerations

1. **Credential Management**
   - Store API keys in environment variables or secure vault
   - Never commit credentials to version control
   - Use encrypted configuration files for sensitive data

2. **API Security**
   - Implement rate limiting to avoid API bans
   - Use HTTPS for all external API calls
   - Validate and sanitize all external data inputs

3. **Access Control**
   - Require explicit confirmation before switching to live mode
   - Log all order submissions with timestamps
   - Implement audit trail for compliance

4. **Data Validation**
   - Validate all order parameters before submission
   - Sanitize symbol inputs to prevent injection attacks
   - Verify data integrity from external sources

## Performance Considerations

1. **Data Caching**
   - Cache historical data locally to reduce API calls
   - Implement TTL (time-to-live) for cached quotes
   - Use efficient data structures (pandas DataFrames)

2. **Async Operations**
   - Use async/await for concurrent API calls
   - Non-blocking order status polling
   - Parallel strategy execution where possible

3. **Resource Management**
   - Limit concurrent API connections
   - Implement connection pooling for database access
   - Clean up resources properly (close connections, files)

4. **Scalability**
   - Design for multiple strategies running concurrently
   - Support multiple symbols per strategy
   - Modular architecture allows horizontal scaling

## Deployment Architecture

### Development Environment
- Local Python environment with virtual env
- SQLite for data persistence
- File-based configuration
- Simulation mode only

### Production Environment
- Containerized deployment (Docker)
- PostgreSQL for production data
- Environment-based configuration
- Support for both simulation and live modes
- Monitoring and alerting integration
- Automated backup of trade history

## Web Interface

### Purpose
Provides a user-friendly web-based dashboard for configuring, monitoring, and controlling the trading system without requiring command-line interaction.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Browser                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Dashboard  │  │   Config   │  │  Backtest  │            │
│  │   View     │  │   Editor   │  │   Runner   │            │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            │
└────────┼────────────────┼────────────────┼──────────────────┘
         │                │                │
         │    HTTP/REST API (JSON)         │
         │                │                │
┌────────▼────────────────▼────────────────▼──────────────────┐
│                    Flask Web Server                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              API Endpoints                            │   │
│  │  • GET  /api/status    - System status               │   │
│  │  • POST /api/start     - Start trading               │   │
│  │  • POST /api/stop      - Stop trading                │   │
│  │  • GET  /api/config    - Get configuration           │   │
│  │  • POST /api/config    - Update configuration        │   │
│  │  • POST /api/backtest  - Run backtest                │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│  ┌────────────────────────▼──────────────────────────────┐  │
│  │         Background Thread Manager                      │  │
│  │  • Runs TradingSystem in separate thread              │  │
│  │  • Manages system lifecycle (start/stop)              │  │
│  │  • Collects real-time status updates                  │  │
│  └────────────────────────────────────────────────────────┘  │
└───────────────────────────┼──────────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   Trading System      │
                │   (Core Components)   │
                └───────────────────────┘
```

### Components

#### 1. Flask Backend (`web_app.py`)

**Purpose**: REST API server that bridges the web UI and trading system core

**Key Endpoints**:

```python
# System Control
POST /api/start
  - Initializes and starts trading system in background thread
  - Returns: {success: bool, message: str}

POST /api/stop
  - Gracefully stops trading system
  - Returns: {success: bool, message: str}

# Status Monitoring
GET /api/status
  - Returns real-time system status
  - Response: {
      running: bool,
      mode: str,
      portfolio: {
        total_value: float,
        cash: float,
        positions_value: float,
        unrealized_pnl: float,
        total_return: float,
        num_positions: int
      },
      positions: [{
        symbol: str,
        quantity: int,
        avg_price: float,
        current_price: float,
        market_value: float,
        unrealized_pnl: float,
        unrealized_pnl_pct: float
      }],
      account: {
        account_id: str,
        cash: float,
        buying_power: float,
        portfolio_value: float
      },
      last_update: str (ISO timestamp)
    }

# Configuration Management
GET /api/config
  - Retrieves current configuration
  - Returns: {success: bool, config: dict}

POST /api/config
  - Updates system configuration
  - Body: Complete config object
  - Returns: {success: bool, message: str}

# Backtesting
POST /api/backtest
  - Runs historical backtest
  - Body: {
      start_date: str (YYYY-MM-DD),
      end_date: str (YYYY-MM-DD),
      symbols: [str],
      initial_capital: float
    }
  - Returns: {
      success: bool,
      result: BacktestResult
    }
```

**Threading Model**:
- Main thread: Flask web server handling HTTP requests
- Background thread: Trading system execution loop
- Thread-safe communication via global state with locks
- Graceful shutdown handling for both threads

#### 2. Frontend UI (`templates/index.html`)

**Purpose**: Single-page application providing interactive dashboard

**Sections**:

1. **Control Panel**
   - Start/Stop buttons with state management
   - System status indicator (running/stopped)
   - Refresh button for manual updates

2. **Portfolio Overview**
   - Real-time metrics display:
     - Total portfolio value
     - Cash balance
     - Positions value
     - Unrealized P&L (color-coded)
     - Total return percentage
     - Number of positions
   - Auto-refresh every 5 seconds when system running

3. **Positions Table**
   - Live position tracking
   - Columns: Symbol, Quantity, Avg Price, Current Price, Market Value, P&L, P&L%
   - Color-coded profit/loss indicators
   - Empty state when no positions

4. **Configuration Editor**
   - Form-based configuration:
     - Trading mode selector (Simulation/Live)
     - Symbols input (comma-separated)
     - Update interval (seconds)
     - Initial capital
   - Save button with validation
   - Success/error notifications

5. **Backtesting Interface**
   - Date range picker
   - Symbol selection
   - Run backtest button
   - Results display:
     - Total return
     - Sharpe ratio
     - Max drawdown
     - Total trades
     - Win rate
     - Final portfolio value

#### 3. Frontend JavaScript (`static/app.js`)

**Purpose**: Client-side logic for API communication and UI updates

**Key Functions**:

```javascript
// System Control
async function startSystem()
  - Calls POST /api/start
  - Disables start button, enables stop button
  - Starts status polling interval
  - Shows success/error notification

async function stopSystem()
  - Calls POST /api/stop
  - Enables start button, disables stop button
  - Stops status polling
  - Shows success/error notification

// Status Updates
async function updateStatus()
  - Calls GET /api/status
  - Updates all dashboard metrics
  - Refreshes positions table
  - Updates status badge
  - Called every 5 seconds when running

// Configuration
async function loadConfig()
  - Calls GET /api/config
  - Populates configuration form

async function saveConfig()
  - Validates form inputs
  - Calls POST /api/config
  - Shows success/error notification

// Backtesting
async function runBacktest()
  - Validates date range and symbols
  - Calls POST /api/backtest
  - Displays results in grid format
  - Shows loading state during execution
```

**State Management**:
- Polling interval ID for auto-refresh
- System running state
- Last update timestamp
- Error handling with user-friendly messages

#### 4. Styling (`static/style.css`)

**Design System**:
- Modern gradient background (purple theme)
- Card-based layout with shadows
- Responsive grid system
- Color-coded metrics:
  - Green: Positive P&L
  - Red: Negative P&L
  - Gray: Neutral/stopped state
- Smooth transitions and hover effects
- Mobile-responsive design

**Key Features**:
- Pulsing status indicator
- Button state transitions
- Table hover effects
- Form input focus states
- Alert notifications (success/error)

### Data Flow

#### Starting the System
```
User clicks "Start" 
  → Frontend: POST /api/start
  → Backend: Initialize TradingSystem
  → Backend: Start background thread
  → Backend: Return success
  → Frontend: Enable polling
  → Frontend: Update UI state
```

#### Real-time Monitoring
```
Every 5 seconds (when running):
  Frontend: GET /api/status
  → Backend: Query TradingSystem state
  → Backend: Collect portfolio data
  → Backend: Collect positions
  → Backend: Return JSON
  → Frontend: Update dashboard
  → Frontend: Refresh positions table
```

#### Configuration Update
```
User edits config and clicks "Save"
  → Frontend: Validate inputs
  → Frontend: POST /api/config
  → Backend: Write to config.yaml
  → Backend: Return success
  → Frontend: Show notification
```

#### Running Backtest
```
User sets parameters and clicks "Run"
  → Frontend: POST /api/backtest
  → Backend: Create Backtester instance
  → Backend: Fetch historical data
  → Backend: Execute backtest
  → Backend: Return results
  → Frontend: Display metrics
```

### Security Considerations

1. **CORS Policy**
   - Flask-CORS enabled for development
   - Restrict origins in production

2. **Input Validation**
   - Server-side validation of all inputs
   - Sanitize symbol inputs
   - Validate date ranges
   - Check numeric bounds

3. **Authentication** (Future Enhancement)
   - Currently no authentication (local use only)
   - Production should add:
     - JWT-based authentication
     - Session management
     - Role-based access control

4. **Rate Limiting** (Future Enhancement)
   - Prevent API abuse
   - Limit backtest frequency
   - Throttle status requests

### Error Handling

**Frontend**:
- Try-catch blocks around all API calls
- User-friendly error messages
- Alert notifications for errors
- Graceful degradation on API failures

**Backend**:
- Exception handling in all endpoints
- Proper HTTP status codes
- Detailed error messages in responses
- Logging of all errors

### Performance Optimization

1. **Efficient Polling**
   - Only poll when system is running
   - 5-second interval balances freshness and load
   - Stop polling when system stopped

2. **Lazy Loading**
   - Load configuration on demand
   - Backtest results only when requested

3. **Caching**
   - Browser caches static assets (CSS, JS)
   - Configuration cached in memory

4. **Async Operations**
   - Non-blocking API calls
   - Background thread for trading system
   - Responsive UI during operations

### Deployment

**Development**:
```bash
python web_app.py
# Runs on http://localhost:5000
# Debug mode enabled
# Auto-reload on code changes
```

**Production** (Future):
- Use production WSGI server (Gunicorn, uWSGI)
- Reverse proxy (Nginx)
- HTTPS with SSL certificates
- Environment-based configuration
- Proper logging and monitoring

### Future Enhancements

1. **Real-time Updates**
   - WebSocket support for live updates
   - Eliminate polling overhead
   - Push notifications for events

2. **Advanced Charting**
   - Equity curve visualization
   - Position performance charts
   - Strategy signal visualization

3. **Trade History**
   - Detailed trade log viewer
   - Export to CSV/Excel
   - Performance analytics

4. **Multi-user Support**
   - User authentication
   - Multiple portfolios
   - Shared strategies

5. **Mobile App**
   - Native iOS/Android apps
   - Push notifications
   - Quick actions

6. **Strategy Builder**
   - Visual strategy editor
   - No-code strategy creation
   - Strategy marketplace

## Technology Stack

- **Language**: Python 3.10+
- **Data Processing**: pandas, numpy
- **Market Data**: yfinance, alpha_vantage
- **Brokerage APIs**: Platform-specific SDKs (to be integrated)
- **Configuration**: PyYAML, python-dotenv
- **Testing**: pytest, pytest-mock
- **Logging**: Python logging module
- **Async**: asyncio, aiohttp
- **Database**: SQLite (dev), PostgreSQL (prod)
- **Scheduling**: APScheduler for periodic tasks
- **Web Framework**: Flask 3.0+
- **Web UI**: HTML5, CSS3, Vanilla JavaScript
- **HTTP Client**: Fetch API
- **CORS**: Flask-CORS
