# Product Overview

A Python-based algorithmic trading platform for developing, backtesting, and executing quantitative trading strategies in the stock market.

## Core Capabilities

- **Strategy Development**: Custom trading strategies using Python interface with base class inheritance
- **Backtesting Engine**: Test strategies against historical data with comprehensive performance metrics
- **Live Trading**: Execute trades through brokerage APIs (Public, Moomoo) or simulation mode
- **Risk Management**: Built-in position sizing, daily loss limits, leverage controls, and symbol restrictions
- **Portfolio Tracking**: Real-time position monitoring and performance analytics
- **Data Integration**: Multiple market data providers (yfinance, Alpha Vantage) with caching support

## Trading Modes

- **Simulation Mode**: Paper trading with live data for risk-free testing (default)
- **Live Mode**: Real money trading through connected brokerage accounts (requires explicit confirmation)

## Key Design Principles

- Configuration-driven system using YAML with environment variable support
- Modular architecture with clear separation of concerns
- Comprehensive logging with separate files for simulation/live modes
- Safety-first approach with risk validation before order submission
