"""Custom exceptions for the trading system."""


class TradingSystemError(Exception):
    """Base exception for trading system."""
    pass


class DataError(TradingSystemError):
    """Market data related errors."""
    pass


class OrderError(TradingSystemError):
    """Order execution related errors."""
    pass


class RiskError(TradingSystemError):
    """Risk limit violations."""
    pass


class BrokerageError(TradingSystemError):
    """Brokerage connection/API errors."""
    pass
