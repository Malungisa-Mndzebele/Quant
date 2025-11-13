"""Brokerage-related error classes."""


class TradingSystemError(Exception):
    """Base exception for trading system."""
    pass


class BrokerageError(TradingSystemError):
    """Brokerage connection/API errors."""
    pass


class OrderError(TradingSystemError):
    """Order execution related errors."""
    pass


class AuthenticationError(BrokerageError):
    """Authentication failures."""
    pass
