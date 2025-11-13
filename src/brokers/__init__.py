"""Brokers module for brokerage integrations."""

from .base import BrokerageAdapter, AccountInfo, OrderStatus, Position
from .simulated_adapter import SimulatedBrokerageAdapter
from .public_adapter import PublicBrokerageAdapter
from .moomoo_adapter import MoomooBrokerageAdapter
from .credentials import (
    BrokerageCredentials,
    CredentialManager,
    ConnectionStatus
)
from .factory import BrokerageFactory
from .errors import (
    TradingSystemError,
    BrokerageError,
    OrderError,
    AuthenticationError
)

# Register real brokerage adapters with factory
# Note: These are placeholders and will raise NotImplementedError until actual API integration
BrokerageFactory.register_adapter('public', PublicBrokerageAdapter)
BrokerageFactory.register_adapter('moomoo', MoomooBrokerageAdapter)

__all__ = [
    'BrokerageAdapter',
    'AccountInfo',
    'OrderStatus',
    'Position',
    'SimulatedBrokerageAdapter',
    'PublicBrokerageAdapter',
    'MoomooBrokerageAdapter',
    'BrokerageCredentials',
    'CredentialManager',
    'ConnectionStatus',
    'BrokerageFactory',
    'TradingSystemError',
    'BrokerageError',
    'OrderError',
    'AuthenticationError',
]
