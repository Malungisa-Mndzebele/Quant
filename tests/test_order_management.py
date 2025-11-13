"""Unit tests for order management system."""
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from src.execution import OrderManager, OrderError
from src.models.signal import Signal, SignalAction, OrderType as SignalOrderType
from src.models.order import Order, OrderAction, OrderType, OrderStatusEnum, OrderStatus
from src.brokers.base import BrokerageAdapter, OrderStatus as BrokerOrderStatus


class MockBrokerageAdapter(BrokerageAdapter):
    """Mock brokerage adapter for testing."""
    
    def __init__(self):
        self.submitted_orders = []
        self.order_statuses = {}
        self.should_fail = False
        self.fail_count = 0
        self.call_count = 0
    
    def authenticate(self, credentials):
        return True
    
    def submit_order(self, order):
        """Mock order submission."""
        self.call_count += 1
        
        if self.should_fail and self.call_count <= self.fail_count:
            raise Exception("Simulated submission failure")
        
        self.submitted_orders.append(order)
        broker_order_id = f"BROKER-{len(self.submitted_orders)}"
        self.order_statuses[broker_order_id] = BrokerOrderStatus(
            order_id=broker_order_id,
            status="SUBMITTED",
            filled_quantity=0,
            average_fill_price=None,
            message="Order submitted"
        )
        return broker_order_id
    
    def get_order_status(self, order_id):
        """Mock order status retrieval."""
        if order_id not in self.order_statuses:
            raise Exception(f"Order {order_id} not found")
        return self.order_statuses[order_id]
    
    def get_account_info(self):
        pass
    
    def get_positions(self):
        pass


class TestOrderCreation:
    """Test order creation from signals."""
    
    def test_create_order_from_buy_signal(self):
        """Test creating a BUY order from a signal."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        order = manager.create_order(signal)
        
        assert order.symbol == "AAPL"
        assert order.action == OrderAction.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatusEnum.PENDING
        assert order.order_id.startswith("ORD-")
    
    def test_create_order_from_sell_signal(self):
        """Test creating a SELL order from a signal."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="GOOGL",
            action=SignalAction.SELL,
            quantity=50,
            order_type=SignalOrderType.MARKET
        )
        
        order = manager.create_order(signal)
        
        assert order.symbol == "GOOGL"
        assert order.action == OrderAction.SELL
        assert order.quantity == 50
    
    def test_create_limit_order_with_price(self):
        """Test creating a LIMIT order with price."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.LIMIT,
            limit_price=150.50
        )
        
        order = manager.create_order(signal)
        
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 150.50
    
    def test_create_stop_loss_order(self):
        """Test creating a STOP_LOSS order."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.SELL,
            quantity=100,
            order_type=SignalOrderType.STOP_LOSS,
            limit_price=145.00
        )
        
        order = manager.create_order(signal)
        
        assert order.order_type == OrderType.STOP_LOSS
        assert order.limit_price == 145.00
    
    def test_create_order_from_hold_signal_raises_error(self):
        """Test that HOLD signal cannot be converted to order."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.HOLD,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        with pytest.raises(OrderError) as exc_info:
            manager.create_order(signal)
        
        assert "HOLD signal" in str(exc_info.value)
    
    def test_created_order_stored_in_manager(self):
        """Test that created orders are stored in manager."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        order = manager.create_order(signal)
        
        assert order.order_id in manager.orders
        assert manager.get_order(order.order_id) == order
    
    def test_unique_order_ids(self):
        """Test that each order gets a unique ID."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        order1 = manager.create_order(signal)
        order2 = manager.create_order(signal)
        
        assert order1.order_id != order2.order_id


class TestOrderValidation:
    """Test order validation logic."""
    
    def test_validate_empty_symbol_raises_error(self):
        """Test that empty symbol raises validation error."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        with pytest.raises(OrderError) as exc_info:
            manager.create_order(signal)
        
        assert "symbol cannot be empty" in str(exc_info.value)
    
    def test_validate_invalid_symbol_format(self):
        """Test that invalid symbol format raises error."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL@#$",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        with pytest.raises(OrderError) as exc_info:
            manager.create_order(signal)
        
        assert "Invalid symbol format" in str(exc_info.value)
    
    def test_validate_zero_quantity_raises_error(self):
        """Test that zero quantity raises validation error."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        # Signal validation will catch this first
        with pytest.raises(ValueError):
            Signal(
                symbol="AAPL",
                action=SignalAction.BUY,
                quantity=0,
                order_type=SignalOrderType.MARKET
            )
    
    def test_validate_negative_quantity_raises_error(self):
        """Test that negative quantity raises validation error."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        # Signal validation will catch this first
        with pytest.raises(ValueError):
            Signal(
                symbol="AAPL",
                action=SignalAction.BUY,
                quantity=-100,
                order_type=SignalOrderType.MARKET
            )
    
    def test_validate_limit_order_without_price_raises_error(self):
        """Test that LIMIT order without price raises error."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        # Signal validation will catch this
        with pytest.raises(ValueError):
            Signal(
                symbol="AAPL",
                action=SignalAction.BUY,
                quantity=100,
                order_type=SignalOrderType.LIMIT,
                limit_price=None
            )
    
    def test_validate_stop_loss_without_price_raises_error(self):
        """Test that STOP_LOSS order without price raises error."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        # Signal validation will catch this
        with pytest.raises(ValueError):
            Signal(
                symbol="AAPL",
                action=SignalAction.SELL,
                quantity=100,
                order_type=SignalOrderType.STOP_LOSS,
                limit_price=None
            )
    
    def test_validate_market_order_accepts_no_price(self):
        """Test that MARKET order doesn't require price."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        order = manager.create_order(signal)
        
        assert order.limit_price is None


class TestOrderSubmission:
    """Test order submission to broker."""
    
    def test_submit_order_success(self):
        """Test successful order submission."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        order = manager.create_order(signal)
        result = manager.submit_order(order)
        
        assert result is True
        assert order.status == OrderStatusEnum.SUBMITTED
        assert len(broker.submitted_orders) == 1
    
    def test_submit_order_updates_order_id(self):
        """Test that broker's order ID is used after submission."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        order = manager.create_order(signal)
        original_id = order.order_id
        
        manager.submit_order(order)
        
        assert order.order_id.startswith("BROKER-")
        assert order.order_id != original_id
    
    def test_submit_multiple_orders(self):
        """Test submitting multiple orders."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal1 = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        signal2 = Signal(
            symbol="GOOGL",
            action=SignalAction.BUY,
            quantity=50,
            order_type=SignalOrderType.MARKET
        )
        
        order1 = manager.create_order(signal1)
        order2 = manager.create_order(signal2)
        
        manager.submit_order(order1)
        manager.submit_order(order2)
        
        assert len(broker.submitted_orders) == 2


class TestOrderRetryLogic:
    """Test retry mechanism for failed submissions."""
    
    def test_retry_on_first_failure(self):
        """Test that order submission retries on failure."""
        broker = MockBrokerageAdapter()
        broker.should_fail = True
        broker.fail_count = 1  # Fail first attempt only
        
        manager = OrderManager(broker, max_retries=3)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        order = manager.create_order(signal)
        result = manager.submit_order(order)
        
        # Should succeed on second attempt
        assert result is True
        assert broker.call_count == 2
    
    def test_retry_exhausted_raises_error(self):
        """Test that error is raised after max retries."""
        broker = MockBrokerageAdapter()
        broker.should_fail = True
        broker.fail_count = 5  # Fail all attempts
        
        manager = OrderManager(broker, max_retries=3)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        order = manager.create_order(signal)
        
        with pytest.raises(OrderError) as exc_info:
            manager.submit_order(order)
        
        assert "failed after 3 attempts" in str(exc_info.value)
        assert order.status == OrderStatusEnum.REJECTED
        assert broker.call_count == 3
    
    def test_retry_with_two_failures(self):
        """Test retry with two failures before success."""
        broker = MockBrokerageAdapter()
        broker.should_fail = True
        broker.fail_count = 2  # Fail first two attempts
        
        manager = OrderManager(broker, max_retries=3)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        order = manager.create_order(signal)
        result = manager.submit_order(order)
        
        # Should succeed on third attempt
        assert result is True
        assert broker.call_count == 3


class TestOrderTracking:
    """Test order status tracking."""
    
    def test_track_order_status(self):
        """Test tracking order status."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        order = manager.create_order(signal)
        manager.submit_order(order)
        
        status = manager.track_order(order.order_id)
        
        assert status.order_id == order.order_id
        assert status.status == OrderStatusEnum.SUBMITTED
    
    def test_track_order_updates_local_order(self):
        """Test that tracking updates local order object."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        order = manager.create_order(signal)
        manager.submit_order(order)
        
        # Update broker status to FILLED
        broker.order_statuses[order.order_id] = BrokerOrderStatus(
            order_id=order.order_id,
            status="FILLED",
            filled_quantity=100,
            average_fill_price=150.25,
            message="Order filled"
        )
        
        status = manager.track_order(order.order_id)
        
        assert order.status == OrderStatusEnum.FILLED
        assert order.filled_quantity == 100
        assert order.filled_price == 150.25
    
    def test_track_nonexistent_order_raises_error(self):
        """Test that tracking non-existent order raises error."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        with pytest.raises(OrderError) as exc_info:
            manager.track_order("NONEXISTENT")
        
        assert "not found" in str(exc_info.value)
    
    def test_get_all_orders(self):
        """Test retrieving all orders."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal1 = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        signal2 = Signal(
            symbol="GOOGL",
            action=SignalAction.BUY,
            quantity=50,
            order_type=SignalOrderType.MARKET
        )
        
        order1 = manager.create_order(signal1)
        order2 = manager.create_order(signal2)
        
        all_orders = manager.get_all_orders()
        
        assert len(all_orders) == 2
        assert order1 in all_orders
        assert order2 in all_orders


class TestOrderTypes:
    """Test different order types."""
    
    def test_market_order_validation(self):
        """Test MARKET order validation."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.MARKET
        )
        
        order = manager.create_order(signal)
        
        # Should not raise error
        manager._validate_order(order)
    
    def test_limit_order_validation(self):
        """Test LIMIT order validation."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            quantity=100,
            order_type=SignalOrderType.LIMIT,
            limit_price=150.00
        )
        
        order = manager.create_order(signal)
        
        # Should not raise error
        manager._validate_order(order)
    
    def test_stop_loss_order_validation(self):
        """Test STOP_LOSS order validation."""
        broker = MockBrokerageAdapter()
        manager = OrderManager(broker)
        
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.SELL,
            quantity=100,
            order_type=SignalOrderType.STOP_LOSS,
            limit_price=145.00
        )
        
        order = manager.create_order(signal)
        
        # Should not raise error
        manager._validate_order(order)
