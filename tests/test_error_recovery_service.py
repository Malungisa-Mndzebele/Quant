"""Tests for error recovery service"""

import pytest
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from services.error_recovery_service import (
    ErrorRecoveryService,
    ErrorSeverity,
    ServiceStatus,
    ErrorRecord,
    SystemState,
    with_error_recovery,
    get_error_recovery_service
)


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for state files"""
    state_dir = tmp_path / "state"
    error_dir = tmp_path / "errors"
    state_dir.mkdir()
    error_dir.mkdir()
    return state_dir, error_dir


@pytest.fixture
def recovery_service(temp_dir):
    """Create error recovery service with temp directories"""
    state_dir, error_dir = temp_dir
    return ErrorRecoveryService(
        state_dir=str(state_dir),
        error_log_dir=str(error_dir),
        max_retries=3,
        base_delay=0.1,  # Fast for testing
        max_delay=1.0,
        backoff_factor=2.0
    )


class TestErrorRecoveryService:
    """Test error recovery service functionality"""
    
    def test_initialization(self, recovery_service, temp_dir):
        """Test service initialization"""
        state_dir, error_dir = temp_dir
        
        assert recovery_service.state_dir == Path(state_dir)
        assert recovery_service.error_log_dir == Path(error_dir)
        assert recovery_service.max_retries == 3
        assert recovery_service.base_delay == 0.1
        assert recovery_service.max_delay == 1.0
        assert recovery_service.backoff_factor == 2.0
    
    def test_retry_success_on_first_attempt(self, recovery_service):
        """Test successful execution on first attempt"""
        mock_func = Mock(return_value="success")
        
        result = recovery_service.retry_with_backoff(
            mock_func,
            service_name="test_service"
        )
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_success_after_failures(self, recovery_service):
        """Test successful execution after some failures"""
        mock_func = Mock(side_effect=[
            Exception("Fail 1"),
            Exception("Fail 2"),
            "success"
        ])
        
        result = recovery_service.retry_with_backoff(
            mock_func,
            service_name="test_service"
        )
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_retry_all_attempts_fail(self, recovery_service):
        """Test failure after all retry attempts"""
        mock_func = Mock(side_effect=Exception("Always fails"))
        
        with pytest.raises(Exception, match="Always fails"):
            recovery_service.retry_with_backoff(
                mock_func,
                service_name="test_service"
            )
        
        assert mock_func.call_count == 3  # max_retries
    
    def test_exponential_backoff_timing(self, recovery_service):
        """Test exponential backoff delays"""
        call_times = []
        
        def failing_func():
            call_times.append(time.time())
            raise Exception("Fail")
        
        with pytest.raises(Exception):
            recovery_service.retry_with_backoff(
                failing_func,
                service_name="test_service"
            )
        
        # Check delays between attempts
        assert len(call_times) == 3
        
        # First retry delay: ~0.1s
        delay1 = call_times[1] - call_times[0]
        assert 0.08 < delay1 < 0.15
        
        # Second retry delay: ~0.2s (0.1 * 2^1)
        delay2 = call_times[2] - call_times[1]
        assert 0.18 < delay2 < 0.25
    
    def test_with_fallback_primary_succeeds(self, recovery_service):
        """Test fallback when primary succeeds"""
        primary = Mock(return_value="primary_result")
        fallback = Mock(return_value="fallback_result")
        
        result = recovery_service.with_fallback(
            primary,
            fallback,
            service_name="test_service"
        )
        
        assert result == "primary_result"
        assert primary.call_count > 0
        assert fallback.call_count == 0
    
    def test_with_fallback_primary_fails(self, recovery_service):
        """Test fallback when primary fails"""
        primary = Mock(side_effect=Exception("Primary failed"))
        fallback = Mock(return_value="fallback_result")
        
        result = recovery_service.with_fallback(
            primary,
            fallback,
            service_name="test_service"
        )
        
        assert result == "fallback_result"
        assert fallback.call_count == 1
    
    def test_with_fallback_both_fail(self, recovery_service):
        """Test when both primary and fallback fail"""
        primary = Mock(side_effect=Exception("Primary failed"))
        fallback = Mock(side_effect=Exception("Fallback failed"))
        
        with pytest.raises(Exception, match="Fallback failed"):
            recovery_service.with_fallback(
                primary,
                fallback,
                service_name="test_service"
            )
    
    def test_service_status_tracking(self, recovery_service):
        """Test service status updates"""
        # Initially operational
        status = recovery_service.get_service_status("test_service")
        assert status == ServiceStatus.OPERATIONAL
        
        # Update to degraded
        recovery_service._update_service_status("test_service", ServiceStatus.DEGRADED)
        status = recovery_service.get_service_status("test_service")
        assert status == ServiceStatus.DEGRADED
        
        # Update to unavailable
        recovery_service._update_service_status("test_service", ServiceStatus.UNAVAILABLE)
        status = recovery_service.get_service_status("test_service")
        assert status == ServiceStatus.UNAVAILABLE
    
    def test_circuit_breaker_opens_after_failures(self, recovery_service):
        """Test circuit breaker opens after threshold failures"""
        mock_func = Mock(side_effect=Exception("Always fails"))
        
        # Trigger multiple failures
        for _ in range(recovery_service._circuit_breaker_threshold):
            try:
                recovery_service.retry_with_backoff(
                    mock_func,
                    service_name="test_service"
                )
            except Exception:
                pass
        
        # Circuit should be open
        assert recovery_service.is_circuit_open("test_service")
    
    def test_circuit_breaker_half_open_after_timeout(self, recovery_service):
        """Test circuit breaker enters half-open state after timeout"""
        # Set short timeout for testing
        recovery_service._circuit_breaker_timeout = 0.2
        
        mock_func = Mock(side_effect=Exception("Always fails"))
        
        # Open circuit
        for _ in range(recovery_service._circuit_breaker_threshold):
            try:
                recovery_service.retry_with_backoff(
                    mock_func,
                    service_name="test_service"
                )
            except Exception:
                pass
        
        assert recovery_service.is_circuit_open("test_service")
        
        # Wait for timeout
        time.sleep(0.3)
        
        # Circuit should be half-open (not fully open)
        assert not recovery_service.is_circuit_open("test_service")
    
    def test_error_recording(self, recovery_service):
        """Test error recording and history"""
        mock_func = Mock(side_effect=Exception("Test error"))
        
        try:
            recovery_service.retry_with_backoff(
                mock_func,
                service_name="test_service"
            )
        except Exception:
            pass
        
        # Check error history
        assert len(recovery_service._error_history) > 0
        
        error = recovery_service._error_history[0]
        assert error.error_type == "Exception"
        assert error.error_message == "Test error"
        assert error.service_name == "test_service"
    
    def test_error_summary(self, recovery_service):
        """Test error summary generation"""
        mock_func = Mock(side_effect=Exception("Test error"))
        
        # Generate some errors
        for _ in range(3):
            try:
                recovery_service.retry_with_backoff(
                    mock_func,
                    service_name="test_service"
                )
            except Exception:
                pass
        
        summary = recovery_service.get_error_summary()
        
        assert summary['total_errors'] > 0
        assert 'test_service' in summary['errors_by_service']
        assert 'Exception' in summary['errors_by_type']
    
    def test_state_persistence(self, recovery_service):
        """Test saving and restoring system state"""
        # Create state
        active_ops = [{'operation': 'test', 'status': 'pending'}]
        cached_data = {'key': 'value'}
        config = {'setting': 'value'}
        
        # Save state
        success = recovery_service.save_state(
            active_operations=active_ops,
            cached_data=cached_data,
            configuration=config,
            force=True
        )
        
        assert success
        
        # Restore state
        restored = recovery_service.restore_state()
        
        assert restored is not None
        assert restored.active_operations == active_ops
        assert restored.cached_data == cached_data
        assert restored.configuration == config
    
    def test_state_save_rate_limiting(self, recovery_service):
        """Test state save rate limiting"""
        # Set short interval for testing
        recovery_service._state_save_interval = 0.2
        
        # First save should succeed
        success1 = recovery_service.save_state(force=False)
        assert success1
        
        # Immediate second save should be skipped
        success2 = recovery_service.save_state(force=False)
        assert not success2
        
        # Wait for interval
        time.sleep(0.3)
        
        # Third save should succeed
        success3 = recovery_service.save_state(force=False)
        assert success3
    
    def test_health_status(self, recovery_service):
        """Test health status reporting"""
        # Set some service statuses
        recovery_service._update_service_status("service1", ServiceStatus.OPERATIONAL)
        recovery_service._update_service_status("service2", ServiceStatus.DEGRADED)
        recovery_service._update_service_status("service3", ServiceStatus.UNAVAILABLE)
        
        health = recovery_service.get_health_status()
        
        assert health['overall_health'] == "unhealthy"  # Has unavailable service
        assert health['service_statuses']['service1'] == 'operational'
        assert health['service_statuses']['service2'] == 'degraded'
        assert health['service_statuses']['service3'] == 'unavailable'
    
    def test_error_log_file_creation(self, recovery_service, temp_dir):
        """Test error logging to file"""
        _, error_dir = temp_dir
        
        mock_func = Mock(side_effect=Exception("Test error"))
        
        try:
            recovery_service.retry_with_backoff(
                mock_func,
                service_name="test_service"
            )
        except Exception:
            pass
        
        # Check error log file exists
        log_files = list(Path(error_dir).glob("errors_*.jsonl"))
        assert len(log_files) > 0
        
        # Check log content
        with open(log_files[0], 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0
            
            # Parse first error
            error_data = json.loads(lines[0])
            assert error_data['error_type'] == 'Exception'
            assert error_data['error_message'] == 'Test error'
    
    def test_circuit_breaker_reset(self, recovery_service):
        """Test manual circuit breaker reset"""
        mock_func = Mock(side_effect=Exception("Always fails"))
        
        # Open circuit
        for _ in range(recovery_service._circuit_breaker_threshold):
            try:
                recovery_service.retry_with_backoff(
                    mock_func,
                    service_name="test_service"
                )
            except Exception:
                pass
        
        assert recovery_service.is_circuit_open("test_service")
        
        # Reset circuit
        recovery_service.reset_circuit_breaker("test_service")
        
        assert not recovery_service.is_circuit_open("test_service")
    
    def test_clear_error_history(self, recovery_service):
        """Test clearing error history"""
        mock_func = Mock(side_effect=Exception("Test error"))
        
        try:
            recovery_service.retry_with_backoff(
                mock_func,
                service_name="test_service"
            )
        except Exception:
            pass
        
        assert len(recovery_service._error_history) > 0
        
        recovery_service.clear_error_history()
        
        assert len(recovery_service._error_history) == 0


class TestErrorRecoveryDecorator:
    """Test error recovery decorator"""
    
    def test_decorator_success(self):
        """Test decorator with successful function"""
        @with_error_recovery("test_service", max_retries=3)
        def successful_func():
            return "success"
        
        result = successful_func()
        assert result == "success"
    
    def test_decorator_with_retries(self):
        """Test decorator with retries"""
        call_count = [0]
        
        @with_error_recovery("test_service", max_retries=3)
        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Fail")
            return "success"
        
        result = flaky_func()
        assert result == "success"
        assert call_count[0] == 3
    
    def test_decorator_with_fallback(self):
        """Test decorator with fallback function"""
        def fallback_func():
            return "fallback"
        
        @with_error_recovery("test_service", fallback=fallback_func)
        def failing_func():
            raise Exception("Always fails")
        
        result = failing_func()
        assert result == "fallback"


class TestGlobalErrorRecoveryService:
    """Test global error recovery service"""
    
    def test_get_global_service(self):
        """Test getting global service instance"""
        service1 = get_error_recovery_service()
        service2 = get_error_recovery_service()
        
        # Should return same instance
        assert service1 is service2


class TestSystemState:
    """Test SystemState dataclass"""
    
    def test_system_state_creation(self):
        """Test creating system state"""
        state = SystemState(
            timestamp=datetime.now(),
            service_statuses={'service1': ServiceStatus.OPERATIONAL},
            active_operations=[{'op': 'test'}],
            cached_data={'key': 'value'},
            configuration={'setting': 'value'}
        )
        
        assert state.service_statuses['service1'] == ServiceStatus.OPERATIONAL
        assert len(state.active_operations) == 1
        assert state.cached_data['key'] == 'value'
    
    def test_system_state_serialization(self):
        """Test system state to/from dict"""
        original = SystemState(
            timestamp=datetime.now(),
            service_statuses={'service1': ServiceStatus.OPERATIONAL},
            active_operations=[{'op': 'test'}],
            cached_data={'key': 'value'},
            configuration={'setting': 'value'}
        )
        
        # Convert to dict
        data = original.to_dict()
        
        # Convert back
        restored = SystemState.from_dict(data)
        
        assert restored.service_statuses == original.service_statuses
        assert restored.active_operations == original.active_operations
        assert restored.cached_data == original.cached_data


class TestErrorRecord:
    """Test ErrorRecord dataclass"""
    
    def test_error_record_creation(self):
        """Test creating error record"""
        record = ErrorRecord(
            timestamp=datetime.now(),
            error_type="ValueError",
            error_message="Invalid input",
            severity=ErrorSeverity.MEDIUM,
            service_name="test_service",
            function_name="test_func",
            retry_count=1
        )
        
        assert record.error_type == "ValueError"
        assert record.severity == ErrorSeverity.MEDIUM
        assert record.retry_count == 1
    
    def test_error_record_serialization(self):
        """Test error record to dict"""
        record = ErrorRecord(
            timestamp=datetime.now(),
            error_type="ValueError",
            error_message="Invalid input",
            severity=ErrorSeverity.MEDIUM,
            service_name="test_service",
            function_name="test_func",
            retry_count=1,
            context={'key': 'value'}
        )
        
        data = record.to_dict()
        
        assert data['error_type'] == "ValueError"
        assert data['severity'] == 'medium'
        assert data['context'] == {'key': 'value'}


class TestErrorRecoveryProperties:
    """Property-based tests for error recovery service"""
    
    def test_property_error_recovery_state_consistency(self, temp_dir):
        """
        Property 14: Error recovery state consistency
        
        For any system restart after error, the restored state should match
        the last successfully saved state.
        
        **Validates: Requirements 14.5**
        
        Feature: ai-trading-agent, Property 14: Error recovery state consistency
        """
        from hypothesis import given, strategies as st, settings
        
        # Strategy for generating service statuses
        service_status_strategy = st.sampled_from([
            ServiceStatus.OPERATIONAL,
            ServiceStatus.DEGRADED,
            ServiceStatus.UNAVAILABLE
        ])
        
        # Strategy for generating system state components
        @st.composite
        def system_state_components(draw):
            # Generate service statuses
            num_services = draw(st.integers(min_value=1, max_value=5))
            service_statuses = {
                f"service_{i}": draw(service_status_strategy)
                for i in range(num_services)
            }
            
            # Generate active operations
            num_operations = draw(st.integers(min_value=0, max_value=3))
            active_operations = [
                {
                    'operation': f'op_{i}',
                    'status': draw(st.sampled_from(['pending', 'running', 'completed'])),
                    'data': draw(st.text(min_size=0, max_size=20))
                }
                for i in range(num_operations)
            ]
            
            # Generate cached data
            num_cache_items = draw(st.integers(min_value=0, max_value=5))
            cached_data = {
                f'key_{i}': draw(st.one_of(
                    st.text(min_size=0, max_size=20),
                    st.integers(),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.booleans()
                ))
                for i in range(num_cache_items)
            }
            
            # Generate configuration
            num_config_items = draw(st.integers(min_value=0, max_value=5))
            configuration = {
                f'setting_{i}': draw(st.one_of(
                    st.text(min_size=0, max_size=20),
                    st.integers(),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.booleans()
                ))
                for i in range(num_config_items)
            }
            
            return service_statuses, active_operations, cached_data, configuration
        
        @given(system_state_components())
        @settings(max_examples=100, deadline=None)
        def property_test(state_components):
            service_statuses, active_operations, cached_data, configuration = state_components
            
            state_dir, error_dir = temp_dir
            
            # Create first service instance
            service1 = ErrorRecoveryService(
                state_dir=str(state_dir),
                error_log_dir=str(error_dir)
            )
            
            # Set service statuses
            for service_name, status in service_statuses.items():
                service1._update_service_status(service_name, status)
            
            # Save state
            success = service1.save_state(
                active_operations=active_operations,
                cached_data=cached_data,
                configuration=configuration,
                force=True
            )
            
            assert success, "State save should succeed"
            
            # Simulate system restart by creating new service instance
            service2 = ErrorRecoveryService(
                state_dir=str(state_dir),
                error_log_dir=str(error_dir)
            )
            
            # Restore state
            restored_state = service2.restore_state()
            
            assert restored_state is not None, "State should be restored"
            
            # Verify service statuses match
            assert restored_state.service_statuses == service_statuses, \
                f"Service statuses should match: expected {service_statuses}, got {restored_state.service_statuses}"
            
            # Verify active operations match
            assert restored_state.active_operations == active_operations, \
                f"Active operations should match: expected {active_operations}, got {restored_state.active_operations}"
            
            # Verify cached data matches
            assert restored_state.cached_data == cached_data, \
                f"Cached data should match: expected {cached_data}, got {restored_state.cached_data}"
            
            # Verify configuration matches
            assert restored_state.configuration == configuration, \
                f"Configuration should match: expected {configuration}, got {restored_state.configuration}"
            
            # Verify service statuses are restored in the service instance
            for service_name, expected_status in service_statuses.items():
                actual_status = service2.get_service_status(service_name)
                assert actual_status == expected_status, \
                    f"Service {service_name} status should be {expected_status}, got {actual_status}"
        
        # Run the property test
        property_test()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
