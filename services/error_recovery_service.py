"""
Error recovery service for handling failures and maintaining system state.

This service provides comprehensive error handling including:
- State persistence on errors
- Automatic retry with exponential backoff
- Graceful degradation for API failures
- Error logging and reporting
- State restoration on restart
"""

import logging
import time
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Callable, TypeVar, List
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ServiceStatus(Enum):
    """Service operational status"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    service_name: str
    function_name: str
    retry_count: int
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['severity'] = self.severity.value
        return data


@dataclass
class SystemState:
    """System state snapshot"""
    timestamp: datetime
    service_statuses: Dict[str, ServiceStatus]
    active_operations: List[Dict[str, Any]]
    cached_data: Dict[str, Any]
    configuration: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'service_statuses': {k: v.value for k, v in self.service_statuses.items()},
            'active_operations': self.active_operations,
            'cached_data': self.cached_data,
            'configuration': self.configuration
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemState':
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            service_statuses={
                k: ServiceStatus(v) for k, v in data['service_statuses'].items()
            },
            active_operations=data['active_operations'],
            cached_data=data['cached_data'],
            configuration=data['configuration']
        )


class ErrorRecoveryService:
    """
    Service for handling errors and maintaining system state.
    
    Provides automatic retry, state persistence, graceful degradation,
    and error reporting capabilities.
    """
    
    def __init__(
        self,
        state_dir: str = "data/state",
        error_log_dir: str = "data/errors",
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0
    ):
        """
        Initialize error recovery service.
        
        Args:
            state_dir: Directory for state persistence
            error_log_dir: Directory for error logs
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_factor: Exponential backoff multiplier
        """
        self.state_dir = Path(state_dir)
        self.error_log_dir = Path(error_log_dir)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        
        # Create directories
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.error_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Service status tracking
        self._service_statuses: Dict[str, ServiceStatus] = {}
        
        # Error history
        self._error_history: List[ErrorRecord] = []
        self._max_error_history = 1000
        
        # Circuit breaker state
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._circuit_breaker_threshold = 5  # failures before opening
        self._circuit_breaker_timeout = 60.0  # seconds
        
        # State persistence
        self._last_state_save = datetime.now()
        self._state_save_interval = 60.0  # seconds
        
        logger.info("Error recovery service initialized")
    
    def retry_with_backoff(
        self,
        func: Callable[..., T],
        *args,
        max_retries: Optional[int] = None,
        service_name: str = "unknown",
        **kwargs
    ) -> T:
        """
        Execute function with automatic retry and exponential backoff.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            max_retries: Override default max retries
            service_name: Name of service for logging
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        max_attempts = max_retries if max_retries is not None else self.max_retries
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                result = func(*args, **kwargs)
                
                # Success - reset circuit breaker if it was open
                if service_name in self._circuit_breakers:
                    self._circuit_breakers[service_name]['failures'] = 0
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Get function name safely
                func_name = getattr(func, '__name__', 'unknown')
                
                # Record error
                self._record_error(
                    error=e,
                    service_name=service_name,
                    function_name=func_name,
                    retry_count=attempt,
                    severity=ErrorSeverity.MEDIUM if attempt < max_attempts - 1 else ErrorSeverity.HIGH
                )
                
                # Check if we should retry
                if attempt < max_attempts - 1:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (self.backoff_factor ** attempt),
                        self.max_delay
                    )
                    
                    logger.warning(
                        f"{service_name}.{func.__name__} failed "
                        f"(attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    time.sleep(delay)
                else:
                    logger.error(
                        f"{service_name}.{func.__name__} failed after "
                        f"{max_attempts} attempts: {e}"
                    )
        
        # All retries failed - update circuit breaker
        if service_name:
            self._update_circuit_breaker(service_name)
            
            # Mark service as degraded or unavailable
            self._update_service_status(service_name, ServiceStatus.UNAVAILABLE)
        
        raise last_exception
    
    def with_fallback(
        self,
        primary_func: Callable[..., T],
        fallback_func: Callable[..., T],
        *args,
        service_name: str = "unknown",
        **kwargs
    ) -> T:
        """
        Execute function with fallback on failure (graceful degradation).
        
        Args:
            primary_func: Primary function to try
            fallback_func: Fallback function if primary fails
            *args: Positional arguments
            service_name: Name of service for logging
            **kwargs: Keyword arguments
            
        Returns:
            Result from primary or fallback function
        """
        try:
            result = self.retry_with_backoff(
                primary_func,
                *args,
                service_name=service_name,
                max_retries=2,  # Fewer retries when fallback available
                **kwargs
            )
            
            # Mark service as operational
            self._update_service_status(service_name, ServiceStatus.OPERATIONAL)
            
            return result
            
        except Exception as e:
            logger.warning(
                f"{service_name} primary function failed, using fallback: {e}"
            )
            
            # Mark service as degraded
            self._update_service_status(service_name, ServiceStatus.DEGRADED)
            
            try:
                result = fallback_func(*args, **kwargs)
                logger.info(f"{service_name} fallback succeeded")
                return result
            except Exception as fallback_error:
                logger.error(
                    f"{service_name} fallback also failed: {fallback_error}"
                )
                self._update_service_status(service_name, ServiceStatus.UNAVAILABLE)
                raise
    
    def _record_error(
        self,
        error: Exception,
        service_name: str,
        function_name: str,
        retry_count: int,
        severity: ErrorSeverity,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record an error occurrence"""
        import traceback
        
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            service_name=service_name,
            function_name=function_name,
            retry_count=retry_count,
            stack_trace=traceback.format_exc(),
            context=context
        )
        
        # Add to history
        self._error_history.append(error_record)
        
        # Trim history if too long
        if len(self._error_history) > self._max_error_history:
            self._error_history = self._error_history[-self._max_error_history:]
        
        # Log to file
        self._log_error_to_file(error_record)
    
    def _log_error_to_file(self, error_record: ErrorRecord):
        """Write error to log file"""
        try:
            # Create daily error log file
            log_file = self.error_log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(error_record.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to write error to log file: {e}")
    
    def _update_circuit_breaker(self, service_name: str):
        """Update circuit breaker state for a service"""
        if service_name not in self._circuit_breakers:
            self._circuit_breakers[service_name] = {
                'failures': 0,
                'last_failure': None,
                'state': 'closed'
            }
        
        breaker = self._circuit_breakers[service_name]
        breaker['failures'] += 1
        breaker['last_failure'] = datetime.now()
        
        # Open circuit if threshold exceeded
        if breaker['failures'] >= self._circuit_breaker_threshold:
            breaker['state'] = 'open'
            logger.warning(
                f"Circuit breaker opened for {service_name} "
                f"after {breaker['failures']} failures"
            )
    
    def is_circuit_open(self, service_name: str) -> bool:
        """
        Check if circuit breaker is open for a service.
        
        Args:
            service_name: Name of service to check
            
        Returns:
            True if circuit is open (service should not be called)
        """
        if service_name not in self._circuit_breakers:
            return False
        
        breaker = self._circuit_breakers[service_name]
        
        if breaker['state'] != 'open':
            return False
        
        # Check if timeout has passed
        if breaker['last_failure']:
            time_since_failure = (datetime.now() - breaker['last_failure']).total_seconds()
            
            if time_since_failure > self._circuit_breaker_timeout:
                # Try half-open state
                breaker['state'] = 'half-open'
                breaker['failures'] = 0
                logger.info(f"Circuit breaker for {service_name} entering half-open state")
                return False
        
        return True
    
    def _update_service_status(self, service_name: str, status: ServiceStatus):
        """Update service operational status"""
        old_status = self._service_statuses.get(service_name)
        
        if old_status != status:
            self._service_statuses[service_name] = status
            logger.info(
                f"Service {service_name} status changed: "
                f"{old_status.value if old_status else 'unknown'} -> {status.value}"
            )
    
    def get_service_status(self, service_name: str) -> ServiceStatus:
        """
        Get current operational status of a service.
        
        Args:
            service_name: Name of service
            
        Returns:
            ServiceStatus enum value
        """
        return self._service_statuses.get(service_name, ServiceStatus.OPERATIONAL)
    
    def get_all_service_statuses(self) -> Dict[str, ServiceStatus]:
        """Get status of all tracked services"""
        return self._service_statuses.copy()
    
    def save_state(
        self,
        active_operations: Optional[List[Dict[str, Any]]] = None,
        cached_data: Optional[Dict[str, Any]] = None,
        configuration: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> bool:
        """
        Save current system state to disk.
        
        Args:
            active_operations: List of active operations to persist
            cached_data: Cached data to persist
            configuration: Configuration to persist
            force: Force save even if interval hasn't elapsed
            
        Returns:
            True if state was saved, False if skipped
        """
        # Check if we should save (rate limiting)
        if not force:
            time_since_last = (datetime.now() - self._last_state_save).total_seconds()
            if time_since_last < self._state_save_interval:
                return False
        
        try:
            state = SystemState(
                timestamp=datetime.now(),
                service_statuses=self._service_statuses.copy(),
                active_operations=active_operations or [],
                cached_data=cached_data or {},
                configuration=configuration or {}
            )
            
            # Save as JSON
            state_file = self.state_dir / "system_state.json"
            with open(state_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            
            # Also save as pickle for complex objects
            pickle_file = self.state_dir / "system_state.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(state, f)
            
            self._last_state_save = datetime.now()
            
            logger.info("System state saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
            return False
    
    def restore_state(self) -> Optional[SystemState]:
        """
        Restore system state from disk.
        
        Returns:
            SystemState object if found, None otherwise
        """
        try:
            # Try pickle first (more reliable for complex objects)
            pickle_file = self.state_dir / "system_state.pkl"
            if pickle_file.exists():
                with open(pickle_file, 'rb') as f:
                    state = pickle.load(f)
                
                # Restore service statuses
                self._service_statuses = state.service_statuses.copy()
                
                logger.info(
                    f"System state restored from {state.timestamp.isoformat()}"
                )
                return state
            
            # Fallback to JSON
            state_file = self.state_dir / "system_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                
                state = SystemState.from_dict(data)
                
                # Restore service statuses
                self._service_statuses = state.service_statuses.copy()
                
                logger.info(
                    f"System state restored from {state.timestamp.isoformat()}"
                )
                return state
            
            logger.info("No saved state found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to restore system state: {e}")
            return None
    
    def get_error_summary(
        self,
        since: Optional[datetime] = None,
        service_name: Optional[str] = None,
        min_severity: Optional[ErrorSeverity] = None
    ) -> Dict[str, Any]:
        """
        Get summary of recent errors.
        
        Args:
            since: Only include errors after this time
            service_name: Filter by service name
            min_severity: Minimum severity level to include
            
        Returns:
            Dictionary with error statistics
        """
        # Filter errors
        filtered_errors = self._error_history
        
        if since:
            filtered_errors = [e for e in filtered_errors if e.timestamp >= since]
        
        if service_name:
            filtered_errors = [e for e in filtered_errors if e.service_name == service_name]
        
        if min_severity:
            severity_order = {
                ErrorSeverity.LOW: 0,
                ErrorSeverity.MEDIUM: 1,
                ErrorSeverity.HIGH: 2,
                ErrorSeverity.CRITICAL: 3
            }
            min_level = severity_order[min_severity]
            filtered_errors = [
                e for e in filtered_errors
                if severity_order[e.severity] >= min_level
            ]
        
        # Calculate statistics
        total_errors = len(filtered_errors)
        
        errors_by_service = {}
        errors_by_type = {}
        errors_by_severity = {s: 0 for s in ErrorSeverity}
        
        for error in filtered_errors:
            # By service
            if error.service_name not in errors_by_service:
                errors_by_service[error.service_name] = 0
            errors_by_service[error.service_name] += 1
            
            # By type
            if error.error_type not in errors_by_type:
                errors_by_type[error.error_type] = 0
            errors_by_type[error.error_type] += 1
            
            # By severity
            errors_by_severity[error.severity] += 1
        
        return {
            'total_errors': total_errors,
            'errors_by_service': errors_by_service,
            'errors_by_type': errors_by_type,
            'errors_by_severity': {k.value: v for k, v in errors_by_severity.items()},
            'recent_errors': [e.to_dict() for e in filtered_errors[-10:]]  # Last 10
        }
    
    def clear_error_history(self):
        """Clear error history (useful for testing)"""
        self._error_history.clear()
        logger.info("Error history cleared")
    
    def reset_circuit_breaker(self, service_name: str):
        """
        Manually reset circuit breaker for a service.
        
        Args:
            service_name: Name of service to reset
        """
        if service_name in self._circuit_breakers:
            self._circuit_breakers[service_name] = {
                'failures': 0,
                'last_failure': None,
                'state': 'closed'
            }
            logger.info(f"Circuit breaker reset for {service_name}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Dictionary with health metrics
        """
        # Count services by status
        status_counts = {
            ServiceStatus.OPERATIONAL: 0,
            ServiceStatus.DEGRADED: 0,
            ServiceStatus.UNAVAILABLE: 0
        }
        
        for status in self._service_statuses.values():
            status_counts[status] += 1
        
        # Determine overall health
        if status_counts[ServiceStatus.UNAVAILABLE] > 0:
            overall_health = "unhealthy"
        elif status_counts[ServiceStatus.DEGRADED] > 0:
            overall_health = "degraded"
        else:
            overall_health = "healthy"
        
        # Recent error rate
        recent_errors = [
            e for e in self._error_history
            if (datetime.now() - e.timestamp).total_seconds() < 300  # Last 5 minutes
        ]
        
        return {
            'overall_health': overall_health,
            'service_statuses': {
                k: v.value for k, v in self._service_statuses.items()
            },
            'status_counts': {k.value: v for k, v in status_counts.items()},
            'recent_error_count': len(recent_errors),
            'circuit_breakers': {
                k: v['state'] for k, v in self._circuit_breakers.items()
            },
            'timestamp': datetime.now().isoformat()
        }


# Decorator for automatic error handling
def with_error_recovery(
    service_name: str,
    max_retries: Optional[int] = None,
    fallback: Optional[Callable] = None
):
    """
    Decorator to add error recovery to a function.
    
    Args:
        service_name: Name of service for logging
        max_retries: Maximum retry attempts
        fallback: Optional fallback function
        
    Example:
        @with_error_recovery("market_data", max_retries=3)
        def fetch_quote(symbol):
            # ... implementation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create error recovery service
            if not hasattr(wrapper, '_recovery_service'):
                wrapper._recovery_service = ErrorRecoveryService()
            
            recovery = wrapper._recovery_service
            
            if fallback:
                return recovery.with_fallback(
                    func, fallback, *args,
                    service_name=service_name,
                    **kwargs
                )
            else:
                return recovery.retry_with_backoff(
                    func, *args,
                    max_retries=max_retries,
                    service_name=service_name,
                    **kwargs
                )
        
        return wrapper
    return decorator


# Global error recovery service instance
_global_recovery_service: Optional[ErrorRecoveryService] = None


def get_error_recovery_service() -> ErrorRecoveryService:
    """Get or create global error recovery service instance"""
    global _global_recovery_service
    
    if _global_recovery_service is None:
        _global_recovery_service = ErrorRecoveryService()
    
    return _global_recovery_service
