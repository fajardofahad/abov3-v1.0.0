"""
Comprehensive Error Handling Framework for ABOV3 Ollama

This module provides a robust error handling system with:
- Hierarchical exception classes with detailed categorization
- Error recovery strategies and retry mechanisms
- User-friendly error messages with localization support
- Error reporting and aggregation capabilities
- Integration with logging and monitoring systems
- Custom error handlers and middleware
- Async-compatible error handling

Features:
- Structured error taxonomy with inheritance
- Error classification and severity levels
- Contextual error information and debugging data
- Error tracking and correlation
- Recovery strategies and fallback mechanisms
- Error metrics and analytics
- Integration with external error reporting services

Author: ABOV3 Enterprise DevOps Agent
Version: 1.0.0
"""

import asyncio
import inspect
import sys
import traceback
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

from pydantic import BaseModel, Field

from .logging import get_logger, get_security_logger, correlation_context


class ErrorSeverity(Enum):
    """Error severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    
    # System errors
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    FILESYSTEM = "filesystem"
    CONFIGURATION = "configuration"
    
    # Application errors
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    BUSINESS_LOGIC = "business_logic"
    
    # AI/ML specific errors
    MODEL = "model"
    INFERENCE = "inference"
    TRAINING = "training"
    DATA_PROCESSING = "data_processing"
    
    # Integration errors
    API = "api"
    EXTERNAL_SERVICE = "external_service"
    PLUGIN = "plugin"
    
    # User errors
    USER_INPUT = "user_input"
    USAGE = "usage"
    
    # Security errors
    SECURITY = "security"
    PRIVACY = "privacy"
    
    # Unknown/Other
    UNKNOWN = "unknown"


class ErrorRecoveryStrategy(Enum):
    """Error recovery strategies."""
    
    NONE = "none"               # No recovery, fail fast
    RETRY = "retry"             # Retry the operation
    FALLBACK = "fallback"       # Use fallback mechanism
    IGNORE = "ignore"           # Ignore the error and continue
    DEGRADE = "degrade"         # Degrade service gracefully
    CIRCUIT_BREAKER = "circuit_breaker"  # Open circuit breaker


@dataclass
class ErrorContext:
    """Contextual information about an error."""
    
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    version: Optional[str] = None
    environment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'correlation_id': self.correlation_id,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'operation': self.operation,
            'component': self.component,
            'version': self.version,
            'environment': self.environment,
            'metadata': self.metadata
        }


class BaseError(Exception):
    """Base exception class for all ABOV3 errors."""
    
    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.NONE,
        user_message: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        
        self.message = message
        self.category = category
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.user_message = user_message or message
        self.error_code = error_code or self._generate_error_code()
        self.context = context or ErrorContext()
        self.cause = cause
        self.details = details or {}
        self.suggestions = suggestions or []
        self.traceback_info = traceback.format_exc() if sys.exc_info()[0] else None
        
        # Set component from the calling module
        frame = inspect.currentframe()
        if frame and frame.f_back:
            self.context.component = frame.f_back.f_globals.get('__name__', 'unknown')
    
    def _generate_error_code(self) -> str:
        """Generate a unique error code."""
        class_name = self.__class__.__name__
        return f"{class_name.upper()}_{self.category.value.upper()}_{uuid.uuid4().hex[:8].upper()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'user_message': self.user_message,
            'category': self.category.value,
            'severity': self.severity.value,
            'recovery_strategy': self.recovery_strategy.value,
            'context': self.context.to_dict(),
            'details': self.details,
            'suggestions': self.suggestions,
            'cause': str(self.cause) if self.cause else None,
            'traceback': self.traceback_info,
            'class': self.__class__.__name__
        }
    
    def __str__(self) -> str:
        """String representation of the error."""
        return f"[{self.error_code}] {self.message}"
    
    def __repr__(self) -> str:
        """Detailed representation of the error."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"category={self.category.value}, "
            f"severity={self.severity.value}, "
            f"error_code='{self.error_code}'"
            f")"
        )


# System Error Classes
class SystemError(BaseError):
    """Base class for system-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SYSTEM)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class NetworkError(SystemError):
    """Network connectivity and communication errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.NETWORK)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.RETRY)
        super().__init__(message, **kwargs)


class DatabaseError(SystemError):
    """Database connection and operation errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATABASE)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.RETRY)
        super().__init__(message, **kwargs)


class FileSystemError(SystemError):
    """File system operation errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.FILESYSTEM)
        super().__init__(message, **kwargs)


class ConfigurationError(SystemError):
    """Configuration and settings errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        super().__init__(message, **kwargs)


# Application Error Classes
class ValidationError(BaseError):
    """Input validation and data validation errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        kwargs.setdefault('user_message', "Please check your input and try again.")
        super().__init__(message, **kwargs)


class AuthenticationError(BaseError):
    """Authentication and identity verification errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.AUTHENTICATION)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('user_message', "Authentication failed. Please check your credentials.")
        super().__init__(message, **kwargs)


class AuthorizationError(BaseError):
    """Authorization and permission errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.AUTHORIZATION)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('user_message', "You don't have permission to perform this action.")
        super().__init__(message, **kwargs)


class BusinessLogicError(BaseError):
    """Business logic and domain rule violations."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.BUSINESS_LOGIC)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


# AI/ML Error Classes
class ModelError(BaseError):
    """AI model related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MODEL)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.FALLBACK)
        super().__init__(message, **kwargs)


class ModelNotFoundError(ModelError):
    """Model not found or not available."""
    
    def __init__(self, model_name: str, **kwargs):
        message = f"Model '{model_name}' not found or not available"
        kwargs.setdefault('details', {'model_name': model_name})
        kwargs.setdefault('suggestions', [
            "Check if the model name is correct",
            "Verify the model is installed and available",
            "Try using a different model"
        ])
        super().__init__(message, **kwargs)


class InferenceError(BaseError):
    """AI model inference errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.INFERENCE)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.RETRY)
        super().__init__(message, **kwargs)


class TrainingError(BaseError):
    """AI model training errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.TRAINING)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class DataProcessingError(BaseError):
    """Data processing and transformation errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATA_PROCESSING)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


# Integration Error Classes
class APIError(BaseError):
    """API operation errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.API)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.RETRY)
        if status_code:
            kwargs.setdefault('details', {}).update({'status_code': status_code})
        super().__init__(message, **kwargs)


class ExternalServiceError(BaseError):
    """External service integration errors."""
    
    def __init__(self, service_name: str, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.EXTERNAL_SERVICE)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.CIRCUIT_BREAKER)
        kwargs.setdefault('details', {'service_name': service_name})
        super().__init__(f"{service_name}: {message}", **kwargs)


class PluginError(BaseError):
    """Plugin loading and execution errors."""
    
    def __init__(self, plugin_name: str, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.PLUGIN)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('details', {'plugin_name': plugin_name})
        super().__init__(f"Plugin '{plugin_name}': {message}", **kwargs)


# User Error Classes
class UserInputError(BaseError):
    """User input validation and format errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.USER_INPUT)
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        kwargs.setdefault('user_message', "Please check your input and try again.")
        super().__init__(message, **kwargs)


class UsageError(BaseError):
    """Incorrect usage or operation errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.USAGE)
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        super().__init__(message, **kwargs)


# Security Error Classes
class SecurityError(BaseError):
    """Security-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SECURITY)
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        kwargs.setdefault('user_message', "A security issue was detected. Please contact support.")
        super().__init__(message, **kwargs)


class PrivacyError(BaseError):
    """Privacy and data protection errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.PRIVACY)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


# Timeout and Rate Limiting Errors
class TimeoutError(BaseError):
    """Operation timeout errors."""
    
    def __init__(self, operation: str, timeout: float, **kwargs):
        message = f"Operation '{operation}' timed out after {timeout} seconds"
        kwargs.setdefault('category', ErrorCategory.SYSTEM)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.RETRY)
        kwargs.setdefault('details', {'operation': operation, 'timeout': timeout})
        super().__init__(message, **kwargs)


class RateLimitError(BaseError):
    """Rate limiting errors."""
    
    def __init__(self, limit: int, window: int, **kwargs):
        message = f"Rate limit exceeded: {limit} requests per {window} seconds"
        kwargs.setdefault('category', ErrorCategory.SYSTEM)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.RETRY)
        kwargs.setdefault('user_message', "Too many requests. Please try again later.")
        kwargs.setdefault('details', {'limit': limit, 'window': window})
        super().__init__(message, **kwargs)


class ErrorHandler:
    """Error handler interface."""
    
    def __init__(self, logger=None, security_logger=None):
        self.logger = logger or get_logger('errors')
        self.security_logger = security_logger or get_security_logger('errors')
        self.error_registry: Dict[str, BaseError] = {}
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> BaseError:
        """Handle and process an error."""
        # Convert to BaseError if necessary
        if isinstance(error, BaseError):
            handled_error = error
        else:
            handled_error = self._convert_to_base_error(error, context)
        
        # Log the error
        self._log_error(handled_error)
        
        # Register the error
        self._register_error(handled_error)
        
        # Handle security errors specially
        if handled_error.category == ErrorCategory.SECURITY:
            self._handle_security_error(handled_error)
        
        return handled_error
    
    def _convert_to_base_error(self, error: Exception, context: Optional[ErrorContext] = None) -> BaseError:
        """Convert a standard exception to BaseError."""
        error_mapping = {
            ValueError: ValidationError,
            TypeError: ValidationError,
            FileNotFoundError: FileSystemError,
            PermissionError: AuthorizationError,
            ConnectionError: NetworkError,
            TimeoutError: TimeoutError,
        }
        
        error_class = error_mapping.get(type(error), BaseError)
        
        return error_class(
            message=str(error),
            context=context,
            cause=error
        )
    
    def _log_error(self, error: BaseError) -> None:
        """Log an error with appropriate level."""
        log_level_mapping = {
            ErrorSeverity.LOW: 'info',
            ErrorSeverity.MEDIUM: 'warning',
            ErrorSeverity.HIGH: 'error',
            ErrorSeverity.CRITICAL: 'critical'
        }
        
        log_level = log_level_mapping.get(error.severity, 'error')
        log_method = getattr(self.logger, log_level)
        
        log_method(
            f"Error handled: {error.message}",
            extra={'extra_fields': error.to_dict()}
        )
    
    def _register_error(self, error: BaseError) -> None:
        """Register error for tracking and analytics."""
        self.error_registry[error.error_code] = error
    
    def _handle_security_error(self, error: BaseError) -> None:
        """Handle security errors with special logging."""
        self.security_logger.log_security_violation(
            violation_type=error.category.value,
            description=error.message,
            severity=error.severity.value,
            error_code=error.error_code,
            context=error.context.to_dict()
        )
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        category_counts = {}
        severity_counts = {}
        
        for error in self.error_registry.values():
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            'total_errors': len(self.error_registry),
            'by_category': category_counts,
            'by_severity': severity_counts,
            'recent_errors': [
                error.to_dict() 
                for error in list(self.error_registry.values())[-10:]
            ]
        }


class RetryableError(BaseError):
    """Base class for errors that support retry logic."""
    
    def __init__(self, message: str, max_retries: int = 3, retry_delay: float = 1.0, **kwargs):
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.RETRY)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_count = 0
        super().__init__(message, **kwargs)
    
    def should_retry(self) -> bool:
        """Check if the error should be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1


class CircuitBreakerError(BaseError):
    """Error for circuit breaker pattern."""
    
    def __init__(self, service: str, **kwargs):
        message = f"Circuit breaker open for service: {service}"
        kwargs.setdefault('category', ErrorCategory.EXTERNAL_SERVICE)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.CIRCUIT_BREAKER)
        kwargs.setdefault('details', {'service': service})
        super().__init__(message, **kwargs)


class RecoveryStrategy:
    """Error recovery strategy implementation."""
    
    @staticmethod
    async def retry(
        func: Callable,
        *args,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        **kwargs
    ) -> Any:
        """Retry a function with exponential backoff."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                if attempt < max_retries:
                    delay = retry_delay * (backoff_factor ** attempt)
                    await asyncio.sleep(delay)
                else:
                    break
        
        raise last_exception
    
    @staticmethod
    async def with_fallback(
        primary_func: Callable,
        fallback_func: Callable,
        *args,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        **kwargs
    ) -> Any:
        """Execute primary function with fallback on failure."""
        try:
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            else:
                return primary_func(*args, **kwargs)
        except exceptions:
            if asyncio.iscoroutinefunction(fallback_func):
                return await fallback_func(*args, **kwargs)
            else:
                return fallback_func(*args, **kwargs)
    
    @staticmethod
    @contextmanager
    def ignore_errors(*exceptions: Type[Exception]):
        """Context manager to ignore specific exceptions."""
        try:
            yield
        except exceptions:
            pass


def error_handler(
    handler: Optional[ErrorHandler] = None,
    log_errors: bool = True,
    reraise: bool = True,
    fallback_value: Any = None,
    context: Optional[ErrorContext] = None
):
    """Decorator for automatic error handling."""
    def decorator(func: Callable) -> Callable:
        error_mgr = handler or ErrorHandler()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handled_error = error_mgr.handle_error(e, context)
                
                if reraise:
                    raise handled_error
                else:
                    return fallback_value
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handled_error = error_mgr.handle_error(e, context)
                
                if reraise:
                    raise handled_error
                else:
                    return fallback_value
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def retry_on_error(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """Decorator for automatic retry on errors."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await RecoveryStrategy.retry(
                func, *args,
                max_retries=max_retries,
                retry_delay=retry_delay,
                backoff_factor=backoff_factor,
                exceptions=exceptions,
                **kwargs
            )
        
        return wrapper
    
    return decorator


def with_error_context(**context_data):
    """Decorator to add context to errors."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            context = ErrorContext(**context_data)
            try:
                return func(*args, **kwargs)
            except BaseError as e:
                if e.context is None:
                    e.context = context
                else:
                    e.context.metadata.update(context.metadata)
                raise
            except Exception as e:
                raise BaseError(
                    message=str(e),
                    context=context,
                    cause=e
                )
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = ErrorContext(**context_data)
            try:
                return await func(*args, **kwargs)
            except BaseError as e:
                if e.context is None:
                    e.context = context
                else:
                    e.context.metadata.update(context.metadata)
                raise
            except Exception as e:
                raise BaseError(
                    message=str(e),
                    context=context,
                    cause=e
                )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_error(error: Exception, context: Optional[ErrorContext] = None) -> BaseError:
    """Handle an error using the global error handler."""
    return get_error_handler().handle_error(error, context)


# Export key classes and functions
__all__ = [
    # Enums
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorRecoveryStrategy',
    
    # Core classes
    'ErrorContext',
    'BaseError',
    'RetryableError',
    'CircuitBreakerError',
    
    # System errors
    'SystemError',
    'NetworkError',
    'DatabaseError',
    'FileSystemError',
    'ConfigurationError',
    
    # Application errors
    'ValidationError',
    'AuthenticationError',
    'AuthorizationError',
    'BusinessLogicError',
    
    # AI/ML errors
    'ModelError',
    'ModelNotFoundError',
    'InferenceError',
    'TrainingError',
    'DataProcessingError',
    
    # Integration errors
    'APIError',
    'ExternalServiceError',
    'PluginError',
    
    # User errors
    'UserInputError',
    'UsageError',
    
    # Security errors
    'SecurityError',
    'PrivacyError',
    
    # Other errors
    'TimeoutError',
    'RateLimitError',
    
    # Utilities
    'ErrorHandler',
    'RecoveryStrategy',
    
    # Decorators
    'error_handler',
    'retry_on_error',
    'with_error_context',
    
    # Functions
    'get_error_handler',
    'handle_error',
]