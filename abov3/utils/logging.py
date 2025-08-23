"""
Advanced Logging Framework for ABOV3 Ollama

This module provides a comprehensive, production-ready logging system with:
- Structured JSON logging with contextual information
- Multiple handlers (console, file, remote, syslog)
- Automatic log rotation and compression
- Correlation IDs for request tracing
- Performance and security event logging
- Integration with existing configuration system
- Async-compatible logging operations

Features:
- Hierarchical loggers with inheritance
- Custom formatters for different output types
- Security-aware logging with PII filtering
- Performance metrics and timing decorators
- Real-time log streaming capabilities
- Log aggregation and centralized collection

Author: ABOV3 Enterprise DevOps Agent
Version: 1.0.0
"""

import asyncio
import gzip
import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, TextIO
from threading import local, Lock
import queue
import socket
import ssl

from pydantic import BaseModel, Field
import structlog

from ..core.config import get_config, Config


class LogLevel:
    """Log level constants and utilities."""
    
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET
    
    @classmethod
    def from_string(cls, level: str) -> int:
        """Convert string level to logging constant."""
        return getattr(logging, level.upper(), logging.INFO)
    
    @classmethod
    def to_string(cls, level: int) -> str:
        """Convert logging constant to string."""
        return logging.getLevelName(level)


class LoggingConfig(BaseModel):
    """Enhanced logging configuration."""
    
    # Basic settings
    level: str = Field(default="INFO", description="Default logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # File logging
    enable_file_logging: bool = Field(default=True, description="Enable file logging")
    log_dir: str = Field(default="logs", description="Log directory path")
    log_filename: str = Field(default="abov3.log", description="Main log filename")
    max_file_size: int = Field(default=50 * 1024 * 1024, description="Max log file size (50MB)")
    backup_count: int = Field(default=10, description="Number of backup files")
    compress_backups: bool = Field(default=True, description="Compress rotated logs")
    
    # Console logging
    enable_console_logging: bool = Field(default=True, description="Enable console logging")
    console_level: str = Field(default="INFO", description="Console log level")
    colored_output: bool = Field(default=True, description="Enable colored console output")
    
    # Structured logging
    enable_json_logging: bool = Field(default=True, description="Enable JSON structured logging")
    json_filename: str = Field(default="abov3.jsonl", description="JSON log filename")
    
    # Performance logging
    enable_performance_logging: bool = Field(default=True, description="Enable performance logging")
    performance_filename: str = Field(default="performance.log", description="Performance log filename")
    slow_query_threshold: float = Field(default=1.0, description="Slow query threshold in seconds")
    
    # Security logging
    enable_security_logging: bool = Field(default=True, description="Enable security event logging")
    security_filename: str = Field(default="security.log", description="Security log filename")
    
    # Remote logging
    enable_remote_logging: bool = Field(default=False, description="Enable remote log shipping")
    remote_host: Optional[str] = Field(default=None, description="Remote log server host")
    remote_port: int = Field(default=514, description="Remote log server port")
    remote_protocol: str = Field(default="UDP", description="Remote protocol (UDP/TCP)")
    
    # Advanced features
    enable_correlation_ids: bool = Field(default=True, description="Enable correlation ID tracking")
    enable_context_logging: bool = Field(default=True, description="Enable context-aware logging")
    buffer_size: int = Field(default=1000, description="Log buffer size for async operations")
    flush_interval: float = Field(default=5.0, description="Buffer flush interval in seconds")
    
    # PII filtering
    enable_pii_filtering: bool = Field(default=True, description="Enable PII data filtering")
    pii_patterns: List[str] = Field(
        default=[
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card
            r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',  # Phone
        ],
        description="PII regex patterns to filter"
    )


class CorrelationContext:
    """Thread-local storage for correlation context."""
    
    _local = local()
    _lock = Lock()
    
    @classmethod
    def get_correlation_id(cls) -> str:
        """Get current correlation ID or generate a new one."""
        if not hasattr(cls._local, 'correlation_id'):
            cls._local.correlation_id = str(uuid.uuid4())
        return cls._local.correlation_id
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set correlation ID for current context."""
        cls._local.correlation_id = correlation_id
    
    @classmethod
    def clear_correlation_id(cls) -> None:
        """Clear correlation ID from current context."""
        if hasattr(cls._local, 'correlation_id'):
            delattr(cls._local, 'correlation_id')
    
    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """Get all context data."""
        context = {
            'correlation_id': cls.get_correlation_id(),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'thread_id': f"{os.getpid()}-{threading.current_thread().ident}",
        }
        
        # Add custom context if available
        if hasattr(cls._local, 'custom_context'):
            context.update(cls._local.custom_context)
        
        return context
    
    @classmethod
    def set_context(cls, **kwargs) -> None:
        """Set custom context data."""
        if not hasattr(cls._local, 'custom_context'):
            cls._local.custom_context = {}
        cls._local.custom_context.update(kwargs)
    
    @classmethod
    def clear_context(cls) -> None:
        """Clear all context data."""
        if hasattr(cls._local, 'custom_context'):
            cls._local.custom_context.clear()


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, enable_pii_filtering: bool = True, pii_patterns: List[str] = None):
        super().__init__()
        self.enable_pii_filtering = enable_pii_filtering
        self.pii_patterns = pii_patterns or []
        
        if self.enable_pii_filtering:
            import re
            self.pii_regex = [re.compile(pattern) for pattern in self.pii_patterns]
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
        }
        
        # Add correlation context
        log_entry.update(CorrelationContext.get_context())
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Filter PII data
        if self.enable_pii_filtering:
            log_entry = self._filter_pii(log_entry)
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)
    
    def _filter_pii(self, data: Any) -> Any:
        """Recursively filter PII data from log entry."""
        if isinstance(data, dict):
            return {k: self._filter_pii(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._filter_pii(item) for item in data]
        elif isinstance(data, str):
            filtered = data
            for regex in self.pii_regex:
                filtered = regex.sub('[REDACTED]', filtered)
            return filtered
        else:
            return data


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        if hasattr(record, 'no_color') and record.no_color:
            return super().format(record)
        
        level_color = self.COLORS.get(record.levelname, '')
        reset_color = self.COLORS['RESET']
        
        # Color the level name
        record.levelname = f"{level_color}{record.levelname}{reset_color}"
        
        # Format the message
        formatted = super().format(record)
        
        return formatted


class AsyncLogHandler:
    """Asynchronous log handler for non-blocking logging."""
    
    def __init__(self, handler: logging.Handler, buffer_size: int = 1000, flush_interval: float = 5.0):
        self.handler = handler
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.queue = queue.Queue(maxsize=buffer_size)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log-worker")
        self.running = True
        
        # Start background worker
        self.executor.submit(self._worker)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record asynchronously."""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # Drop oldest records if buffer is full
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(record)
            except queue.Empty:
                pass
    
    def _worker(self) -> None:
        """Background worker to process log records."""
        records = []
        last_flush = time.time()
        
        while self.running:
            try:
                # Get record with timeout
                try:
                    record = self.queue.get(timeout=1.0)
                    records.append(record)
                except queue.Empty:
                    pass
                
                # Flush if buffer is full or interval elapsed
                current_time = time.time()
                if (len(records) >= 100 or 
                    (records and current_time - last_flush >= self.flush_interval)):
                    self._flush_records(records)
                    records.clear()
                    last_flush = current_time
                    
            except Exception as e:
                # Log worker errors to stderr
                print(f"Log worker error: {e}", file=sys.stderr)
        
        # Flush remaining records on shutdown
        if records:
            self._flush_records(records)
    
    def _flush_records(self, records: List[logging.LogRecord]) -> None:
        """Flush a batch of log records."""
        for record in records:
            try:
                self.handler.emit(record)
            except Exception as e:
                print(f"Error emitting log record: {e}", file=sys.stderr)
        
        try:
            self.handler.flush()
        except Exception as e:
            print(f"Error flushing handler: {e}", file=sys.stderr)
    
    def close(self) -> None:
        """Close the async handler."""
        self.running = False
        self.executor.shutdown(wait=True)
        self.handler.close()


class PerformanceLogger:
    """Performance logging utilities."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.slow_query_threshold = get_config().logging.slow_query_threshold
    
    def log_timing(self, operation: str, duration: float, **kwargs) -> None:
        """Log operation timing."""
        level = logging.WARNING if duration > self.slow_query_threshold else logging.INFO
        
        self.logger.log(
            level,
            f"Performance: {operation}",
            extra={
                'extra_fields': {
                    'operation': operation,
                    'duration_ms': round(duration * 1000, 2),
                    'slow_query': duration > self.slow_query_threshold,
                    **kwargs
                }
            }
        )
    
    @contextmanager
    def timer(self, operation: str, **kwargs):
        """Context manager for timing operations."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.log_timing(operation, duration, **kwargs)
    
    def timing_decorator(self, operation: Optional[str] = None):
        """Decorator for timing function calls."""
        def decorator(func: Callable) -> Callable:
            op_name = operation or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.timer(op_name):
                    return func(*args, **kwargs)
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.timer(op_name):
                    return await func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator


class SecurityLogger:
    """Security event logging utilities."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_authentication(self, success: bool, user_id: str = None, ip_address: str = None, **kwargs) -> None:
        """Log authentication events."""
        self.logger.info(
            f"Authentication {'successful' if success else 'failed'}",
            extra={
                'extra_fields': {
                    'event_type': 'authentication',
                    'success': success,
                    'user_id': user_id,
                    'ip_address': ip_address,
                    **kwargs
                }
            }
        )
    
    def log_authorization(self, success: bool, user_id: str = None, resource: str = None, action: str = None, **kwargs) -> None:
        """Log authorization events."""
        level = logging.INFO if success else logging.WARNING
        
        self.logger.log(
            level,
            f"Authorization {'granted' if success else 'denied'}",
            extra={
                'extra_fields': {
                    'event_type': 'authorization',
                    'success': success,
                    'user_id': user_id,
                    'resource': resource,
                    'action': action,
                    **kwargs
                }
            }
        )
    
    def log_security_violation(self, violation_type: str, description: str, severity: str = 'medium', **kwargs) -> None:
        """Log security violations."""
        level_map = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }
        
        level = level_map.get(severity.lower(), logging.WARNING)
        
        self.logger.log(
            level,
            f"Security violation: {description}",
            extra={
                'extra_fields': {
                    'event_type': 'security_violation',
                    'violation_type': violation_type,
                    'severity': severity,
                    'description': description,
                    **kwargs
                }
            }
        )
    
    def log_data_access(self, resource: str, user_id: str = None, action: str = 'read', **kwargs) -> None:
        """Log data access events."""
        self.logger.info(
            f"Data access: {action} {resource}",
            extra={
                'extra_fields': {
                    'event_type': 'data_access',
                    'resource': resource,
                    'action': action,
                    'user_id': user_id,
                    **kwargs
                }
            }
        )


class LoggerManager:
    """Central logger management and configuration."""
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        self.config = config or LoggingConfig()
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: List[logging.Handler] = []
        self.async_handlers: List[AsyncLogHandler] = []
        self._initialized = False
        self._lock = Lock()
    
    def initialize(self) -> None:
        """Initialize the logging system."""
        with self._lock:
            if self._initialized:
                return
            
            # Create log directory
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(LogLevel.from_string(self.config.level))
            
            # Clear existing handlers
            root_logger.handlers.clear()
            
            # Setup third-party library logging suppression FIRST
            self._setup_third_party_logging()
            
            # Setup handlers
            if self.config.enable_console_logging:
                self._setup_console_handler()
            
            if self.config.enable_file_logging:
                self._setup_file_handler()
            
            if self.config.enable_json_logging:
                self._setup_json_handler()
            
            if self.config.enable_performance_logging:
                self._setup_performance_handler()
            
            if self.config.enable_security_logging:
                self._setup_security_handler()
            
            if self.config.enable_remote_logging and self.config.remote_host:
                self._setup_remote_handler()
            
            self._initialized = True
    
    def _setup_third_party_logging(self) -> None:
        """Setup third-party library logging to prevent noise in user interface."""
        # Comprehensive list of third-party loggers that can generate noise
        third_party_loggers = [
            'httpx',           # HTTP client library (very verbose)
            'httpcore',        # HTTP core library (connection logs)  
            'urllib3',         # HTTP library
            'requests',        # HTTP library
            'aiohttp',         # Async HTTP library
            'asyncio',         # Asyncio debug logs
            'ollama',          # Ollama library logs
            'websockets',      # WebSocket library
            'prompt_toolkit',  # Terminal library
            'pygments',        # Syntax highlighting
            'rich',            # Rich text library
            'markdown',        # Markdown parsing
            'watchdog',        # File watching
            'gitpython',       # Git library
            'git',             # Git command logs
            'paramiko',        # SSH library
            'cryptography',    # Crypto library
            'ssl',             # SSL library
            'chardet',         # Character detection
            'multipart',       # Multipart parsing
            'h11',             # HTTP/1.1 library
            'h2',              # HTTP/2 library
            'hpack',           # HTTP/2 header compression
            'hyperframe',      # HTTP/2 framing
        ]
        
        # Set WARNING level for most third-party libraries
        for logger_name in third_party_loggers:
            third_party_logger = logging.getLogger(logger_name)
            third_party_logger.setLevel(logging.WARNING)
            third_party_logger.propagate = True  # Allow propagation to file handlers
        
        # Special handling for extremely verbose libraries
        # httpx is particularly noisy with HTTP request details
        httpx_logger = logging.getLogger('httpx')
        httpx_logger.setLevel(logging.ERROR)
        
        # httpcore generates connection-level logs
        httpcore_logger = logging.getLogger('httpcore')  
        httpcore_logger.setLevel(logging.ERROR)
        
        # urllib3 can be verbose with connection pooling
        urllib3_logger = logging.getLogger('urllib3')
        urllib3_logger.setLevel(logging.WARNING)
        urllib3_logger.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
        
        # Suppress asyncio debug logs which can be very verbose
        asyncio_logger = logging.getLogger('asyncio')
        asyncio_logger.setLevel(logging.WARNING)
        
        # Ollama library should show warnings and errors only
        ollama_logger = logging.getLogger('ollama')
        ollama_logger.setLevel(logging.WARNING)
        
        # Ensure ABOV3 logs are preserved at configured level
        abov3_logger = logging.getLogger('abov3')
        abov3_logger.setLevel(LogLevel.from_string(self.config.level))
    
    def _setup_console_handler(self) -> None:
        """Setup console logging handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(LogLevel.from_string(self.config.console_level))
        
        if self.config.colored_output:
            formatter = ColoredFormatter(self.config.format)
        else:
            formatter = logging.Formatter(self.config.format)
        
        handler.setFormatter(formatter)
        
        # Wrap in async handler if configured
        if self.config.buffer_size > 0:
            async_handler = AsyncLogHandler(handler, self.config.buffer_size, self.config.flush_interval)
            self.async_handlers.append(async_handler)
            logging.getLogger().addHandler(logging.Handler())
            logging.getLogger().handlers[-1].emit = async_handler.emit
        else:
            logging.getLogger().addHandler(handler)
        
        self.handlers.append(handler)
    
    def _setup_file_handler(self) -> None:
        """Setup file logging handler."""
        log_file = Path(self.config.log_dir) / self.config.log_filename
        
        if self.config.compress_backups:
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            # Add compression on rotation
            handler.rotator = self._compress_rotated_log
        else:
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
        
        handler.setLevel(LogLevel.from_string(self.config.level))
        handler.setFormatter(logging.Formatter(self.config.format))
        
        logging.getLogger().addHandler(handler)
        self.handlers.append(handler)
    
    def _setup_json_handler(self) -> None:
        """Setup JSON structured logging handler."""
        log_file = Path(self.config.log_dir) / self.config.json_filename
        
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count
        )
        
        handler.setLevel(LogLevel.from_string(self.config.level))
        handler.setFormatter(JSONFormatter(
            self.config.enable_pii_filtering,
            self.config.pii_patterns
        ))
        
        logging.getLogger().addHandler(handler)
        self.handlers.append(handler)
    
    def _setup_performance_handler(self) -> None:
        """Setup performance logging handler."""
        log_file = Path(self.config.log_dir) / self.config.performance_filename
        
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count
        )
        
        handler.setLevel(logging.INFO)
        handler.setFormatter(JSONFormatter(
            self.config.enable_pii_filtering,
            self.config.pii_patterns
        ))
        
        # Create performance logger
        perf_logger = logging.getLogger('abov3.performance')
        perf_logger.addHandler(handler)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False
        
        self.handlers.append(handler)
    
    def _setup_security_handler(self) -> None:
        """Setup security logging handler."""
        log_file = Path(self.config.log_dir) / self.config.security_filename
        
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count
        )
        
        handler.setLevel(logging.INFO)
        handler.setFormatter(JSONFormatter(
            self.config.enable_pii_filtering,
            self.config.pii_patterns
        ))
        
        # Create security logger
        sec_logger = logging.getLogger('abov3.security')
        sec_logger.addHandler(handler)
        sec_logger.setLevel(logging.INFO)
        sec_logger.propagate = False
        
        self.handlers.append(handler)
    
    def _setup_remote_handler(self) -> None:
        """Setup remote logging handler."""
        try:
            if self.config.remote_protocol.upper() == 'TCP':
                handler = logging.handlers.SocketHandler(
                    self.config.remote_host,
                    self.config.remote_port
                )
            else:  # UDP
                handler = logging.handlers.DatagramHandler(
                    self.config.remote_host,
                    self.config.remote_port
                )
            
            handler.setLevel(LogLevel.from_string(self.config.level))
            handler.setFormatter(JSONFormatter(
                self.config.enable_pii_filtering,
                self.config.pii_patterns
            ))
            
            logging.getLogger().addHandler(handler)
            self.handlers.append(handler)
            
        except Exception as e:
            # Log error to console if remote logging fails
            print(f"Failed to setup remote logging: {e}", file=sys.stderr)
    
    def _compress_rotated_log(self, source: str, dest: str) -> None:
        """Compress rotated log files."""
        try:
            with open(source, 'rb') as f_in:
                with gzip.open(f"{dest}.gz", 'wb') as f_out:
                    f_out.writelines(f_in)
            os.remove(source)
        except Exception as e:
            print(f"Failed to compress log file {source}: {e}", file=sys.stderr)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the given name."""
        if not self._initialized:
            self.initialize()
        
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def get_performance_logger(self, name: str = None) -> PerformanceLogger:
        """Get performance logger instance."""
        logger_name = f"abov3.performance.{name}" if name else "abov3.performance"
        logger = self.get_logger(logger_name)
        return PerformanceLogger(logger)
    
    def get_security_logger(self, name: str = None) -> SecurityLogger:
        """Get security logger instance."""
        logger_name = f"abov3.security.{name}" if name else "abov3.security"
        logger = self.get_logger(logger_name)
        return SecurityLogger(logger)
    
    def shutdown(self) -> None:
        """Shutdown the logging system."""
        # Close async handlers
        for async_handler in self.async_handlers:
            async_handler.close()
        
        # Close regular handlers
        for handler in self.handlers:
            handler.close()
        
        # Clear loggers
        self.loggers.clear()
        self.handlers.clear()
        self.async_handlers.clear()
        self._initialized = False


# Global logger manager instance
_logger_manager: Optional[LoggerManager] = None
_manager_lock = Lock()


def get_logger_manager() -> LoggerManager:
    """Get the global logger manager instance."""
    global _logger_manager
    
    if _logger_manager is None:
        with _manager_lock:
            if _logger_manager is None:
                config = get_config()
                logging_config = LoggingConfig(**config.logging.model_dump())
                _logger_manager = LoggerManager(logging_config)
                _logger_manager.initialize()
    
    return _logger_manager


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance."""
    if name is None:
        name = 'abov3'
    
    manager = get_logger_manager()
    return manager.get_logger(name)


def get_performance_logger(name: str = None) -> PerformanceLogger:
    """Get a performance logger instance."""
    manager = get_logger_manager()
    return manager.get_performance_logger(name)


def get_security_logger(name: str = None) -> SecurityLogger:
    """Get a security logger instance."""
    manager = get_logger_manager()
    return manager.get_security_logger(name)


@contextmanager
def log_context(**kwargs):
    """Context manager for adding context to logs."""
    CorrelationContext.set_context(**kwargs)
    try:
        yield
    finally:
        CorrelationContext.clear_context()


@contextmanager
def correlation_context(correlation_id: str = None):
    """Context manager for correlation ID tracking."""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    old_id = getattr(CorrelationContext._local, 'correlation_id', None)
    CorrelationContext.set_correlation_id(correlation_id)
    
    try:
        yield correlation_id
    finally:
        if old_id is not None:
            CorrelationContext.set_correlation_id(old_id)
        else:
            CorrelationContext.clear_correlation_id()


def log_function_call(logger: logging.Logger = None, level: int = logging.DEBUG):
    """Decorator to log function calls."""
    def decorator(func: Callable) -> Callable:
        func_logger = logger or get_logger(func.__module__)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_logger.log(
                level,
                f"Calling {func.__name__}",
                extra={'extra_fields': {'function': func.__name__, 'args_count': len(args), 'kwargs_count': len(kwargs)}}
            )
            try:
                result = func(*args, **kwargs)
                func_logger.log(level, f"Completed {func.__name__}")
                return result
            except Exception as e:
                func_logger.error(f"Error in {func.__name__}: {e}")
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_logger.log(
                level,
                f"Calling {func.__name__}",
                extra={'extra_fields': {'function': func.__name__, 'args_count': len(args), 'kwargs_count': len(kwargs)}}
            )
            try:
                result = await func(*args, **kwargs)
                func_logger.log(level, f"Completed {func.__name__}")
                return result
            except Exception as e:
                func_logger.error(f"Error in {func.__name__}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Convenience functions for common logging operations
def log_info(message: str, **kwargs) -> None:
    """Log an info message."""
    get_logger().info(message, extra={'extra_fields': kwargs} if kwargs else None)


def log_warning(message: str, **kwargs) -> None:
    """Log a warning message."""
    get_logger().warning(message, extra={'extra_fields': kwargs} if kwargs else None)


def log_error(message: str, **kwargs) -> None:
    """Log an error message."""
    get_logger().error(message, extra={'extra_fields': kwargs} if kwargs else None)


def log_debug(message: str, **kwargs) -> None:
    """Log a debug message."""
    get_logger().debug(message, extra={'extra_fields': kwargs} if kwargs else None)


def log_exception(message: str, **kwargs) -> None:
    """Log an exception with traceback."""
    get_logger().exception(message, extra={'extra_fields': kwargs} if kwargs else None)


# Initialize logging on module import
try:
    import threading
    # Initialize logging on first import, but delay getting config until needed
    pass
except Exception as e:
    print(f"Failed to initialize logging system: {e}", file=sys.stderr)


# Export key classes and functions
__all__ = [
    'LogLevel',
    'LoggingConfig',
    'CorrelationContext',
    'JSONFormatter',
    'ColoredFormatter',
    'AsyncLogHandler',
    'PerformanceLogger',
    'SecurityLogger',
    'LoggerManager',
    'get_logger_manager',
    'get_logger',
    'get_performance_logger',
    'get_security_logger',
    'log_context',
    'correlation_context',
    'log_function_call',
    'log_info',
    'log_warning',
    'log_error',
    'log_debug',
    'log_exception',
]