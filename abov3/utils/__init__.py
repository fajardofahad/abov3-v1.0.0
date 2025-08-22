"""
ABOV3 Ollama Security and Validation Utilities

This module provides comprehensive security and validation utilities for the ABOV3 platform,
ensuring safe code execution, input validation, and protection against various security threats.

Modules:
    security: Core security utilities for authentication, authorization, and threat protection
    validation: Input/output validation functions and schema enforcement
    sanitize: Input sanitization and code safety validation utilities

Security Features:
    - Input sanitization and validation
    - Code execution sandboxing
    - Path traversal protection
    - Malicious code detection
    - Rate limiting and session management
    - Comprehensive security logging

Author: ABOV3 Enterprise Cybersecurity Agent
Version: 1.0.0
"""

from .security import (
    SecurityManager,
    CodeSandbox,
    RateLimiter,
    SessionManager,
    SecurityLogger,
    validate_file_permissions,
    is_safe_path,
    detect_malicious_patterns,
    hash_password,
    verify_password,
    generate_secure_token,
    validate_token
)

from .validation import (
    InputValidator,
    OutputValidator,
    ModelValidator,
    ConfigValidator,
    PathValidator,
    CommandValidator,
    PythonCodeValidator,
    validate_model_name,
    validate_config_value,
    validate_file_path,
    validate_command,
    validate_python_syntax,
    enforce_security_policy
)

from .sanitize import (
    InputSanitizer,
    CodeSanitizer,
    PromptSanitizer,
    FileSanitizer,
    sanitize_user_input,
    sanitize_code_output,
    sanitize_file_path,
    sanitize_prompt,
    escape_shell_command,
    validate_environment_vars,
    clean_configuration_values
)

from .updater import UpdateChecker
# Note: SetupWizard should be imported directly from .utils.setup to avoid circular imports

# Logging utilities
from .logging import (
    LogLevel,
    LoggingConfig,
    CorrelationContext,
    JSONFormatter,
    ColoredFormatter,
    AsyncLogHandler,
    PerformanceLogger,
    SecurityLogger,
    LoggerManager,
    get_logger_manager,
    get_logger,
    get_performance_logger,
    get_security_logger,
    log_context,
    correlation_context,
    log_function_call,
    log_info,
    log_warning,
    log_error,
    log_debug,
    log_exception,
)

# Error handling utilities
from .errors import (
    ErrorSeverity,
    ErrorCategory,
    ErrorRecoveryStrategy,
    ErrorContext,
    BaseError,
    RetryableError,
    CircuitBreakerError,
    SystemError,
    NetworkError,
    DatabaseError,
    FileSystemError,
    ConfigurationError,
    ValidationError as ErrorValidationError,
    AuthenticationError,
    AuthorizationError,
    BusinessLogicError,
    ModelError,
    ModelNotFoundError,
    InferenceError,
    TrainingError,
    DataProcessingError,
    APIError,
    ExternalServiceError,
    PluginError,
    UserInputError,
    UsageError,
    SecurityError,
    PrivacyError,
    TimeoutError as ErrorTimeoutError,
    RateLimitError,
    ErrorHandler,
    RecoveryStrategy,
    error_handler,
    retry_on_error,
    with_error_context,
    get_error_handler,
    handle_error,
)

# Monitoring utilities
from .monitoring import (
    MetricType,
    AlertSeverity,
    HealthStatus,
    MetricPoint,
    Alert,
    HealthCheck,
    MetricsCollector,
    SystemMetricsCollector,
    ApplicationMetricsCollector,
    AlertManager,
    HealthCheckManager,
    PrometheusExporter,
    MonitoringAPI,
    MonitoringSystem,
    get_monitoring_system,
    get_system_metrics,
    get_app_metrics,
    get_alert_manager,
    get_health_manager,
    monitor_performance,
    count_calls,
)

# Code analysis utilities
from .code_analysis import (
    CodeMetrics,
    FunctionInfo,
    ClassInfo,
    ImportInfo,
    SecurityIssue,
    StyleIssue,
    AnalysisResult,
    LanguageDetector,
    PythonASTAnalyzer,
    CodeQualityAnalyzer,
    CodeSimilarityAnalyzer,
    ProjectAnalyzer as CodeProjectAnalyzer
)

# File operations utilities
from .file_ops import (
    FileInfo,
    DirectoryInfo,
    ProjectStructure,
    BackupInfo,
    ChangeEvent,
    FileOperationError,
    SecurityViolationError,
    PathTraversalError,
    SafeFileOperations,
    DirectoryAnalyzer,
    ProjectAnalyzer as FileProjectAnalyzer,
    FileWatcher,
    ArchiveOperations,
    FileWatcherCallback
)

# Git integration utilities
from .git_integration import (
    GitConfig,
    CommitInfo,
    BranchInfo,
    DiffInfo,
    MergeConflict,
    RepositoryStats,
    RemoteInfo,
    GitOperationError,
    RepositoryNotFoundError,
    BranchNotFoundError,
    MergeConflictError,
    GitRepository,
    GitWorkflow
)

__all__ = [
    # Security utilities
    'SecurityManager',
    'CodeSandbox',
    'RateLimiter',
    'SessionManager',
    'SecurityLogger',
    'validate_file_permissions',
    'is_safe_path',
    'detect_malicious_patterns',
    'hash_password',
    'verify_password',
    'generate_secure_token',
    'validate_token',
    
    # Validation utilities
    'InputValidator',
    'OutputValidator',
    'ModelValidator',
    'ConfigValidator',
    'PathValidator',
    'CommandValidator',
    'PythonCodeValidator',
    'validate_model_name',
    'validate_config_value',
    'validate_file_path',
    'validate_command',
    'validate_python_syntax',
    'enforce_security_policy',
    
    # Sanitization utilities
    'InputSanitizer',
    'CodeSanitizer',
    'PromptSanitizer',
    'FileSanitizer',
    'sanitize_user_input',
    'sanitize_code_output',
    'sanitize_file_path',
    'sanitize_prompt',
    'escape_shell_command',
    'validate_environment_vars',
    'clean_configuration_values',
    
    # CLI utilities
    'UpdateChecker',
    # 'SetupWizard',  # Import directly from .utils.setup to avoid circular imports
    
    # Logging utilities
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
    
    # Error handling utilities
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorRecoveryStrategy',
    'ErrorContext',
    'BaseError',
    'RetryableError',
    'CircuitBreakerError',
    'SystemError',
    'NetworkError',
    'DatabaseError',
    'FileSystemError',
    'ConfigurationError',
    'ErrorValidationError',
    'AuthenticationError',
    'AuthorizationError',
    'BusinessLogicError',
    'ModelError',
    'ModelNotFoundError',
    'InferenceError',
    'TrainingError',
    'DataProcessingError',
    'APIError',
    'ExternalServiceError',
    'PluginError',
    'UserInputError',
    'UsageError',
    'SecurityError',
    'PrivacyError',
    'ErrorTimeoutError',
    'RateLimitError',
    'ErrorHandler',
    'RecoveryStrategy',
    'error_handler',
    'retry_on_error',
    'with_error_context',
    'get_error_handler',
    'handle_error',
    
    # Monitoring utilities
    'MetricType',
    'AlertSeverity',
    'HealthStatus',
    'MetricPoint',
    'Alert',
    'HealthCheck',
    'MetricsCollector',
    'SystemMetricsCollector',
    'ApplicationMetricsCollector',
    'AlertManager',
    'HealthCheckManager',
    'PrometheusExporter',
    'MonitoringAPI',
    'MonitoringSystem',
    'get_monitoring_system',
    'get_system_metrics',
    'get_app_metrics',
    'get_alert_manager',
    'get_health_manager',
    'monitor_performance',
    'count_calls',
    
    # Code analysis utilities
    'CodeMetrics',
    'FunctionInfo',
    'ClassInfo',
    'ImportInfo',
    'SecurityIssue',
    'StyleIssue',
    'AnalysisResult',
    'LanguageDetector',
    'PythonASTAnalyzer',
    'CodeQualityAnalyzer',
    'CodeSimilarityAnalyzer',
    'CodeProjectAnalyzer',
    
    # File operations utilities
    'FileInfo',
    'DirectoryInfo',
    'ProjectStructure',
    'BackupInfo',
    'ChangeEvent',
    'FileOperationError',
    'SecurityViolationError',
    'PathTraversalError',
    'SafeFileOperations',
    'DirectoryAnalyzer',
    'FileProjectAnalyzer',
    'FileWatcher',
    'ArchiveOperations',
    'FileWatcherCallback',
    
    # Git integration utilities
    'GitConfig',
    'CommitInfo',
    'BranchInfo',
    'DiffInfo',
    'MergeConflict',
    'RepositoryStats',
    'RemoteInfo',
    'GitOperationError',
    'RepositoryNotFoundError',
    'BranchNotFoundError',
    'MergeConflictError',
    'GitRepository',
    'GitWorkflow'
]

# Version information
__version__ = '1.0.0'
__author__ = 'ABOV3 Enterprise Cybersecurity Agent'
__email__ = 'security@abov3.ai'
__license__ = 'Proprietary'

# Security configuration
SECURITY_CONFIG = {
    'max_input_length': 10000,
    'max_output_length': 50000,
    'rate_limit_requests': 100,
    'rate_limit_window': 3600,  # 1 hour
    'session_timeout': 1800,    # 30 minutes
    'token_expiry': 86400,      # 24 hours
    'max_file_size': 10485760,  # 10MB
    'allowed_file_extensions': ['.py', '.txt', '.json', '.yaml', '.yml', '.toml'],
    'blocked_patterns': [
        'eval\\s*\\(',
        'exec\\s*\\(',
        '__import__',
        'subprocess\\.',
        'os\\.(system|popen|execv)',
        'open\\s*\\(',
        'file\\s*\\(',
        'input\\s*\\(',
        'raw_input\\s*\\('
    ]
}