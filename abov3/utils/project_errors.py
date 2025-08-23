"""
Project-specific error handling for ABOV3.

This module provides specialized exceptions and error handling for project management,
file operations, and context-aware AI assistance.
"""

import logging
from typing import Optional, Dict, Any, List
from enum import Enum


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ABOV3ProjectError(Exception):
    """Base exception for ABOV3 project-related errors."""
    
    def __init__(self, 
                 message: str,
                 error_code: str = "UNKNOWN",
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 details: Optional[Dict[str, Any]] = None,
                 suggestions: Optional[List[str]] = None):
        """Initialize project error."""
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.details = details or {}
        self.suggestions = suggestions or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "details": self.details,
            "suggestions": self.suggestions
        }


class ProjectNotFoundError(ABOV3ProjectError):
    """Raised when a project directory is not found."""
    
    def __init__(self, project_path: str):
        super().__init__(
            message=f"Project directory not found: {project_path}",
            error_code="PROJECT_NOT_FOUND",
            severity=ErrorSeverity.HIGH,
            details={"project_path": project_path},
            suggestions=[
                "Verify the project path is correct",
                "Ensure you have read permissions for the directory",
                "Use an absolute path instead of a relative path"
            ]
        )


class ProjectNotSelectedException(ABOV3ProjectError):
    """Raised when attempting project operations without selecting a project."""
    
    def __init__(self, operation: str = "operation"):
        super().__init__(
            message=f"No project selected for {operation}",
            error_code="PROJECT_NOT_SELECTED",
            severity=ErrorSeverity.MEDIUM,
            suggestions=[
                "Use '/project <path>' to select a project directory",
                "Verify the project was selected successfully"
            ]
        )


class ProjectPermissionError(ABOV3ProjectError):
    """Raised when lacking permissions for project operations."""
    
    def __init__(self, operation: str, path: str):
        super().__init__(
            message=f"Permission denied for {operation} on {path}",
            error_code="PROJECT_PERMISSION_DENIED",
            severity=ErrorSeverity.HIGH,
            details={"operation": operation, "path": path},
            suggestions=[
                "Check file and directory permissions",
                "Run ABOV3 with appropriate privileges",
                "Ensure the project directory is not read-only"
            ]
        )


class FileOperationError(ABOV3ProjectError):
    """Raised when file operations fail."""
    
    def __init__(self, operation: str, file_path: str, reason: str):
        super().__init__(
            message=f"File {operation} failed for {file_path}: {reason}",
            error_code="FILE_OPERATION_FAILED",
            severity=ErrorSeverity.MEDIUM,
            details={
                "operation": operation,
                "file_path": file_path,
                "reason": reason
            },
            suggestions=[
                "Check if the file exists and is accessible",
                "Verify file permissions",
                "Ensure sufficient disk space for write operations"
            ]
        )


class ProjectAnalysisError(ABOV3ProjectError):
    """Raised when project analysis fails."""
    
    def __init__(self, project_path: str, reason: str):
        super().__init__(
            message=f"Project analysis failed for {project_path}: {reason}",
            error_code="PROJECT_ANALYSIS_FAILED",
            severity=ErrorSeverity.MEDIUM,
            details={"project_path": project_path, "reason": reason},
            suggestions=[
                "Verify project structure is valid",
                "Check for corrupted or inaccessible files",
                "Ensure project contains recognizable file types"
            ]
        )


class ContextSyncError(ABOV3ProjectError):
    """Raised when context synchronization fails."""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"Context synchronization failed: {reason}",
            error_code="CONTEXT_SYNC_FAILED",
            severity=ErrorSeverity.MEDIUM,
            details={"reason": reason},
            suggestions=[
                "Check project manager status",
                "Verify file system access",
                "Restart context synchronization"
            ]
        )


class AIAssistanceError(ABOV3ProjectError):
    """Raised when AI assistance operations fail."""
    
    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"AI assistance failed for {operation}: {reason}",
            error_code="AI_ASSISTANCE_FAILED",
            severity=ErrorSeverity.MEDIUM,
            details={"operation": operation, "reason": reason},
            suggestions=[
                "Check Ollama server connection",
                "Verify model availability",
                "Try again with a simpler request"
            ]
        )


class ProjectConfigurationError(ABOV3ProjectError):
    """Raised when project configuration is invalid."""
    
    def __init__(self, setting: str, reason: str):
        super().__init__(
            message=f"Invalid project configuration for {setting}: {reason}",
            error_code="PROJECT_CONFIG_INVALID",
            severity=ErrorSeverity.HIGH,
            details={"setting": setting, "reason": reason},
            suggestions=[
                "Check configuration file syntax",
                "Verify setting values are within valid ranges",
                "Reset to default configuration if needed"
            ]
        )


class FileWatchingError(ABOV3ProjectError):
    """Raised when file watching operations fail."""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"File watching failed: {reason}",
            error_code="FILE_WATCHING_FAILED",
            severity=ErrorSeverity.LOW,
            details={"reason": reason},
            suggestions=[
                "File watching is optional and can be disabled",
                "Check if watchdog package is installed",
                "Verify filesystem supports file watching"
            ]
        )


class ProjectErrorHandler:
    """Centralized error handler for project operations."""
    
    def __init__(self):
        """Initialize the error handler."""
        self.logger = logging.getLogger(f"{__name__}.ProjectErrorHandler")
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an error and return formatted error information.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            
        Returns:
            Dict containing error information
        """
        context = context or {}
        
        if isinstance(error, ABOV3ProjectError):
            # Handle project-specific errors
            error_info = error.to_dict()
            error_info["context"] = context
            
            # Log based on severity
            if error.severity == ErrorSeverity.CRITICAL:
                self.logger.critical(f"{error.error_code}: {error.message}", extra={"details": error.details})
            elif error.severity == ErrorSeverity.HIGH:
                self.logger.error(f"{error.error_code}: {error.message}", extra={"details": error.details})
            elif error.severity == ErrorSeverity.MEDIUM:
                self.logger.warning(f"{error.error_code}: {error.message}", extra={"details": error.details})
            else:
                self.logger.info(f"{error.error_code}: {error.message}", extra={"details": error.details})
            
            return error_info
        
        else:
            # Handle generic errors
            error_info = {
                "error_type": error.__class__.__name__,
                "message": str(error),
                "error_code": "GENERIC_ERROR",
                "severity": ErrorSeverity.MEDIUM.value,
                "details": {"original_error": str(error)},
                "suggestions": ["Check logs for more details", "Contact support if issue persists"],
                "context": context
            }
            
            self.logger.error(f"Unexpected error: {error}", exc_info=True)
            return error_info
    
    def format_user_message(self, error_info: Dict[str, Any]) -> str:
        """
        Format error information for user display.
        
        Args:
            error_info: Error information dictionary
            
        Returns:
            User-friendly error message
        """
        message_parts = []
        
        # Main error message
        message_parts.append(f"Error: {error_info['message']}")
        
        # Add suggestions if available
        suggestions = error_info.get('suggestions', [])
        if suggestions:
            message_parts.append("\nSuggestions:")
            for suggestion in suggestions:
                message_parts.append(f"  • {suggestion}")
        
        # Add error code for reference
        error_code = error_info.get('error_code')
        if error_code and error_code != "GENERIC_ERROR":
            message_parts.append(f"\nError Code: {error_code}")
        
        return "\n".join(message_parts)


# Global error handler instance
error_handler = ProjectErrorHandler()


def handle_project_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to handle project errors.
    
    Args:
        error: The exception that occurred
        context: Additional context information
        
    Returns:
        Dict containing error information
    """
    return error_handler.handle_error(error, context)


def format_error_for_user(error_info: Dict[str, Any]) -> str:
    """
    Convenience function to format error for user display.
    
    Args:
        error_info: Error information dictionary
        
    Returns:
        User-friendly error message
    """
    return error_handler.format_user_message(error_info)