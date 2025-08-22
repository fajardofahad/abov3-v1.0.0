"""
Custom exceptions for API operations.
"""

from typing import Optional, Any


class APIError(Exception):
    """Base exception for API-related errors."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        response_data: Optional[Any] = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class ConnectionError(APIError):
    """Exception raised when unable to connect to the API."""
    pass


class ModelNotFoundError(APIError):
    """Exception raised when a requested model is not available."""
    pass


class AuthenticationError(APIError):
    """Exception raised when authentication fails."""
    pass


class RateLimitError(APIError):
    """Exception raised when rate limit is exceeded."""
    pass


class TimeoutError(APIError):
    """Exception raised when a request times out."""
    pass


class ValidationError(APIError):
    """Exception raised when request validation fails."""
    pass