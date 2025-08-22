"""
API integration modules for ABOV3 4 Ollama.

This package contains the API clients and adapters for communicating with
Ollama models and other external services.
"""

from .ollama_client import (
    OllamaClient,
    ChatMessage,
    ChatResponse,
    ModelInfo,
    RetryConfig,
    get_ollama_client,
    quick_chat,
    quick_generate,
)
from .exceptions import (
    APIError,
    ModelNotFoundError,
    ConnectionError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    ValidationError,
)

__all__ = [
    # Client and data classes
    "OllamaClient",
    "ChatMessage",
    "ChatResponse", 
    "ModelInfo",
    "RetryConfig",
    
    # Context manager and convenience functions
    "get_ollama_client",
    "quick_chat",
    "quick_generate",
    
    # Exceptions
    "APIError",
    "ModelNotFoundError",
    "ConnectionError",
    "AuthenticationError",
    "RateLimitError",
    "TimeoutError",
    "ValidationError",
]