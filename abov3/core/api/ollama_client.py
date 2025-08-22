"""
Comprehensive Ollama API client for ABOV3 project.

This module provides a production-ready async client for interacting with the Ollama API,
including model management, chat completion with streaming, error handling, and retry logic.
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import aiohttp
import ollama
from ollama import AsyncClient

from .exceptions import (
    APIError,
    ConnectionError,
    ModelNotFoundError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    ValidationError,
)
from ..config import Config, get_config


logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an available model."""
    name: str
    size: int
    modified_at: str
    digest: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str
    content: str
    images: Optional[List[str]] = None


@dataclass
class ChatResponse:
    """Response from chat completion."""
    message: ChatMessage
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


class OllamaClient:
    """
    Production-ready async Ollama API client.
    
    Features:
    - Async communication with connection pooling
    - Model management (list, pull, delete)
    - Chat completion with streaming support
    - Comprehensive error handling and retries
    - Request timeout and connection management
    - Model parameter validation
    - Response streaming and parsing
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        retry_config: Optional[RetryConfig] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        """
        Initialize the Ollama client.
        
        Args:
            config: Configuration object, defaults to global config
            retry_config: Retry configuration, defaults to standard retry policy
            session: Optional aiohttp session for connection pooling
        """
        self.config = config or get_config()
        self.retry_config = retry_config or RetryConfig(
            max_retries=self.config.ollama.max_retries
        )
        self._session = session
        self._owned_session = session is None
        self._client: Optional[AsyncClient] = None
        self._available_models: Dict[str, ModelInfo] = {}
        self._last_model_refresh = 0.0
        self._model_cache_ttl = 300.0  # 5 minutes
        
        # Connection settings
        self._timeout = aiohttp.ClientTimeout(total=self.config.ollama.timeout)
        self._connector_limit = 100
        self._connector_limit_per_host = 10
    
    async def __aenter__(self) -> "OllamaClient":
        """Async context manager entry."""
        await self._ensure_session()
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self) -> None:
        """Ensure aiohttp session is available."""
        if self._session is None:
            connector = aiohttp.TCPConnector(
                limit=self._connector_limit,
                limit_per_host=self._connector_limit_per_host,
                verify_ssl=self.config.ollama.verify_ssl,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
            )
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=self._timeout,
                headers={"User-Agent": "ABOV3-Ollama-Client/1.0.0"},
            )
    
    async def _ensure_client(self) -> None:
        """Ensure Ollama client is available."""
        if self._client is None:
            self._client = AsyncClient(host=self.config.ollama.host)
    
    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._client:
            # Try to close the ollama client if it has a close method
            try:
                if hasattr(self._client, 'close'):
                    await self._client.close()
                elif hasattr(self._client, 'aclose'):
                    await self._client.aclose()
            except Exception:
                pass  # Ignore errors during cleanup
            
        if self._owned_session and self._session:
            await self._session.close()
            self._session = None
        self._client = None
    
    async def _retry_with_backoff(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute a function with exponential backoff retry logic.
        
        Args:
            func: Async function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            APIError: When all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                ConnectionError,
                RateLimitError,
            ) as e:
                last_exception = e
                
                if attempt == self.retry_config.max_retries:
                    break
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.retry_config.base_delay * (
                        self.retry_config.exponential_base ** attempt
                    ),
                    self.retry_config.max_delay,
                )
                
                # Add jitter to prevent thundering herd
                if self.retry_config.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)
                
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                await asyncio.sleep(delay)
        
        # All retries exhausted
        raise APIError(f"All retry attempts failed: {last_exception}")
    
    def _handle_ollama_error(self, error: Exception) -> APIError:
        """
        Convert Ollama exceptions to our custom exceptions.
        
        Args:
            error: The original exception
            
        Returns:
            Appropriate APIError subclass
        """
        error_msg = str(error)
        
        if "connection" in error_msg.lower():
            return ConnectionError(f"Failed to connect to Ollama: {error_msg}")
        elif "not found" in error_msg.lower() or "model" in error_msg.lower():
            return ModelNotFoundError(f"Model not found: {error_msg}")
        elif "timeout" in error_msg.lower():
            return TimeoutError(f"Request timed out: {error_msg}")
        elif "rate limit" in error_msg.lower():
            return RateLimitError(f"Rate limit exceeded: {error_msg}")
        elif "auth" in error_msg.lower():
            return AuthenticationError(f"Authentication failed: {error_msg}")
        else:
            return APIError(f"Ollama API error: {error_msg}")
    
    def _validate_model_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize model parameters.
        
        Args:
            params: Raw parameters dictionary
            
        Returns:
            Validated parameters dictionary
            
        Raises:
            ValidationError: When parameters are invalid
        """
        validated = {}
        
        # Temperature validation
        if "temperature" in params:
            temp = params["temperature"]
            if not 0.0 <= temp <= 2.0:
                raise ValidationError(f"Temperature must be between 0.0 and 2.0, got {temp}")
            validated["temperature"] = temp
        
        # Top-p validation
        if "top_p" in params:
            top_p = params["top_p"]
            if not 0.0 <= top_p <= 1.0:
                raise ValidationError(f"Top-p must be between 0.0 and 1.0, got {top_p}")
            validated["top_p"] = top_p
        
        # Top-k validation
        if "top_k" in params:
            top_k = params["top_k"]
            if not isinstance(top_k, int) or top_k < 1:
                raise ValidationError(f"Top-k must be a positive integer, got {top_k}")
            validated["top_k"] = top_k
        
        # Repeat penalty validation
        if "repeat_penalty" in params:
            penalty = params["repeat_penalty"]
            if penalty < 0.0:
                raise ValidationError(f"Repeat penalty must be non-negative, got {penalty}")
            validated["repeat_penalty"] = penalty
        
        # Seed validation
        if "seed" in params and params["seed"] is not None:
            seed = params["seed"]
            if not isinstance(seed, int):
                raise ValidationError(f"Seed must be an integer, got {type(seed)}")
            validated["seed"] = seed
        
        # Max tokens validation
        if "max_tokens" in params:
            max_tokens = params["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens < 1:
                raise ValidationError(f"Max tokens must be a positive integer, got {max_tokens}")
            validated["num_predict"] = max_tokens  # Ollama uses num_predict
        
        return validated
    
    async def health_check(self) -> bool:
        """
        Check if the Ollama server is healthy and responsive.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            await self._ensure_client()
            # Try to list models as a health check - this is more reliable
            # than just checking the /api/tags endpoint
            await asyncio.wait_for(self._client.list(), timeout=5.0)
            return True
        except asyncio.TimeoutError:
            logger.warning("Health check timed out")
            return False
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def list_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """
        List available models from Ollama.
        
        Args:
            force_refresh: Force refresh of model cache
            
        Returns:
            List of available models
            
        Raises:
            APIError: When the request fails
        """
        now = time.time()
        
        # Use cache if not forcing refresh and cache is still valid
        if (
            not force_refresh
            and self._available_models
            and (now - self._last_model_refresh) < self._model_cache_ttl
        ):
            return list(self._available_models.values())
        
        await self._ensure_client()
        
        async def _list_models():
            try:
                response = await self._client.list()
                models = []
                
                # Handle the ollama library's ListResponse object
                if hasattr(response, 'models'):
                    # It's a ListResponse object from the ollama library
                    model_list = response.models
                else:
                    # Fallback for dict response (shouldn't happen with current library)
                    model_list = response.get("models", []) if isinstance(response, dict) else []
                
                for model_data in model_list:
                    # The ollama library returns Model objects with 'model' attribute (not 'name')
                    if hasattr(model_data, 'model'):
                        # It's an ollama Model object
                        name = model_data.model  # Note: attribute is 'model' not 'name'
                        size = model_data.size if hasattr(model_data, 'size') else 0
                        modified_at = str(model_data.modified_at) if hasattr(model_data, 'modified_at') else ""
                        digest = model_data.digest if hasattr(model_data, 'digest') else ""
                        
                        # Handle details which is a ModelDetails object
                        details = {}
                        if hasattr(model_data, 'details'):
                            details_obj = model_data.details
                            if hasattr(details_obj, '__dict__'):
                                details = details_obj.__dict__
                            elif isinstance(details_obj, dict):
                                details = details_obj
                    else:
                        # Handle dict format (fallback for compatibility)
                        name = model_data.get("name", model_data.get("model", "unknown"))
                        size = model_data.get("size", 0)
                        modified_at = model_data.get("modified_at", "")
                        digest = model_data.get("digest", "")
                        details = model_data.get("details", {})
                    
                    model_info = ModelInfo(
                        name=name,
                        size=size,
                        modified_at=modified_at,
                        digest=digest,
                        details=details,
                    )
                    models.append(model_info)
                    self._available_models[model_info.name] = model_info
                
                self._last_model_refresh = now
                return models
                
            except Exception as e:
                raise self._handle_ollama_error(e)
        
        return await self._retry_with_backoff(_list_models)
    
    async def pull_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            APIError: When the request fails
        """
        await self._ensure_client()
        
        async def _pull_model():
            try:
                async for chunk in await self._client.pull(model_name, stream=True):
                    if progress_callback:
                        progress_callback(chunk)
                
                # Refresh model cache after successful pull
                await self.list_models(force_refresh=True)
                return True
                
            except Exception as e:
                raise self._handle_ollama_error(e)
        
        return await self._retry_with_backoff(_pull_model)
    
    async def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from local storage.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            APIError: When the request fails
        """
        await self._ensure_client()
        
        async def _delete_model():
            try:
                await self._client.delete(model_name)
                
                # Remove from cache
                self._available_models.pop(model_name, None)
                return True
                
            except Exception as e:
                raise self._handle_ollama_error(e)
        
        return await self._retry_with_backoff(_delete_model)
    
    async def model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists locally.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model exists, False otherwise
        """
        try:
            models = await self.list_models()
            return any(model.name == model_name for model in models)
        except APIError:
            return False
    
    async def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        """
        Send a chat completion request.
        
        Args:
            model: Model name to use
            messages: List of chat messages
            stream: Whether to stream the response
            **kwargs: Additional parameters for the model
            
        Returns:
            Single response or async iterator of responses
            
        Raises:
            APIError: When the request fails
            ModelNotFoundError: When the model is not available
            ValidationError: When parameters are invalid
        """
        await self._ensure_client()
        
        # Validate model exists
        if not await self.model_exists(model):
            raise ModelNotFoundError(f"Model '{model}' is not available")
        
        # Merge config parameters with provided kwargs
        params = self.config.get_model_params()
        params.update(kwargs)
        
        # Validate parameters
        validated_params = self._validate_model_params(params)
        
        # Convert messages to Ollama format
        ollama_messages = [
            {
                "role": msg.role,
                "content": msg.content,
                **({"images": msg.images} if msg.images else {}),
            }
            for msg in messages
        ]
        
        async def _chat():
            try:
                if stream:
                    return self._stream_chat(model, ollama_messages, validated_params)
                else:
                    response = await self._client.chat(
                        model=model,
                        messages=ollama_messages,
                        options=validated_params,
                    )
                    return self._parse_chat_response(response)
                    
            except Exception as e:
                raise self._handle_ollama_error(e)
        
        return await self._retry_with_backoff(_chat)
    
    async def _stream_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        options: Dict[str, Any],
    ) -> AsyncIterator[ChatResponse]:
        """
        Handle streaming chat responses.
        
        Args:
            model: Model name
            messages: List of messages in Ollama format
            options: Model options
            
        Yields:
            ChatResponse objects for each chunk
        """
        try:
            async for chunk in await self._client.chat(
                model=model,
                messages=messages,
                options=options,
                stream=True,
            ):
                yield self._parse_chat_response(chunk)
                
        except Exception as e:
            raise self._handle_ollama_error(e)
    
    def _parse_chat_response(self, response) -> ChatResponse:
        """
        Parse Ollama chat response into ChatResponse object.
        
        Args:
            response: Response from Ollama (either dict or ChatResponse object)
            
        Returns:
            Parsed ChatResponse object
        """
        # Handle both dict format (legacy) and ChatResponse object from ollama library
        if hasattr(response, 'message'):
            # It's an ollama ChatResponse object
            message = ChatMessage(
                role=response.message.role if hasattr(response.message, 'role') else "assistant",
                content=response.message.content if hasattr(response.message, 'content') else "",
                images=getattr(response.message, 'images', None),
            )
            return ChatResponse(
                message=message,
                done=getattr(response, 'done', False),
                total_duration=getattr(response, 'total_duration', None),
                load_duration=getattr(response, 'load_duration', None),
                prompt_eval_count=getattr(response, 'prompt_eval_count', None),
                prompt_eval_duration=getattr(response, 'prompt_eval_duration', None),
                eval_count=getattr(response, 'eval_count', None),
                eval_duration=getattr(response, 'eval_duration', None),
            )
        else:
            # It's a dict format (fallback)
            message_data = response.get("message", {})
            message = ChatMessage(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content", ""),
                images=message_data.get("images"),
            )
            
            return ChatResponse(
                message=message,
                done=response.get("done", False),
                total_duration=response.get("total_duration"),
                load_duration=response.get("load_duration"),
                prompt_eval_count=response.get("prompt_eval_count"),
                prompt_eval_duration=response.get("prompt_eval_duration"),
                eval_count=response.get("eval_count"),
                eval_duration=response.get("eval_duration"),
            )
    
    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generate text completion.
        
        Args:
            model: Model name to use
            prompt: Input prompt
            stream: Whether to stream the response
            **kwargs: Additional parameters for the model
            
        Returns:
            Complete text or async iterator of text chunks
            
        Raises:
            APIError: When the request fails
            ModelNotFoundError: When the model is not available
            ValidationError: When parameters are invalid
        """
        await self._ensure_client()
        
        # Validate model exists
        if not await self.model_exists(model):
            raise ModelNotFoundError(f"Model '{model}' is not available")
        
        # Merge config parameters with provided kwargs
        params = self.config.get_model_params()
        params.update(kwargs)
        
        # Validate parameters
        validated_params = self._validate_model_params(params)
        
        async def _generate():
            try:
                if stream:
                    return self._stream_generate(model, prompt, validated_params)
                else:
                    response = await self._client.generate(
                        model=model,
                        prompt=prompt,
                        options=validated_params,
                    )
                    return response.get("response", "")
                    
            except Exception as e:
                raise self._handle_ollama_error(e)
        
        return await self._retry_with_backoff(_generate)
    
    async def _stream_generate(
        self,
        model: str,
        prompt: str,
        options: Dict[str, Any],
    ) -> AsyncIterator[str]:
        """
        Handle streaming text generation.
        
        Args:
            model: Model name
            prompt: Input prompt
            options: Model options
            
        Yields:
            Text chunks
        """
        try:
            async for chunk in await self._client.generate(
                model=model,
                prompt=prompt,
                options=options,
                stream=True,
            ):
                if "response" in chunk:
                    yield chunk["response"]
                    
        except Exception as e:
            raise self._handle_ollama_error(e)
    
    async def embed(
        self,
        model: str,
        input_text: Union[str, List[str]],
    ) -> List[List[float]]:
        """
        Generate embeddings for input text.
        
        Args:
            model: Model name to use for embeddings
            input_text: Single string or list of strings
            
        Returns:
            List of embedding vectors
            
        Raises:
            APIError: When the request fails
            ModelNotFoundError: When the model is not available
        """
        await self._ensure_client()
        
        # Validate model exists
        if not await self.model_exists(model):
            raise ModelNotFoundError(f"Model '{model}' is not available")
        
        # Ensure input is a list
        if isinstance(input_text, str):
            input_text = [input_text]
        
        async def _embed():
            try:
                embeddings = []
                for text in input_text:
                    response = await self._client.embeddings(
                        model=model,
                        prompt=text,
                    )
                    embeddings.append(response.get("embedding", []))
                return embeddings
                
            except Exception as e:
                raise self._handle_ollama_error(e)
        
        return await self._retry_with_backoff(_embed)
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get current model parameters from config.
        
        Returns:
            Dictionary of model parameters
        """
        return self.config.get_model_params()
    
    def update_model_params(self, **kwargs) -> None:
        """
        Update model parameters in the config.
        
        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config.model, key):
                setattr(self.config.model, key, value)
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo object or None if not found
        """
        models = await self.list_models()
        return next((model for model in models if model.name == model_name), None)


@asynccontextmanager
async def get_ollama_client(
    config: Optional[Config] = None,
    retry_config: Optional[RetryConfig] = None,
) -> OllamaClient:
    """
    Async context manager for getting an Ollama client.
    
    Args:
        config: Optional configuration object
        retry_config: Optional retry configuration
        
    Yields:
        Configured OllamaClient instance
    """
    client = OllamaClient(config=config, retry_config=retry_config)
    try:
        async with client:
            yield client
    finally:
        await client.close()


# Convenience functions for common operations
async def quick_chat(
    prompt: str,
    model: Optional[str] = None,
    config: Optional[Config] = None,
) -> str:
    """
    Quick chat completion for simple use cases.
    
    Args:
        prompt: User prompt
        model: Model name (uses default if not specified)
        config: Optional configuration
        
    Returns:
        Response text
    """
    config = config or get_config()
    model = model or config.model.default_model
    
    async with get_ollama_client(config) as client:
        messages = [ChatMessage(role="user", content=prompt)]
        response = await client.chat(model, messages)
        return response.message.content


async def quick_generate(
    prompt: str,
    model: Optional[str] = None,
    config: Optional[Config] = None,
) -> str:
    """
    Quick text generation for simple use cases.
    
    Args:
        prompt: Input prompt
        model: Model name (uses default if not specified)
        config: Optional configuration
        
    Returns:
        Generated text
    """
    config = config or get_config()
    model = model or config.model.default_model
    
    async with get_ollama_client(config) as client:
        return await client.generate(model, prompt)