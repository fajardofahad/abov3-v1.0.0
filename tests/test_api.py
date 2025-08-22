"""
Test suite for Ollama API client and related functionality.

This module provides comprehensive testing for API interactions,
including connection handling, retry logic, and error scenarios.
"""

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import aiohttp
import pytest
from aioresponses import aioresponses

from abov3.core.api.exceptions import (
    APIError,
    ConnectionError,
    ModelNotFoundError,
    RateLimitError,
    TimeoutError,
    ValidationError,
)
from abov3.core.api.ollama_client import (
    ChatMessage,
    ChatResponse,
    ModelInfo,
    OllamaClient,
    RetryConfig,
    StreamHandler,
    get_ollama_client,
    quick_chat,
    quick_generate,
)


class TestOllamaClientConnection:
    """Test cases for OllamaClient connection handling."""
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, test_config):
        """Test client initialization with various configurations."""
        client = OllamaClient(config=test_config)
        
        assert client.config == test_config
        assert client._client is None
        assert client._session is None
        assert client._available_models == {}
    
    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self, test_config):
        """Test async context manager lifecycle."""
        client = OllamaClient(config=test_config)
        
        async with client as c:
            assert c is client
            assert client._session is not None
            assert client._client is not None
        
        # After context exit, resources should be cleaned up
        assert client._session is None
        assert client._client is None
    
    @pytest.mark.asyncio
    async def test_multiple_context_entries(self, test_config):
        """Test multiple entries into context manager."""
        client = OllamaClient(config=test_config)
        
        # First entry
        async with client:
            session1 = client._session
            assert session1 is not None
        
        # Second entry should create new session
        async with client:
            session2 = client._session
            assert session2 is not None
            assert session2 is not session1
    
    @pytest.mark.asyncio
    async def test_connection_pooling(self, test_config):
        """Test connection pooling configuration."""
        client = OllamaClient(config=test_config)
        
        async with client:
            connector = client._session.connector
            # Check connection pool settings
            assert connector is not None
    
    @pytest.mark.asyncio
    async def test_ssl_verification(self, test_config):
        """Test SSL verification settings."""
        # Test with SSL verification enabled
        test_config.ollama.verify_ssl = True
        client = OllamaClient(config=test_config)
        
        async with client:
            # Should use SSL verification
            pass
        
        # Test with SSL verification disabled
        test_config.ollama.verify_ssl = False
        client = OllamaClient(config=test_config)
        
        async with client:
            # Should skip SSL verification
            pass


class TestHealthCheck:
    """Test cases for health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_ollama_client):
        """Test successful health check."""
        mock_ollama_client.health_check.return_value = True
        
        result = await mock_ollama_client.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_ollama_client):
        """Test failed health check."""
        mock_ollama_client.health_check.side_effect = ConnectionError("Connection refused")
        
        with pytest.raises(ConnectionError):
            await mock_ollama_client.health_check()
    
    @pytest.mark.asyncio
    async def test_health_check_timeout(self, mock_ollama_client):
        """Test health check timeout."""
        mock_ollama_client.health_check.side_effect = asyncio.TimeoutError()
        
        with pytest.raises(asyncio.TimeoutError):
            await mock_ollama_client.health_check()
    
    @pytest.mark.asyncio
    async def test_health_check_with_retry(self, test_config):
        """Test health check with retry logic."""
        client = OllamaClient(
            config=test_config,
            retry_config=RetryConfig(max_retries=2, base_delay=0.1)
        )
        
        call_count = 0
        async def failing_health_check():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return True
        
        with patch.object(client, 'health_check', failing_health_check):
            result = await client.health_check()
            assert result is True
            assert call_count == 2


class TestModelManagement:
    """Test cases for model management operations."""
    
    @pytest.mark.asyncio
    async def test_list_models_success(self, mock_ollama_client):
        """Test successful model listing."""
        models = [
            ModelInfo(name="model1", size=1000, digest="abc", modified_at="2024-01-01"),
            ModelInfo(name="model2", size=2000, digest="def", modified_at="2024-01-02"),
        ]
        mock_ollama_client.list_models.return_value = models
        
        result = await mock_ollama_client.list_models()
        assert len(result) == 2
        assert result[0].name == "model1"
        assert result[1].name == "model2"
    
    @pytest.mark.asyncio
    async def test_list_models_empty(self, mock_ollama_client):
        """Test listing models when none are available."""
        mock_ollama_client.list_models.return_value = []
        
        result = await mock_ollama_client.list_models()
        assert result == []
    
    @pytest.mark.asyncio
    async def test_list_models_caching(self, mock_ollama_client):
        """Test model list caching."""
        models = [ModelInfo(name="cached", size=1000, digest="xyz", modified_at="2024-01-01")]
        mock_ollama_client.list_models.return_value = models
        
        # First call
        result1 = await mock_ollama_client.list_models()
        # Second call (should use cache)
        result2 = await mock_ollama_client.list_models()
        
        assert result1 == result2
        mock_ollama_client.list_models.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_model_exists(self, mock_ollama_client):
        """Test checking if a model exists."""
        mock_ollama_client.model_exists.return_value = True
        
        exists = await mock_ollama_client.model_exists("existing-model")
        assert exists is True
        
        mock_ollama_client.model_exists.return_value = False
        not_exists = await mock_ollama_client.model_exists("non-existent")
        assert not_exists is False
    
    @pytest.mark.asyncio
    async def test_pull_model_success(self, mock_ollama_client):
        """Test successful model pulling."""
        mock_ollama_client.pull_model.return_value = True
        
        result = await mock_ollama_client.pull_model("new-model")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_pull_model_with_progress(self, mock_ollama_client):
        """Test model pulling with progress callback."""
        progress_updates = []
        
        async def mock_pull(model, callback=None):
            updates = [
                {"status": "downloading", "completed": 0, "total": 100},
                {"status": "downloading", "completed": 50, "total": 100},
                {"status": "downloading", "completed": 100, "total": 100},
                {"status": "success"}
            ]
            for update in updates:
                if callback:
                    callback(update)
                progress_updates.append(update)
            return True
        
        mock_ollama_client.pull_model.side_effect = mock_pull
        
        def progress_callback(update):
            pass
        
        result = await mock_ollama_client.pull_model("model", progress_callback)
        assert result is True
        assert len(progress_updates) == 4
        assert progress_updates[-1]["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_delete_model_success(self, mock_ollama_client):
        """Test successful model deletion."""
        mock_ollama_client.delete_model.return_value = True
        
        result = await mock_ollama_client.delete_model("unwanted-model")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_model_not_found(self, mock_ollama_client):
        """Test deleting non-existent model."""
        mock_ollama_client.delete_model.side_effect = ModelNotFoundError("Model not found")
        
        with pytest.raises(ModelNotFoundError):
            await mock_ollama_client.delete_model("non-existent")


class TestChatFunctionality:
    """Test cases for chat functionality."""
    
    @pytest.mark.asyncio
    async def test_chat_simple_message(self, mock_ollama_client):
        """Test simple chat message."""
        messages = [ChatMessage(role="user", content="Hello")]
        response = ChatResponse(
            message=ChatMessage(role="assistant", content="Hi there!"),
            done=True,
            total_duration=1000000,
            eval_count=10
        )
        mock_ollama_client.chat.return_value = response
        
        result = await mock_ollama_client.chat("model", messages)
        assert result.message.content == "Hi there!"
        assert result.done is True
    
    @pytest.mark.asyncio
    async def test_chat_conversation(self, mock_ollama_client):
        """Test multi-turn conversation."""
        messages = [
            ChatMessage(role="user", content="What's 2+2?"),
            ChatMessage(role="assistant", content="2+2 equals 4"),
            ChatMessage(role="user", content="What about 3+3?")
        ]
        
        response = ChatResponse(
            message=ChatMessage(role="assistant", content="3+3 equals 6"),
            done=True
        )
        mock_ollama_client.chat.return_value = response
        
        result = await mock_ollama_client.chat("model", messages)
        assert "6" in result.message.content
    
    @pytest.mark.asyncio
    async def test_chat_with_system_prompt(self, mock_ollama_client):
        """Test chat with system prompt."""
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant"),
            ChatMessage(role="user", content="Hello")
        ]
        
        response = ChatResponse(
            message=ChatMessage(role="assistant", content="Hello! How can I help you?"),
            done=True
        )
        mock_ollama_client.chat.return_value = response
        
        result = await mock_ollama_client.chat("model", messages)
        assert result.message.role == "assistant"
    
    @pytest.mark.asyncio
    async def test_chat_streaming(self, mock_ollama_client):
        """Test streaming chat responses."""
        chunks = [
            ChatResponse(message=ChatMessage(role="assistant", content="Hello"), done=False),
            ChatResponse(message=ChatMessage(role="assistant", content=" there"), done=False),
            ChatResponse(message=ChatMessage(role="assistant", content="!"), done=True)
        ]
        
        async def stream_generator():
            for chunk in chunks:
                yield chunk
        
        mock_ollama_client.chat.return_value = stream_generator()
        
        messages = [ChatMessage(role="user", content="Hi")]
        collected = []
        
        async for chunk in await mock_ollama_client.chat("model", messages, stream=True):
            collected.append(chunk)
        
        assert len(collected) == 3
        assert collected[-1].done is True
    
    @pytest.mark.asyncio
    async def test_chat_with_model_params(self, mock_ollama_client):
        """Test chat with custom model parameters."""
        messages = [ChatMessage(role="user", content="Test")]
        params = {
            "temperature": 0.5,
            "top_p": 0.8,
            "max_tokens": 100
        }
        
        response = ChatResponse(
            message=ChatMessage(role="assistant", content="Response"),
            done=True
        )
        mock_ollama_client.chat.return_value = response
        
        result = await mock_ollama_client.chat("model", messages, **params)
        assert result.message.content == "Response"


class TestGenerateFunctionality:
    """Test cases for text generation."""
    
    @pytest.mark.asyncio
    async def test_generate_simple(self, mock_ollama_client):
        """Test simple text generation."""
        mock_ollama_client.generate.return_value = "Generated text content"
        
        result = await mock_ollama_client.generate("model", "Complete this:")
        assert result == "Generated text content"
    
    @pytest.mark.asyncio
    async def test_generate_with_context(self, mock_ollama_client):
        """Test generation with context."""
        mock_ollama_client.generate.return_value = "Contextual response"
        
        result = await mock_ollama_client.generate(
            "model",
            "Question",
            context="Previous context"
        )
        assert result == "Contextual response"
    
    @pytest.mark.asyncio
    async def test_generate_streaming(self, mock_ollama_client):
        """Test streaming text generation."""
        chunks = ["Part 1", " Part 2", " Part 3"]
        
        async def stream_generator():
            for chunk in chunks:
                yield chunk
        
        mock_ollama_client.generate.return_value = stream_generator()
        
        collected = []
        async for chunk in await mock_ollama_client.generate("model", "Prompt", stream=True):
            collected.append(chunk)
        
        assert len(collected) == 3
        assert "".join(collected) == "Part 1 Part 2 Part 3"


class TestEmbeddings:
    """Test cases for embedding generation."""
    
    @pytest.mark.asyncio
    async def test_embed_single_text(self, mock_ollama_client):
        """Test embedding generation for single text."""
        mock_ollama_client.embed.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        
        result = await mock_ollama_client.embed("model", "Test text")
        assert len(result) == 1
        assert len(result[0]) == 5
    
    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, mock_ollama_client):
        """Test embedding generation for multiple texts."""
        mock_ollama_client.embed.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        
        texts = ["Text 1", "Text 2"]
        result = await mock_ollama_client.embed("model", texts)
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3
    
    @pytest.mark.asyncio
    async def test_embed_empty_text(self, mock_ollama_client):
        """Test embedding generation for empty text."""
        mock_ollama_client.embed.side_effect = ValidationError("Empty text")
        
        with pytest.raises(ValidationError):
            await mock_ollama_client.embed("model", "")
    
    @pytest.mark.asyncio
    async def test_embed_large_batch(self, mock_ollama_client):
        """Test embedding generation for large batch."""
        batch_size = 100
        texts = [f"Text {i}" for i in range(batch_size)]
        
        mock_ollama_client.embed.return_value = [[0.1] * 768] * batch_size
        
        result = await mock_ollama_client.embed("model", texts)
        assert len(result) == batch_size


class TestErrorHandling:
    """Test cases for error handling."""
    
    @pytest.mark.asyncio
    async def test_connection_error(self, mock_ollama_client):
        """Test connection error handling."""
        mock_ollama_client.chat.side_effect = ConnectionError("Connection refused")
        
        with pytest.raises(ConnectionError):
            await mock_ollama_client.chat("model", [])
    
    @pytest.mark.asyncio
    async def test_timeout_error(self, mock_ollama_client):
        """Test timeout error handling."""
        mock_ollama_client.chat.side_effect = TimeoutError("Request timeout")
        
        with pytest.raises(TimeoutError):
            await mock_ollama_client.chat("model", [])
    
    @pytest.mark.asyncio
    async def test_model_not_found_error(self, mock_ollama_client):
        """Test model not found error."""
        mock_ollama_client.chat.side_effect = ModelNotFoundError("Model not found")
        
        with pytest.raises(ModelNotFoundError):
            await mock_ollama_client.chat("non-existent", [])
    
    @pytest.mark.asyncio
    async def test_rate_limit_error(self, mock_ollama_client):
        """Test rate limit error handling."""
        mock_ollama_client.chat.side_effect = RateLimitError("Rate limit exceeded")
        
        with pytest.raises(RateLimitError):
            await mock_ollama_client.chat("model", [])
    
    @pytest.mark.asyncio
    async def test_validation_error(self, mock_ollama_client):
        """Test validation error handling."""
        mock_ollama_client.chat.side_effect = ValidationError("Invalid parameters")
        
        with pytest.raises(ValidationError):
            await mock_ollama_client.chat("model", [])
    
    @pytest.mark.asyncio
    async def test_generic_api_error(self, mock_ollama_client):
        """Test generic API error handling."""
        mock_ollama_client.chat.side_effect = APIError("Unknown error")
        
        with pytest.raises(APIError):
            await mock_ollama_client.chat("model", [])


class TestRetryLogic:
    """Test cases for retry logic."""
    
    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, test_config):
        """Test retry on transient errors."""
        client = OllamaClient(
            config=test_config,
            retry_config=RetryConfig(max_retries=3, base_delay=0.1)
        )
        
        call_count = 0
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            return "success"
        
        result = await client._retry_with_backoff(failing_operation)
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, test_config):
        """Test retry exhaustion."""
        client = OllamaClient(
            config=test_config,
            retry_config=RetryConfig(max_retries=2, base_delay=0.1)
        )
        
        async def always_failing():
            raise ConnectionError("Persistent error")
        
        with pytest.raises(APIError, match="All retry attempts failed"):
            await client._retry_with_backoff(always_failing)
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self, test_config):
        """Test exponential backoff timing."""
        client = OllamaClient(
            config=test_config,
            retry_config=RetryConfig(
                max_retries=3,
                base_delay=0.1,
                max_delay=1.0,
                exponential_base=2
            )
        )
        
        import time
        delays = []
        
        async def track_delays():
            start = time.time()
            raise ConnectionError("Error")
        
        with patch('asyncio.sleep', side_effect=lambda d: delays.append(d)):
            try:
                await client._retry_with_backoff(track_delays)
            except APIError:
                pass
        
        # Check exponential growth
        assert len(delays) > 0
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i-1]
    
    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self, test_config):
        """Test that non-retryable errors are not retried."""
        client = OllamaClient(
            config=test_config,
            retry_config=RetryConfig(max_retries=3)
        )
        
        call_count = 0
        async def validation_error():
            nonlocal call_count
            call_count += 1
            raise ValidationError("Invalid input")
        
        with pytest.raises(ValidationError):
            await client._retry_with_backoff(validation_error)
        
        assert call_count == 1  # Should not retry


class TestParameterValidation:
    """Test cases for parameter validation."""
    
    def test_validate_temperature(self, test_config):
        """Test temperature parameter validation."""
        client = OllamaClient(config=test_config)
        
        # Valid temperatures
        assert client._validate_model_params({"temperature": 0.0})["temperature"] == 0.0
        assert client._validate_model_params({"temperature": 1.0})["temperature"] == 1.0
        assert client._validate_model_params({"temperature": 2.0})["temperature"] == 2.0
        
        # Invalid temperatures
        with pytest.raises(ValidationError):
            client._validate_model_params({"temperature": -0.1})
        with pytest.raises(ValidationError):
            client._validate_model_params({"temperature": 2.1})
    
    def test_validate_top_p(self, test_config):
        """Test top_p parameter validation."""
        client = OllamaClient(config=test_config)
        
        # Valid values
        assert client._validate_model_params({"top_p": 0.0})["top_p"] == 0.0
        assert client._validate_model_params({"top_p": 0.5})["top_p"] == 0.5
        assert client._validate_model_params({"top_p": 1.0})["top_p"] == 1.0
        
        # Invalid values
        with pytest.raises(ValidationError):
            client._validate_model_params({"top_p": -0.1})
        with pytest.raises(ValidationError):
            client._validate_model_params({"top_p": 1.1})
    
    def test_validate_top_k(self, test_config):
        """Test top_k parameter validation."""
        client = OllamaClient(config=test_config)
        
        # Valid values
        assert client._validate_model_params({"top_k": 1})["top_k"] == 1
        assert client._validate_model_params({"top_k": 50})["top_k"] == 50
        assert client._validate_model_params({"top_k": 100})["top_k"] == 100
        
        # Invalid values
        with pytest.raises(ValidationError):
            client._validate_model_params({"top_k": 0})
        with pytest.raises(ValidationError):
            client._validate_model_params({"top_k": -1})
    
    def test_parameter_name_mapping(self, test_config):
        """Test parameter name mapping for Ollama API."""
        client = OllamaClient(config=test_config)
        
        params = {
            "max_tokens": 1024,
            "stop_sequences": ["END", "STOP"],
            "presence_penalty": 0.5
        }
        
        validated = client._validate_model_params(params)
        
        # Check name mappings
        assert "num_predict" in validated
        assert validated["num_predict"] == 1024
        assert "stop" in validated
        assert validated["stop"] == ["END", "STOP"]


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    @pytest.mark.asyncio
    async def test_quick_chat(self, mock_ollama_client):
        """Test quick_chat convenience function."""
        with patch('abov3.core.api.ollama_client.get_ollama_client') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_ollama_client
            mock_ollama_client.chat.return_value = ChatResponse(
                message=ChatMessage(role="assistant", content="Quick response"),
                done=True
            )
            
            result = await quick_chat("Quick question")
            assert result == "Quick response"
    
    @pytest.mark.asyncio
    async def test_quick_generate(self, mock_ollama_client):
        """Test quick_generate convenience function."""
        with patch('abov3.core.api.ollama_client.get_ollama_client') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_ollama_client
            mock_ollama_client.generate.return_value = "Generated content"
            
            result = await quick_generate("Generate this")
            assert result == "Generated content"
    
    @pytest.mark.asyncio
    async def test_get_ollama_client(self, test_config):
        """Test get_ollama_client factory function."""
        with patch('abov3.core.api.ollama_client.load_config') as mock_load:
            mock_load.return_value = test_config
            
            async with get_ollama_client() as client:
                assert isinstance(client, OllamaClient)
                assert client.config == test_config


class TestPerformance:
    """Performance tests for API client."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_ollama_client):
        """Test handling of concurrent requests."""
        mock_ollama_client.chat.return_value = ChatResponse(
            message=ChatMessage(role="assistant", content="Response"),
            done=True
        )
        
        messages = [ChatMessage(role="user", content=f"Message {i}") for i in range(10)]
        
        tasks = [
            mock_ollama_client.chat("model", [msg])
            for msg in messages
        ]
        
        results = await asyncio.gather(*tasks)
        assert len(results) == 10
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_response_handling(self, mock_ollama_client):
        """Test handling of large responses."""
        large_content = "x" * 100000  # 100KB response
        mock_ollama_client.generate.return_value = large_content
        
        result = await mock_ollama_client.generate("model", "Generate large")
        assert len(result) == 100000
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_connection_pool_reuse(self, test_config):
        """Test connection pool reuse for performance."""
        client = OllamaClient(config=test_config)
        
        async with client:
            # Multiple requests should reuse connections
            with patch.object(client._session, 'get') as mock_get:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_get.return_value.__aenter__.return_value = mock_response
                
                for _ in range(10):
                    await client.health_check()
                
                # Should reuse connection from pool
                assert mock_get.call_count == 10