"""
Unit tests for the Ollama API client.

This module contains comprehensive tests for the OllamaClient functionality
including mocking external dependencies for reliable testing.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from abov3.core.api.ollama_client import (
    OllamaClient,
    ChatMessage,
    ChatResponse,
    ModelInfo,
    RetryConfig,
    get_ollama_client,
    quick_chat,
    quick_generate,
)
from abov3.core.api.exceptions import (
    APIError,
    ConnectionError,
    ModelNotFoundError,
    ValidationError,
    TimeoutError,
)
from abov3.core.config import Config, OllamaConfig, ModelConfig


class TestOllamaClient:
    """Test cases for OllamaClient."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            ollama=OllamaConfig(
                host="http://localhost:11434",
                timeout=30,
                max_retries=2
            ),
            model=ModelConfig(
                default_model="llama3.2:latest",
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024
            )
        )
    
    @pytest.fixture
    def retry_config(self):
        """Create test retry configuration."""
        return RetryConfig(max_retries=1, base_delay=0.1)
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Create mock Ollama AsyncClient."""
        mock_client = AsyncMock()
        return mock_client
    
    @pytest.fixture
    def client(self, config, retry_config):
        """Create OllamaClient instance for testing."""
        return OllamaClient(config=config, retry_config=retry_config)
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, client, config):
        """Test client initialization."""
        assert client.config == config
        assert client.retry_config.max_retries == 1
        assert client._client is None
        assert client._session is None
    
    @pytest.mark.asyncio
    async def test_context_manager(self, client):
        """Test async context manager functionality."""
        async with client as c:
            assert c is client
            # Should have initialized session and client
            assert client._session is not None
            assert client._client is not None
    
    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_health_check_success(self, mock_get, client):
        """Test successful health check."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with client:
            result = await client.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_health_check_failure(self, mock_get, client):
        """Test failed health check."""
        mock_get.side_effect = Exception("Connection failed")
        
        async with client:
            result = await client.health_check()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_list_models_success(self, client, mock_ollama_client):
        """Test successful model listing."""
        mock_response = {
            "models": [
                {
                    "name": "llama3.2:latest",
                    "size": 4000000000,
                    "modified_at": "2024-01-01T00:00:00Z",
                    "digest": "abc123",
                    "details": {"family": "llama"}
                }
            ]
        }
        mock_ollama_client.list.return_value = mock_response
        
        with patch.object(client, '_client', mock_ollama_client):
            models = await client.list_models()
            
            assert len(models) == 1
            assert models[0].name == "llama3.2:latest"
            assert models[0].size == 4000000000
            assert models[0].digest == "abc123"
    
    @pytest.mark.asyncio
    async def test_list_models_with_cache(self, client, mock_ollama_client):
        """Test model listing with caching."""
        mock_response = {
            "models": [{"name": "test-model", "size": 1000, "modified_at": "", "digest": ""}]
        }
        mock_ollama_client.list.return_value = mock_response
        
        with patch.object(client, '_client', mock_ollama_client):
            # First call should hit the API
            models1 = await client.list_models()
            assert len(models1) == 1
            
            # Second call should use cache
            models2 = await client.list_models()
            assert len(models2) == 1
            
            # Should only have called the API once
            assert mock_ollama_client.list.call_count == 1
    
    @pytest.mark.asyncio
    async def test_model_exists(self, client, mock_ollama_client):
        """Test model existence checking."""
        mock_response = {
            "models": [{"name": "existing-model", "size": 1000, "modified_at": "", "digest": ""}]
        }
        mock_ollama_client.list.return_value = mock_response
        
        with patch.object(client, '_client', mock_ollama_client):
            exists = await client.model_exists("existing-model")
            assert exists is True
            
            not_exists = await client.model_exists("non-existent-model")
            assert not_exists is False
    
    @pytest.mark.asyncio
    async def test_pull_model_success(self, client, mock_ollama_client):
        """Test successful model pulling."""
        mock_chunks = [
            {"status": "downloading", "completed": 50, "total": 100},
            {"status": "downloading", "completed": 100, "total": 100},
            {"status": "success"}
        ]
        
        async def mock_pull(*args, **kwargs):
            for chunk in mock_chunks:
                yield chunk
        
        mock_ollama_client.pull.return_value = mock_pull()
        mock_ollama_client.list.return_value = {"models": []}  # For cache refresh
        
        progress_calls = []
        def progress_callback(chunk):
            progress_calls.append(chunk)
        
        with patch.object(client, '_client', mock_ollama_client):
            result = await client.pull_model("new-model", progress_callback)
            
            assert result is True
            assert len(progress_calls) == 3
            assert progress_calls[0]["status"] == "downloading"
    
    @pytest.mark.asyncio
    async def test_delete_model_success(self, client, mock_ollama_client):
        """Test successful model deletion."""
        mock_ollama_client.delete.return_value = None
        
        # Pre-populate cache
        client._available_models["test-model"] = ModelInfo(
            name="test-model", size=1000, modified_at="", digest=""
        )
        
        with patch.object(client, '_client', mock_ollama_client):
            result = await client.delete_model("test-model")
            
            assert result is True
            assert "test-model" not in client._available_models
            mock_ollama_client.delete.assert_called_once_with("test-model")
    
    @pytest.mark.asyncio
    async def test_chat_success(self, client, mock_ollama_client):
        """Test successful chat completion."""
        mock_response = {
            "message": {"role": "assistant", "content": "Hello! How can I help you?"},
            "done": True,
            "total_duration": 1000000,
            "eval_count": 10
        }
        mock_ollama_client.chat.return_value = mock_response
        mock_ollama_client.list.return_value = {"models": [{"name": "test-model", "size": 1000, "modified_at": "", "digest": ""}]}
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch.object(client, '_client', mock_ollama_client):
            response = await client.chat("test-model", messages)
            
            assert isinstance(response, ChatResponse)
            assert response.message.role == "assistant"
            assert response.message.content == "Hello! How can I help you?"
            assert response.done is True
    
    @pytest.mark.asyncio
    async def test_chat_model_not_found(self, client, mock_ollama_client):
        """Test chat with non-existent model."""
        mock_ollama_client.list.return_value = {"models": []}
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch.object(client, '_client', mock_ollama_client):
            with pytest.raises(ModelNotFoundError):
                await client.chat("non-existent-model", messages)
    
    @pytest.mark.asyncio
    async def test_chat_streaming(self, client, mock_ollama_client):
        """Test streaming chat completion."""
        mock_chunks = [
            {"message": {"role": "assistant", "content": "Hello"}, "done": False},
            {"message": {"role": "assistant", "content": " there!"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True}
        ]
        
        async def mock_chat(*args, **kwargs):
            for chunk in mock_chunks:
                yield chunk
        
        mock_ollama_client.chat.return_value = mock_chat()
        mock_ollama_client.list.return_value = {"models": [{"name": "test-model", "size": 1000, "modified_at": "", "digest": ""}]}
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch.object(client, '_client', mock_ollama_client):
            chunks = []
            async for chunk in await client.chat("test-model", messages, stream=True):
                chunks.append(chunk)
            
            assert len(chunks) == 3
            assert chunks[0].message.content == "Hello"
            assert chunks[1].message.content == " there!"
            assert chunks[2].done is True
    
    @pytest.mark.asyncio
    async def test_generate_success(self, client, mock_ollama_client):
        """Test successful text generation."""
        mock_response = {"response": "This is a generated response."}
        mock_ollama_client.generate.return_value = mock_response
        mock_ollama_client.list.return_value = {"models": [{"name": "test-model", "size": 1000, "modified_at": "", "digest": ""}]}
        
        with patch.object(client, '_client', mock_ollama_client):
            response = await client.generate("test-model", "Complete this sentence:")
            
            assert response == "This is a generated response."
    
    @pytest.mark.asyncio
    async def test_embed_success(self, client, mock_ollama_client):
        """Test successful embedding generation."""
        mock_response = {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
        mock_ollama_client.embeddings.return_value = mock_response
        mock_ollama_client.list.return_value = {"models": [{"name": "test-model", "size": 1000, "modified_at": "", "digest": ""}]}
        
        with patch.object(client, '_client', mock_ollama_client):
            embeddings = await client.embed("test-model", "Test text")
            
            assert len(embeddings) == 1
            assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, client, mock_ollama_client):
        """Test embedding generation for multiple texts."""
        mock_response = {"embedding": [0.1, 0.2, 0.3]}
        mock_ollama_client.embeddings.return_value = mock_response
        mock_ollama_client.list.return_value = {"models": [{"name": "test-model", "size": 1000, "modified_at": "", "digest": ""}]}
        
        texts = ["Text 1", "Text 2"]
        
        with patch.object(client, '_client', mock_ollama_client):
            embeddings = await client.embed("test-model", texts)
            
            assert len(embeddings) == 2
            assert mock_ollama_client.embeddings.call_count == 2
    
    def test_validate_model_params_valid(self, client):
        """Test parameter validation with valid parameters."""
        params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "seed": 42,
            "max_tokens": 1024
        }
        
        validated = client._validate_model_params(params)
        
        assert validated["temperature"] == 0.7
        assert validated["top_p"] == 0.9
        assert validated["top_k"] == 40
        assert validated["repeat_penalty"] == 1.1
        assert validated["seed"] == 42
        assert validated["num_predict"] == 1024  # max_tokens -> num_predict
    
    def test_validate_model_params_invalid_temperature(self, client):
        """Test parameter validation with invalid temperature."""
        params = {"temperature": 3.0}  # Too high
        
        with pytest.raises(ValidationError, match="Temperature must be between"):
            client._validate_model_params(params)
    
    def test_validate_model_params_invalid_top_p(self, client):
        """Test parameter validation with invalid top_p."""
        params = {"top_p": 1.5}  # Too high
        
        with pytest.raises(ValidationError, match="Top-p must be between"):
            client._validate_model_params(params)
    
    def test_validate_model_params_invalid_top_k(self, client):
        """Test parameter validation with invalid top_k."""
        params = {"top_k": 0}  # Too low
        
        with pytest.raises(ValidationError, match="Top-k must be a positive integer"):
            client._validate_model_params(params)
    
    def test_handle_ollama_error(self, client):
        """Test error handling and conversion."""
        # Connection error
        conn_error = Exception("connection refused")
        result = client._handle_ollama_error(conn_error)
        assert isinstance(result, ConnectionError)
        
        # Model not found error
        model_error = Exception("model not found")
        result = client._handle_ollama_error(model_error)
        assert isinstance(result, ModelNotFoundError)
        
        # Generic error
        generic_error = Exception("unknown error")
        result = client._handle_ollama_error(generic_error)
        assert isinstance(result, APIError)
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self, client):
        """Test retry logic with eventual success."""
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await client._retry_with_backoff(failing_func)
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_exhausted(self, client):
        """Test retry logic with all attempts failing."""
        async def always_failing_func():
            raise ConnectionError("Persistent failure")
        
        with pytest.raises(APIError, match="All retry attempts failed"):
            await client._retry_with_backoff(always_failing_func)
    
    def test_get_model_params(self, client):
        """Test getting model parameters from config."""
        params = client.get_model_params()
        
        assert "temperature" in params
        assert "top_p" in params
        assert "top_k" in params
        assert params["temperature"] == 0.7
    
    def test_update_model_params(self, client):
        """Test updating model parameters."""
        client.update_model_params(temperature=0.9, max_tokens=2048)
        
        assert client.config.model.temperature == 0.9
        assert client.config.model.max_tokens == 2048


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    @patch("abov3.core.api.ollama_client.get_ollama_client")
    async def test_quick_chat(self, mock_get_client):
        """Test quick_chat convenience function."""
        mock_client = AsyncMock()
        mock_response = ChatResponse(
            message=ChatMessage(role="assistant", content="Hello there!"),
            done=True
        )
        mock_client.chat.return_value = mock_response
        mock_get_client.return_value.__aenter__.return_value = mock_client
        
        result = await quick_chat("Hello")
        
        assert result == "Hello there!"
        mock_client.chat.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("abov3.core.api.ollama_client.get_ollama_client")
    async def test_quick_generate(self, mock_get_client):
        """Test quick_generate convenience function."""
        mock_client = AsyncMock()
        mock_client.generate.return_value = "Generated text"
        mock_get_client.return_value.__aenter__.return_value = mock_client
        
        result = await quick_generate("Complete this:")
        
        assert result == "Generated text"
        mock_client.generate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])