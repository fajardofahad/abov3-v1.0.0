"""
Pytest configuration and shared fixtures for ABOV3 test suite.

This module provides common fixtures, test data factories, and utilities
for all test modules in the suite.
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from faker import Faker

from abov3.core.api.ollama_client import ChatMessage, ChatResponse, ModelInfo, OllamaClient
from abov3.core.config import (
    Config,
    HistoryConfig,
    LoggingConfig,
    ModelConfig,
    OllamaConfig,
    PluginConfig,
    UIConfig,
)
from abov3.core.context.manager import ContextManager
from abov3.core.context.session import Session
from abov3.models.registry import ModelRegistry
from abov3.plugins.base.manager import PluginManager
from abov3.utils.security import SecurityValidator

# Initialize Faker for test data generation
fake = Faker()


# ========================
# Configuration Fixtures
# ========================

@pytest.fixture
def test_config() -> Config:
    """Create a test configuration with all components."""
    return Config(
        ollama=OllamaConfig(
            host="http://localhost:11434",
            timeout=30,
            verify_ssl=False,
            max_retries=2
        ),
        model=ModelConfig(
            default_model="test-model:latest",
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=1024,
            context_length=4096
        ),
        ui=UIConfig(
            theme="dark",
            syntax_highlighting=True,
            auto_complete=True
        ),
        history=HistoryConfig(
            max_conversations=50,
            auto_save=True
        ),
        plugins=PluginConfig(
            enabled=["test_plugin"],
            auto_load=False
        ),
        logging=LoggingConfig(
            level="DEBUG",
            file_path=None
        )
    )


@pytest.fixture
def ollama_config() -> OllamaConfig:
    """Create a test Ollama configuration."""
    return OllamaConfig(
        host="http://test-ollama:11434",
        timeout=10,
        verify_ssl=False,
        max_retries=1
    )


@pytest.fixture
def model_config() -> ModelConfig:
    """Create a test model configuration."""
    return ModelConfig(
        default_model="llama3.2:latest",
        temperature=0.5,
        max_tokens=512
    )


# ========================
# Mock Fixtures
# ========================

@pytest.fixture
def mock_ollama_client() -> AsyncMock:
    """Create a mock Ollama client with common responses."""
    client = AsyncMock(spec=OllamaClient)
    
    # Setup default responses
    client.health_check.return_value = True
    client.list_models.return_value = [
        ModelInfo(
            name="test-model:latest",
            size=1000000,
            digest="abc123",
            modified_at=datetime.now().isoformat()
        )
    ]
    client.model_exists.return_value = True
    
    # Chat response
    client.chat.return_value = ChatResponse(
        message=ChatMessage(role="assistant", content="Test response"),
        done=True,
        total_duration=1000000,
        eval_count=10
    )
    
    # Generate response
    client.generate.return_value = "Generated text"
    
    # Embed response
    client.embed.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    
    return client


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp session."""
    session = AsyncMock()
    response = AsyncMock()
    response.status = 200
    response.json.return_value = {"status": "ok"}
    response.text.return_value = "OK"
    
    session.get.return_value.__aenter__.return_value = response
    session.post.return_value.__aenter__.return_value = response
    session.put.return_value.__aenter__.return_value = response
    session.delete.return_value.__aenter__.return_value = response
    
    return session


@pytest.fixture
def mock_context_manager() -> MagicMock:
    """Create a mock context manager."""
    manager = MagicMock(spec=ContextManager)
    manager.get_context.return_value = {
        "conversation_id": "test-conv-123",
        "messages": [],
        "metadata": {}
    }
    manager.add_message.return_value = None
    manager.clear_context.return_value = None
    return manager


@pytest.fixture
def mock_model_registry() -> MagicMock:
    """Create a mock model registry."""
    registry = MagicMock(spec=ModelRegistry)
    registry.get_model.return_value = {
        "name": "test-model",
        "family": "llama",
        "parameters": {}
    }
    registry.list_models.return_value = ["test-model", "other-model"]
    registry.register_model.return_value = True
    return registry


@pytest.fixture
def mock_plugin_manager() -> MagicMock:
    """Create a mock plugin manager."""
    manager = MagicMock(spec=PluginManager)
    manager.load_plugin.return_value = True
    manager.unload_plugin.return_value = True
    manager.get_plugin.return_value = MagicMock()
    manager.list_plugins.return_value = ["test_plugin"]
    return manager


@pytest.fixture
def mock_security_validator() -> MagicMock:
    """Create a mock security validator."""
    validator = MagicMock(spec=SecurityValidator)
    validator.validate_input.return_value = True
    validator.sanitize_output.return_value = "sanitized output"
    validator.check_permissions.return_value = True
    return validator


# ========================
# File System Fixtures
# ========================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_file(temp_dir: Path) -> Path:
    """Create a temporary config file."""
    config_file = temp_dir / "config.toml"
    config_data = {
        "ollama": {
            "host": "http://localhost:11434",
            "timeout": 30
        },
        "model": {
            "default_model": "test-model",
            "temperature": 0.7
        }
    }
    
    import toml
    with open(config_file, "w") as f:
        toml.dump(config_data, f)
    
    return config_file


@pytest.fixture
def temp_history_file(temp_dir: Path) -> Path:
    """Create a temporary history file."""
    history_file = temp_dir / "history.json"
    history_data = {
        "conversations": [
            {
                "id": "conv-1",
                "timestamp": datetime.now().isoformat(),
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        ]
    }
    
    with open(history_file, "w") as f:
        json.dump(history_data, f)
    
    return history_file


# ========================
# Test Data Factories
# ========================

class TestDataFactory:
    """Factory for generating test data."""
    
    @staticmethod
    def create_chat_message(
        role: str = "user",
        content: Optional[str] = None
    ) -> ChatMessage:
        """Create a test chat message."""
        if content is None:
            content = fake.text(max_nb_chars=200)
        return ChatMessage(role=role, content=content)
    
    @staticmethod
    def create_chat_response(
        content: Optional[str] = None,
        done: bool = True
    ) -> ChatResponse:
        """Create a test chat response."""
        if content is None:
            content = fake.text(max_nb_chars=500)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            done=done,
            total_duration=fake.random_int(min=100000, max=10000000),
            eval_count=fake.random_int(min=10, max=1000)
        )
    
    @staticmethod
    def create_model_info(
        name: Optional[str] = None,
        size: Optional[int] = None
    ) -> ModelInfo:
        """Create test model info."""
        if name is None:
            name = f"{fake.word()}:latest"
        if size is None:
            size = fake.random_int(min=1000000, max=10000000000)
        
        return ModelInfo(
            name=name,
            size=size,
            digest=fake.sha256(),
            modified_at=datetime.now().isoformat()
        )
    
    @staticmethod
    def create_session_data() -> Dict[str, Any]:
        """Create test session data."""
        return {
            "session_id": fake.uuid4(),
            "user_id": fake.uuid4(),
            "started_at": datetime.now().isoformat(),
            "messages": [
                TestDataFactory.create_chat_message("user").dict(),
                TestDataFactory.create_chat_message("assistant").dict()
            ],
            "metadata": {
                "model": "test-model",
                "temperature": 0.7
            }
        }
    
    @staticmethod
    def create_plugin_metadata() -> Dict[str, Any]:
        """Create test plugin metadata."""
        return {
            "name": fake.word(),
            "version": "1.0.0",
            "description": fake.text(max_nb_chars=100),
            "author": fake.name(),
            "enabled": True,
            "config": {
                "api_key": fake.uuid4(),
                "endpoint": fake.url()
            }
        }


@pytest.fixture
def test_factory() -> TestDataFactory:
    """Provide test data factory."""
    return TestDataFactory()


# ========================
# Async Test Utilities
# ========================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def async_client(test_config: Config) -> AsyncMock:
    """Create an async client for testing."""
    client = AsyncMock()
    client.config = test_config
    return client


# ========================
# Performance Testing
# ========================

@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.times = []
        
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.end = time.perf_counter()
            self.duration = self.end - self.start
            self.times.append(self.duration)
        
        @property
        def average(self):
            return sum(self.times) / len(self.times) if self.times else 0
        
        @property
        def total(self):
            return sum(self.times)
    
    return Timer()


# ========================
# Security Testing
# ========================

@pytest.fixture
def malicious_inputs() -> List[str]:
    """Provide common malicious input patterns for security testing."""
    return [
        # SQL Injection
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        
        # Command Injection
        "; ls -la",
        "| cat /etc/passwd",
        "$(rm -rf /)",
        
        # Path Traversal
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        
        # XSS
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        
        # LDAP Injection
        "*)(uid=*",
        "*)(|(uid=*))",
        
        # XXE
        "<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>",
        
        # Template Injection
        "{{7*7}}",
        "${7*7}",
        
        # Format String
        "%x %x %x %x",
        "%s %s %s %s",
        
        # Buffer Overflow Attempts
        "A" * 10000,
        "\x00" * 1000,
        
        # Unicode/Encoding Issues
        "\u202e\u0041\u0042\u0043",  # Right-to-left override
        "\ufeff",  # Zero-width no-break space
    ]


@pytest.fixture
def edge_case_inputs() -> List[Any]:
    """Provide edge case inputs for robustness testing."""
    return [
        None,
        "",
        " ",
        "\n",
        "\t",
        0,
        -1,
        float('inf'),
        float('-inf'),
        float('nan'),
        [],
        {},
        {"": ""},
        [None],
        "null",
        "undefined",
        "NaN",
        "Infinity",
        True,
        False,
        b"bytes",
        "🚀" * 1000,  # Emoji spam
        "a" * 1000000,  # Very long string
    ]


# ========================
# Mock Network Responses
# ========================

@pytest.fixture
def mock_network_responses():
    """Provide mock network responses for different scenarios."""
    return {
        "success": {
            "status": 200,
            "json": {"status": "success", "data": "test"},
            "text": "OK"
        },
        "error_400": {
            "status": 400,
            "json": {"error": "Bad Request"},
            "text": "Bad Request"
        },
        "error_401": {
            "status": 401,
            "json": {"error": "Unauthorized"},
            "text": "Unauthorized"
        },
        "error_403": {
            "status": 403,
            "json": {"error": "Forbidden"},
            "text": "Forbidden"
        },
        "error_404": {
            "status": 404,
            "json": {"error": "Not Found"},
            "text": "Not Found"
        },
        "error_500": {
            "status": 500,
            "json": {"error": "Internal Server Error"},
            "text": "Internal Server Error"
        },
        "timeout": {
            "exception": asyncio.TimeoutError("Request timeout")
        },
        "connection_error": {
            "exception": ConnectionError("Connection refused")
        }
    }


# ========================
# Pytest Configuration
# ========================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "asyncio: marks tests as async tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test file names
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        else:
            item.add_marker(pytest.mark.unit)
        
        # Add async marker for async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)


# ========================
# Test Result Capture
# ========================

@pytest.fixture
def capture_logs():
    """Capture logs during test execution."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger()
    logger.addHandler(handler)
    
    yield log_capture
    
    logger.removeHandler(handler)


# ========================
# Cleanup Fixtures
# ========================

@pytest.fixture(autouse=True)
def cleanup_env_vars():
    """Clean up environment variables after each test."""
    import os
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Add any singleton reset logic here
    yield