# Developer Guide

Comprehensive guide for contributing to and extending ABOV3 4 Ollama.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Architecture Overview](#architecture-overview)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Plugin Development](#plugin-development)
- [Contributing Process](#contributing-process)
- [API Design Guidelines](#api-design-guidelines)
- [Performance Optimization](#performance-optimization)
- [Security Considerations](#security-considerations)
- [Documentation Standards](#documentation-standards)
- [Release Process](#release-process)

## Getting Started

### Prerequisites

- Python 3.8+ (3.10+ recommended)
- Git
- Ollama installed and running
- Basic understanding of async/await patterns
- Familiarity with type hints and modern Python practices

### Quick Development Setup

```bash
# Clone the repository
git clone https://github.com/abov3/abov3-ollama.git
cd abov3-ollama

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest

# Start development server
python -m abov3.cli chat --debug
```

## Development Environment

### Recommended IDE Setup

#### Visual Studio Code

Install recommended extensions:
```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.isort",
        "ms-python.mypy-type-checker",
        "ms-python.pytest"
    ]
}
```

Workspace settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm

1. Open project in PyCharm
2. Configure Python interpreter: File → Settings → Project → Python Interpreter
3. Enable Black formatter: Settings → Tools → External Tools
4. Configure pytest: Settings → Tools → Python Integrated Tools

### Environment Variables

```bash
# Development environment
export ABOV3_DEBUG=true
export ABOV3_CONFIG_PATH="./dev_config.toml"
export ABOV3_DATA_DIR="./dev_data"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Testing environment
export ABOV3_TEST_MODE=true
export ABOV3_OLLAMA_HOST="http://localhost:11434"
```

### Development Configuration

Create `dev_config.toml`:
```toml
[model]
default_model = "llama3.2:latest"
temperature = 0.7

[ollama]
host = "http://localhost:11434"
timeout = 60

[ui]
theme = "dark"
debug_mode = true

[security]
sandbox_mode = true
enable_content_filter = false

[performance]
async_processing = true
cache_enabled = false  # Disable for development
```

## Architecture Overview

### Project Structure

```
abov3/
├── core/                   # Core application logic
│   ├── app.py             # Main application class
│   ├── config.py          # Configuration management
│   ├── api/               # API clients
│   │   ├── ollama_client.py
│   │   └── exceptions.py
│   └── context/           # Context management
│       ├── manager.py
│       ├── memory.py
│       └── session.py
├── models/                # Model management
│   ├── manager.py         # Model lifecycle
│   ├── info.py           # Model metadata
│   ├── evaluation.py     # Model evaluation
│   └── fine_tuning.py    # Fine-tuning support
├── ui/                    # User interface
│   ├── console/          # Terminal UI
│   │   ├── repl.py       # REPL implementation
│   │   ├── formatters.py # Output formatting
│   │   └── completers.py # Auto-completion
│   └── components/       # Reusable UI components
├── plugins/              # Plugin system
│   ├── base/            # Plugin framework
│   │   ├── plugin.py    # Base plugin class
│   │   ├── manager.py   # Plugin manager
│   │   └── registry.py  # Plugin registry
│   └── builtin/         # Built-in plugins
├── utils/               # Utility modules
│   ├── security.py     # Security utilities
│   ├── file_ops.py     # File operations
│   ├── git_integration.py
│   ├── export.py       # Export functionality
│   └── monitoring.py   # Performance monitoring
└── tests/              # Test suite
    ├── unit/           # Unit tests
    ├── integration/    # Integration tests
    └── fixtures/       # Test fixtures
```

### Key Design Patterns

#### Async/Await Architecture

ABOV3 is built on asynchronous programming for responsive UI and efficient resource usage:

```python
class ABOV3App:
    async def run(self) -> None:
        """Main application loop using async/await."""
        async with self.context_manager:
            await self.initialize_subsystems()
            await self.start_repl()
    
    async def send_message(self, message: str) -> AsyncIterator[str]:
        """Async message processing with streaming."""
        async for chunk in self.ollama_client.chat(message):
            yield chunk
```

#### Plugin Architecture

Extensible plugin system with dependency injection:

```python
class PluginManager:
    def __init__(self, config: Config):
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, List[Callable]] = defaultdict(list)
    
    async def load_plugin(self, plugin_class: Type[Plugin]) -> None:
        """Load and initialize a plugin."""
        plugin = plugin_class()
        await plugin.initialize()
        self.plugins[plugin.name] = plugin
```

#### Configuration Management

Hierarchical configuration with validation:

```python
class Config(BaseModel):
    """Pydantic-based configuration with validation."""
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v
```

### Data Flow

```
User Input → REPL → Context Manager → Ollama Client → Model
     ↑                                                    ↓
UI Formatter ← Plugin System ← Response Processor ← Streaming Response
```

## Code Standards

### Python Style Guide

We follow PEP 8 with these specific guidelines:

#### Formatting

```python
# Use Black formatter with 88-character line length
# Configured in pyproject.toml

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
```

#### Import Organization

```python
# Standard library imports
import asyncio
import logging
from typing import Optional, List, Dict, Any

# Third-party imports
import aiohttp
from rich.console import Console

# Local imports
from .config import Config
from ..utils.security import SecurityManager
```

#### Type Hints

Always use type hints for public APIs:

```python
async def send_message(
    self,
    message: str,
    session_id: Optional[str] = None,
    stream: bool = True
) -> AsyncIterator[str]:
    """Send message with proper type annotations."""
    ...
```

#### Error Handling

Use specific exception types and proper error context:

```python
try:
    result = await self.ollama_client.chat(message)
except ConnectionError as e:
    logger.error(f"Failed to connect to Ollama: {e}")
    raise APIError(f"Connection failed: {e}") from e
except Exception as e:
    logger.exception("Unexpected error in chat")
    raise
```

### Documentation Standards

#### Docstring Format

Use Google-style docstrings:

```python
async def process_message(
    self,
    message: str,
    context: Optional[Dict[str, Any]] = None
) -> ChatResponse:
    """
    Process a user message and generate AI response.
    
    Args:
        message: The user's input message.
        context: Optional context dictionary for the conversation.
        
    Returns:
        ChatResponse object containing the AI's response.
        
    Raises:
        ValidationError: If the message is invalid.
        APIError: If the API request fails.
        
    Example:
        >>> response = await app.process_message("Hello, world!")
        >>> print(response.message.content)
        "Hello! How can I help you today?"
    """
```

#### Code Comments

Use comments sparingly and focus on "why" not "what":

```python
# Use exponential backoff to handle rate limiting gracefully
await asyncio.sleep(self.retry_config.base_delay * (2 ** attempt))

# Cache model info to avoid repeated API calls during session
if model_name not in self._model_cache:
    self._model_cache[model_name] = await self._fetch_model_info(model_name)
```

### Naming Conventions

- **Classes**: PascalCase (`ModelManager`, `ChatResponse`)
- **Functions/Methods**: snake_case (`send_message`, `get_config`)
- **Variables**: snake_case (`session_id`, `max_retries`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_MODEL`, `MAX_CONTEXT_LENGTH`)
- **Private members**: Leading underscore (`_cache`, `_validate_config`)

## Testing Guidelines

### Test Structure

Organize tests by type and module:

```
tests/
├── unit/                  # Fast, isolated unit tests
│   ├── test_config.py
│   ├── test_ollama_client.py
│   └── test_models/
├── integration/           # Integration tests with external services
│   ├── test_full_workflow.py
│   └── test_ollama_integration.py
├── fixtures/             # Shared test data
│   ├── sample_configs.py
│   └── mock_responses.py
└── conftest.py          # Pytest configuration
```

### Writing Tests

#### Unit Tests

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from abov3.core.ollama_client import OllamaClient
from abov3.core.config import Config

class TestOllamaClient:
    @pytest.fixture
    def mock_config(self):
        return Config(
            ollama=OllamaConfig(host="http://test:11434")
        )
    
    @pytest.fixture
    def client(self, mock_config):
        return OllamaClient(mock_config)
    
    @pytest.mark.asyncio
    async def test_chat_success(self, client):
        # Arrange
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = AsyncMock()
            mock_request.return_value.__aiter__.return_value = [
                {"message": {"content": "Hello back!"}, "done": True}
            ]
            
            # Act
            responses = []
            async for response in client.chat(messages, "llama3.2"):
                responses.append(response)
            
            # Assert
            assert len(responses) == 1
            assert responses[0].message.content == "Hello back!"
```

#### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_chat_workflow():
    """Test complete chat workflow with real Ollama instance."""
    config = Config()
    app = ABOV3App(config)
    
    # Requires running Ollama instance
    if not await app.ollama_client.health_check():
        pytest.skip("Ollama not available")
    
    session_id = await app.start_session()
    
    responses = []
    async for chunk in app.send_message("Hello", session_id):
        responses.append(chunk)
    
    assert len(responses) > 0
    assert any("hello" in chunk.lower() for chunk in responses)
```

### Test Fixtures

```python
# conftest.py
import pytest
from abov3.core.config import Config
from abov3.core.app import ABOV3App

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return Config(
        model=ModelConfig(default_model="test-model"),
        ollama=OllamaConfig(host="http://test:11434"),
        security=SecurityConfig(sandbox_mode=True)
    )

@pytest.fixture
async def app_instance(test_config):
    """Provide app instance with test config."""
    app = ABOV3App(test_config)
    yield app
    await app.cleanup()

@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "message": {"role": "assistant", "content": "Test response"},
        "done": True,
        "total_duration": 1000000,
        "eval_count": 10
    }
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test types
pytest tests/unit/
pytest tests/integration/ -m integration

# Run with coverage
pytest --cov=abov3 --cov-report=html

# Run in parallel
pytest -n auto

# Run specific test
pytest tests/unit/test_config.py::TestConfig::test_load_from_file

# Run with verbose output
pytest -v -s
```

### Test Markers

```python
# Mark integration tests
@pytest.mark.integration
def test_ollama_connection():
    pass

# Mark slow tests
@pytest.mark.slow
def test_model_download():
    pass

# Mark tests requiring GPU
@pytest.mark.gpu
def test_gpu_acceleration():
    pass

# Skip tests conditionally
@pytest.mark.skipif(not has_ollama(), reason="Ollama not available")
def test_model_list():
    pass
```

## Plugin Development

### Creating a Plugin

#### Basic Plugin Structure

```python
# my_plugin.py
from abov3.plugins.base import Plugin
from typing import Dict, Any

class MyPlugin(Plugin):
    """Example plugin demonstrating core functionality."""
    
    name = "my_plugin"
    version = "1.0.0"
    description = "Example plugin for demonstration"
    author = "Developer Name"
    
    async def initialize(self) -> None:
        """Initialize the plugin."""
        # Register commands
        self.register_command("my-command", self.handle_my_command)
        self.register_command("status", self.show_status)
        
        # Register event hooks
        self.register_hook("message_sent", self.on_message_sent)
        
        # Load plugin configuration
        self.config = self.get_config("my_plugin", {
            "enabled": True,
            "api_key": None
        })
        
        self.log_info("Plugin initialized successfully")
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self.log_info("Plugin cleaned up")
    
    async def handle_my_command(self, args: str) -> str:
        """Handle the my-command."""
        if not self.config.get("enabled"):
            return "Plugin is disabled"
        
        # Process command
        result = f"Processed: {args}"
        
        # Log activity
        self.log_info(f"Command executed: {args}")
        
        return result
    
    async def show_status(self, args: str) -> str:
        """Show plugin status."""
        status = {
            "name": self.name,
            "version": self.version,
            "enabled": self.config.get("enabled", False),
            "last_used": getattr(self, 'last_used', 'Never')
        }
        
        return f"Status: {status}"
    
    async def on_message_sent(self, message: str) -> None:
        """Hook called when a message is sent."""
        self.last_used = datetime.now().isoformat()
        
        # Optional: Modify or log the message
        if "error" in message.lower():
            self.log_info("Error message detected")
```

#### Advanced Plugin Features

```python
class AdvancedPlugin(Plugin):
    """Advanced plugin with complex functionality."""
    
    name = "advanced_plugin"
    version = "2.0.0"
    dependencies = ["base_plugin"]  # Plugin dependencies
    
    async def initialize(self) -> None:
        # Initialize with async setup
        self.api_client = await self.create_api_client()
        
        # Register multiple command types
        self.register_command("async-task", self.async_task)
        self.register_command("stream-data", self.stream_data)
        
        # Register configuration validator
        self.register_config_validator(self.validate_config)
    
    async def create_api_client(self):
        """Create API client with authentication."""
        api_key = self.get_config("api_key")
        if not api_key:
            raise PluginError("API key required")
        
        return APIClient(api_key)
    
    async def async_task(self, args: str) -> str:
        """Execute an asynchronous task."""
        try:
            result = await self.api_client.process(args)
            return f"Task completed: {result}"
        except Exception as e:
            self.log_error(f"Task failed: {e}")
            return f"Task failed: {str(e)}"
    
    async def stream_data(self, args: str):
        """Stream data response."""
        async for chunk in self.api_client.stream(args):
            yield f"Data: {chunk}"
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        required_keys = ["api_key", "endpoint"]
        return all(key in config for key in required_keys)
```

### Plugin Configuration

```python
# Plugin-specific configuration in main config
[plugins.my_plugin]
enabled = true
api_key = "your-api-key"
endpoint = "https://api.example.com"
timeout = 30

[plugins.advanced_plugin]
enabled = false
debug_mode = true
```

### Plugin Installation

```bash
# Install from file
abov3 plugins install path/to/my_plugin.py

# Install from URL
abov3 plugins install https://github.com/user/plugin.git

# Install from package
pip install abov3-my-plugin
```

### Plugin Testing

```python
# test_my_plugin.py
import pytest
from unittest.mock import AsyncMock
from abov3.plugins.base import PluginManager
from my_plugin import MyPlugin

class TestMyPlugin:
    @pytest.fixture
    async def plugin(self):
        plugin = MyPlugin()
        await plugin.initialize()
        return plugin
    
    @pytest.mark.asyncio
    async def test_command_execution(self, plugin):
        result = await plugin.handle_my_command("test input")
        assert "Processed: test input" in result
    
    @pytest.mark.asyncio
    async def test_disabled_plugin(self, plugin):
        plugin.config["enabled"] = False
        result = await plugin.handle_my_command("test")
        assert result == "Plugin is disabled"
```

## Contributing Process

### Git Workflow

We use the GitHub Flow model:

1. **Fork** the repository
2. **Create** a feature branch from `main`
3. **Commit** changes with descriptive messages
4. **Push** branch to your fork
5. **Create** a Pull Request
6. **Review** and address feedback
7. **Merge** after approval

### Branch Naming

```bash
# Feature branches
feature/add-new-plugin-system
feature/improve-error-handling

# Bug fix branches
fix/memory-leak-in-context-manager
fix/config-validation-error

# Documentation branches
docs/update-api-reference
docs/add-plugin-tutorial

# Hotfix branches
hotfix/critical-security-issue
```

### Commit Messages

Follow the Conventional Commits specification:

```bash
# Format: type(scope): description

feat(plugins): add support for async plugin commands
fix(ollama): handle connection timeout gracefully
docs(api): update ModelManager documentation
test(core): add integration tests for chat workflow
refactor(ui): simplify REPL command parsing
perf(context): optimize memory usage in context manager
```

### Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure all tests pass** locally
4. **Update CHANGELOG.md** if applicable
5. **Fill out PR template** completely

#### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] CHANGELOG.md updated
```

### Code Review Guidelines

#### For Authors

- Keep PRs focused and small
- Write descriptive commit messages
- Add comprehensive tests
- Update documentation
- Respond promptly to feedback

#### For Reviewers

- Focus on code correctness and design
- Check for security issues
- Verify test coverage
- Ensure documentation accuracy
- Be constructive in feedback

## API Design Guidelines

### Consistency Principles

1. **Naming**: Use consistent naming patterns across modules
2. **Error Handling**: Use specific exception types
3. **Async/Await**: Use async for I/O operations
4. **Type Hints**: Always include type annotations
5. **Documentation**: Document all public APIs

### API Evolution

#### Backward Compatibility

```python
# Good: Optional parameter with default
async def send_message(
    self,
    message: str,
    timeout: Optional[int] = None  # New optional parameter
) -> str:
    pass

# Bad: Breaking change
async def send_message(
    self,
    message: str,
    required_new_param: str  # Breaking change
) -> str:
    pass
```

#### Deprecation Process

```python
import warnings
from typing import Optional

async def old_method(self, param: str) -> str:
    """
    Old method (deprecated).
    
    .. deprecated:: 1.2.0
        Use :func:`new_method` instead.
    """
    warnings.warn(
        "old_method is deprecated, use new_method instead",
        DeprecationWarning,
        stacklevel=2
    )
    return await self.new_method(param)
```

### Version Management

Use semantic versioning (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

```python
# Version bumping examples
1.0.0 → 1.0.1  # Bug fix
1.0.1 → 1.1.0  # New feature
1.1.0 → 2.0.0  # Breaking change
```

## Performance Optimization

### Profiling

```python
import cProfile
import pstats
from abov3.core.app import ABOV3App

# Profile application startup
profiler = cProfile.Profile()
profiler.enable()

app = ABOV3App()
# Run operations to profile

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats()
```

### Memory Optimization

```python
import tracemalloc
import gc

# Monitor memory usage
tracemalloc.start()

# Run memory-intensive operations
await app.process_large_conversation()

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

tracemalloc.stop()
```

### Async Optimization

```python
# Good: Concurrent operations
async def process_multiple_files(files: List[Path]) -> List[str]:
    tasks = [process_file(file) for file in files]
    return await asyncio.gather(*tasks)

# Bad: Sequential operations
async def process_multiple_files_slow(files: List[Path]) -> List[str]:
    results = []
    for file in files:
        result = await process_file(file)  # Blocks other operations
        results.append(result)
    return results
```

### Caching Strategies

```python
from functools import lru_cache
import asyncio

class ModelManager:
    def __init__(self):
        self._model_cache = {}
        self._cache_lock = asyncio.Lock()
    
    @lru_cache(maxsize=128)
    def get_model_metadata(self, model_name: str) -> ModelInfo:
        """LRU cache for model metadata."""
        return self._fetch_model_metadata(model_name)
    
    async def get_model_with_cache(self, model_name: str) -> Model:
        """Async cache with lock protection."""
        async with self._cache_lock:
            if model_name not in self._model_cache:
                self._model_cache[model_name] = await self._load_model(model_name)
            return self._model_cache[model_name]
```

## Security Considerations

### Input Validation

```python
from abov3.utils.validation import validate_input, ValidationError

async def process_user_input(user_input: str) -> str:
    """Process user input with validation."""
    try:
        validated_input = validate_input(user_input, max_length=10000)
    except ValidationError as e:
        raise APIError(f"Invalid input: {e}")
    
    # Sanitize for security
    sanitized_input = sanitize_user_input(validated_input)
    
    return await self.model.process(sanitized_input)
```

### Code Execution Safety

```python
import ast
import sys
from types import CodeType

def safe_code_execution(code: str) -> bool:
    """Check if code is safe to execute."""
    try:
        # Parse code to AST
        tree = ast.parse(code)
    except SyntaxError:
        return False
    
    # Check for dangerous operations
    dangerous_nodes = (
        ast.Import, ast.ImportFrom, ast.Exec, ast.Eval,
        ast.Call  # Would need more sophisticated checking
    )
    
    for node in ast.walk(tree):
        if isinstance(node, dangerous_nodes):
            return False
    
    return True
```

### Secure Configuration

```python
import os
from cryptography.fernet import Fernet

class SecureConfig:
    """Secure configuration management."""
    
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self) -> bytes:
        """Get encryption key from environment or create new one."""
        key = os.environ.get('ABOV3_ENCRYPTION_KEY')
        if not key:
            key = Fernet.generate_key()
            print(f"Generated new encryption key: {key.decode()}")
        return key.encode() if isinstance(key, str) else key
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt sensitive configuration value."""
        return self.cipher.encrypt(value.encode()).decode()
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt configuration value."""
        return self.cipher.decrypt(encrypted_value.encode()).decode()
```

## Documentation Standards

### Code Documentation

```python
class ModelManager:
    """
    Manages AI models for ABOV3.
    
    The ModelManager handles model lifecycle operations including installation,
    removal, and performance monitoring. It provides both synchronous and
    asynchronous interfaces for different use cases.
    
    Attributes:
        config: Configuration object containing model settings.
        models: Dictionary of loaded models keyed by name.
        performance_tracker: Tracks model performance metrics.
    
    Example:
        >>> manager = ModelManager(config)
        >>> await manager.install_model("llama3.2:latest")
        >>> models = await manager.list_models()
        >>> print(f"Available models: {[m.name for m in models]}")
    """
```

### API Documentation

Use Sphinx with proper formatting:

```python
def send_message(
    self,
    message: str,
    session_id: Optional[str] = None,
    stream: bool = True
) -> AsyncIterator[str]:
    """
    Send a message to the AI model and receive a streaming response.
    
    This method handles the complete message lifecycle including context
    management, model inference, and response streaming.
    
    Args:
        message: The user's input message. Must be non-empty.
        session_id: Optional session identifier. If None, uses current session.
        stream: Whether to stream the response. Defaults to True.
    
    Returns:
        An async iterator yielding response chunks as they're generated.
    
    Raises:
        ValidationError: If the message is empty or invalid.
        APIError: If the model request fails.
        TimeoutError: If the request times out.
    
    Example:
        >>> async for chunk in app.send_message("Hello, world!"):
        ...     print(chunk, end='', flush=True)
        Hello! How can I help you today?
    
    Note:
        The response is streamed in real-time, so the complete response
        is not available until the iterator is exhausted.
    
    .. versionadded:: 1.0.0
    .. versionchanged:: 1.1.0
        Added support for session_id parameter.
    """
```

### README Updates

Keep README.md current with:
- Latest features
- Updated installation instructions
- Current system requirements
- Recent examples

## Release Process

### Version Planning

1. **Plan release scope** in GitHub milestones
2. **Update version** in `pyproject.toml` and `__init__.py`
3. **Update CHANGELOG.md** with new features and fixes
4. **Run full test suite** including integration tests
5. **Build documentation** and verify accuracy

### Pre-release Checklist

```bash
# 1. Update version numbers
sed -i 's/version = "1.0.0"/version = "1.1.0"/' pyproject.toml
sed -i 's/__version__ = "1.0.0"/__version__ = "1.1.0"/' abov3/__init__.py

# 2. Update changelog
# Edit CHANGELOG.md manually

# 3. Run tests
pytest tests/
pytest tests/integration/ -m integration

# 4. Check code quality
black --check abov3/
isort --check-only abov3/
flake8 abov3/
mypy abov3/

# 5. Build package
python -m build

# 6. Test package installation
pip install dist/abov3-*.whl
abov3 --version

# 7. Generate documentation
cd docs/
make html
```

### Release Steps

```bash
# 1. Create release branch
git checkout -b release/v1.1.0

# 2. Commit version updates
git add pyproject.toml abov3/__init__.py CHANGELOG.md
git commit -m "chore: bump version to 1.1.0"

# 3. Create pull request for release branch
gh pr create --title "Release v1.1.0" --body "Release preparation"

# 4. After approval and merge, tag release
git checkout main
git pull origin main
git tag -a v1.1.0 -m "Release version 1.1.0"
git push origin v1.1.0

# 5. Build and publish (automated in CI)
python -m build
twine upload dist/*

# 6. Create GitHub release
gh release create v1.1.0 --title "ABOV3 v1.1.0" --notes-file RELEASE_NOTES.md
```

### Post-release

1. **Update documentation** website
2. **Announce release** in community channels
3. **Monitor** for issues and user feedback
4. **Plan next release** based on feedback

---

This developer guide covers the essential aspects of contributing to ABOV3 4 Ollama. For specific questions or clarifications, please:

- 📖 Check the [API Reference](api_reference.md)
- 💬 Ask in [GitHub Discussions](https://github.com/abov3/abov3-ollama/discussions)
- 🐛 Report issues in [GitHub Issues](https://github.com/abov3/abov3-ollama/issues)
- 📧 Contact the team at contact@abov3.dev

Thank you for contributing to ABOV3! 🚀