# API Reference

Complete API documentation for ABOV3 4 Ollama.

## Table of Contents

- [Core Classes](#core-classes)
  - [ABOV3App](#abov3app)
  - [Config](#config)
  - [OllamaClient](#ollamaclient)
  - [ModelManager](#modelmanager)
- [Chat & Messaging](#chat--messaging)
  - [ChatMessage](#chatmessage)
  - [ChatResponse](#chatresponse)
  - [ABOV3REPL](#abov3repl)
- [Context Management](#context-management)
  - [ContextManager](#contextmanager)
  - [MemoryManager](#memorymanager)
  - [SessionManager](#sessionmanager)
- [Plugin System](#plugin-system)
  - [Plugin](#plugin)
  - [PluginManager](#pluginmanager)
  - [PluginRegistry](#pluginregistry)
- [Security](#security)
  - [SecurityManager](#securitymanager)
- [Utilities](#utilities)
  - [File Operations](#file-operations)
  - [Git Integration](#git-integration)
  - [Export Functions](#export-functions)
- [Exceptions](#exceptions)
- [Configuration Schema](#configuration-schema)
- [CLI Reference](#cli-reference)

## Core Classes

### ABOV3App

The main application class that orchestrates all subsystems.

```python
class ABOV3App:
    """
    Main ABOV3 application class.
    
    Coordinates all subsystems including API communication, context management,
    UI components, security, and plugins.
    """
    
    def __init__(self, config: Optional[Config] = None) -> None:
        """
        Initialize the ABOV3 application.
        
        Args:
            config: Configuration object. If None, loads from default location.
        """
```

#### Methods

```python
async def run(self) -> None:
    """Start the interactive application."""

async def start_session(self, model: Optional[str] = None) -> str:
    """
    Start a new chat session.
    
    Args:
        model: Model to use for the session.
        
    Returns:
        Session ID.
    """

async def send_message(
    self, 
    message: str, 
    session_id: Optional[str] = None,
    stream: bool = True
) -> AsyncIterator[str]:
    """
    Send a message and get streaming response.
    
    Args:
        message: User message.
        session_id: Session to send message to.
        stream: Whether to stream the response.
        
    Yields:
        Response chunks.
    """

def set_system_prompt(self, prompt: str) -> None:
    """Set the system prompt for conversations."""

def load_last_conversation(self) -> None:
    """Load the most recent conversation."""

async def export_conversation(
    self, 
    session_id: str, 
    format: str = "markdown",
    output_path: Optional[Path] = None
) -> str:
    """
    Export conversation to various formats.
    
    Args:
        session_id: Session to export.
        format: Export format (markdown, json, text).
        output_path: Output file path.
        
    Returns:
        Exported content or file path.
    """
```

#### Properties

```python
@property
def current_model(self) -> str:
    """Get the currently active model."""

@property
def session_count(self) -> int:
    """Get the number of active sessions."""

@property
def metrics(self) -> AppMetrics:
    """Get application performance metrics."""
```

### Config

Configuration management class with validation and persistence.

```python
class Config(BaseModel):
    """Main configuration class with nested configurations."""
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
```

#### Methods

```python
@classmethod
def load_from_file(cls, file_path: Path) -> "Config":
    """Load configuration from TOML file."""

def save_to_file(self, file_path: Optional[Path] = None) -> None:
    """Save configuration to TOML file."""

@classmethod
def get_config_dir(cls) -> Path:
    """Get the configuration directory path."""

@classmethod
def get_data_dir(cls) -> Path:
    """Get the data directory path."""

@classmethod
def get_cache_dir(cls) -> Path:
    """Get the cache directory path."""

def validate_model_config(self) -> bool:
    """Validate model configuration settings."""

def get_effective_config(self) -> Dict[str, Any]:
    """Get configuration with environment variable overrides."""
```

#### Configuration Subclasses

```python
class ModelConfig(BaseModel):
    """Model-related configuration."""
    default_model: str = "llama3.2:latest"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1)
    max_tokens: int = Field(default=4096, ge=1)
    context_length: int = Field(default=8192, ge=1)
    repeat_penalty: float = Field(default=1.1, ge=0.0)
    seed: Optional[int] = None

class OllamaConfig(BaseModel):
    """Ollama API configuration."""
    host: str = "http://localhost:11434"
    timeout: int = 120
    verify_ssl: bool = True
    max_retries: int = 3

class UIConfig(BaseModel):
    """User interface configuration."""
    theme: str = "dark"
    show_timestamps: bool = True
    syntax_highlighting: bool = True
    auto_scroll: bool = True
    word_wrap: bool = True
    max_history_lines: int = 1000
```

### OllamaClient

Async client for Ollama API communication.

```python
class OllamaClient:
    """
    Production-ready async Ollama API client.
    
    Features:
    - Async communication with connection pooling
    - Model management (list, pull, delete)
    - Chat completion with streaming support
    - Comprehensive error handling and retries
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        retry_config: Optional[RetryConfig] = None
    ) -> None:
        """Initialize the Ollama client."""
```

#### Methods

```python
async def list_models(self) -> List[ModelInfo]:
    """List all available models."""

async def pull_model(
    self, 
    model_name: str,
    progress_callback: Optional[Callable[[Dict], None]] = None
) -> None:
    """
    Pull a model from the Ollama registry.
    
    Args:
        model_name: Name of the model to pull.
        progress_callback: Optional callback for progress updates.
    """

async def delete_model(self, model_name: str) -> None:
    """Delete a model."""

async def get_model_info(self, model_name: str) -> Dict[str, Any]:
    """Get detailed information about a model."""

async def chat(
    self,
    messages: List[ChatMessage],
    model: str,
    stream: bool = True,
    **options
) -> AsyncIterator[ChatResponse]:
    """
    Send chat messages and get streaming response.
    
    Args:
        messages: List of chat messages.
        model: Model to use for completion.
        stream: Whether to stream the response.
        **options: Additional model parameters.
        
    Yields:
        Chat response chunks.
    """

async def generate(
    self,
    prompt: str,
    model: str,
    stream: bool = True,
    **options
) -> AsyncIterator[str]:
    """
    Generate text from a prompt.
    
    Args:
        prompt: Input prompt.
        model: Model to use.
        stream: Whether to stream response.
        **options: Additional parameters.
        
    Yields:
        Generated text chunks.
    """

async def embed(
    self,
    text: str,
    model: str = "nomic-embed-text"
) -> List[float]:
    """
    Generate embeddings for text.
    
    Args:
        text: Text to embed.
        model: Embedding model to use.
        
    Returns:
        Embedding vector.
    """

async def health_check(self) -> bool:
    """Check if the Ollama server is accessible."""

async def close(self) -> None:
    """Close the client and cleanup resources."""
```

### ModelManager

Advanced model management with performance tracking.

```python
class ModelManager:
    """
    Comprehensive model manager with performance tracking and optimization.
    """
    
    def __init__(self, config: Config) -> None:
        """Initialize the model manager."""
```

#### Methods

```python
async def list_models(self) -> List[ModelInfo]:
    """List available models with metadata."""

async def install_model(
    self,
    model_name: str,
    progress_callback: Optional[Callable] = None
) -> None:
    """Install a model with progress tracking."""

async def remove_model(self, model_name: str) -> None:
    """Remove a model and cleanup resources."""

def is_model_available(self, model_name: str) -> bool:
    """Check if a model is available locally."""

async def get_model_performance(self, model_name: str) -> ModelPerformanceMetrics:
    """Get performance metrics for a model."""

async def recommend_model(self, task_type: str) -> List[str]:
    """
    Recommend models for a specific task type.
    
    Args:
        task_type: Type of task (coding, chat, analysis, etc.).
        
    Returns:
        List of recommended model names.
    """

async def benchmark_model(self, model_name: str) -> Dict[str, float]:
    """Run performance benchmarks on a model."""

def get_model_info(self, model_name: str) -> ModelInfo:
    """Get detailed model information."""

async def update_models(self) -> List[str]:
    """Check for and install model updates."""

def set_default_model(self, model_name: str) -> None:
    """Set the default model for new sessions."""
```

## Chat & Messaging

### ChatMessage

Represents a single chat message.

```python
@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # "user", "assistant", "system"
    content: str
    images: Optional[List[str]] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
```

### ChatResponse

Response from chat completion.

```python
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
    model: Optional[str] = None
```

### ABOV3REPL

Interactive Read-Eval-Print Loop interface.

```python
class ABOV3REPL:
    """
    Advanced REPL interface with rich formatting and command support.
    """
    
    def __init__(self, config: REPLConfig, app: ABOV3App) -> None:
        """Initialize the REPL."""
```

#### Methods

```python
async def run(self) -> None:
    """Start the REPL loop."""

def register_command(self, name: str, handler: Callable) -> None:
    """Register a new REPL command."""

def set_prompt(self, prompt: str) -> None:
    """Set the REPL prompt."""

def add_completer(self, completer: Callable) -> None:
    """Add a custom completer."""

async def execute_command(self, command: str, args: str) -> str:
    """Execute a REPL command."""

def display_help(self) -> None:
    """Display help information."""

def clear_screen(self) -> None:
    """Clear the terminal screen."""
```

## Context Management

### ContextManager

Manages conversation context and memory.

```python
class ContextManager:
    """
    Intelligent context management for conversations.
    
    Features:
    - Context window management
    - Relevance scoring
    - File inclusion tracking
    - Memory optimization
    """
    
    def __init__(self, config: Config) -> None:
        """Initialize the context manager."""
```

#### Methods

```python
def add_message(self, message: ChatMessage, session_id: str) -> None:
    """Add a message to the context."""

def get_context(self, session_id: str, max_tokens: int) -> List[ChatMessage]:
    """
    Get optimized context for a session.
    
    Args:
        session_id: Session identifier.
        max_tokens: Maximum tokens to include.
        
    Returns:
        List of messages within token limit.
    """

def include_file(self, file_path: Path, session_id: str) -> None:
    """Include a file in the session context."""

def exclude_file(self, file_path: Path, session_id: str) -> None:
    """Remove a file from session context."""

def get_included_files(self, session_id: str) -> List[Path]:
    """Get list of files included in context."""

def clear_context(self, session_id: str) -> None:
    """Clear all context for a session."""

def estimate_tokens(self, text: str) -> int:
    """Estimate token count for text."""

def optimize_context(self, messages: List[ChatMessage], max_tokens: int) -> List[ChatMessage]:
    """Optimize context to fit within token limit."""
```

### MemoryManager

Manages conversation memory and persistence.

```python
class MemoryManager:
    """Manages conversation memory and persistence."""
    
    def __init__(self, config: Config) -> None:
        """Initialize the memory manager."""
```

#### Methods

```python
async def save_conversation(self, session_id: str) -> None:
    """Save conversation to persistent storage."""

async def load_conversation(self, session_id: str) -> List[ChatMessage]:
    """Load conversation from storage."""

async def search_conversations(
    self, 
    query: str, 
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search through conversation history.
    
    Args:
        query: Search query.
        limit: Maximum results to return.
        
    Returns:
        List of matching conversations with metadata.
    """

async def delete_conversation(self, session_id: str) -> None:
    """Delete a conversation from storage."""

async def get_conversation_metadata(self, session_id: str) -> Dict[str, Any]:
    """Get metadata for a conversation."""

async def cleanup_old_conversations(self, days: int = 30) -> int:
    """
    Clean up conversations older than specified days.
    
    Returns:
        Number of conversations deleted.
    """
```

## Plugin System

### Plugin

Base class for all plugins.

```python
class Plugin(ABC):
    """
    Abstract base class for ABOV3 plugins.
    
    All plugins must inherit from this class and implement required methods.
    """
    
    name: str
    version: str
    description: str
    author: str
    
    def __init__(self) -> None:
        """Initialize the plugin."""
```

#### Abstract Methods

```python
@abstractmethod
async def initialize(self) -> None:
    """Initialize the plugin. Called when plugin is loaded."""

@abstractmethod
async def cleanup(self) -> None:
    """Cleanup plugin resources. Called when plugin is unloaded."""
```

#### Available Methods

```python
def register_command(self, name: str, handler: Callable) -> None:
    """Register a new command."""

def register_hook(self, event: str, handler: Callable) -> None:
    """Register an event hook."""

def get_config(self, key: str, default: Any = None) -> Any:
    """Get plugin configuration value."""

def set_config(self, key: str, value: Any) -> None:
    """Set plugin configuration value."""

async def emit_event(self, event: str, data: Any = None) -> None:
    """Emit an event to other plugins."""

def log_info(self, message: str) -> None:
    """Log an info message."""

def log_error(self, message: str, exception: Optional[Exception] = None) -> None:
    """Log an error message."""
```

### PluginManager

Manages plugin lifecycle and communication.

```python
class PluginManager:
    """
    Manages plugin loading, unloading, and communication.
    """
    
    def __init__(self, config: Config) -> None:
        """Initialize the plugin manager."""
```

#### Methods

```python
async def load_plugin(self, plugin_path: Path) -> None:
    """Load a plugin from file path."""

async def unload_plugin(self, plugin_name: str) -> None:
    """Unload a plugin by name."""

def list_plugins(self) -> List[Dict[str, Any]]:
    """List all loaded plugins."""

def get_plugin(self, name: str) -> Optional[Plugin]:
    """Get a plugin by name."""

async def enable_plugin(self, name: str) -> None:
    """Enable a plugin."""

async def disable_plugin(self, name: str) -> None:
    """Disable a plugin."""

def is_plugin_enabled(self, name: str) -> bool:
    """Check if a plugin is enabled."""

async def execute_command(self, command: str, args: str) -> str:
    """Execute a plugin command."""

async def emit_event(self, event: str, data: Any = None) -> None:
    """Emit an event to all plugins."""
```

## Security

### SecurityManager

Provides security features and threat protection.

```python
class SecurityManager:
    """
    Comprehensive security manager for ABOV3.
    
    Features:
    - Code execution sandboxing
    - Malicious pattern detection
    - File access validation
    - Session security
    """
    
    def __init__(self, config: SecurityConfig) -> None:
        """Initialize the security manager."""
```

#### Methods

```python
def validate_file_access(self, file_path: Path, operation: str = "read") -> bool:
    """
    Validate file access permissions.
    
    Args:
        file_path: Path to validate.
        operation: Type of operation (read, write, execute).
        
    Returns:
        True if access is allowed.
    """

def scan_code_for_threats(self, code: str) -> List[str]:
    """
    Scan code for potential security threats.
    
    Args:
        code: Code to scan.
        
    Returns:
        List of detected threats.
    """

def sanitize_input(self, user_input: str) -> str:
    """Sanitize user input for safe processing."""

async def execute_in_sandbox(
    self, 
    code: str, 
    language: str = "python",
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Execute code in a sandboxed environment.
    
    Args:
        code: Code to execute.
        language: Programming language.
        timeout: Execution timeout in seconds.
        
    Returns:
        Execution result with output and errors.
    """

def generate_session_token(self) -> str:
    """Generate a secure session token."""

def validate_session_token(self, token: str) -> bool:
    """Validate a session token."""

def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
    """Log a security event."""

def is_path_safe(self, path: Path) -> bool:
    """Check if a path is safe for access."""
```

## Utilities

### File Operations

```python
async def read_file(file_path: Path) -> str:
    """Read file contents safely."""

async def write_file(file_path: Path, content: str) -> None:
    """Write content to file safely."""

async def copy_file(src: Path, dst: Path) -> None:
    """Copy file from source to destination."""

async def backup_file(file_path: Path) -> Path:
    """Create a backup of a file."""

def get_file_info(file_path: Path) -> Dict[str, Any]:
    """Get file metadata and information."""

async def find_files(pattern: str, directory: Path) -> List[Path]:
    """Find files matching a pattern."""
```

### Git Integration

```python
class GitIntegration:
    """Git integration utilities."""
    
    @staticmethod
    def get_repository_root(path: Path) -> Optional[Path]:
        """Get the root of a git repository."""
    
    @staticmethod
    def get_current_branch(repo_path: Path) -> str:
        """Get the current git branch."""
    
    @staticmethod
    def get_modified_files(repo_path: Path) -> List[Path]:
        """Get list of modified files."""
    
    @staticmethod
    def get_diff(repo_path: Path, file_path: Optional[Path] = None) -> str:
        """Get git diff for repository or specific file."""
    
    @staticmethod
    async def commit(repo_path: Path, message: str, files: List[Path]) -> None:
        """Commit changes to git repository."""
```

### Export Functions

```python
async def export_to_markdown(
    messages: List[ChatMessage],
    output_path: Path,
    include_metadata: bool = True
) -> None:
    """Export conversation to markdown format."""

async def export_to_json(
    messages: List[ChatMessage],
    output_path: Path,
    pretty: bool = True
) -> None:
    """Export conversation to JSON format."""

async def export_code_blocks(
    messages: List[ChatMessage],
    output_path: Path,
    language: Optional[str] = None
) -> None:
    """Extract and export code blocks."""

async def export_to_pdf(
    messages: List[ChatMessage],
    output_path: Path,
    theme: str = "default"
) -> None:
    """Export conversation to PDF format."""
```

## Exceptions

Custom exception classes for error handling.

```python
class ABOV3Error(Exception):
    """Base exception for ABOV3 errors."""

class APIError(ABOV3Error):
    """API communication error."""

class ConnectionError(ABOV3Error):
    """Connection error."""

class ModelNotFoundError(ABOV3Error):
    """Model not found error."""

class AuthenticationError(ABOV3Error):
    """Authentication error."""

class ValidationError(ABOV3Error):
    """Configuration validation error."""

class SecurityError(ABOV3Error):
    """Security-related error."""

class PluginError(ABOV3Error):
    """Plugin-related error."""

class ContextError(ABOV3Error):
    """Context management error."""
```

## Configuration Schema

Complete configuration schema with validation rules.

```toml
[model]
default_model = "llama3.2:latest"  # string
temperature = 0.7                  # float, 0.0-2.0
top_p = 0.9                       # float, 0.0-1.0
top_k = 40                        # int, >= 1
max_tokens = 4096                 # int, >= 1
context_length = 8192             # int, >= 1
repeat_penalty = 1.1              # float, >= 0.0
seed = null                       # int or null

[ollama]
host = "http://localhost:11434"   # string, URL
timeout = 120                     # int, seconds
verify_ssl = true                 # boolean
max_retries = 3                   # int, >= 0

[ui]
theme = "dark"                    # string: dark, light, auto
show_timestamps = true            # boolean
syntax_highlighting = true        # boolean
auto_scroll = true                # boolean
word_wrap = true                  # boolean
max_history_lines = 1000          # int, >= 0

[security]
enable_content_filter = true      # boolean
sandbox_mode = false              # boolean
max_file_size = 10485760          # int, bytes
allowed_extensions = [".py", ".js", ".ts"]  # array of strings
blocked_paths = ["/etc", "/root"] # array of strings

[plugins]
enabled = ["git", "file_ops"]     # array of strings
auto_load = true                  # boolean
plugin_paths = []                 # array of strings

[history]
max_sessions = 100                # int, >= 0
auto_save = true                  # boolean
search_index = true               # boolean
retention_days = 30               # int, >= 0

[performance]
async_processing = true           # boolean
max_concurrent_requests = 5       # int, >= 1
cache_enabled = true              # boolean
cache_size = 1000                 # int, >= 0
```

## CLI Reference

Complete command-line interface reference.

### Global Options

```bash
--config, -c PATH     # Configuration file path
--debug, -d           # Enable debug mode
--version, -v         # Show version
--no-banner          # Don't show banner
```

### Commands

#### chat

Start interactive chat session.

```bash
abov3 chat [OPTIONS]

Options:
  --model, -m TEXT          # Model to use
  --system, -s TEXT         # System prompt
  --temperature, -t FLOAT   # Temperature setting
  --no-history             # Don't save to history
  --continue-last          # Continue last conversation
```

#### config

Manage configuration.

```bash
abov3 config COMMAND [OPTIONS]

Commands:
  show                     # Show configuration
  get KEY                  # Get configuration value
  set KEY VALUE           # Set configuration value
  reset                   # Reset to defaults
  validate               # Validate configuration

Options:
  --format FORMAT         # Output format (table, json, yaml)
```

#### models

Manage AI models.

```bash
abov3 models COMMAND [OPTIONS]

Commands:
  list                    # List available models
  install MODEL_NAME      # Install a model
  remove MODEL_NAME       # Remove a model
  info MODEL_NAME         # Show model information
  set-default MODEL_NAME  # Set default model

Options:
  --format FORMAT         # Output format (table, json)
  --progress             # Show progress for install
  --confirm              # Skip confirmation for remove
```

#### history

Manage conversation history.

```bash
abov3 history COMMAND [OPTIONS]

Commands:
  list                    # List conversations
  search QUERY            # Search conversations
  show SESSION_ID         # Show conversation
  export SESSION_ID       # Export conversation
  delete SESSION_ID       # Delete conversation
  clear                   # Clear all history

Options:
  --limit, -l INT         # Limit results
  --format FORMAT         # Output format
```

#### plugins

Manage plugins.

```bash
abov3 plugins COMMAND [OPTIONS]

Commands:
  list                    # List plugins
  enable PLUGIN_NAME      # Enable plugin
  disable PLUGIN_NAME     # Disable plugin
  info PLUGIN_NAME        # Show plugin info
  install PATH            # Install plugin

Options:
  --enabled-only          # Show only enabled plugins
```

#### doctor

Run health checks.

```bash
abov3 doctor [OPTIONS]

# Checks:
# - Ollama connection
# - Model availability
# - Configuration validity
# - Directory permissions
# - Plugin status
```

#### update

Check for updates.

```bash
abov3 update [OPTIONS]

Options:
  --check-only            # Only check, don't install
```

### Environment Variables

```bash
ABOV3_CONFIG_PATH       # Configuration file path
ABOV3_DEBUG            # Enable debug mode
ABOV3_OLLAMA_HOST      # Ollama server host
ABOV3_DEFAULT_MODEL    # Default model name
ABOV3_THEME           # UI theme
ABOV3_DATA_DIR        # Data directory path
```

---

For more examples and usage patterns, see:
- [User Guide](user_guide.md)
- [Developer Guide](developer_guide.md)
- [Examples Directory](../examples/)