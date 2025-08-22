# ABOV3 4 Ollama - User Manual

**Version 1.0.0**  
**Advanced Interactive AI Coding Assistant for Ollama**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Core Features](#core-features)
4. [Interactive Chat Interface](#interactive-chat-interface)
5. [Code Generation & Analysis](#code-generation--analysis)
6. [Model Management](#model-management)
7. [Configuration System](#configuration-system)
8. [History & Session Management](#history--session-management)
9. [Plugin System](#plugin-system)
10. [Advanced Features](#advanced-features)
11. [Examples & Use Cases](#examples--use-cases)
12. [Best Practices](#best-practices)
13. [Integration Guide](#integration-guide)

---

## Introduction

ABOV3 4 Ollama is a powerful Python-based console application that provides an interactive CLI interface for AI-powered code generation, debugging, and refactoring using local Ollama models. Built for developers who want enterprise-grade AI assistance without cloud dependencies.

### Key Benefits

- **Local Processing**: All AI operations run on your machine
- **Privacy First**: No data sent to external services
- **Multi-Model Support**: Works with any Ollama-compatible model
- **Rich Terminal UI**: Beautiful, responsive console interface
- **Extensible**: Plugin system for custom functionality
- **Enterprise Ready**: Comprehensive logging, monitoring, and security features

### System Requirements

**Minimum Requirements:**
- Python 3.8+
- 4GB RAM
- 1GB disk space
- Ollama server running

**Recommended Requirements:**
- Python 3.10+
- 8GB+ RAM
- 10GB+ disk space
- GPU with 8GB+ VRAM (for larger models)

---

## Getting Started

### Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Verify Ollama Connection:**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   ```

3. **First Run:**
   ```bash
   # Start ABOV3 (setup wizard runs automatically on first launch)
   abov3
   ```

### Quick Start

```bash
# Basic chat session
abov3 chat

# Use specific model
abov3 chat -m llama3.2:latest

# Continue last conversation
abov3 chat --continue-last

# Set custom temperature
abov3 chat -t 0.9 -s "You are a Python expert"
```

---

## Core Features

### 1. Interactive Chat Interface

The main interface provides a rich terminal experience with:

- **Syntax Highlighting**: Code blocks are automatically highlighted
- **Auto-completion**: Intelligent command completion
- **History Navigation**: Use arrow keys to navigate command history
- **Search**: Search through conversation history
- **Export Options**: Save conversations in multiple formats

**Starting a Chat Session:**
```bash
abov3 chat
```

**Chat Interface Commands:**
- `/help` - Show available commands
- `/clear` - Clear screen
- `/history` - View command history
- `/save <filename>` - Save current session
- `/exit` - Exit chat session

### 2. Streaming Responses

All AI responses stream in real-time for immediate feedback:

```bash
# Responses appear as they're generated
👤 User: Explain Python decorators
🤖 Assistant: Python decorators are a powerful feature...
```

### 3. Context Management

ABOV3 maintains conversation context automatically:

- **Session Persistence**: Conversations continue across interactions
- **Context Windows**: Intelligent management of context limits
- **Memory Optimization**: Efficient handling of long conversations

---

## Interactive Chat Interface

### Basic Usage

1. **Start Chat:**
   ```bash
   abov3 chat
   ```

2. **Send Messages:**
   Simply type your message and press Enter:
   ```
   ABOV3> How do I create a REST API in Python?
   ```

3. **Multi-line Input:**
   For longer inputs, the system automatically detects and handles multi-line content.

### REPL Commands

The interactive environment supports various commands:

| Command | Description | Example |
|---------|-------------|---------|
| `/help` | Show help information | `/help` |
| `/clear` | Clear the screen | `/clear` |
| `/history` | Show command history | `/history` |
| `/save` | Save current session | `/save my_session.json` |
| `/load` | Load saved session | `/load my_session.json` |
| `/config` | Show/modify configuration | `/config show` |
| `/theme` | Change color theme | `/theme monokai` |
| `/mode` | Switch key binding mode | `/mode vi` |
| `/debug` | Toggle debug mode | `/debug` |
| `/context` | Show current context | `/context` |
| `/reset` | Reset session | `/reset` |
| `/export` | Export session to markdown | `/export report.md` |
| `/status` | Show system status | `/status` |
| `/models` | List available models | `/models` |
| `/exit` | Exit REPL | `/exit` |

### Interface Customization

**Key Bindings:**
- **Emacs Mode (default)**: Standard readline shortcuts
- **Vi Mode**: Vim-style editing
- **Custom Mode**: User-defined shortcuts

**Themes:**
- `monokai` (default)
- `github`
- `solarized`
- `material`
- `dracula`

**Change Theme:**
```bash
# Via command line
abov3 config set ui.theme github

# Via REPL command
/theme github
```

---

## Code Generation & Analysis

### Code Generation

ABOV3 excels at generating code in any programming language:

**Example Request:**
```
Create a Python function that validates email addresses using regex.
Include error handling and type hints.
```

**Generated Response:**
```python
import re
from typing import bool

def validate_email(email: str) -> bool:
    """
    Validate email address using regex pattern.
    
    Args:
        email (str): Email address to validate
        
    Returns:
        bool: True if email is valid, False otherwise
        
    Raises:
        TypeError: If email is not a string
    """
    if not isinstance(email, str):
        raise TypeError("Email must be a string")
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
```

### Code Analysis

**Debugging Assistance:**
```
ABOV3> I'm getting a KeyError in this Python code: [paste your code]
```

**Code Review:**
```
ABOV3> Review this function for performance and best practices: [paste code]
```

**Refactoring Suggestions:**
```
ABOV3> How can I refactor this code to be more maintainable? [paste code]
```

### Language Support

ABOV3 supports code generation and analysis for:

- **Python** - Full ecosystem support
- **JavaScript/TypeScript** - Frontend and Node.js
- **Java** - Enterprise applications
- **C#** - .NET development
- **Go** - Systems programming
- **Rust** - Systems programming
- **C/C++** - Low-level programming
- **PHP** - Web development
- **Ruby** - Web applications
- **SQL** - Database queries
- **Shell/Bash** - System administration
- **HTML/CSS** - Web markup and styling
- **And many more...**

### Advanced Code Features

**1. Unit Test Generation:**
```
Generate unit tests for this Python function: [paste function]
```

**2. Documentation Generation:**
```
Create comprehensive docstrings for this class: [paste class]
```

**3. Performance Optimization:**
```
Optimize this algorithm for better performance: [paste code]
```

**4. Security Analysis:**
```
Analyze this code for security vulnerabilities: [paste code]
```

---

## Model Management

### Listing Models

```bash
# List all available models
abov3 models list

# JSON format output
abov3 models list -f json
```

### Installing Models

```bash
# Install a model
abov3 models install codellama:latest

# Install with progress display
abov3 models install llama3.2:latest --progress
```

### Model Information

```bash
# Get detailed model information
abov3 models info llama3.2:latest
```

### Setting Default Model

```bash
# Set default model for all sessions
abov3 models set-default llama3.2:latest
```

### Removing Models

```bash
# Remove a model (with confirmation)
abov3 models remove llama2:7b

# Skip confirmation
abov3 models remove llama2:7b --confirm
```

### Recommended Models

**For Code Generation:**
- `codellama:latest` - Specialized for code
- `deepseek-coder:latest` - Advanced code understanding
- `magicoder:latest` - Code-focused model

**For General Purpose:**
- `llama3.2:latest` - Balanced performance
- `mistral:latest` - Fast and efficient
- `mixtral:latest` - Large context window

**For Specialized Tasks:**
- `sql-coder:latest` - Database queries
- `python-coder:latest` - Python-specific
- `web-dev:latest` - Web development

---

## Configuration System

ABOV3 uses a hierarchical configuration system supporting:
- TOML configuration files
- Environment variables
- Command-line overrides
- Interactive configuration

### Configuration File

**Location:**
- Windows: `%APPDATA%\abov3\config.toml`
- Linux/macOS: `~/.config/abov3/config.toml`

**Example Configuration:**
```toml
# Model Configuration
[model]
default_model = "llama3.2:latest"
temperature = 0.7
max_tokens = 4096
context_length = 8192
top_p = 0.9
top_k = 40
repeat_penalty = 1.1

# Ollama Connection
[ollama]
host = "http://localhost:11434"
timeout = 120
verify_ssl = true
max_retries = 3

# User Interface
[ui]
theme = "monokai"
syntax_highlighting = true
line_numbers = true
word_wrap = true
auto_complete = true
vim_mode = false
streaming_output = true

# History Management
[history]
max_conversations = 100
auto_save = true
compression = true
search_index = true

# Plugin System
[plugins]
enabled = ["git", "file_ops"]
auto_load = true
directories = ["~/.abov3/plugins"]

# Logging Configuration
[logging]
level = "INFO"
enable_file_logging = true
enable_console_logging = true
enable_json_logging = true
max_file_size = 52428800  # 50MB
backup_count = 10
```

### Configuration Commands

**View Configuration:**
```bash
# Show all settings
abov3 config show

# JSON format
abov3 config show -f json

# YAML format
abov3 config show -f yaml
```

**Get Specific Setting:**
```bash
abov3 config get model.default_model
abov3 config get ollama.host
abov3 config get ui.theme
```

**Set Configuration:**
```bash
# Set model settings
abov3 config set model.temperature 0.8
abov3 config set model.max_tokens 2048

# Set UI preferences
abov3 config set ui.theme github
abov3 config set ui.vim_mode true

# Set Ollama connection
abov3 config set ollama.host http://192.168.1.100:11434
```

**Reset Configuration:**
```bash
# Reset to defaults (with confirmation)
abov3 config reset

# Skip confirmation
abov3 config reset --confirm
```

**Validate Configuration:**
```bash
# Check configuration validity
abov3 config validate
```

### Environment Variables

Override settings using environment variables:

```bash
# Model settings
export ABOV3_DEFAULT_MODEL="codellama:latest"
export ABOV3_TEMPERATURE="0.9"
export ABOV3_MAX_TOKENS="2048"

# Connection settings
export ABOV3_OLLAMA_HOST="http://localhost:11434"
export ABOV3_OLLAMA_TIMEOUT="180"

# Application settings
export ABOV3_DEBUG="true"
export ABOV3_LOG_LEVEL="DEBUG"
```

---

## History & Session Management

### Conversation History

ABOV3 automatically saves all conversations with features for:
- **Persistent Storage**: Conversations saved across sessions
- **Search Capability**: Find specific conversations
- **Export Options**: Multiple export formats
- **Session Management**: Organize related conversations

### History Commands

```bash
# List conversation history
abov3 history list

# Limit results
abov3 history list --limit 20

# Search conversations
abov3 history search "python function"
abov3 history search "REST API" --limit 10

# Show specific conversation
abov3 history show <conversation_id>

# Export conversation
abov3 history export <conversation_id> output.json
abov3 history export <conversation_id> output.md

# Clear all history
abov3 history clear
```

### Session Management

**Starting Sessions:**
```bash
# Start new chat session
abov3 chat

# Continue last session
abov3 chat --continue-last

# Start with custom settings
abov3 chat -m codellama:latest -t 0.9 -s "You are a senior developer"
```

**Saving Sessions:**
```bash
# In REPL
/save my_coding_session.json

# Export as markdown
/export my_session_report.md
```

**Loading Sessions:**
```bash
# In REPL
/load my_coding_session.json

# From command line
abov3 chat --load my_session.json
```

---

## Plugin System

ABOV3 features an extensible plugin architecture for custom functionality:

### Built-in Plugins

**Available Plugins:**
- `git` - Git integration and automation
- `file_ops` - File operations and management
- `web_search` - Web search capabilities
- `code_analysis` - Advanced code analysis
- `documentation` - Documentation generation

### Plugin Management

**List Plugins:**
```bash
# List all plugins
abov3 plugins list

# Show only enabled plugins
abov3 plugins list --enabled-only
```

**Enable/Disable Plugins:**
```bash
# Enable a plugin
abov3 plugins enable git

# Disable a plugin
abov3 plugins disable git
```

**Plugin Information:**
```bash
# Show plugin details
abov3 plugins info git
```

### Custom Plugin Development

**Basic Plugin Structure:**
```python
from abov3.plugins.base import Plugin

class MyPlugin(Plugin):
    name = "my_plugin"
    version = "1.0.0"
    description = "My custom ABOV3 plugin"
    
    def initialize(self):
        """Initialize plugin"""
        self.register_command("my-command", self.handle_command)
        self.register_hook("pre_response", self.pre_response_hook)
    
    async def handle_command(self, args: str) -> str:
        """Handle custom command"""
        return f"Plugin command executed with args: {args}"
    
    async def pre_response_hook(self, message: str) -> str:
        """Hook called before AI response"""
        return message  # Modify message if needed
```

**Plugin Installation:**
```bash
# Install from directory
abov3 plugins install /path/to/plugin

# Install from git repository
abov3 plugins install git+https://github.com/user/plugin.git
```

---

## Advanced Features

### 1. Fine-tuning Support

ABOV3 supports model fine-tuning for specialized tasks:

```python
from abov3.models.fine_tuning import FineTuner

# Create fine-tuner
fine_tuner = FineTuner(base_model="codellama:7b")

# Add training data
fine_tuner.add_training_data("examples/training_data.jsonl")

# Start fine-tuning
await fine_tuner.start_training(
    output_name="my-custom-model",
    epochs=3,
    learning_rate=0.0001
)
```

### 2. Context Management

**Smart Context Handling:**
- Automatic context window management
- Conversation summarization for long chats
- Context preservation across sessions
- Intelligent context pruning

### 3. Performance Monitoring

**Built-in Metrics:**
```bash
# System status
abov3 doctor

# Performance metrics
abov3 status --metrics

# Resource monitoring
abov3 monitor
```

### 4. Security Features

**Security Options:**
- Content filtering
- Input sanitization
- Audit logging
- Sandbox mode (optional)
- Local processing guarantee

**Security Configuration:**
```toml
[security]
enable_content_filter = true
enable_audit_logging = true
sandbox_mode = false
max_input_length = 10000
blocked_patterns = ["system:", "eval("]
```

### 5. API Integration

**REST API Mode:**
```bash
# Start API server
abov3 serve --host 0.0.0.0 --port 8080

# API endpoints available:
# POST /api/v1/chat - Send chat message
# GET /api/v1/models - List models
# GET /api/v1/status - System status
```

### 6. Batch Processing

**Process Multiple Files:**
```python
from abov3.core.app import ABOV3App
from abov3.utils.batch import BatchProcessor

app = ABOV3App(config)
processor = BatchProcessor(app)

# Process multiple files
results = await processor.process_files([
    "file1.py",
    "file2.js", 
    "file3.go"
], task="Add comprehensive docstrings")
```

---

## Examples & Use Cases

### 1. Code Generation

**Generate a REST API:**
```
ABOV3> Create a Python Flask REST API for a todo list application with the following features:
- CRUD operations for todos
- SQLite database
- Input validation
- Error handling
- Unit tests
```

**Generate React Component:**
```
ABOV3> Create a React component for a user profile card with:
- Props for user data
- Edit mode toggle
- Form validation
- TypeScript types
```

### 2. Debugging Assistant

**Debug Python Code:**
```
ABOV3> I'm getting this error: [paste error message and code]
Can you help me understand what's wrong and how to fix it?
```

**Performance Analysis:**
```
ABOV3> This function is running slowly on large datasets:
[paste function]
Can you suggest optimizations?
```

### 3. Learning & Education

**Explain Concepts:**
```
ABOV3> Explain async/await in Python with practical examples
```

**Code Review:**
```
ABOV3> Review this code for best practices and suggest improvements:
[paste code]
```

### 4. Documentation Generation

**Generate Documentation:**
```
ABOV3> Create comprehensive documentation for this API:
[paste API code]
Include examples and usage instructions.
```

**Generate README:**
```
ABOV3> Create a README.md for this Python project:
[describe project]
```

### 5. Testing

**Generate Unit Tests:**
```
ABOV3> Create comprehensive unit tests for this class:
[paste class code]
Use pytest and include edge cases.
```

**Generate Integration Tests:**
```
ABOV3> Create integration tests for this REST API:
[paste API code]
```

### 6. Refactoring

**Modernize Code:**
```
ABOV3> Refactor this legacy Python code to use modern practices:
[paste legacy code]
Use type hints, proper error handling, and clean architecture.
```

**Extract Functions:**
```
ABOV3> This function is too long. Help me break it down:
[paste long function]
```

---

## Best Practices

### 1. Effective Prompting

**Be Specific:**
```
❌ Bad: "Write a function"
✅ Good: "Write a Python function that validates email addresses using regex, includes type hints, and proper error handling"
```

**Provide Context:**
```
✅ Good: "I'm building a Django web application for inventory management. Create a model for tracking products with these fields: name, SKU, quantity, price, category."
```

**Include Requirements:**
```
✅ Good: "Create a React component that:
- Displays user profile information
- Allows editing in place
- Validates form inputs
- Handles loading and error states
- Uses TypeScript"
```

### 2. Model Selection

**For Code Generation:**
- Use `codellama:latest` or `deepseek-coder:latest`
- Higher temperature (0.7-0.9) for creative solutions
- Lower temperature (0.3-0.5) for precise implementations

**For Debugging:**
- Use models with larger context windows
- Lower temperature (0.1-0.3) for focused analysis
- Provide complete error messages and stack traces

**For Learning:**
- Use balanced models like `llama3.2:latest`
- Medium temperature (0.5-0.7) for comprehensive explanations
- Ask for examples and practical applications

### 3. Session Management

**Organize Conversations:**
- Use descriptive session names
- Save important conversations
- Export key discussions as documentation

**Context Optimization:**
- Start new sessions for different projects
- Use `/reset` to clear context when switching topics
- Provide relevant context at the beginning of sessions

### 4. Configuration Optimization

**Performance Settings:**
```toml
[model]
max_tokens = 4096  # Balance between completeness and speed
temperature = 0.7   # Good balance for most tasks
context_length = 8192  # Adequate for most conversations

[ui]
streaming_output = true  # Real-time feedback
auto_complete = true     # Faster input
```

**Development Settings:**
```toml
[logging]
level = "DEBUG"  # Detailed logging during development
enable_file_logging = true

[security]
enable_audit_logging = true  # Track all interactions
```

### 5. Security Considerations

**Best Practices:**
- Never paste sensitive credentials or API keys
- Use environment variables for configuration secrets
- Enable content filtering in production environments
- Regular security updates and model updates
- Monitor audit logs for unusual activity

**Safe Coding Practices:**
- Validate all generated code before use
- Test generated code thoroughly
- Review security implications of generated code
- Use version control for all changes

---

## Integration Guide

### 1. IDE Integration

**VS Code Extension:**
Create a custom VS Code extension that integrates ABOV3:

```json
{
  "contributes": {
    "commands": [
      {
        "command": "abov3.generateCode",
        "title": "Generate Code with ABOV3"
      },
      {
        "command": "abov3.explainCode",
        "title": "Explain Code with ABOV3"
      }
    ]
  }
}
```

### 2. CI/CD Integration

**GitHub Actions:**
```yaml
name: ABOV3 Code Review
on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup ABOV3
      run: |
        pip install abov3-ollama
        abov3 config set ollama.host ${{ secrets.OLLAMA_HOST }}
    - name: Code Review
      run: |
        abov3 batch-review --files "$(git diff --name-only HEAD~1)"
```

### 3. Custom Applications

**Python Integration:**
```python
import asyncio
from abov3.core.app import ABOV3App
from abov3.core.config import Config

async def integrate_abov3():
    # Configure ABOV3
    config = Config()
    config.model.default_model = "codellama:latest"
    
    # Create app instance
    app = ABOV3App(config)
    
    # Send message and get response
    session_id = await app.start_session()
    
    async for chunk in app.send_message("Generate a Python function to sort a list", session_id):
        print(chunk, end='')
    
    await app.cleanup()

# Run integration
asyncio.run(integrate_abov3())
```

### 4. Web Integration

**Flask Web Interface:**
```python
from flask import Flask, request, jsonify, render_template
from abov3.core.app import ABOV3App

app = Flask(__name__)
abov3_app = None

@app.route('/api/chat', methods=['POST'])
async def chat():
    message = request.json.get('message')
    session_id = request.json.get('session_id')
    
    if not session_id:
        session_id = await abov3_app.start_session()
    
    response = ""
    async for chunk in abov3_app.send_message(message, session_id):
        response += chunk
    
    return jsonify({
        'response': response,
        'session_id': session_id
    })

@app.route('/')
def index():
    return render_template('chat.html')
```

### 5. Database Integration

**Store Conversations:**
```python
import sqlite3
from datetime import datetime

class ConversationStore:
    def __init__(self, db_path="conversations.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                message TEXT,
                response TEXT,
                timestamp DATETIME,
                model TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_conversation(self, session_id, message, response, model):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO conversations (session_id, message, response, timestamp, model)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, message, response, datetime.now(), model))
        conn.commit()
        conn.close()
```

---

## Conclusion

ABOV3 4 Ollama provides a powerful, privacy-focused AI coding assistant that runs entirely on your local machine. With its rich feature set, extensible plugin system, and comprehensive configuration options, it's designed to enhance your development workflow while maintaining complete control over your code and data.

For additional support and resources:
- Check the [Command Reference](COMMAND_REFERENCE.md) for detailed command documentation
- Review [Troubleshooting](TROUBLESHOOTING.md) for common issues
- See the [FAQ](FAQ.md) for frequently asked questions
- Visit the examples directory for practical usage scenarios

Happy coding with ABOV3!

---

*This manual covers ABOV3 4 Ollama version 1.0.0. For the latest updates and features, check the project documentation.*