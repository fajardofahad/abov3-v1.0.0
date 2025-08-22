# Quick Start Guide

Welcome to ABOV3 4 Ollama! This guide will get you up and running in just a few minutes.

## Table of Contents

- [Prerequisites Check](#prerequisites-check)
- [5-Minute Setup](#5-minute-setup)
- [First Chat Session](#first-chat-session)
- [Essential Commands](#essential-commands)
- [Basic Configuration](#basic-configuration)
- [Your First Coding Task](#your-first-coding-task)
- [Next Steps](#next-steps)

## Prerequisites Check

Before starting, ensure you have:

1. **Python 3.8+** installed
   ```bash
   python --version
   # Should show Python 3.8.0 or higher
   ```

2. **Ollama** installed and running
   ```bash
   ollama --version
   # Should show Ollama version
   ```

3. **At least one model** available
   ```bash
   ollama list
   # Should show at least one model
   ```

If any of these are missing, see the [Installation Guide](installation.md).

## 5-Minute Setup

### Step 1: Install ABOV3

```bash
# Install from PyPI
pip install abov3-ollama

# Verify installation
abov3 --version
```

### Step 2: Pull a Recommended Model

```bash
# Pull a good coding model (this may take a few minutes)
ollama pull llama3.2:latest

# Or for better coding performance
ollama pull codellama:latest
```

### Step 3: Start Ollama Server

```bash
# Start Ollama (if not already running)
ollama serve
```

### Step 4: First Run

```bash
# Launch ABOV3 (will run setup wizard)
abov3
```

That's it! ABOV3 is now ready to use.

## First Chat Session

### Starting a Chat

```bash
# Start interactive chat
abov3 chat

# Or start with specific options
abov3 chat -m llama3.2:latest -t 0.7
```

### Your First Interaction

Once in the chat interface, try these commands:

```
> /help
# Shows available commands

> Hello! Can you help me write a Python function?
# Start with a simple request

> /model list
# See available models

> /exit
# Exit the chat
```

### Basic Chat Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/help` | Show help menu | `/help` |
| `/model <name>` | Switch models | `/model codellama:latest` |
| `/temperature <value>` | Set temperature | `/temperature 0.8` |
| `/clear` | Clear conversation | `/clear` |
| `/history` | Show chat history | `/history` |
| `/export` | Export conversation | `/export conversation.md` |
| `/exit` | Exit chat | `/exit` |

## Essential Commands

### Model Management

```bash
# List available models
abov3 models list

# Install a new model
abov3 models install deepseek-coder:latest

# Get model information
abov3 models info llama3.2:latest

# Set default model
abov3 models set-default codellama:latest

# Remove a model
abov3 models remove old-model:tag
```

### Configuration

```bash
# Show current configuration
abov3 config show

# Get specific setting
abov3 config get model.default_model

# Set a configuration value
abov3 config set model.temperature 0.8

# Reset to defaults
abov3 config reset
```

### Health Check

```bash
# Run system diagnostics
abov3 doctor

# Check specific components
abov3 config validate
```

## Basic Configuration

### Set Your Preferred Model

```bash
# Set your default model
abov3 config set model.default_model codellama:latest

# Adjust creativity level
abov3 config set model.temperature 0.7

# Set response length
abov3 config set model.max_tokens 4096
```

### Customize UI

```bash
# Set theme (dark/light/auto)
abov3 config set ui.theme dark

# Enable timestamps
abov3 config set ui.show_timestamps true

# Enable syntax highlighting
abov3 config set ui.syntax_highlighting true
```

### Quick Configuration File

Create `~/.abov3/config.toml`:

```toml
[model]
default_model = "codellama:latest"
temperature = 0.7
max_tokens = 4096

[ui]
theme = "dark"
show_timestamps = true
syntax_highlighting = true

[ollama]
host = "http://localhost:11434"
timeout = 120
```

## Your First Coding Task

Let's create a simple Python function together:

### Step 1: Start Chat

```bash
abov3 chat -m codellama:latest
```

### Step 2: Request Code

```
> Create a Python function that calculates the factorial of a number
```

### Step 3: Ask for Improvements

```
> Can you add error handling and type hints to this function?
```

### Step 4: Request Tests

```
> Please write unit tests for this function using pytest
```

### Step 5: Export Your Work

```
> /export factorial_project.md
```

### Example Session Output

```python
def factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer.
    
    Args:
        n (int): Non-negative integer
        
    Returns:
        int: Factorial of n
        
    Raises:
        ValueError: If n is negative
        TypeError: If n is not an integer
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    if n < 0:
        raise ValueError("Input must be non-negative")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

# Tests
import pytest

def test_factorial_basic():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120

def test_factorial_errors():
    with pytest.raises(ValueError):
        factorial(-1)
    with pytest.raises(TypeError):
        factorial(3.14)
```

## Advanced Quick Tips

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Cancel current operation |
| `Ctrl+D` | Exit chat |
| `Tab` | Auto-complete commands |
| `Up/Down` | Navigate command history |
| `Ctrl+L` | Clear screen |

### Context Management

```bash
# Include files in conversation context
> /include src/main.py

# Set working directory
> /cd /path/to/project

# Show current context
> /context
```

### Code Analysis

```bash
# Analyze code file
> /analyze src/main.py

# Review code for issues
> /review src/main.py

# Suggest improvements
> /optimize src/main.py
```

### Export Options

```bash
# Export as markdown
> /export conversation.md

# Export code only
> /export --code-only functions.py

# Export with metadata
> /export --full session_data.json
```

## Troubleshooting Quick Fixes

### Can't Connect to Ollama?

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Check ABOV3 configuration
abov3 config get ollama.host
```

### Model Not Found?

```bash
# List available models
ollama list

# Pull the model you want
ollama pull llama3.2:latest

# Update ABOV3 default
abov3 config set model.default_model llama3.2:latest
```

### Performance Issues?

```bash
# Run diagnostics
abov3 doctor

# Reduce model size
abov3 config set model.max_tokens 2048

# Lower temperature
abov3 config set model.temperature 0.3
```

### Permission Errors?

```bash
# Check installation location
pip show abov3-ollama

# Install in user space
pip install --user abov3-ollama

# Fix PATH (Linux/macOS)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

## Integration Examples

### With VS Code

1. Install the ABOV3 VS Code extension (coming soon)
2. Or use ABOV3 in integrated terminal:
   ```bash
   # Open VS Code
   code .
   
   # In terminal
   abov3 chat
   ```

### With Git

```bash
# Analyze git diff
git diff | abov3 chat --input -

# Generate commit messages
abov3 chat "Generate a commit message for these changes: $(git diff --cached)"

# Code review
abov3 chat "Review this pull request: $(git show HEAD)"
```

### With Jupyter

```python
# In Jupyter notebook
import subprocess

def ask_abov3(question):
    result = subprocess.run(
        ['abov3', 'chat', '--input', '-'],
        input=question,
        text=True,
        capture_output=True
    )
    return result.stdout

# Use it
response = ask_abov3("Explain this pandas code: df.groupby('column').mean()")
print(response)
```

## Next Steps

Now that you're up and running, explore these areas:

### Learn More
- 📖 [User Guide](user_guide.md) - Comprehensive documentation
- 🔧 [API Reference](api_reference.md) - Technical details
- 👨‍💻 [Developer Guide](developer_guide.md) - Contributing and customization

### Explore Features
- 🧩 **Plugins**: Extend functionality with custom plugins
- 🔄 **History**: Search and manage conversation history
- 🏷️ **Fine-tuning**: Customize models for your specific needs
- 🔒 **Security**: Enterprise-grade security features

### Join the Community
- 💬 [GitHub Discussions](https://github.com/abov3/abov3-ollama/discussions)
- 🐛 [Report Issues](https://github.com/abov3/abov3-ollama/issues)
- 📧 [Contact Us](mailto:contact@abov3.dev)

### Real-World Examples

Check out our examples directory for practical use cases:

```bash
# Clone examples
git clone https://github.com/abov3/abov3-ollama.git
cd abov3-ollama/examples

# Try basic usage
python basic_usage.py

# Explore advanced features
python advanced_features.py
```

## Quick Reference Card

### Most Used Commands

```bash
# Start chat
abov3 chat

# Set model
abov3 config set model.default_model codellama:latest

# Install model
abov3 models install deepseek-coder:latest

# Check health
abov3 doctor

# Show config
abov3 config show

# Get help
abov3 --help
```

### Best Practices

1. **Start with small requests** - Build up to complex tasks
2. **Use specific models** - Choose the right model for your task
3. **Save your work** - Export important conversations
4. **Provide context** - Include relevant code files
5. **Iterate** - Refine requests based on responses

Congratulations! You're now ready to harness the power of AI-assisted coding with ABOV3 4 Ollama. Happy coding! 🚀