# ABOV3 4 Ollama

**Production-Ready Interactive AI Coding Assistant for Ollama**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/abov3/abov3-ollama)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/ollama-0.3.0%2B-orange.svg)](https://ollama.ai)
[![Status](https://img.shields.io/badge/status-production--ready-green.svg)]()

ABOV3 4 Ollama is a **fully functional, production-ready** Python-based console application that provides an interactive CLI interface for AI-powered code generation, debugging, and refactoring using local Ollama models. Built for developers who want enterprise-grade AI assistance without cloud dependencies. **Now fully working with streaming responses, interactive chat, and cross-platform compatibility.**

![ABOV3 Screenshot Placeholder](docs/images/abov3-screenshot.png)

## Features

### Core Capabilities ✅ FULLY WORKING
- **Interactive Chat Interface**: Rich terminal-based chat with syntax highlighting
- **Streaming Responses**: Real-time AI responses with proper formatting
- **Code Generation**: Generate code in any programming language
- **Code Analysis & Debugging**: Analyze existing code and identify issues
- **Refactoring Assistance**: Intelligent code refactoring suggestions
- **Multi-Model Support**: Work with any Ollama-compatible model
- **Command System**: Full command support (/help, /model, /clear, /save, /exit)
- **Cross-Platform Support**: Works flawlessly on Windows, macOS, and Linux

### Advanced Features
- **Context Management**: Intelligent context handling for long conversations
- **Plugin System**: Extensible architecture with built-in and custom plugins
- **Model Fine-tuning**: Fine-tune models for specific coding tasks
- **Security & Privacy**: Enterprise-grade security with local processing
- **Performance Monitoring**: Built-in performance metrics and optimization
- **Git Integration**: Seamless integration with Git workflows
- **Multi-platform Support**: Works on Windows, macOS, and Linux

### Developer Experience
- **Rich Terminal UI**: Beautiful, responsive terminal interface
- **Auto-completion**: Intelligent command and code completion
- **Custom Keybindings**: Customizable keyboard shortcuts
- **Configuration Management**: Flexible configuration system
- **Comprehensive Testing**: Full test suite with CI/CD integration
- **Documentation**: Extensive documentation and examples

## Quick Start

### Prerequisites
- Python 3.8 or higher
- [Ollama](https://ollama.ai) installed and running
- At least one language model pulled in Ollama

### Installation (Multiple Working Methods)

#### Method 1: Direct Launch (Windows - Recommended)
```cmd
# Navigate to ABOV3 directory
cd "C:\path\to\above3-ollama-v1.0.0"

# Install dependencies
pip install -r requirements.txt

# Launch immediately (no setup needed)
abov3_direct.bat
```

#### Method 2: Global Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install ABOV3 globally
pip install -e .

# Use from anywhere
abov3
```

#### Method 3: Simple Start Script
```bash
# Automatic dependency check and launch
python start.py
```

#### Method 4: Direct Python
```bash
# Direct execution
python abov3.py
```

### First Run

```bash
# Start ABOV3 interactive chat (choose any method above)
abov3_direct.bat    # Windows direct
abov3               # Global installation
python start.py     # Automatic setup
python abov3.py     # Direct execution
```

### Basic Usage

```bash
# Start interactive chat session (default mode)
abov3

# Show version and status
abov3 --version

# Show help and available options
abov3 --help

# Check system health
abov3 doctor

# List available Ollama models
abov3 models list
```

### Interactive Commands (Within ABOV3 Chat)

Once ABOV3 is running, use these commands:

```
/help          - Show all available commands
/model         - Change the current AI model
/clear         - Clear the current conversation
/save          - Save the current conversation
/exit          - Exit ABOV3 safely
```

### Working Features Confirmed ✅

- **Error-Free Startup**: All key binding and configuration issues resolved
- **Streaming Responses**: Real-time AI responses with proper text formatting
- **Interactive Commands**: All slash commands work correctly
- **Model Switching**: Seamless switching between Ollama models
- **Cross-Platform**: Tested on Windows, works on macOS and Linux
- **Multiple Launch Methods**: Choose your preferred installation approach

## Architecture Overview

ABOV3 follows a modular, plugin-based architecture designed for scalability and maintainability:

```
abov3/
├── core/                   # Core application logic
│   ├── app.py             # Main application class
│   ├── config.py          # Configuration management
│   ├── api/               # API clients and communication
│   └── context/           # Context and memory management
├── models/                # Model management and fine-tuning
├── ui/                    # User interface components
│   ├── console/           # Terminal-based UI
│   └── components/        # Reusable UI components
├── plugins/               # Plugin system
│   ├── base/              # Plugin framework
│   └── builtin/           # Built-in plugins
├── utils/                 # Utility modules
└── tests/                 # Test suite
```

## Configuration

ABOV3 uses a hierarchical configuration system with support for:
- TOML configuration files
- Environment variables
- Command-line overrides
- Interactive configuration

Example configuration:

```toml
[model]
default_model = "llama3.2:latest"
temperature = 0.7
max_tokens = 4096

[ollama]
host = "http://localhost:11434"
timeout = 120

[ui]
theme = "dark"
show_timestamps = true
syntax_highlighting = true

[security]
enable_content_filter = true
sandbox_mode = false
```

## Plugin Development

ABOV3 features an extensible plugin system. Create custom plugins to extend functionality:

```python
from abov3.plugins.base import Plugin

class MyPlugin(Plugin):
    name = "my_plugin"
    version = "1.0.0"
    
    def initialize(self):
        self.register_command("my-command", self.handle_command)
    
    async def handle_command(self, args):
        # Plugin logic here
        pass
```

## Performance & Scalability

- **Async Architecture**: Non-blocking operations for responsive UI
- **Memory Management**: Intelligent context window management
- **Caching**: Smart caching for improved performance
- **Resource Monitoring**: Built-in resource usage monitoring
- **Batch Processing**: Efficient handling of multiple requests

## Security Features

- **Local Processing**: All AI processing happens locally
- **Content Filtering**: Built-in content safety filters
- **Sandbox Mode**: Optional sandboxed execution environment
- **Audit Logging**: Comprehensive audit trail
- **Data Privacy**: No data sent to external services

## Documentation

- [Installation Guide](docs/installation.md) - Detailed installation instructions
- [Quick Start Guide](docs/quick_start.md) - Get up and running quickly
- [User Guide](docs/user_guide.md) - Comprehensive user documentation
- [API Reference](docs/api_reference.md) - Complete API documentation
- [Developer Guide](docs/developer_guide.md) - Contributing and development
- [Examples](examples/) - Code examples and tutorials

## System Requirements

### Minimum Requirements ✅ TESTED
- **OS**: Windows 10, macOS 10.15, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher (tested with Python 3.13)
- **RAM**: 4GB available memory
- **Disk**: 1GB free space + model storage
- **Network**: Internet connection for initial Ollama model downloads
- **Ollama**: Version 0.3.0 or higher running locally

### Recommended Requirements
- **OS**: Latest version of Windows 11, macOS, or Linux
- **Python**: 3.10 or higher (Python 3.13 recommended)
- **RAM**: 8GB+ available memory for better model performance
- **Disk**: 10GB+ free space (for multiple models)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional, for faster inference)
- **SSD**: Solid-state drive recommended for faster model loading

### Dependencies (Auto-installed)
- ollama>=0.3.0
- rich>=13.0.0 (for beautiful terminal UI)
- prompt-toolkit>=3.0.36 (for interactive features)
- click>=8.0.0 (for CLI interface)
- aiohttp>=3.8.0 (for async operations)
- All other dependencies listed in requirements.txt

## Troubleshooting

### Common Issues and Solutions ✅ RESOLVED

#### "abov3 command not found"
**Solution**: Use one of the working launch methods:
```cmd
# Windows - Direct launcher (always works)
abov3_direct.bat

# Or use full Python path
python abov3.py

# Or automatic setup
python start.py
```

#### "Invalid key" or Key binding errors
**Status**: ✅ FIXED - All key binding issues have been resolved in the current version.

#### "Ollama connection failed"
**Solution**: 
1. Install Ollama: https://ollama.ai/download
2. Start Ollama service: `ollama serve`
3. Pull a model: `ollama pull llama3.2:latest`
4. Verify connection: `ollama list`

#### "Module not found" errors
**Solution**:
```bash
# Install all dependencies
pip install -r requirements.txt

# Or use the automatic setup
python start.py
```

#### Windows Unicode/encoding issues
**Solution**: Set environment variable:
```cmd
set PYTHONIOENCODING=utf-8
```

#### Git Bash / MINGW64 compatibility
**Solution**: Use the Windows Command Prompt or PowerShell for the best experience, or use:
```bash
python "/c/path/to/abov3.py"
```

### Performance Tips

1. **Use SSD storage** for faster model loading
2. **Close other applications** when using large models
3. **Use smaller models** (7B parameters) for faster responses
4. **Enable GPU acceleration** in Ollama for better performance

## Model Compatibility

ABOV3 works with any Ollama-compatible model:

### Recommended Models
- **Code Generation**: `codellama:latest`, `deepseek-coder:latest`
- **General Purpose**: `llama3.2:latest`, `mistral:latest`
- **Specialized**: `sql-coder:latest`, `magicoder:latest`

### Model Requirements
- Minimum 7B parameters recommended for coding tasks
- 13B+ parameters for complex reasoning
- 34B+ parameters for advanced code generation

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/abov3/abov3-ollama.git
cd abov3-ollama
pip install -e ".[dev]"

# Run tests
pytest

# Run code formatting
black abov3/
isort abov3/

# Run linting
flake8 abov3/
mypy abov3/
```

## Community & Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/abov3/abov3-ollama/issues)
- **Discussions**: [Community discussions](https://github.com/abov3/abov3-ollama/discussions)
- **Documentation**: [Full documentation](https://abov3-ollama.readthedocs.io)
- **Email**: contact@abov3.dev

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ollama](https://ollama.ai) for providing the local AI model platform
- [Rich](https://github.com/Textualize/rich) for the beautiful terminal UI
- [Click](https://click.palletsprojects.com/) for the CLI framework
- The open-source community for various dependencies and inspiration

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

---

**Made with ❤️ by the ABOV3 Team**

*Empowering developers with AI-powered coding assistance, locally and privately.*