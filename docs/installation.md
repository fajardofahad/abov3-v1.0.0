# Installation Guide

This guide provides comprehensive installation instructions for ABOV3 4 Ollama across different platforms and deployment scenarios.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Installation](#quick-installation)
- [Platform-Specific Installation](#platform-specific-installation)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
- [Docker Installation](#docker-installation)
- [Development Installation](#development-installation)
- [Verification](#verification)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Uninstallation](#uninstallation)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- Operating System: Windows 10, macOS 10.15, or Linux (Ubuntu 18.04+)
- Python: 3.8 or higher
- RAM: 4GB available memory
- Disk Space: 1GB free space
- Network: Internet connection for downloads

**Recommended Requirements:**
- Python: 3.10 or higher
- RAM: 8GB+ available memory
- Disk Space: 10GB+ free space (for multiple models)
- GPU: NVIDIA GPU with 8GB+ VRAM (optional, for larger models)

### Required Software

1. **Python 3.8+**
   - Download from [python.org](https://python.org)
   - Ensure `pip` is included in the installation

2. **Ollama**
   - Download from [ollama.ai](https://ollama.ai)
   - Must be running and accessible

3. **Git** (optional but recommended)
   - For cloning the repository and version control integration

## Quick Installation

For users who want to get started quickly:

```bash
# 1. Install Ollama (if not already installed)
# Visit https://ollama.ai and follow platform-specific instructions

# 2. Start Ollama service
ollama serve

# 3. Pull a language model
ollama pull llama3.2:latest

# 4. Install ABOV3
pip install abov3-ollama

# 5. Run ABOV3
abov3
```

## Platform-Specific Installation

### Windows

#### Method 1: Using pip (Recommended)

```powershell
# Open PowerShell as Administrator
# Check Python version
python --version

# Upgrade pip
python -m pip install --upgrade pip

# Install ABOV3
pip install abov3-ollama

# Verify installation
abov3 --version
```

#### Method 2: From Source

```powershell
# Clone repository
git clone https://github.com/abov3/abov3-ollama.git
cd abov3-ollama

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

#### Installing Ollama on Windows

1. Download Ollama installer from [ollama.ai](https://ollama.ai)
2. Run the installer with administrator privileges
3. Start Ollama from the Start menu or command line:
   ```powershell
   ollama serve
   ```

#### Windows-Specific Configuration

```powershell
# Set environment variables (optional)
setx ABOV3_CONFIG_PATH "%USERPROFILE%\.abov3\config.toml"
setx ABOV3_OLLAMA_HOST "http://localhost:11434"

# Create config directory
mkdir "%USERPROFILE%\.abov3"
```

### macOS

#### Method 1: Using pip

```bash
# Check Python version
python3 --version

# Install using pip
pip3 install abov3-ollama

# Add to PATH if needed
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### Method 2: Using Homebrew

```bash
# Install Python (if needed)
brew install python

# Install ABOV3
pip3 install abov3-ollama
```

#### Installing Ollama on macOS

```bash
# Using Homebrew
brew install ollama

# Or download from https://ollama.ai
# Start Ollama
ollama serve
```

#### macOS-Specific Configuration

```bash
# Set environment variables
echo 'export ABOV3_CONFIG_PATH="$HOME/.abov3/config.toml"' >> ~/.zshrc
echo 'export ABOV3_OLLAMA_HOST="http://localhost:11434"' >> ~/.zshrc
source ~/.zshrc

# Create config directory
mkdir -p ~/.abov3
```

### Linux

#### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv git

# Install ABOV3
pip3 install abov3-ollama

# Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### Red Hat/CentOS/Fedora

```bash
# Install Python and pip
sudo dnf install python3 python3-pip git

# Install ABOV3
pip3 install abov3-ollama
```

#### Arch Linux

```bash
# Install Python and pip
sudo pacman -S python python-pip git

# Install ABOV3
pip install abov3-ollama
```

#### Installing Ollama on Linux

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
systemctl --user start ollama
systemctl --user enable ollama

# Or run manually
ollama serve
```

#### Linux-Specific Configuration

```bash
# Set environment variables
echo 'export ABOV3_CONFIG_PATH="$HOME/.abov3/config.toml"' >> ~/.bashrc
echo 'export ABOV3_OLLAMA_HOST="http://localhost:11434"' >> ~/.bashrc
source ~/.bashrc

# Create config directory
mkdir -p ~/.abov3

# Set up systemd service (optional)
sudo tee /etc/systemd/system/abov3.service << EOF
[Unit]
Description=ABOV3 Service
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=$(which abov3) serve
Restart=always

[Install]
WantedBy=multi-user.target
EOF
```

## Docker Installation

For containerized deployment:

### Using Pre-built Image

```bash
# Pull the official image
docker pull abov3/abov3-ollama:latest

# Run with Ollama
docker run -d --name abov3 \
  -p 8080:8080 \
  -v abov3-data:/app/data \
  -e ABOV3_OLLAMA_HOST=http://host.docker.internal:11434 \
  abov3/abov3-ollama:latest
```

### Building from Source

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN pip install -e .

# Create user
RUN useradd -m -u 1000 abov3
USER abov3

EXPOSE 8080

CMD ["abov3", "serve", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
# Build image
docker build -t abov3-ollama .

# Run container
docker run -d --name abov3 \
  -p 8080:8080 \
  -v abov3-data:/app/data \
  abov3-ollama
```

### Docker Compose

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
  
  abov3:
    image: abov3/abov3-ollama:latest
    ports:
      - "8080:8080"
    volumes:
      - abov3-data:/app/data
    environment:
      - ABOV3_OLLAMA_HOST=http://ollama:11434
    depends_on:
      - ollama

volumes:
  ollama-data:
  abov3-data:
```

## Development Installation

For contributors and developers:

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/abov3/abov3-ollama.git
cd abov3-ollama

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

### Development Dependencies

The development installation includes:
- **Testing**: pytest, pytest-asyncio, pytest-cov, pytest-mock
- **Code Quality**: black, isort, flake8, mypy
- **Documentation**: sphinx, sphinx-rtd-theme, myst-parser
- **Development Tools**: pre-commit

### IDE Setup

#### Visual Studio Code

```json
// .vscode/settings.json
{
    "python.defaultInterpreter": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.sortImports.args": ["--profile", "black"]
}
```

#### PyCharm

1. Open project in PyCharm
2. Configure Python interpreter to use virtual environment
3. Enable code style: Settings → Code Style → Python → Black
4. Configure run configurations for tests and application

## Verification

### Basic Verification

```bash
# Check ABOV3 installation
abov3 --version

# Check configuration
abov3 config show

# Test Ollama connection
abov3 doctor

# Run health check
abov3 models list
```

### Full System Test

```bash
# Run comprehensive tests
pytest tests/

# Test chat functionality
abov3 chat --model llama3.2:latest

# Test model management
abov3 models install codellama:latest
abov3 models list

# Test configuration
abov3 config set model.temperature 0.8
abov3 config get model.temperature
```

## Configuration

### Initial Configuration

Run the setup wizard on first launch:

```bash
abov3
# Follow the interactive setup wizard
```

### Manual Configuration

Create configuration file:

```bash
# Create config directory
mkdir -p ~/.abov3

# Create config file
cat > ~/.abov3/config.toml << EOF
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

[security]
enable_content_filter = true
EOF
```

### Environment Variables

```bash
# Ollama configuration
export ABOV3_OLLAMA_HOST="http://localhost:11434"
export ABOV3_DEFAULT_MODEL="llama3.2:latest"

# Application configuration
export ABOV3_CONFIG_PATH="$HOME/.abov3/config.toml"
export ABOV3_DEBUG="false"

# UI configuration
export ABOV3_THEME="dark"
```

## Troubleshooting

### Common Issues

#### "Command not found: abov3"

**Solution:**
```bash
# Check if pip installation directory is in PATH
pip show -f abov3-ollama

# Add to PATH (Linux/macOS)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Add to PATH (Windows)
setx PATH "%PATH%;%APPDATA%\Python\Python311\Scripts"
```

#### "Cannot connect to Ollama server"

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Check firewall settings
# Ensure port 11434 is open
```

#### "Model not found"

**Solution:**
```bash
# List available models
ollama list

# Pull required model
ollama pull llama3.2:latest

# Set default model
abov3 config set model.default_model llama3.2:latest
```

#### Python version compatibility

**Solution:**
```bash
# Check Python version
python --version

# If using Python 3.7 or lower, upgrade:
# Windows: Download from python.org
# macOS: brew install python@3.11
# Linux: sudo apt install python3.11
```

### Logging and Debugging

```bash
# Enable debug mode
export ABOV3_DEBUG=true
abov3 --debug

# Check logs
tail -f ~/.abov3/logs/abov3.log

# Run with verbose output
abov3 -vvv chat
```

### Performance Issues

```bash
# Check system resources
abov3 doctor

# Monitor performance
abov3 config set monitoring.enabled true

# Optimize model settings
abov3 config set model.max_tokens 2048
abov3 config set model.context_length 4096
```

## Uninstallation

### Complete Uninstallation

```bash
# Uninstall package
pip uninstall abov3-ollama

# Remove configuration and data
rm -rf ~/.abov3

# Remove environment variables (Linux/macOS)
# Edit ~/.bashrc and remove ABOV3_* variables

# Remove environment variables (Windows)
# Use System Properties → Environment Variables
```

### Partial Uninstallation

```bash
# Keep configuration, remove only package
pip uninstall abov3-ollama

# Keep data, reset configuration
abov3 config reset
```

## Next Steps

After successful installation:

1. **Read the [Quick Start Guide](quick_start.md)** for your first steps
2. **Review the [User Guide](user_guide.md)** for detailed usage instructions
3. **Explore [Examples](../examples/)** for practical use cases
4. **Join the [Community](https://github.com/abov3/abov3-ollama/discussions)** for support and discussions

---

For additional help, please:
- Check the [FAQ](user_guide.md#faq)
- Search [GitHub Issues](https://github.com/abov3/abov3-ollama/issues)
- Ask in [GitHub Discussions](https://github.com/abov3/abov3-ollama/discussions)
- Contact support at contact@abov3.dev