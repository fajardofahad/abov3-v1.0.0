# 📦 ABOV3 4 Ollama - Complete Installation Guide

**Comprehensive installation instructions for all platforms and use cases**

## 🎯 Installation Overview

ABOV3 4 Ollama offers multiple installation methods to fit your workflow:

1. **🚀 Direct Launcher** (Windows - Recommended)
2. **⚡ Auto Setup Script** (Cross-platform)  
3. **🔧 Global Installation** (Advanced users)
4. **🐳 Docker Installation** (Containerized)
5. **💻 Development Installation** (Contributors)

## 📋 System Requirements

### Minimum Requirements ✅ TESTED
| Component | Requirement | Notes |
|-----------|-------------|-------|
| **Operating System** | Windows 10, macOS 10.15, Ubuntu 18.04+ | Fully tested on Windows 11 |
| **Python** | 3.8 or higher | Tested with Python 3.13 |
| **Memory** | 4GB RAM | 8GB+ recommended |
| **Storage** | 2GB free space | + space for AI models |
| **Network** | Internet connection | For downloading models |

### Dependencies (Auto-installed)
```
ollama>=0.3.0          # AI model platform
rich>=13.0.0           # Terminal UI
prompt-toolkit>=3.0.36 # Interactive features
click>=8.0.0           # CLI framework
aiohttp>=3.8.0         # Async HTTP client
pygments>=2.10.0       # Syntax highlighting
pydantic>=2.0.0        # Data validation
```

## 🚀 Method 1: Direct Launcher (Windows - Recommended)

**Best for**: Windows users who want immediate results

### Step 1: Download ABOV3
```cmd
# If you haven't already, download/clone ABOV3
git clone https://github.com/abov3/abov3-ollama.git
cd abov3-ollama
```

### Step 2: Install Dependencies
```cmd
# Install all required packages
pip install -r requirements.txt
```

### Step 3: Launch with Direct Launcher ✅
```cmd
# Use the direct launcher - always works!
abov3_direct.bat --version
abov3_direct.bat
```

**Why this works**: The direct launcher (`abov3_direct.bat`) uses the full path to the installed ABOV3 executable, bypassing PATH issues.

## ⚡ Method 2: Auto Setup Script (Cross-platform)

**Best for**: Users who want automatic dependency management

### One Command Installation
```bash
# Navigate to ABOV3 directory
cd path/to/above3-ollama-v1.0.0

# Run auto setup - handles everything!
python start.py
```

**What it does**:
1. Checks Python version
2. Installs missing dependencies
3. Verifies Ollama connection
4. Launches ABOV3 automatically

## 🔧 Method 3: Global Installation (Advanced)

**Best for**: Users who want `abov3` command available system-wide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install ABOV3 Globally
```bash
# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### Step 3: Use Global Command
```cmd
# Now available system-wide
abov3 --version
abov3
```

### Windows PATH Alternative
If the global command doesn't work, use one of these solutions:

#### Option A: Direct Launcher
```cmd
# Use the always-working direct launcher
abov3_direct.bat
```

#### Option B: Add to Windows PATH
```cmd
# Run as Administrator to add to system PATH
make_global.bat
```

#### Option C: PowerShell PATH Fix
```powershell
# Run PowerShell script to fix PATH
.\fix_path.ps1
```

## 🐳 Method 4: Docker Installation

**Best for**: Containerized deployments or isolated environments

### Create Dockerfile
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy ABOV3
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirements.txt
RUN pip install -e .

# Expose port for Ollama
EXPOSE 11434

# Start Ollama and ABOV3
CMD ["python", "start.py"]
```

### Build and Run
```bash
# Build Docker image
docker build -t abov3-ollama .

# Run container
docker run -it -p 11434:11434 abov3-ollama
```

## 💻 Method 5: Development Installation

**Best for**: Contributors and developers

### Setup Development Environment
```bash
# Clone repository
git clone https://github.com/abov3/abov3-ollama.git
cd abov3-ollama

# Create virtual environment
python -m venv abov3-env

# Activate virtual environment
# Windows:
abov3-env\Scripts\activate
# Linux/macOS:
source abov3-env/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
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

## 🔧 Ollama Setup (Required for All Methods)

### Step 1: Install Ollama
1. **Download**: https://ollama.ai/download
2. **Install**: Run the installer for your platform
3. **Start**: Service starts automatically (or run `ollama serve`)

### Step 2: Download AI Models
```bash
# Essential models for coding
ollama pull llama3.2:latest      # General purpose (4.9GB)
ollama pull codellama:latest     # Code generation (3.8GB)
ollama pull deepseek-coder:latest # Advanced coding (6.9GB)

# Smaller models for lower resources
ollama pull llama3.2:1b          # Lightweight (1.3GB)
ollama pull codellama:7b         # Medium coding (3.8GB)
```

### Step 3: Verify Ollama
```bash
# Check Ollama is running
ollama serve

# List installed models
ollama list

# Test model
ollama run llama3.2:latest "Hello world"
```

## ✅ Verification & Testing

After installation, verify everything works:

### Basic Functionality Test
```cmd
# Check version
abov3_direct.bat --version
# Should show: ABOV3 4 Ollama v1.0.0

# Check help
abov3_direct.bat --help
# Should show command options

# Start interactive chat
abov3_direct.bat
# Should start chat interface
```

### Interactive Commands Test
Once ABOV3 is running:
```
> /help
# Should show available commands

> /model  
# Should show available Ollama models

> Hello, can you write a Python function?
# Should get streaming AI response

> /exit
# Should exit cleanly
```

### System Health Check
```cmd
# Run system diagnostics
abov3 doctor
```

## 🚨 Troubleshooting Guide

### Common Installation Issues

#### 1. "abov3 command not found" ❌
**Solutions (choose one)**:
```cmd
# A) Use direct launcher (always works)
abov3_direct.bat

# B) Use Python directly
python abov3.py

# C) Use auto setup
python start.py

# D) Fix PATH (Windows)
make_global.bat
```

#### 2. "Permission denied" or "Access denied" ❌
**Solutions**:
```cmd
# Windows: Run Command Prompt as Administrator
# Install globally
pip install -e . --user

# Or use direct launcher (no admin needed)
abov3_direct.bat
```

#### 3. "Module not found" errors ❌
**Solutions**:
```bash
# Install dependencies
pip install -r requirements.txt

# Upgrade pip if needed
python -m pip install --upgrade pip

# Check Python version
python --version  # Should be 3.8+
```

#### 4. "Ollama connection failed" ❌
**Solutions**:
1. **Install Ollama**: https://ollama.ai/download
2. **Start service**: `ollama serve`
3. **Download model**: `ollama pull llama3.2:latest`
4. **Check status**: `ollama list`
5. **Test connection**: `curl http://localhost:11434/api/tags`

#### 5. "Invalid key" or Key binding errors ❌
**Status**: ✅ **RESOLVED** - This issue has been completely fixed in the current version.

#### 6. Windows Unicode/encoding issues ❌
**Solutions**:
```cmd
# Set encoding
set PYTHONIOENCODING=utf-8
chcp 65001

# Or use Windows Terminal instead of Command Prompt
```

#### 7. Git Bash compatibility issues ❌
**Solutions**:
```bash
# A) Use full Python path
python "/c/path/to/abov3.py"

# B) Set alias
alias abov3='python "/c/path/to/abov3.py"'

# C) Use Windows Command Prompt (recommended)
```

### Platform-Specific Notes

#### Windows
- **Recommended**: Use Windows Command Prompt or PowerShell
- **Direct launcher** (`abov3_direct.bat`) always works
- **Windows Terminal** provides the best experience
- **Git Bash** has limited compatibility

#### macOS
```bash
# Install Python via Homebrew
brew install python

# Install Ollama
brew install ollama

# Use standard installation
python start.py
```

#### Linux (Ubuntu/Debian)
```bash
# Install dependencies
sudo apt update
sudo apt install python3 python3-pip curl

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Install ABOV3
python3 start.py
```

#### Linux (CentOS/RHEL)
```bash
# Install dependencies
sudo yum install python3 python3-pip curl

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Install ABOV3
python3 start.py
```

## 🚀 Post-Installation Setup

### 1. Configuration
ABOV3 creates a config file at:
- **Windows**: `%APPDATA%\abov3\config.toml`
- **macOS**: `~/Library/Application Support/abov3/config.toml`
- **Linux**: `~/.config/abov3/config.toml`

### 2. Model Management
```cmd
# List available models
abov3 models list

# Set default model
abov3 config set model.default_model "llama3.2:latest"

# Set temperature
abov3 config set model.temperature 0.7
```

### 3. IDE Integration

#### VS Code
Add to your tasks.json:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Start ABOV3",
            "type": "shell",
            "command": "abov3_direct.bat",
            "group": "build"
        }
    ]
}
```

#### Command Line Shortcuts
Create shortcuts for frequent use:
```cmd
# Windows shortcut
doskey abov3=abov3_direct.bat $*

# Bash alias  
alias abov3="python /path/to/abov3.py"
```

## 🎉 Success! Installation Complete

You now have ABOV3 4 Ollama installed and ready to use! 

### Quick Start
```cmd
# Launch ABOV3 (choose your installed method)
abov3_direct.bat    # Direct launcher
abov3               # Global command
python start.py     # Auto setup
python abov3.py     # Direct Python
```

### Next Steps
1. Read the [Quick Start Guide](QUICK_START.md) for usage instructions
2. Check [README.md](README.md) for complete documentation
3. Try the interactive chat and explore `/help` commands
4. Switch models with `/model` for different tasks

---

**🚀 Welcome to AI-powered software development!**

Need more help? Check our [Community & Support](#community--support) section in the README.