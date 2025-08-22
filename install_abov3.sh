#!/bin/bash
#===============================================================================
# ABOV3 4 Ollama - Unix/Linux Installation Script
#===============================================================================
#
# This script provides a comprehensive installer for ABOV3 4 Ollama on Unix-like
# systems including Linux, macOS, and BSD variants. It automatically detects
# the system configuration, installs dependencies, and sets up the complete
# ABOV3 development environment with Ollama integration.
#
# Features:
# - Cross-platform Unix compatibility (Linux, macOS, BSD)
# - Automatic package manager detection (apt, yum, dnf, pacman, brew)
# - Python version validation and virtual environment support
# - Comprehensive error handling and recovery mechanisms
# - Progress reporting with colored output
# - System integration (desktop files, PATH setup)
# - Security-conscious installation with privilege checks
#
# Author: ABOV3 Enterprise DevOps Agent
# Version: 1.0.0
#===============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly LOG_FILE="${TMPDIR:-/tmp}/abov3_install_$(date +%Y%m%d_%H%M%S).log"
readonly PYTHON_REQUIRED="3.8"
readonly DEFAULT_MODEL="llama3.2"
readonly INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/.local}"

# Color codes for terminal output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly MAGENTA='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[0;37m'
readonly BOLD='\033[1m'
readonly UNDERLINE='\033[4m'
readonly RESET='\033[0m'

# Global variables
PYTHON_EXE=""
PYTHON_VERSION=""
SYSTEM_OS=""
SYSTEM_ARCH=""
PACKAGE_MANAGER=""
INSTALL_SUCCESS=0
SUDO_CMD=""
USE_VENV=0

# Logging function
log() {
    local level="${1:-INFO}"
    local message="${2:-}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] $level: $message" >> "$LOG_FILE"
    
    case "$level" in
        ERROR)   echo -e "${RED}[$timestamp] ERROR: $message${RESET}" ;;
        WARNING) echo -e "${YELLOW}[$timestamp] WARNING: $message${RESET}" ;;
        SUCCESS) echo -e "${GREEN}[$timestamp] SUCCESS: $message${RESET}" ;;
        INFO)    echo -e "${CYAN}[$timestamp] INFO: $message${RESET}" ;;
        *)       echo "[$timestamp] $level: $message" ;;
    esac
}

# Error handler
error_exit() {
    local message="${1:-Unknown error occurred}"
    local exit_code="${2:-1}"
    
    log "ERROR" "$message"
    echo -e "\n${RED}${BOLD}Installation failed!${RESET}"
    echo -e "${YELLOW}Check the log file for details: $LOG_FILE${RESET}"
    
    cleanup_on_failure
    exit "$exit_code"
}

# Cleanup function for failed installations
cleanup_on_failure() {
    log "INFO" "Performing cleanup after installation failure"
    
    # Remove any created shortcuts
    rm -f "$HOME/Desktop/ABOV3 4 Ollama.desktop" 2>/dev/null || true
    rm -f "$HOME/.local/share/applications/abov3-ollama.desktop" 2>/dev/null || true
    
    # Don't remove config directory as it contains logs
    log "INFO" "Cleanup completed"
}

# Print installation banner
print_banner() {
    cat << 'EOF'

                    ╔══════════════════════════════════════════════════════════════════╗
                    ║                    ABOV3 4 OLLAMA INSTALLER                     ║
                    ║                                                                  ║
                    ║  Revolutionary AI-Powered Coding Platform                       ║
                    ║  Enterprise-Grade Infrastructure & Deployment                   ║
                    ║                                                                  ║
                    ║  Version: 1.0.0 | Platform: Unix/Linux                         ║
                    ║                                                                  ║
                    ╚══════════════════════════════════════════════════════════════════╝

EOF
}

# Detect system information
detect_system() {
    log "INFO" "Detecting system information"
    
    SYSTEM_OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    SYSTEM_ARCH=$(uname -m)
    
    log "INFO" "System: $SYSTEM_OS $SYSTEM_ARCH"
    echo -e "${CYAN}System detected: ${GREEN}$SYSTEM_OS $SYSTEM_ARCH${RESET}"
    
    # Detect package manager
    if command -v apt-get >/dev/null 2>&1; then
        PACKAGE_MANAGER="apt"
    elif command -v yum >/dev/null 2>&1; then
        PACKAGE_MANAGER="yum"
    elif command -v dnf >/dev/null 2>&1; then
        PACKAGE_MANAGER="dnf"
    elif command -v pacman >/dev/null 2>&1; then
        PACKAGE_MANAGER="pacman"
    elif command -v brew >/dev/null 2>&1; then
        PACKAGE_MANAGER="brew"
    elif command -v zypper >/dev/null 2>&1; then
        PACKAGE_MANAGER="zypper"
    else
        log "WARNING" "No supported package manager found"
        PACKAGE_MANAGER="unknown"
    fi
    
    log "INFO" "Package manager: $PACKAGE_MANAGER"
    echo -e "${CYAN}Package manager: ${GREEN}$PACKAGE_MANAGER${RESET}"
    
    # Check for sudo/doas
    if command -v sudo >/dev/null 2>&1; then
        SUDO_CMD="sudo"
    elif command -v doas >/dev/null 2>&1; then
        SUDO_CMD="doas"
    else
        log "WARNING" "No privilege escalation command found (sudo/doas)"
        SUDO_CMD=""
    fi
}

# Check system requirements
check_system_requirements() {
    log "INFO" "Checking system requirements"
    echo -e "\n${CYAN}${BOLD}Checking system requirements...${RESET}"
    
    # Check available disk space (minimum 2GB)
    local available_space
    available_space=$(df "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    
    if [ "$available_gb" -lt 2 ]; then
        error_exit "Insufficient disk space. Required: 2GB, Available: ${available_gb}GB"
    fi
    
    log "SUCCESS" "Disk space check passed: ${available_gb}GB available"
    echo -e "${GREEN}✓ Disk space: ${available_gb}GB available${RESET}"
    
    # Check network connectivity
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        log "SUCCESS" "Network connectivity verified"
        echo -e "${GREEN}✓ Network connectivity verified${RESET}"
    else
        log "WARNING" "Network connectivity issues detected"
        echo -e "${YELLOW}⚠ Network connectivity issues detected${RESET}"
        echo -e "${YELLOW}Some installation steps may fail without internet access${RESET}"
        
        read -p "Continue anyway? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error_exit "Installation aborted by user due to network issues"
        fi
    fi
    
    # Check for essential build tools
    local missing_tools=()
    for tool in curl wget git; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log "WARNING" "Missing essential tools: ${missing_tools[*]}"
        echo -e "${YELLOW}⚠ Missing essential tools: ${missing_tools[*]}${RESET}"
        
        if [[ "$PACKAGE_MANAGER" != "unknown" ]]; then
            echo -e "${CYAN}Attempting to install missing tools...${RESET}"
            install_system_dependencies "${missing_tools[@]}"
        else
            echo -e "${YELLOW}Please install the following tools manually: ${missing_tools[*]}${RESET}"
            read -p "Continue anyway? (y/N): " -r
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                error_exit "Installation aborted - missing essential tools"
            fi
        fi
    fi
}

# Install system dependencies
install_system_dependencies() {
    local packages=("$@")
    log "INFO" "Installing system dependencies: ${packages[*]}"
    
    case "$PACKAGE_MANAGER" in
        apt)
            if [[ -n "$SUDO_CMD" ]]; then
                $SUDO_CMD apt-get update -qq
                $SUDO_CMD apt-get install -y "${packages[@]}"
            else
                error_exit "Cannot install system dependencies without sudo privileges"
            fi
            ;;
        yum|dnf)
            if [[ -n "$SUDO_CMD" ]]; then
                $SUDO_CMD $PACKAGE_MANAGER install -y "${packages[@]}"
            else
                error_exit "Cannot install system dependencies without sudo privileges"
            fi
            ;;
        pacman)
            if [[ -n "$SUDO_CMD" ]]; then
                $SUDO_CMD pacman -S --noconfirm "${packages[@]}"
            else
                error_exit "Cannot install system dependencies without sudo privileges"
            fi
            ;;
        brew)
            brew install "${packages[@]}"
            ;;
        zypper)
            if [[ -n "$SUDO_CMD" ]]; then
                $SUDO_CMD zypper install -y "${packages[@]}"
            else
                error_exit "Cannot install system dependencies without sudo privileges"
            fi
            ;;
        *)
            log "WARNING" "Cannot automatically install dependencies with unknown package manager"
            return 1
            ;;
    esac
    
    log "SUCCESS" "System dependencies installed successfully"
}

# Detect Python installation
detect_python() {
    log "INFO" "Detecting Python installation"
    echo -e "\n${CYAN}${BOLD}Detecting Python installation...${RESET}"
    
    # Try common Python executables
    for python_cmd in python3 python python3.12 python3.11 python3.10 python3.9 python3.8; do
        if command -v "$python_cmd" >/dev/null 2>&1; then
            local version
            version=$("$python_cmd" --version 2>&1 | cut -d' ' -f2)
            
            if validate_python_version "$version"; then
                PYTHON_EXE="$python_cmd"
                PYTHON_VERSION="$version"
                log "SUCCESS" "Python found: $python_cmd ($version)"
                echo -e "${GREEN}✓ Python found: $python_cmd${RESET}"
                echo -e "${GREEN}✓ Version: $version${RESET}"
                return 0
            else
                log "INFO" "Python $python_cmd ($version) is too old"
            fi
        fi
    done
    
    # Python not found or too old
    error_exit "Python $PYTHON_REQUIRED+ not found. Please install Python $PYTHON_REQUIRED or higher."
}

# Validate Python version
validate_python_version() {
    local version="$1"
    local required="$PYTHON_REQUIRED"
    
    # Convert versions to comparable format
    local version_num
    local required_num
    version_num=$(echo "$version" | cut -d. -f1,2 | tr -d '.')
    required_num=$(echo "$required" | cut -d. -f1,2 | tr -d '.')
    
    [ "$version_num" -ge "$required_num" ]
}

# Setup Python virtual environment
setup_python_environment() {
    log "INFO" "Setting up Python environment"
    echo -e "\n${CYAN}${BOLD}Setting up Python environment...${RESET}"
    
    # Check if we should use virtual environment
    if [[ "$USE_VENV" -eq 1 ]]; then
        local venv_dir="$SCRIPT_DIR/venv"
        
        echo -e "${CYAN}Creating virtual environment at $venv_dir...${RESET}"
        "$PYTHON_EXE" -m venv "$venv_dir"
        
        # Activate virtual environment
        source "$venv_dir/bin/activate"
        PYTHON_EXE="$venv_dir/bin/python"
        
        log "SUCCESS" "Virtual environment created and activated"
        echo -e "${GREEN}✓ Virtual environment activated${RESET}"
    fi
    
    # Upgrade pip
    echo -e "${CYAN}Upgrading pip...${RESET}"
    "$PYTHON_EXE" -m pip install --upgrade pip
    
    log "SUCCESS" "Python environment setup completed"
}

# Install Python dependencies
install_python_dependencies() {
    log "INFO" "Installing Python dependencies"
    echo -e "\n${CYAN}${BOLD}Installing Python dependencies...${RESET}"
    
    local dependencies=(
        "requests>=2.25.0"
        "click>=8.0.0"
        "colorama>=0.4.4"
        "tqdm>=4.60.0"
        "pyyaml>=5.4.0"
        "python-dotenv>=0.19.0"
        "psutil>=5.8.0"
        "rich>=12.0.0"
        "httpx>=0.23.0"
        "websockets>=10.0"
        "fastapi>=0.68.0"
        "uvicorn>=0.15.0"
        "pydantic>=1.8.0"
        "numpy>=1.21.0"
        "packaging>=21.0"
    )
    
    for dep in "${dependencies[@]}"; do
        echo -e "${CYAN}Installing $dep...${RESET}"
        if ! "$PYTHON_EXE" -m pip install "$dep"; then
            error_exit "Failed to install Python dependency: $dep"
        fi
    done
    
    log "SUCCESS" "All Python dependencies installed successfully"
    echo -e "${GREEN}✓ All Python dependencies installed${RESET}"
}

# Install ABOV3 in development mode
install_abov3_development() {
    log "INFO" "Installing ABOV3 in development mode"
    echo -e "\n${CYAN}${BOLD}Installing ABOV3 in development mode...${RESET}"
    
    # Check if setup.py exists, create if needed
    if [[ ! -f "$SCRIPT_DIR/setup.py" && ! -f "$SCRIPT_DIR/pyproject.toml" ]]; then
        log "INFO" "Creating setup.py for development installation"
        create_setup_py
    fi
    
    # Install in editable mode
    if ! "$PYTHON_EXE" -m pip install -e "$SCRIPT_DIR"; then
        error_exit "Failed to install ABOV3 in development mode"
    fi
    
    log "SUCCESS" "ABOV3 installed in development mode successfully"
    echo -e "${GREEN}✓ ABOV3 installed in development mode${RESET}"
}

# Create setup.py if it doesn't exist
create_setup_py() {
    cat > "$SCRIPT_DIR/setup.py" << 'EOF'
from setuptools import setup, find_packages

setup(
    name="abov3-ollama",
    version="1.0.0",
    description="ABOV3 4 Ollama - Revolutionary AI-Powered Coding Platform",
    long_description="Enterprise-grade AI coding platform with Ollama integration",
    author="ABOV3 Enterprise DevOps Team",
    author_email="devops@abov3.com",
    url="https://github.com/abov3/abov3-ollama",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
        "click>=8.0.0",
        "colorama>=0.4.4",
        "tqdm>=4.60.0",
        "pyyaml>=5.4.0",
        "python-dotenv>=0.19.0",
        "psutil>=5.8.0",
        "rich>=12.0.0",
        "httpx>=0.23.0",
        "websockets>=10.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "numpy>=1.21.0",
        "packaging>=21.0"
    ],
    entry_points={
        "console_scripts": [
            "abov3=abov3.cli:main",
            "abov3-server=abov3.server:main",
            "abov3-setup=abov3.setup:main"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="ai coding assistant ollama enterprise development",
)
EOF
    
    log "SUCCESS" "Created setup.py"
}

# Setup PATH environment
setup_path_environment() {
    log "INFO" "Setting up PATH environment"
    echo -e "\n${CYAN}${BOLD}Setting up PATH environment...${RESET}"
    
    # Get the directory where pip installs executables
    local bin_dir
    if [[ "$USE_VENV" -eq 1 ]]; then
        bin_dir="$SCRIPT_DIR/venv/bin"
    else
        bin_dir=$("$PYTHON_EXE" -c "import site; print(site.USER_BASE + '/bin')")
        
        # Ensure the directory exists
        mkdir -p "$bin_dir"
    fi
    
    # Add to shell profiles
    local shell_profiles=("$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile")
    local path_export="export PATH=\"$bin_dir:\$PATH\""
    
    for profile in "${shell_profiles[@]}"; do
        if [[ -f "$profile" ]]; then
            if ! grep -Fq "$bin_dir" "$profile"; then
                echo "" >> "$profile"
                echo "# Added by ABOV3 installer" >> "$profile"
                echo "$path_export" >> "$profile"
                log "SUCCESS" "Added PATH export to $profile"
            else
                log "INFO" "PATH already configured in $profile"
            fi
        fi
    done
    
    # Also add to current session
    export PATH="$bin_dir:$PATH"
    
    log "SUCCESS" "PATH environment setup completed"
    echo -e "${GREEN}✓ PATH environment configured${RESET}"
}

# Check Ollama installation
check_ollama_installation() {
    log "INFO" "Checking Ollama installation"
    echo -e "\n${CYAN}${BOLD}Checking Ollama installation...${RESET}"
    
    if command -v ollama >/dev/null 2>&1; then
        local version
        version=$(ollama --version 2>/dev/null || echo "unknown")
        log "SUCCESS" "Ollama found: $version"
        echo -e "${GREEN}✓ Ollama found: $version${RESET}"
        return 0
    else
        log "WARNING" "Ollama not found"
        echo -e "${YELLOW}⚠ Ollama not found${RESET}"
        return 1
    fi
}

# Install Ollama
install_ollama() {
    log "INFO" "Installing Ollama"
    echo -e "\n${CYAN}${BOLD}Installing Ollama...${RESET}"
    
    case "$SYSTEM_OS" in
        linux)
            # Use official installation script
            echo -e "${CYAN}Downloading Ollama installation script...${RESET}"
            if curl -fsSL https://ollama.com/install.sh | sh; then
                log "SUCCESS" "Ollama installed successfully"
                echo -e "${GREEN}✓ Ollama installed successfully${RESET}"
            else
                error_exit "Failed to install Ollama"
            fi
            ;;
        darwin)
            if command -v brew >/dev/null 2>&1; then
                echo -e "${CYAN}Installing Ollama via Homebrew...${RESET}"
                if brew install ollama; then
                    log "SUCCESS" "Ollama installed via Homebrew"
                    echo -e "${GREEN}✓ Ollama installed via Homebrew${RESET}"
                else
                    # Fallback to direct download
                    echo -e "${CYAN}Homebrew failed, trying direct download...${RESET}"
                    install_ollama_direct_macos
                fi
            else
                install_ollama_direct_macos
            fi
            ;;
        *)
            log "WARNING" "Automatic Ollama installation not supported on $SYSTEM_OS"
            echo -e "${YELLOW}⚠ Automatic Ollama installation not supported on $SYSTEM_OS${RESET}"
            echo -e "${CYAN}Please install Ollama manually from: https://ollama.com${RESET}"
            return 1
            ;;
    esac
}

# Install Ollama directly on macOS
install_ollama_direct_macos() {
    echo -e "${CYAN}Installing Ollama via direct download...${RESET}"
    
    local install_dir="/usr/local/bin"
    local ollama_url="https://ollama.com/download/ollama-darwin"
    
    if [[ -w "$install_dir" ]]; then
        curl -fsSL "$ollama_url" -o "$install_dir/ollama"
        chmod +x "$install_dir/ollama"
        log "SUCCESS" "Ollama installed to $install_dir"
        echo -e "${GREEN}✓ Ollama installed successfully${RESET}"
    else
        if [[ -n "$SUDO_CMD" ]]; then
            $SUDO_CMD curl -fsSL "$ollama_url" -o "$install_dir/ollama"
            $SUDO_CMD chmod +x "$install_dir/ollama"
            log "SUCCESS" "Ollama installed to $install_dir (with sudo)"
            echo -e "${GREEN}✓ Ollama installed successfully${RESET}"
        else
            error_exit "Cannot install Ollama to $install_dir without sudo privileges"
        fi
    fi
}

# Start Ollama service
start_ollama_service() {
    log "INFO" "Starting Ollama service"
    echo -e "\n${CYAN}${BOLD}Starting Ollama service...${RESET}"
    
    # Check if already running
    if ollama list >/dev/null 2>&1; then
        log "INFO" "Ollama service already running"
        echo -e "${GREEN}✓ Ollama service already running${RESET}"
        return 0
    fi
    
    # Start service in background
    echo -e "${CYAN}Starting Ollama service...${RESET}"
    nohup ollama serve >/dev/null 2>&1 &
    
    # Wait for service to start
    for i in {1..10}; do
        if ollama list >/dev/null 2>&1; then
            log "SUCCESS" "Ollama service started"
            echo -e "${GREEN}✓ Ollama service started${RESET}"
            return 0
        fi
        sleep 1
    done
    
    log "WARNING" "Ollama service may not have started properly"
    echo -e "${YELLOW}⚠ Ollama service may not have started properly${RESET}"
    return 1
}

# Setup default model
setup_default_model() {
    log "INFO" "Setting up default model: $DEFAULT_MODEL"
    echo -e "\n${CYAN}${BOLD}Setting up default model: $DEFAULT_MODEL...${RESET}"
    
    # Start Ollama service if needed
    start_ollama_service
    
    # Check if model is already available
    if ollama list 2>/dev/null | grep -q "$DEFAULT_MODEL"; then
        log "SUCCESS" "Model $DEFAULT_MODEL already available"
        echo -e "${GREEN}✓ Model $DEFAULT_MODEL already available${RESET}"
        return 0
    fi
    
    # Download the model
    echo -e "${CYAN}Downloading $DEFAULT_MODEL model... This may take a while.${RESET}"
    
    if ollama pull "$DEFAULT_MODEL"; then
        log "SUCCESS" "Model $DEFAULT_MODEL downloaded successfully"
        echo -e "${GREEN}✓ Model $DEFAULT_MODEL downloaded successfully${RESET}"
        
        # Test the model
        test_model "$DEFAULT_MODEL"
    else
        log "ERROR" "Failed to download model $DEFAULT_MODEL"
        echo -e "${RED}✗ Failed to download model $DEFAULT_MODEL${RESET}"
        return 1
    fi
}

# Test model functionality
test_model() {
    local model="$1"
    log "INFO" "Testing model $model"
    echo -e "${CYAN}Testing model $model...${RESET}"
    
    local test_response
    test_response=$(echo "Hello, respond with 'Model working'" | ollama run "$model" 2>/dev/null | head -1)
    
    if [[ "$test_response" == *"working"* ]]; then
        log "SUCCESS" "Model $model test passed"
        echo -e "${GREEN}✓ Model $model test passed${RESET}"
        return 0
    else
        log "WARNING" "Model $model test failed"
        echo -e "${YELLOW}⚠ Model $model test failed${RESET}"
        return 1
    fi
}

# Create configuration files
create_configuration_files() {
    log "INFO" "Creating configuration files"
    echo -e "\n${CYAN}${BOLD}Creating configuration files...${RESET}"
    
    local config_dir
    case "$SYSTEM_OS" in
        darwin)
            config_dir="$HOME/Library/Application Support/ABOV3"
            ;;
        *)
            config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/abov3"
            ;;
    esac
    
    mkdir -p "$config_dir"
    
    # Create main config file
    cat > "$config_dir/config.json" << EOF
{
  "version": "1.0.0",
  "installation_date": "$(date -Iseconds)",
  "system": "$SYSTEM_OS",
  "architecture": "$SYSTEM_ARCH",
  "python_executable": "$PYTHON_EXE",
  "install_directory": "$SCRIPT_DIR",
  "config_directory": "$config_dir",
  "ollama": {
    "enabled": true,
    "default_model": "$DEFAULT_MODEL",
    "api_endpoint": "http://localhost:11434"
  },
  "features": {
    "code_completion": true,
    "code_analysis": true,
    "documentation_generation": true,
    "test_generation": true,
    "code_optimization": true
  },
  "ui": {
    "theme": "dark",
    "font_size": 12,
    "auto_save": true
  }
}
EOF
    
    # Create environment file
    cat > "$config_dir/.env" << EOF
# ABOV3 4 Ollama Environment Configuration
ABOV3_VERSION=1.0.0
ABOV3_CONFIG_DIR=$config_dir
ABOV3_INSTALL_DIR=$SCRIPT_DIR
ABOV3_LOG_LEVEL=INFO
ABOV3_OLLAMA_ENDPOINT=http://localhost:11434
ABOV3_DEFAULT_MODEL=$DEFAULT_MODEL
ABOV3_AUTO_UPDATE=true
ABOV3_TELEMETRY=false
EOF
    
    log "SUCCESS" "Configuration files created: $config_dir"
    echo -e "${GREEN}✓ Configuration files created${RESET}"
}

# Create desktop shortcuts
create_desktop_shortcuts() {
    log "INFO" "Creating desktop shortcuts"
    echo -e "\n${CYAN}${BOLD}Creating desktop shortcuts...${RESET}"
    
    case "$SYSTEM_OS" in
        darwin)
            create_macos_shortcuts
            ;;
        *)
            create_linux_shortcuts
            ;;
    esac
}

# Create macOS shortcuts
create_macos_shortcuts() {
    local app_dir="/Applications/ABOV3 4 Ollama.app"
    local contents_dir="$app_dir/Contents"
    local macos_dir="$contents_dir/MacOS"
    local resources_dir="$contents_dir/Resources"
    
    # Create directories (requires admin for /Applications)
    if [[ -w "/Applications" ]]; then
        mkdir -p "$macos_dir" "$resources_dir"
    elif [[ -n "$SUDO_CMD" ]]; then
        $SUDO_CMD mkdir -p "$macos_dir" "$resources_dir"
    else
        log "WARNING" "Cannot create macOS app bundle without write access to /Applications"
        return 1
    fi
    
    # Create Info.plist
    local plist_content='<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>ABOV3</string>
    <key>CFBundleIdentifier</key>
    <string>com.abov3.ollama</string>
    <key>CFBundleName</key>
    <string>ABOV3 4 Ollama</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>'
    
    if [[ -w "$contents_dir" ]]; then
        echo "$plist_content" > "$contents_dir/Info.plist"
    else
        echo "$plist_content" | $SUDO_CMD tee "$contents_dir/Info.plist" >/dev/null
    fi
    
    # Create executable script
    local script_content="#!/bin/bash
cd \"$SCRIPT_DIR\"
\"$PYTHON_EXE\" -m abov3 \"\$@\""
    
    local executable_path="$macos_dir/ABOV3"
    if [[ -w "$macos_dir" ]]; then
        echo "$script_content" > "$executable_path"
        chmod +x "$executable_path"
    else
        echo "$script_content" | $SUDO_CMD tee "$executable_path" >/dev/null
        $SUDO_CMD chmod +x "$executable_path"
    fi
    
    log "SUCCESS" "macOS application created: $app_dir"
    echo -e "${GREEN}✓ macOS application created${RESET}"
}

# Create Linux shortcuts
create_linux_shortcuts() {
    local desktop_file_content="[Desktop Entry]
Name=ABOV3 4 Ollama
Comment=Revolutionary AI-Powered Coding Platform
Exec=$PYTHON_EXE -m abov3
Icon=$SCRIPT_DIR/assets/icon.png
Terminal=false
Type=Application
Categories=Development;IDE;
StartupWMClass=ABOV3"
    
    # Create in applications directory
    local apps_dir="$HOME/.local/share/applications"
    mkdir -p "$apps_dir"
    
    local desktop_file_path="$apps_dir/abov3-ollama.desktop"
    echo "$desktop_file_content" > "$desktop_file_path"
    chmod +x "$desktop_file_path"
    
    log "SUCCESS" "Linux desktop file created: $desktop_file_path"
    echo -e "${GREEN}✓ Linux desktop file created${RESET}"
    
    # Also create on desktop if it exists
    if [[ -d "$HOME/Desktop" ]]; then
        local desktop_shortcut="$HOME/Desktop/abov3-ollama.desktop"
        echo "$desktop_file_content" > "$desktop_shortcut"
        chmod +x "$desktop_shortcut"
        log "SUCCESS" "Desktop shortcut created: $desktop_shortcut"
        echo -e "${GREEN}✓ Desktop shortcut created${RESET}"
    fi
}

# Run verification tests
run_verification_tests() {
    log "INFO" "Running verification tests"
    echo -e "\n${CYAN}${BOLD}Running verification tests...${RESET}"
    
    local tests_passed=0
    local total_tests=5
    
    # Test 1: ABOV3 module import
    echo -e "${CYAN}Testing ABOV3 module import...${RESET}"
    if "$PYTHON_EXE" -c "import abov3" 2>/dev/null; then
        echo -e "${GREEN}✓ ABOV3 module import test passed${RESET}"
        log "SUCCESS" "ABOV3 module import test passed"
        ((tests_passed++))
    else
        echo -e "${RED}✗ ABOV3 module import test failed${RESET}"
        log "ERROR" "ABOV3 module import test failed"
    fi
    
    # Test 2: Configuration files
    local config_dir
    case "$SYSTEM_OS" in
        darwin)
            config_dir="$HOME/Library/Application Support/ABOV3"
            ;;
        *)
            config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/abov3"
            ;;
    esac
    
    echo -e "${CYAN}Testing configuration files...${RESET}"
    if [[ -f "$config_dir/config.json" ]]; then
        if "$PYTHON_EXE" -c "import json; json.load(open('$config_dir/config.json'))" 2>/dev/null; then
            echo -e "${GREEN}✓ Configuration file test passed${RESET}"
            log "SUCCESS" "Configuration file test passed"
            ((tests_passed++))
        else
            echo -e "${RED}✗ Configuration file test failed${RESET}"
            log "ERROR" "Configuration file test failed"
        fi
    else
        echo -e "${RED}✗ Configuration file not found${RESET}"
        log "ERROR" "Configuration file not found"
    fi
    
    # Test 3: Ollama connectivity
    echo -e "${CYAN}Testing Ollama connectivity...${RESET}"
    if ollama list >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama connectivity test passed${RESET}"
        log "SUCCESS" "Ollama connectivity test passed"
        ((tests_passed++))
    else
        echo -e "${RED}✗ Ollama connectivity test failed${RESET}"
        log "ERROR" "Ollama connectivity test failed"
    fi
    
    # Test 4: Default model availability
    echo -e "${CYAN}Testing default model availability...${RESET}"
    if ollama list 2>/dev/null | grep -q "$DEFAULT_MODEL"; then
        echo -e "${GREEN}✓ Default model availability test passed${RESET}"
        log "SUCCESS" "Default model availability test passed"
        ((tests_passed++))
    else
        echo -e "${RED}✗ Default model availability test failed${RESET}"
        log "ERROR" "Default model availability test failed"
    fi
    
    # Test 5: Command line interface
    echo -e "${CYAN}Testing command line interface...${RESET}"
    if "$PYTHON_EXE" -m abov3 --help >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Command line interface test passed${RESET}"
        log "SUCCESS" "Command line interface test passed"
        ((tests_passed++))
    else
        echo -e "${RED}✗ Command line interface test failed${RESET}"
        log "ERROR" "Command line interface test failed"
    fi
    
    # Summary
    local success_rate=$((tests_passed * 100 / total_tests))
    log "INFO" "Verification tests completed: $tests_passed/$total_tests passed ($success_rate%)"
    echo -e "\n${CYAN}Verification Summary: $tests_passed/$total_tests tests passed ($success_rate%)${RESET}"
    
    if [[ $tests_passed -ge $((total_tests * 80 / 100)) ]]; then
        echo -e "${GREEN}✓ Verification tests PASSED${RESET}"
        return 0
    else
        echo -e "${YELLOW}⚠ Some verification tests failed${RESET}"
        return 1
    fi
}

# Print success message
print_success_message() {
    cat << 'EOF'

                    ╔══════════════════════════════════════════════════════════════════╗
                    ║                    INSTALLATION SUCCESSFUL!                     ║
                    ║                                                                  ║
                    ║  🎉 ABOV3 4 Ollama has been installed successfully!             ║
                    ║                                                                  ║
                    ║  Quick Start:                                                    ║
                    ║  • Run 'abov3 --help' to see available commands                 ║
                    ║  • Run 'abov3 serve' to start the development server            ║
                    ║  • Run 'abov3 chat' to start an interactive session             ║
                    ║                                                                  ║
                    ║  Documentation: https://docs.abov3.com                          ║
                    ║  Support: support@abov3.com                                     ║
                    ║                                                                  ║
                    ╚══════════════════════════════════════════════════════════════════╝

EOF

    echo -e "${CYAN}Next Steps:${RESET}"
    echo -e "1. ${YELLOW}Restart your terminal${RESET} or run ${GREEN}source ~/.bashrc${RESET} to ensure PATH changes take effect"
    echo -e "2. ${YELLOW}Check firewall settings${RESET} if you experience connectivity issues"
    
    local config_dir
    case "$SYSTEM_OS" in
        darwin)
            config_dir="$HOME/Library/Application Support/ABOV3"
            ;;
        *)
            config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/abov3"
            ;;
    esac
    
    echo -e "3. ${YELLOW}Review configuration${RESET} in $config_dir/config.json"
    echo ""
    
    read -p "Open documentation in browser? (Y/n): " -r
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        case "$SYSTEM_OS" in
            darwin) open https://docs.abov3.com ;;
            linux) xdg-open https://docs.abov3.com 2>/dev/null || echo "Visit https://docs.abov3.com" ;;
            *) echo "Visit https://docs.abov3.com" ;;
        esac
    fi
    
    read -p "Start ABOV3 now? (Y/n): " -r
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo -e "${CYAN}Starting ABOV3...${RESET}"
        "$PYTHON_EXE" -m abov3 --help
        echo ""
        echo -e "${GREEN}Type 'abov3 serve' to start the server${RESET}"
    fi
}

# Print failure message
print_failure_message() {
    cat << EOF

                    ╔══════════════════════════════════════════════════════════════════╗
                    ║                    INSTALLATION FAILED!                         ║
                    ║                                                                  ║
                    ║  ❌ ABOV3 4 Ollama installation encountered errors.              ║
                    ║                                                                  ║
                    ║  Please check the log file for details:                         ║
                    ║  $LOG_FILE                           ║
                    ║                                                                  ║
                    ║  Common Solutions:                                               ║
                    ║  • Ensure you have Python 3.8+ installed                       ║
                    ║  • Check your internet connection                               ║
                    ║  • Install missing system dependencies                          ║
                    ║  • Ensure sufficient disk space (2GB+ required)                 ║
                    ║                                                                  ║
                    ║  For support, visit: https://docs.abov3.com/troubleshooting     ║
                    ║  Or contact: support@abov3.com                                  ║
                    ║                                                                  ║
                    ╚══════════════════════════════════════════════════════════════════╝

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-ollama)
                SKIP_OLLAMA=1
                shift
                ;;
            --use-venv)
                USE_VENV=1
                shift
                ;;
            --model)
                DEFAULT_MODEL="$2"
                shift 2
                ;;
            --prefix)
                INSTALL_PREFIX="$2"
                shift 2
                ;;
            --no-shortcuts)
                NO_SHORTCUTS=1
                shift
                ;;
            --verbose|-v)
                set -x
                shift
                ;;
            --help|-h)
                cat << 'EOF'
ABOV3 4 Ollama Unix/Linux Installation Script

Usage: ./install_abov3.sh [OPTIONS]

Options:
  --skip-ollama        Skip Ollama installation
  --use-venv           Install in a Python virtual environment
  --model MODEL        Default model to install (default: llama3.2)
  --prefix PREFIX      Installation prefix (default: ~/.local)
  --no-shortcuts       Skip creating desktop shortcuts
  --verbose, -v        Enable verbose output
  --help, -h           Show this help message

Examples:
  ./install_abov3.sh                    # Standard installation
  ./install_abov3.sh --skip-ollama      # Skip Ollama installation
  ./install_abov3.sh --use-venv         # Install in virtual environment
  ./install_abov3.sh --model codellama  # Use different default model

EOF
                exit 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                echo "Use --help for usage information" >&2
                exit 1
                ;;
        esac
    done
}

# Main installation function
main() {
    # Initialize log
    {
        echo "ABOV3 4 Ollama Unix/Linux Installation Log"
        echo "=========================================="
        echo "Start Time: $(date)"
        echo "Script Location: $SCRIPT_DIR"
        echo "User: $USER"
        echo "System: $(uname -a)"
        echo ""
    } > "$LOG_FILE"
    
    log "INFO" "Installation started by $USER"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Run installation steps
    print_banner
    detect_system
    check_system_requirements
    detect_python
    setup_python_environment
    install_python_dependencies
    install_abov3_development
    setup_path_environment
    
    # Ollama installation
    if [[ "${SKIP_OLLAMA:-0}" -eq 0 ]]; then
        if ! check_ollama_installation; then
            install_ollama
        fi
        setup_default_model
    else
        log "INFO" "Skipping Ollama installation as requested"
        echo -e "${YELLOW}Skipping Ollama installation as requested${RESET}"
    fi
    
    create_configuration_files
    
    if [[ "${NO_SHORTCUTS:-0}" -eq 0 ]]; then
        create_desktop_shortcuts
    else
        log "INFO" "Skipping desktop shortcuts as requested"
        echo -e "${YELLOW}Skipping desktop shortcuts as requested${RESET}"
    fi
    
    # Verification
    if run_verification_tests; then
        INSTALL_SUCCESS=1
        print_success_message
        log "SUCCESS" "Installation completed successfully"
    else
        print_failure_message
        log "ERROR" "Installation completed with errors"
        exit 1
    fi
    
    echo -e "\n${CYAN}Installation log saved to: $LOG_FILE${RESET}"
}

# Execute main function with all arguments
main "$@"