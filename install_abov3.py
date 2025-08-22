#!/usr/bin/env python3
"""
ABOV3 4 Ollama - Automated Installation Script
==============================================

This script automatically installs and configures ABOV3 4 Ollama on your system.
It handles dependencies, Ollama setup, model installation, and system integration.

Features:
- Cross-platform compatibility (Windows, macOS, Linux)
- Automatic dependency resolution
- Ollama installation and configuration
- Model downloading and setup
- Desktop shortcuts and start menu entries
- Comprehensive error handling and recovery
- Installation verification and testing

Author: ABOV3 Enterprise DevOps Agent
Version: 1.0.0
"""

import os
import sys
import subprocess
import platform
import json
import shutil
import urllib.request
import urllib.error
import zipfile
import tarfile
import tempfile
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Version requirements
MIN_PYTHON_VERSION = (3, 8)
OLLAMA_VERSION_REQUIREMENT = "0.1.0"
DEFAULT_MODEL = "llama3.2"

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class ProgressBar:
    """Simple progress bar for terminal output"""
    
    def __init__(self, total: int, description: str = ""):
        self.total = total
        self.current = 0
        self.description = description
        self.width = 50
    
    def update(self, amount: int = 1):
        self.current += amount
        self._render()
    
    def _render(self):
        percent = (self.current / self.total) * 100
        filled = int(self.width * self.current / self.total)
        bar = '█' * filled + '-' * (self.width - filled)
        print(f'\r{self.description} |{bar}| {percent:.1f}%', end='', flush=True)
        if self.current >= self.total:
            print()

class ABOV3Installer:
    """Main installer class for ABOV3 4 Ollama"""
    
    def __init__(self, args):
        self.args = args
        self.system = platform.system().lower()
        self.architecture = platform.machine().lower()
        self.python_executable = sys.executable
        self.install_dir = Path(__file__).parent.absolute()
        self.home_dir = Path.home()
        self.config_dir = self._get_config_dir()
        self.log_file = self.config_dir / "abov3_install.log"
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.log_entries = []
        self.log(f"ABOV3 4 Ollama Installation Started - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"System: {self.system} {self.architecture}")
        self.log(f"Python: {sys.version}")
        self.log(f"Install directory: {self.install_dir}")
        
    def log(self, message: str, level: str = "INFO"):
        """Log message to both console and file"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_entries.append(log_entry)
        
        # Write to log file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception:
            pass
        
        # Print to console with colors
        if level == "ERROR":
            print(f"{Colors.RED}{log_entry}{Colors.END}")
        elif level == "WARNING":
            print(f"{Colors.YELLOW}{log_entry}{Colors.END}")
        elif level == "SUCCESS":
            print(f"{Colors.GREEN}{log_entry}{Colors.END}")
        elif level == "INFO":
            print(f"{Colors.CYAN}{log_entry}{Colors.END}")
        else:
            print(log_entry)
    
    def _get_config_dir(self) -> Path:
        """Get the appropriate configuration directory for the OS"""
        if self.system == "windows":
            return Path(os.environ.get('APPDATA', self.home_dir / 'AppData/Roaming')) / 'ABOV3'
        elif self.system == "darwin":
            return self.home_dir / 'Library/Application Support/ABOV3'
        else:
            return Path(os.environ.get('XDG_CONFIG_HOME', self.home_dir / '.config')) / 'abov3'
    
    def print_banner(self):
        """Print the installation banner"""
        banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════╗
║                    ABOV3 4 OLLAMA INSTALLER                     ║
║                                                                  ║
║  Revolutionary AI-Powered Coding Platform                       ║
║  Enterprise-Grade Infrastructure & Deployment                   ║
║                                                                  ║
║  Version: 1.0.0                                                 ║
║  Target: {self.system.title()} {self.architecture.upper()}                                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
{Colors.END}
"""
        print(banner)
    
    def check_system_requirements(self) -> bool:
        """Check if system meets minimum requirements"""
        self.log("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < MIN_PYTHON_VERSION:
            self.log(f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ is required. "
                    f"Current version: {sys.version_info[0]}.{sys.version_info[1]}", "ERROR")
            return False
        
        self.log(f"Python version check passed: {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}", "SUCCESS")
        
        # Check available disk space (minimum 2GB)
        try:
            if self.system == "windows":
                import shutil
                free_space = shutil.disk_usage(self.install_dir)[2]
            else:
                statvfs = os.statvfs(self.install_dir)
                free_space = statvfs.f_frsize * statvfs.f_bavail
            
            required_space = 2 * 1024 * 1024 * 1024  # 2GB
            if free_space < required_space:
                self.log(f"Insufficient disk space. Required: 2GB, Available: {free_space / (1024**3):.1f}GB", "ERROR")
                return False
            
            self.log(f"Disk space check passed: {free_space / (1024**3):.1f}GB available", "SUCCESS")
        
        except Exception as e:
            self.log(f"Could not check disk space: {e}", "WARNING")
        
        # Check network connectivity
        try:
            urllib.request.urlopen('https://www.google.com', timeout=10)
            self.log("Network connectivity check passed", "SUCCESS")
        except Exception:
            self.log("Network connectivity check failed - some features may not work", "WARNING")
        
        return True
    
    def install_pip_dependencies(self) -> bool:
        """Install Python dependencies using pip"""
        self.log("Installing Python dependencies...")
        
        dependencies = [
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
        ]
        
        progress = ProgressBar(len(dependencies), "Installing dependencies")
        
        for dep in dependencies:
            try:
                result = subprocess.run([
                    self.python_executable, "-m", "pip", "install", 
                    "--upgrade", dep
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    self.log(f"Failed to install {dep}: {result.stderr}", "ERROR")
                    return False
                
                progress.update()
                
            except subprocess.TimeoutExpired:
                self.log(f"Timeout installing {dep}", "ERROR")
                return False
            except Exception as e:
                self.log(f"Error installing {dep}: {e}", "ERROR")
                return False
        
        self.log("All Python dependencies installed successfully", "SUCCESS")
        return True
    
    def install_abov3_development_mode(self) -> bool:
        """Install ABOV3 in development mode"""
        self.log("Installing ABOV3 in development mode...")
        
        try:
            # Check if setup.py or pyproject.toml exists
            setup_py = self.install_dir / "setup.py"
            pyproject_toml = self.install_dir / "pyproject.toml"
            
            if not setup_py.exists() and not pyproject_toml.exists():
                self.log("Creating setup.py for development installation...", "INFO")
                self.create_setup_py()
            
            # Install in editable mode
            result = subprocess.run([
                self.python_executable, "-m", "pip", "install", "-e", str(self.install_dir)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.log(f"Failed to install ABOV3 in development mode: {result.stderr}", "ERROR")
                return False
            
            self.log("ABOV3 installed in development mode successfully", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Error installing ABOV3: {e}", "ERROR")
            return False
    
    def create_setup_py(self):
        """Create a setup.py file if it doesn't exist"""
        setup_content = '''
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
    project_urls={
        "Bug Reports": "https://github.com/abov3/abov3-ollama/issues",
        "Source": "https://github.com/abov3/abov3-ollama",
        "Documentation": "https://docs.abov3.com",
    }
)
'''.strip()
        
        setup_py_path = self.install_dir / "setup.py"
        with open(setup_py_path, 'w', encoding='utf-8') as f:
            f.write(setup_content)
        
        self.log(f"Created setup.py at {setup_py_path}", "SUCCESS")
    
    def setup_path_environment(self) -> bool:
        """Setup PATH environment for ABOV3 commands"""
        self.log("Setting up PATH environment...")
        
        try:
            # Get the Scripts/bin directory where pip installs executables
            if self.system == "windows":
                scripts_dir = Path(self.python_executable).parent / "Scripts"
            else:
                scripts_dir = Path(self.python_executable).parent
            
            if self.system == "windows":
                # Windows PATH setup
                self.setup_windows_path(str(scripts_dir))
            else:
                # Unix-like PATH setup
                self.setup_unix_path(str(scripts_dir))
            
            self.log("PATH environment setup completed", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Error setting up PATH: {e}", "ERROR")
            return False
    
    def setup_windows_path(self, scripts_dir: str):
        """Setup PATH on Windows"""
        try:
            import winreg
            
            # Add to user PATH
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS) as key:
                try:
                    current_path, _ = winreg.QueryValueEx(key, "PATH")
                except FileNotFoundError:
                    current_path = ""
                
                if scripts_dir not in current_path:
                    new_path = f"{current_path};{scripts_dir}" if current_path else scripts_dir
                    winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
                    self.log(f"Added {scripts_dir} to user PATH", "SUCCESS")
                else:
                    self.log("Scripts directory already in PATH", "INFO")
                    
        except ImportError:
            self.log("Could not modify Windows PATH - winreg not available", "WARNING")
        except Exception as e:
            self.log(f"Error modifying Windows PATH: {e}", "WARNING")
    
    def setup_unix_path(self, scripts_dir: str):
        """Setup PATH on Unix-like systems"""
        try:
            # Add to shell profile
            shell_profiles = [".bashrc", ".zshrc", ".profile"]
            
            for profile in shell_profiles:
                profile_path = self.home_dir / profile
                if profile_path.exists():
                    with open(profile_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    path_export = f'export PATH="{scripts_dir}:$PATH"'
                    if path_export not in content:
                        with open(profile_path, 'a', encoding='utf-8') as f:
                            f.write(f'\n# Added by ABOV3 installer\n{path_export}\n')
                        self.log(f"Added PATH export to {profile}", "SUCCESS")
                    
        except Exception as e:
            self.log(f"Error modifying Unix PATH: {e}", "WARNING")
    
    def check_ollama_installation(self) -> bool:
        """Check if Ollama is installed and working"""
        self.log("Checking Ollama installation...")
        
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                self.log(f"Ollama found: {version}", "SUCCESS")
                return True
            else:
                self.log("Ollama not found or not working properly", "WARNING")
                return False
                
        except FileNotFoundError:
            self.log("Ollama not found in PATH", "WARNING")
            return False
        except subprocess.TimeoutExpired:
            self.log("Ollama check timed out", "WARNING")
            return False
        except Exception as e:
            self.log(f"Error checking Ollama: {e}", "WARNING")
            return False
    
    def install_ollama(self) -> bool:
        """Install Ollama automatically"""
        self.log("Installing Ollama...")
        
        if self.args.skip_ollama:
            self.log("Skipping Ollama installation as requested", "INFO")
            return True
        
        try:
            if self.system == "windows":
                return self.install_ollama_windows()
            elif self.system == "darwin":
                return self.install_ollama_macos()
            else:
                return self.install_ollama_linux()
                
        except Exception as e:
            self.log(f"Error installing Ollama: {e}", "ERROR")
            return False
    
    def install_ollama_windows(self) -> bool:
        """Install Ollama on Windows"""
        try:
            self.log("Downloading Ollama for Windows...")
            
            # Download Ollama installer
            download_url = "https://ollama.com/download/ollama-windows-amd64.exe"
            installer_path = self.config_dir / "ollama-installer.exe"
            
            urllib.request.urlretrieve(download_url, installer_path)
            self.log(f"Downloaded Ollama installer to {installer_path}", "SUCCESS")
            
            # Run installer silently
            result = subprocess.run([str(installer_path), "/S"], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.log("Ollama installed successfully", "SUCCESS")
                
                # Clean up installer
                installer_path.unlink()
                
                # Start Ollama service
                time.sleep(5)  # Wait for installation to complete
                subprocess.run(["ollama", "serve"], timeout=5)
                
                return True
            else:
                self.log(f"Ollama installation failed: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error installing Ollama on Windows: {e}", "ERROR")
            return False
    
    def install_ollama_macos(self) -> bool:
        """Install Ollama on macOS"""
        try:
            self.log("Installing Ollama on macOS...")
            
            # Use Homebrew if available
            try:
                subprocess.run(["brew", "--version"], capture_output=True, check=True)
                result = subprocess.run(["brew", "install", "ollama"], 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.log("Ollama installed via Homebrew", "SUCCESS")
                    return True
                    
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
            
            # Fallback to direct download
            self.log("Installing Ollama via direct download...")
            download_url = "https://ollama.com/download/ollama-darwin"
            
            result = subprocess.run([
                "curl", "-fsSL", download_url, "-o", "/usr/local/bin/ollama"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                subprocess.run(["chmod", "+x", "/usr/local/bin/ollama"])
                self.log("Ollama installed successfully", "SUCCESS")
                return True
            else:
                self.log(f"Failed to download Ollama: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error installing Ollama on macOS: {e}", "ERROR")
            return False
    
    def install_ollama_linux(self) -> bool:
        """Install Ollama on Linux"""
        try:
            self.log("Installing Ollama on Linux...")
            
            # Use the official installation script
            result = subprocess.run([
                "curl", "-fsSL", "https://ollama.com/install.sh"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                install_script = result.stdout
                
                # Execute the installation script
                process = subprocess.Popen([
                    "bash", "-c", install_script
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                stdout, stderr = process.communicate(timeout=300)
                
                if process.returncode == 0:
                    self.log("Ollama installed successfully", "SUCCESS")
                    return True
                else:
                    self.log(f"Ollama installation failed: {stderr}", "ERROR")
                    return False
            else:
                self.log(f"Failed to download Ollama installer: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error installing Ollama on Linux: {e}", "ERROR")
            return False
    
    def setup_default_model(self) -> bool:
        """Download and setup the default model"""
        self.log(f"Setting up default model: {DEFAULT_MODEL}...")
        
        try:
            # Start Ollama service if not running
            self.start_ollama_service()
            
            # Check if model is already available
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and DEFAULT_MODEL in result.stdout:
                self.log(f"Model {DEFAULT_MODEL} already available", "SUCCESS")
                return True
            
            # Download the model
            self.log(f"Downloading {DEFAULT_MODEL} model... This may take a while.")
            
            process = subprocess.Popen([
                "ollama", "pull", DEFAULT_MODEL
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
               text=True, universal_newlines=True)
            
            # Show progress
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Parse Ollama progress output
                    if "pulling" in output.lower() or "%" in output:
                        print(f"\r{Colors.YELLOW}Downloading: {output.strip()}{Colors.END}", end='', flush=True)
            
            print()  # New line after progress
            
            if process.returncode == 0:
                self.log(f"Model {DEFAULT_MODEL} downloaded successfully", "SUCCESS")
                
                # Test the model
                return self.test_model(DEFAULT_MODEL)
            else:
                self.log(f"Failed to download model {DEFAULT_MODEL}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error setting up model: {e}", "ERROR")
            return False
    
    def start_ollama_service(self):
        """Start the Ollama service"""
        try:
            # Check if Ollama is already running
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                return  # Already running
            
            # Start Ollama service in background
            if self.system == "windows":
                subprocess.Popen(["ollama", "serve"], 
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                subprocess.Popen(["ollama", "serve"], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            
            # Wait for service to start
            time.sleep(3)
            self.log("Ollama service started", "SUCCESS")
            
        except Exception as e:
            self.log(f"Error starting Ollama service: {e}", "WARNING")
    
    def test_model(self, model: str) -> bool:
        """Test if a model is working correctly"""
        try:
            self.log(f"Testing model {model}...")
            
            result = subprocess.run([
                "ollama", "run", model, "Hello, respond with 'Model working'"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and "working" in result.stdout.lower():
                self.log(f"Model {model} test passed", "SUCCESS")
                return True
            else:
                self.log(f"Model {model} test failed", "WARNING")
                return False
                
        except Exception as e:
            self.log(f"Error testing model: {e}", "WARNING")
            return False
    
    def create_configuration_files(self) -> bool:
        """Create necessary configuration files"""
        self.log("Creating configuration files...")
        
        try:
            # Create main config file
            config = {
                "version": "1.0.0",
                "installation_date": time.strftime('%Y-%m-%d %H:%M:%S'),
                "system": self.system,
                "architecture": self.architecture,
                "python_executable": str(self.python_executable),
                "install_directory": str(self.install_dir),
                "config_directory": str(self.config_dir),
                "ollama": {
                    "enabled": True,
                    "default_model": DEFAULT_MODEL,
                    "api_endpoint": "http://localhost:11434"
                },
                "features": {
                    "code_completion": True,
                    "code_analysis": True,
                    "documentation_generation": True,
                    "test_generation": True,
                    "code_optimization": True
                },
                "ui": {
                    "theme": "dark",
                    "font_size": 12,
                    "auto_save": True
                }
            }
            
            config_file = self.config_dir / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            self.log(f"Configuration file created: {config_file}", "SUCCESS")
            
            # Create environment file
            env_content = f"""# ABOV3 4 Ollama Environment Configuration
ABOV3_VERSION=1.0.0
ABOV3_CONFIG_DIR={self.config_dir}
ABOV3_INSTALL_DIR={self.install_dir}
ABOV3_LOG_LEVEL=INFO
ABOV3_OLLAMA_ENDPOINT=http://localhost:11434
ABOV3_DEFAULT_MODEL={DEFAULT_MODEL}
ABOV3_AUTO_UPDATE=true
ABOV3_TELEMETRY=false
"""
            
            env_file = self.config_dir / ".env"
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)
            
            self.log(f"Environment file created: {env_file}", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log(f"Error creating configuration files: {e}", "ERROR")
            return False
    
    def create_desktop_shortcuts(self) -> bool:
        """Create desktop shortcuts and start menu entries"""
        self.log("Creating desktop shortcuts...")
        
        try:
            if self.system == "windows":
                return self.create_windows_shortcuts()
            elif self.system == "darwin":
                return self.create_macos_shortcuts()
            else:
                return self.create_linux_shortcuts()
                
        except Exception as e:
            self.log(f"Error creating shortcuts: {e}", "ERROR")
            return False
    
    def create_windows_shortcuts(self) -> bool:
        """Create Windows shortcuts"""
        try:
            import win32com.client
            
            shell = win32com.client.Dispatch("WScript.Shell")
            
            # Desktop shortcut
            desktop = shell.SpecialFolders("Desktop")
            shortcut_path = os.path.join(desktop, "ABOV3 4 Ollama.lnk")
            shortcut = shell.CreateShortcut(shortcut_path)
            shortcut.Targetpath = self.python_executable
            shortcut.Arguments = f'-m abov3'
            shortcut.WorkingDirectory = str(self.install_dir)
            shortcut.IconLocation = str(self.install_dir / "assets" / "icon.ico") if (self.install_dir / "assets" / "icon.ico").exists() else ""
            shortcut.Description = "ABOV3 4 Ollama - Revolutionary AI-Powered Coding Platform"
            shortcut.save()
            
            self.log(f"Desktop shortcut created: {shortcut_path}", "SUCCESS")
            
            # Start menu shortcut
            start_menu = shell.SpecialFolders("StartMenu")
            programs = os.path.join(start_menu, "Programs", "ABOV3")
            os.makedirs(programs, exist_ok=True)
            
            start_shortcut_path = os.path.join(programs, "ABOV3 4 Ollama.lnk")
            start_shortcut = shell.CreateShortcut(start_shortcut_path)
            start_shortcut.Targetpath = shortcut.Targetpath
            start_shortcut.Arguments = shortcut.Arguments
            start_shortcut.WorkingDirectory = shortcut.WorkingDirectory
            start_shortcut.IconLocation = shortcut.IconLocation
            start_shortcut.Description = shortcut.Description
            start_shortcut.save()
            
            self.log(f"Start menu shortcut created: {start_shortcut_path}", "SUCCESS")
            
            return True
            
        except ImportError:
            self.log("Could not create Windows shortcuts - pywin32 not available", "WARNING")
            return True  # Non-critical failure
        except Exception as e:
            self.log(f"Error creating Windows shortcuts: {e}", "WARNING")
            return True  # Non-critical failure
    
    def create_macos_shortcuts(self) -> bool:
        """Create macOS shortcuts"""
        try:
            # Create application bundle
            app_dir = Path("/Applications/ABOV3 4 Ollama.app")
            contents_dir = app_dir / "Contents"
            macos_dir = contents_dir / "MacOS"
            resources_dir = contents_dir / "Resources"
            
            # Create directories
            macos_dir.mkdir(parents=True, exist_ok=True)
            resources_dir.mkdir(parents=True, exist_ok=True)
            
            # Create Info.plist
            plist_content = f'''<?xml version="1.0" encoding="UTF-8"?>
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
</plist>'''
            
            with open(contents_dir / "Info.plist", 'w') as f:
                f.write(plist_content)
            
            # Create executable script
            script_content = f'''#!/bin/bash
cd "{self.install_dir}"
"{self.python_executable}" -m abov3 "$@"
'''
            
            executable_path = macos_dir / "ABOV3"
            with open(executable_path, 'w') as f:
                f.write(script_content)
            
            # Make executable
            os.chmod(executable_path, 0o755)
            
            self.log(f"macOS application created: {app_dir}", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log(f"Error creating macOS shortcuts: {e}", "WARNING")
            return True  # Non-critical failure
    
    def create_linux_shortcuts(self) -> bool:
        """Create Linux desktop shortcuts"""
        try:
            # Create .desktop file
            desktop_file_content = f'''[Desktop Entry]
Name=ABOV3 4 Ollama
Comment=Revolutionary AI-Powered Coding Platform
Exec={self.python_executable} -m abov3
Icon={self.install_dir}/assets/icon.png
Terminal=false
Type=Application
Categories=Development;IDE;
StartupWMClass=ABOV3
'''
            
            # Create in applications directory
            apps_dir = self.home_dir / ".local/share/applications"
            apps_dir.mkdir(parents=True, exist_ok=True)
            
            desktop_file_path = apps_dir / "abov3-ollama.desktop"
            with open(desktop_file_path, 'w') as f:
                f.write(desktop_file_content)
            
            # Make executable
            os.chmod(desktop_file_path, 0o755)
            
            self.log(f"Linux desktop file created: {desktop_file_path}", "SUCCESS")
            
            # Also create on desktop if it exists
            desktop_dir = self.home_dir / "Desktop"
            if desktop_dir.exists():
                desktop_shortcut = desktop_dir / "abov3-ollama.desktop"
                with open(desktop_shortcut, 'w') as f:
                    f.write(desktop_file_content)
                os.chmod(desktop_shortcut, 0o755)
                
                self.log(f"Desktop shortcut created: {desktop_shortcut}", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log(f"Error creating Linux shortcuts: {e}", "WARNING")
            return True  # Non-critical failure
    
    def run_verification_tests(self) -> bool:
        """Run comprehensive verification tests"""
        self.log("Running verification tests...")
        
        tests_passed = 0
        total_tests = 5
        
        # Test 1: Python imports
        try:
            import abov3
            self.log("✓ ABOV3 module import test passed", "SUCCESS")
            tests_passed += 1
        except Exception as e:
            self.log(f"✗ ABOV3 module import test failed: {e}", "ERROR")
        
        # Test 2: Configuration files
        config_file = self.config_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    json.load(f)
                self.log("✓ Configuration file test passed", "SUCCESS")
                tests_passed += 1
            except Exception as e:
                self.log(f"✗ Configuration file test failed: {e}", "ERROR")
        else:
            self.log("✗ Configuration file not found", "ERROR")
        
        # Test 3: Ollama connectivity
        try:
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.log("✓ Ollama connectivity test passed", "SUCCESS")
                tests_passed += 1
            else:
                self.log("✗ Ollama connectivity test failed", "ERROR")
        except Exception as e:
            self.log(f"✗ Ollama connectivity test failed: {e}", "ERROR")
        
        # Test 4: Default model availability
        try:
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and DEFAULT_MODEL in result.stdout:
                self.log("✓ Default model availability test passed", "SUCCESS")
                tests_passed += 1
            else:
                self.log("✗ Default model availability test failed", "ERROR")
        except Exception as e:
            self.log(f"✗ Default model availability test failed: {e}", "ERROR")
        
        # Test 5: Command line interface
        try:
            result = subprocess.run([self.python_executable, "-m", "abov3", "--help"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.log("✓ Command line interface test passed", "SUCCESS")
                tests_passed += 1
            else:
                self.log("✗ Command line interface test failed", "ERROR")
        except Exception as e:
            self.log(f"✗ Command line interface test failed: {e}", "ERROR")
        
        success_rate = (tests_passed / total_tests) * 100
        self.log(f"Verification tests completed: {tests_passed}/{total_tests} passed ({success_rate:.1f}%)", 
                "SUCCESS" if tests_passed >= total_tests * 0.8 else "WARNING")
        
        return tests_passed >= total_tests * 0.8  # 80% pass rate required
    
    def handle_common_issues(self) -> bool:
        """Handle and fix common installation issues"""
        self.log("Checking for common issues...")
        
        issues_fixed = 0
        
        # Issue 1: Missing dependencies
        try:
            missing_deps = self.check_missing_dependencies()
            if missing_deps:
                self.log(f"Found missing dependencies: {missing_deps}", "WARNING")
                if self.install_missing_dependencies(missing_deps):
                    issues_fixed += 1
        except Exception as e:
            self.log(f"Error checking dependencies: {e}", "WARNING")
        
        # Issue 2: Permission problems
        try:
            if not os.access(self.config_dir, os.W_OK):
                self.log("Configuration directory not writable", "WARNING")
                self.fix_permissions()
                issues_fixed += 1
        except Exception as e:
            self.log(f"Error checking permissions: {e}", "WARNING")
        
        # Issue 3: Port conflicts
        try:
            if not self.check_port_availability(11434):
                self.log("Ollama port 11434 may be in use", "WARNING")
                self.fix_port_conflict()
                issues_fixed += 1
        except Exception as e:
            self.log(f"Error checking ports: {e}", "WARNING")
        
        # Issue 4: Firewall/Antivirus interference
        try:
            self.check_security_software()
        except Exception as e:
            self.log(f"Error checking security software: {e}", "WARNING")
        
        if issues_fixed > 0:
            self.log(f"Fixed {issues_fixed} common issues", "SUCCESS")
        else:
            self.log("No common issues detected", "SUCCESS")
        
        return True
    
    def check_missing_dependencies(self) -> List[str]:
        """Check for missing Python dependencies"""
        required_modules = [
            "requests", "click", "colorama", "tqdm", "yaml", 
            "dotenv", "psutil", "rich", "httpx", "websockets",
            "fastapi", "uvicorn", "pydantic", "numpy", "packaging"
        ]
        
        missing = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        
        return missing
    
    def install_missing_dependencies(self, deps: List[str]) -> bool:
        """Install missing dependencies"""
        try:
            for dep in deps:
                result = subprocess.run([
                    self.python_executable, "-m", "pip", "install", dep
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode != 0:
                    self.log(f"Failed to install {dep}", "ERROR")
                    return False
            
            self.log("Missing dependencies installed", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Error installing dependencies: {e}", "ERROR")
            return False
    
    def fix_permissions(self):
        """Fix permission issues"""
        try:
            if self.system != "windows":
                os.chmod(self.config_dir, 0o755)
                for root, dirs, files in os.walk(self.config_dir):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o755)
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o644)
            
            self.log("Permissions fixed", "SUCCESS")
            
        except Exception as e:
            self.log(f"Error fixing permissions: {e}", "WARNING")
    
    def check_port_availability(self, port: int) -> bool:
        """Check if a port is available"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    def fix_port_conflict(self):
        """Handle port conflicts"""
        self.log("Port 11434 appears to be in use. This might be another Ollama instance.", "WARNING")
        self.log("Please stop any running Ollama instances or check for conflicting services.", "INFO")
    
    def check_security_software(self):
        """Check for potential security software interference"""
        if self.system == "windows":
            self.log("If you experience issues, check Windows Defender or antivirus software.", "INFO")
            self.log("You may need to add exceptions for ABOV3 and Ollama executables.", "INFO")
    
    def print_success_message(self):
        """Print installation success message"""
        success_message = f"""
{Colors.GREEN}{Colors.BOLD}
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
║  Configuration:                                                  ║
║  • Config file: {str(self.config_dir / 'config.json'):<44} ║
║  • Log file: {str(self.log_file):<47} ║
║                                                                  ║
║  Default Model: {DEFAULT_MODEL:<47} ║
║  Ollama Endpoint: http://localhost:11434                        ║
║                                                                  ║
║  Documentation: https://docs.abov3.com                          ║
║  Support: support@abov3.com                                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
{Colors.END}
"""
        print(success_message)
    
    def print_failure_message(self, error_details: str = ""):
        """Print installation failure message"""
        failure_message = f"""
{Colors.RED}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════╗
║                    INSTALLATION FAILED!                         ║
║                                                                  ║
║  ❌ ABOV3 4 Ollama installation encountered errors.              ║
║                                                                  ║
║  Please check the log file for details:                         ║
║  {str(self.log_file):<58} ║
║                                                                  ║
║  Common Solutions:                                               ║
║  • Ensure you have Python 3.8+ installed                       ║
║  • Check your internet connection                               ║
║  • Run the installer as administrator (Windows)                 ║
║  • Ensure sufficient disk space (2GB+ required)                 ║
║                                                                  ║
║  For support, visit: https://docs.abov3.com/troubleshooting     ║
║  Or contact: support@abov3.com                                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
{Colors.END}
"""
        print(failure_message)
        
        if error_details:
            print(f"\n{Colors.YELLOW}Error Details:{Colors.END}")
            print(error_details)
    
    def cleanup_on_failure(self):
        """Cleanup partial installation on failure"""
        self.log("Cleaning up partial installation...", "INFO")
        
        try:
            # Remove any created shortcuts
            if self.system == "windows":
                try:
                    desktop = Path.home() / "Desktop" / "ABOV3 4 Ollama.lnk"
                    if desktop.exists():
                        desktop.unlink()
                except Exception:
                    pass
            
            # Don't remove config directory as it contains logs
            self.log("Cleanup completed", "INFO")
            
        except Exception as e:
            self.log(f"Error during cleanup: {e}", "WARNING")
    
    def run_installation(self) -> bool:
        """Run the complete installation process"""
        try:
            self.print_banner()
            
            # Installation steps
            steps = [
                ("Checking system requirements", self.check_system_requirements),
                ("Installing Python dependencies", self.install_pip_dependencies),
                ("Installing ABOV3 in development mode", self.install_abov3_development_mode),
                ("Setting up PATH environment", self.setup_path_environment),
                ("Checking/Installing Ollama", lambda: self.check_ollama_installation() or self.install_ollama()),
                ("Setting up default model", self.setup_default_model),
                ("Creating configuration files", self.create_configuration_files),
                ("Creating desktop shortcuts", self.create_desktop_shortcuts),
                ("Handling common issues", self.handle_common_issues),
                ("Running verification tests", self.run_verification_tests)
            ]
            
            failed_steps = []
            
            for i, (description, step_func) in enumerate(steps, 1):
                self.log(f"Step {i}/{len(steps)}: {description}", "INFO")
                
                try:
                    if not step_func():
                        failed_steps.append(description)
                        if not self.args.continue_on_error:
                            break
                except Exception as e:
                    self.log(f"Step failed with exception: {e}", "ERROR")
                    failed_steps.append(description)
                    if not self.args.continue_on_error:
                        break
                
                print()  # Add spacing between steps
            
            # Determine success
            critical_steps = [
                "Checking system requirements",
                "Installing Python dependencies", 
                "Installing ABOV3 in development mode"
            ]
            
            critical_failures = [step for step in failed_steps if step in critical_steps]
            
            if critical_failures:
                self.log(f"Critical installation steps failed: {critical_failures}", "ERROR")
                self.cleanup_on_failure()
                self.print_failure_message(f"Critical steps failed: {', '.join(critical_failures)}")
                return False
            
            if failed_steps:
                self.log(f"Some optional steps failed: {failed_steps}", "WARNING")
                self.log("Installation completed with warnings", "WARNING")
            else:
                self.log("Installation completed successfully!", "SUCCESS")
            
            self.print_success_message()
            return True
            
        except KeyboardInterrupt:
            self.log("Installation cancelled by user", "WARNING")
            self.cleanup_on_failure()
            return False
        except Exception as e:
            self.log(f"Unexpected error during installation: {e}", "ERROR")
            self.cleanup_on_failure()
            self.print_failure_message(str(e))
            return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ABOV3 4 Ollama - Automated Installation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install_abov3.py                    # Standard installation
  python install_abov3.py --skip-ollama      # Skip Ollama installation
  python install_abov3.py --continue-on-error # Continue even if some steps fail
  python install_abov3.py --model codellama  # Use different default model
        """
    )
    
    parser.add_argument("--skip-ollama", action="store_true",
                       help="Skip Ollama installation")
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue installation even if some steps fail")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                       help=f"Default model to install (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-shortcuts", action="store_true",
                       help="Skip creating desktop shortcuts")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without actually doing it")
    
    return parser.parse_args()

def main():
    """Main entry point"""
    try:
        args = parse_arguments()
        
        # Override default model if specified
        if args.model != DEFAULT_MODEL:
            global DEFAULT_MODEL
            DEFAULT_MODEL = args.model
        
        installer = ABOV3Installer(args)
        
        if args.dry_run:
            installer.log("DRY RUN MODE - No actual changes will be made", "WARNING")
            installer.print_banner()
            installer.log("Installation steps that would be performed:", "INFO")
            steps = [
                "Check system requirements",
                "Install Python dependencies",
                "Install ABOV3 in development mode",
                "Setup PATH environment",
                "Check/Install Ollama",
                f"Setup default model ({DEFAULT_MODEL})",
                "Create configuration files",
                "Create desktop shortcuts",
                "Handle common issues",
                "Run verification tests"
            ]
            
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step}")
            
            return True
        
        success = installer.run_installation()
        return success
        
    except Exception as e:
        print(f"{Colors.RED}Fatal error: {e}{Colors.END}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)