# ABOV3 Entry Points Installation Guide

This document explains how to use the various entry points created for ABOV3 4 Ollama to make it easily accessible via the `abov3` command.

## Available Entry Points

Multiple entry points have been created to provide flexible access to ABOV3:

### 1. Python Module Entry Point (`__main__.py`)
Allows running ABOV3 as a Python module:
```bash
python -m abov3
python -m abov3 --version
python -m abov3 chat
```

### 2. Direct Python Script (`abov3.py`)
Direct execution of the ABOV3 script:
```bash
python abov3.py
python abov3.py --version
python abov3.py chat
```

### 3. Windows Batch Script (`abov3.bat`)
Windows batch file for direct command access:
```bash
abov3.bat
abov3.bat --version
abov3.bat chat
```

### 4. Unix Shell Script (`abov3.sh`)
Unix/Linux/macOS shell script:
```bash
./abov3.sh
./abov3.sh --version
./abov3.sh chat
```

### 5. Package Installation (Recommended)
Install as a Python package with console script entry point.

## Installation Methods

### Method 1: Package Installation (Recommended)

Install ABOV3 as a Python package to get the `abov3` command globally:

```bash
# Install from current directory
pip install .

# Or install in development mode
pip install -e .

# Now you can use the abov3 command anywhere
abov3
abov3 --version
abov3 chat
```

After installation, the `abov3` command will be available system-wide.

### Method 2: Direct Usage

You can use ABOV3 directly without installation:

**Windows:**
```bash
# Using Python directly
python abov3.py

# Using Windows batch script  
abov3.bat

# Using Python module
python -m abov3
```

**Unix/Linux/macOS:**
```bash
# Using Python directly
python3 abov3.py

# Using shell script (make executable first)
chmod +x abov3.sh
./abov3.sh

# Using Python module
python3 -m abov3
```

### Method 3: Add to PATH

You can add the ABOV3 directory to your system PATH to make the scripts globally accessible.

**Windows:**
1. Add the ABOV3 directory to your PATH environment variable
2. Rename `abov3.bat` to `abov3.bat` (or create a symbolic link)
3. Now you can run `abov3` from anywhere

**Unix/Linux/macOS:**
1. Add the ABOV3 directory to your PATH in `~/.bashrc` or `~/.zshrc`:
   ```bash
   export PATH="/path/to/abov3_directory:$PATH"
   ```
2. Create a symbolic link or alias:
   ```bash
   ln -s /path/to/abov3_directory/abov3.sh /usr/local/bin/abov3
   ```

## Default Behavior

When running ABOV3 without any command, it automatically starts the interactive chat session:

```bash
abov3                    # Starts chat session
abov3 --version         # Shows version
abov3 --help           # Shows help
abov3 chat             # Explicitly starts chat
abov3 models list      # Lists available models
```

## Requirements

Ensure all dependencies are installed before using ABOV3:

```bash
pip install -r requirements.txt
```

Required dependencies:
- ollama>=0.3.0
- rich>=13.0.0
- prompt-toolkit>=3.0.36
- click>=8.0.0
- pydantic>=2.0.0
- aiohttp>=3.8.0
- pygments>=2.10.0
- toml>=0.10.2
- gitpython>=3.1.30
- watchdog>=3.0.0
- python-dotenv>=1.0.0
- colorama>=0.4.6
- packaging>=21.0
- aiosqlite>=0.17.0
- numpy>=1.21.0
- pyyaml>=6.0.0
- psutil>=5.9.0

## Testing Entry Points

A test script is provided to verify all entry points work correctly:

```bash
python test_entry_points.py
```

This will test all entry points and report any issues.

## Cross-Platform Compatibility

All entry points are designed to work across different platforms:

- **Windows**: Uses `abov3.bat` and standard Python execution
- **Unix/Linux/macOS**: Uses `abov3.sh` with automatic Python version detection
- **All Platforms**: Python module and direct script execution work universally

## Troubleshooting

### Python Not Found
If you get "Python not found" errors:
1. Ensure Python 3.8+ is installed
2. Make sure Python is in your PATH
3. Try using `python3` instead of `python` on Unix systems

### Import Errors
If you get import errors:
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure you're in the correct directory
3. Try using `python -m abov3` instead of direct execution

### Permission Errors (Unix/Linux/macOS)
If you get permission errors:
1. Make the shell script executable: `chmod +x abov3.sh`
2. Check directory permissions
3. Use `sudo` if necessary for system-wide installation

## Support

For additional support or issues:
1. Check the main README.md file
2. Review the documentation in the `docs/` directory
3. Run `abov3 doctor` to diagnose common issues