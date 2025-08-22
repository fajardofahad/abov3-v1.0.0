#!/bin/bash
#
# ABOV3 4 Ollama - Unix Shell Script Entry Point
#
# This shell script provides a Unix/Linux/macOS entry point for ABOV3.
# It handles various Python installation scenarios and ensures proper execution.
#
# Usage: ./abov3.sh [commands...]
#        abov3 [commands...]
#
# Author: ABOV3 Team
# Version: 1.0.0
# License: MIT

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to find Python executable
find_python() {
    local python_exe=""
    
    # Check for python3 command first (preferred)
    if command -v python3 &> /dev/null; then
        # Verify it's Python 3.8+
        if python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
            echo "python3"
            return 0
        fi
    fi
    
    # Check for python command
    if command -v python &> /dev/null; then
        # Verify it's Python 3.8+
        if python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
            echo "python"
            return 0
        fi
    fi
    
    # Check for specific Python versions
    for version in 3.12 3.11 3.10 3.9 3.8; do
        if command -v "python$version" &> /dev/null; then
            echo "python$version"
            return 0
        fi
    done
    
    return 1
}

# Function to display error message
show_python_error() {
    echo "Error: Python 3.8+ is not installed or not in PATH." >&2
    echo "Please install Python 3.8 or later and ensure it's accessible from command line." >&2
    echo "" >&2
    echo "On Ubuntu/Debian: sudo apt install python3 python3-pip" >&2
    echo "On CentOS/RHEL: sudo yum install python3 python3-pip" >&2
    echo "On macOS: brew install python3" >&2
    echo "Or visit: https://www.python.org/downloads/" >&2
    exit 1
}

# Main execution
main() {
    # Find Python executable
    PYTHON_EXE=$(find_python)
    
    if [ $? -ne 0 ]; then
        show_python_error
    fi
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Run ABOV3 with the found Python executable
    exec "$PYTHON_EXE" abov3.py "$@"
}

# Handle interrupts gracefully
trap 'echo -e "\nOperation cancelled by user." >&2; exit 130' INT TERM

# Run main function with all arguments
main "$@"