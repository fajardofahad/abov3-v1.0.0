#!/usr/bin/env python3
"""
ABOV3 4 Ollama - Quick Start Script

This script provides a simple way to start ABOV3 with automatic setup.
Just run: python start.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def check_dependencies():
    """Check if required dependencies are installed."""
    required = ['ollama', 'rich', 'click', 'prompt_toolkit']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def install_dependencies():
    """Install missing dependencies."""
    print("Installing ABOV3 dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Dependencies installed successfully!")

def check_ollama():
    """Check if Ollama is running."""
    try:
        import ollama
        client = ollama.Client()
        client.list()
        return True
    except:
        return False

def main():
    """Main entry point."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     █████  ██████   ██████  ██    ██ ██████                 ║
║    ██   ██ ██   ██ ██    ██ ██    ██      ██                ║
║    ███████ ██████  ██    ██ ██    ██  █████                 ║
║    ██   ██ ██   ██ ██    ██  ██  ██       ██                ║
║    ██   ██ ██████   ██████    ████   ██████                 ║
║                                                               ║
║         ABOV3 4 Ollama - AI Coding Assistant                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        response = input("Would you like to install them? (y/n): ")
        if response.lower() == 'y':
            install_dependencies()
        else:
            print("Please install dependencies manually: pip install -r requirements.txt")
            sys.exit(1)
    
    # Check Ollama
    if not check_ollama():
        print("\n⚠️  Ollama server is not running!")
        print("\nTo use ABOV3, you need to:")
        print("1. Install Ollama from: https://ollama.ai/download")
        print("2. Start Ollama: ollama serve")
        print("3. Download a model: ollama pull llama3.2:latest")
        print("\nWould you like to start the demo anyway? (y/n): ", end="")
        response = input()
        if response.lower() != 'y':
            sys.exit(0)
    
    # Start ABOV3
    print("\n🚀 Starting ABOV3 4 Ollama...")
    print("Type 'help' for available commands or 'exit' to quit.\n")
    
    # Import and run
    try:
        from abov3.cli import cli
        cli()
    except ImportError:
        # Fallback to demo if full CLI fails
        print("Running in demo mode...")
        from demo import main as demo_main
        demo_main()

if __name__ == "__main__":
    main()