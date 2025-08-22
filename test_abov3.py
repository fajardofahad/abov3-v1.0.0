#!/usr/bin/env python3
"""
Test script to verify ABOV3 can be started directly.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the CLI
from abov3.cli import cli

if __name__ == "__main__":
    # When run without arguments, it should start the chat session
    cli()