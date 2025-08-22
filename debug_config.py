#!/usr/bin/env python3
"""Debug script to check configuration."""

import sys
from pathlib import Path

# Add the parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from abov3.core.config import get_config

def debug_config():
    """Check the current configuration."""
    print("Checking ABOV3 configuration...")
    
    config = get_config()
    
    print(f"\nOllama Configuration:")
    print(f"  Host: {config.ollama.host}")
    print(f"  Timeout: {config.ollama.timeout}")
    print(f"  Max Retries: {config.ollama.max_retries}")
    print(f"  Required for Startup: {config.ollama.required_for_startup}")
    
    print(f"\nModel Configuration:")
    print(f"  Default Model: {config.model.default_model}")
    print(f"  Temperature: {config.model.temperature}")
    
    # Check if host URL is valid
    from urllib.parse import urlparse
    try:
        parsed = urlparse(config.ollama.host)
        print(f"\nParsed URL:")
        print(f"  Scheme: {parsed.scheme}")
        print(f"  Hostname: {parsed.hostname}")
        print(f"  Port: {parsed.port}")
        
        if parsed.port and (parsed.port < 0 or parsed.port > 65535):
            print(f"  ERROR: Port {parsed.port} is out of range!")
    except Exception as e:
        print(f"  ERROR parsing URL: {e}")

if __name__ == "__main__":
    debug_config()