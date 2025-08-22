#!/usr/bin/env python3
"""Fix the invalid port in configuration."""

import sys
from pathlib import Path

# Add the parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from abov3.core.config import Config, save_config

def fix_config():
    """Fix the invalid port in the config."""
    print("Fixing ABOV3 configuration...")
    
    # Load the current config
    config = Config.load_from_file()
    
    print(f"Current Ollama host: {config.ollama.host}")
    
    # Fix the host
    config.ollama.host = "http://localhost:11434"
    
    print(f"New Ollama host: {config.ollama.host}")
    
    # Save the config
    save_config(config)
    
    print("Configuration fixed and saved!")

if __name__ == "__main__":
    fix_config()