#!/usr/bin/env python3
"""Find the config file location."""

import sys
from pathlib import Path

# Add the parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from abov3.core.config import Config

def find_config():
    """Find where the config file is located."""
    config_dir = Config.get_config_dir()
    config_file = config_dir / "config.toml"
    
    print(f"Config directory: {config_dir}")
    print(f"Config file path: {config_file}")
    print(f"Config file exists: {config_file.exists()}")
    
    if config_file.exists():
        print(f"\nConfig file contents:")
        with open(config_file, 'r') as f:
            print(f.read())

if __name__ == "__main__":
    find_config()