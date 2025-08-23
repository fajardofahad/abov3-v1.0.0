#!/usr/bin/env python3
"""
ABOV3 4 Ollama - Main Module Entry Point

This module provides the main entry point for running ABOV3 as a module.
It enables running ABOV3 using: python -m abov3

Author: ABOV3 Team
Version: 1.0.0
License: MIT
"""

import sys
import os
import logging

def suppress_third_party_logging():
    """Early suppression of third-party logging noise."""
    # Configure root logger first to ensure proper inheritance
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    noisy_loggers = [
        'httpx', 'httpcore', 'urllib3', 'requests', 'aiohttp',
        'ollama', 'asyncio', 'websockets', 'prompt_toolkit'
    ]
    
    # Set ERROR level for the noisiest loggers
    for logger_name in ['httpx', 'httpcore']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = True
    
    # Set WARNING level for other noisy loggers  
    for logger_name in noisy_loggers:
        if logger_name not in ['httpx', 'httpcore']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            logger.propagate = True

def main():
    """Main entry point when running as module."""
    # Suppress third-party logging as early as possible
    suppress_third_party_logging()
    
    # Add the parent directory to sys.path to ensure imports work correctly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        from .cli import main as cli_main
        cli_main()
    except ImportError as e:
        # Fallback import method
        try:
            import abov3.cli
            abov3.cli.main()
        except ImportError:
            print(f"Error: Failed to import ABOV3 CLI: {e}", file=sys.stderr)
            print("Please ensure ABOV3 is properly installed.", file=sys.stderr)
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()