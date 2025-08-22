#!/usr/bin/env python3
"""
ABOV3 4 Ollama - Direct Script Entry Point

This script provides a direct entry point for running ABOV3.
It can be executed directly or used as a console script entry point.

Usage:
    python abov3.py [commands...]
    abov3 [commands...]

Author: ABOV3 Team
Version: 1.0.0
License: MIT
"""

import sys
import os

def main():
    """Main entry point for the ABOV3 application."""
    # Ensure the current directory is in Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        # Import and run the CLI
        from abov3.cli import main as cli_main
        cli_main()
    except ImportError as e:
        # Fallback for different import scenarios
        try:
            import abov3.cli
            abov3.cli.main()
        except ImportError:
            print(f"Error: Failed to import ABOV3 CLI: {e}", file=sys.stderr)
            print("Please ensure all dependencies are installed:", file=sys.stderr)
            print("  pip install -r requirements.txt", file=sys.stderr)
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        print("Run with --debug flag for more details.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()