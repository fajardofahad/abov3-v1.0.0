#!/usr/bin/env python3
"""
ABOV3 Project Management Entry Point

Launch ABOV3 with full project management capabilities including:
- Project directory selection and management
- File operations and analysis
- Context-aware AI coding assistance
- Enhanced REPL with project commands
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from abov3.core.project_app import run_project_aware_app


def main():
    """Main entry point for project-aware ABOV3."""
    parser = argparse.ArgumentParser(
        description="ABOV3 - AI-Powered Coding Assistant with Project Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python abov3_project.py                    # Start ABOV3 in project mode
  python abov3_project.py ~/my-project      # Start with specific project
  python abov3_project.py --config custom.toml  # Use custom configuration

Project Commands (once running):
  /project <path>    # Select a project directory
  /files [pattern]   # List project files
  /read <file>       # Read file contents
  /search <query>    # Search across project files
  /tree              # Show directory structure
  /help              # Show all available commands

For detailed documentation, see PROJECT_MANAGEMENT_GUIDE.md
        """
    )
    
    parser.add_argument(
        "project_path",
        nargs="?",
        help="Initial project directory to load (optional)"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test suite to validate functionality"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="ABOV3 Project Management v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Handle test mode
    if args.test:
        print("Running ABOV3 project management test suite...")
        test_script = project_root / "test_project_features.py"
        if test_script.exists():
            os.system(f"python {test_script}")
        else:
            print("Test script not found. Please ensure test_project_features.py exists.")
        return
    
    # Validate project path if provided
    if args.project_path:
        project_path = Path(args.project_path).resolve()
        if not project_path.exists():
            print(f"Error: Project directory does not exist: {project_path}")
            sys.exit(1)
        if not project_path.is_dir():
            print(f"Error: Path is not a directory: {project_path}")
            sys.exit(1)
        args.project_path = str(project_path)
    
    # Welcome message
    print("ABOV3 - AI-Powered Coding Assistant with Project Management")
    print("=" * 60)
    
    if args.project_path:
        print(f"Loading project: {args.project_path}")
    else:
        print("Tip: Use '/project <path>' to select a project directory")
    
    print("Type '/help' for available commands")
    print("Press Ctrl+C or type '/exit' to quit")
    print("=" * 60)
    
    try:
        # Run the project-aware application
        asyncio.run(run_project_aware_app(
            project_path=args.project_path,
            config_path=args.config
        ))
    except KeyboardInterrupt:
        print("\nGoodbye! Thanks for using ABOV3.")
    except Exception as e:
        print(f"\nError starting ABOV3: {e}")
        print("Please check your installation and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()