"""
ABOV3 4 Ollama Test Suite.

This package contains comprehensive tests for the ABOV3 platform,
ensuring enterprise-grade quality and reliability.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to allow imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Test configuration constants
TEST_TIMEOUT = 30  # Default timeout for async tests
TEST_DATA_DIR = Path(__file__).parent / "test_data"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Create test directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
FIXTURES_DIR.mkdir(exist_ok=True)

# Test environment variables
os.environ.setdefault("ABOV3_TEST_MODE", "true")
os.environ.setdefault("ABOV3_LOG_LEVEL", "DEBUG")

__version__ = "1.0.0"
__all__ = ["TEST_TIMEOUT", "TEST_DATA_DIR", "FIXTURES_DIR"]