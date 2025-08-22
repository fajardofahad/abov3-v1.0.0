#!/usr/bin/env python3
"""
ABOV3 4 Ollama - Setup Script

This setup script provides installation configuration for ABOV3.
It supports both pip installation and development setup.

Author: ABOV3 Team
Version: 1.0.0
License: MIT
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure we're running on Python 3.8+
if sys.version_info < (3, 8):
    print("Error: ABOV3 requires Python 3.8 or later.", file=sys.stderr)
    print(f"You are running Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}", file=sys.stderr)
    sys.exit(1)

# Get the long description from README
here = Path(__file__).parent
long_description_file = here / "README.md"
if long_description_file.exists():
    with open(long_description_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Advanced Interactive AI Coding Assistant for Ollama"

# Read version from abov3/__init__.py
def get_version():
    """Get version from abov3/__init__.py"""
    init_file = here / "abov3" / "__init__.py"
    if init_file.exists():
        with open(init_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

# Define requirements
install_requires = [
    "ollama>=0.3.0",
    "rich>=13.0.0", 
    "prompt-toolkit>=3.0.36",
    "click>=8.0.0",
    "pydantic>=2.0.0",
    "aiohttp>=3.8.0",
    "pygments>=2.10.0",
    "toml>=0.10.2",
    "gitpython>=3.1.30",
    "watchdog>=3.0.0",
    "python-dotenv>=1.0.0",
    "colorama>=0.4.6",
    "packaging>=21.0",
    "aiosqlite>=0.17.0",
    "numpy>=1.21.0",
    "pyyaml>=6.0.0",
    "psutil>=5.9.0"
]

dev_requires = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.20.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=5.0.0",
    "mypy>=1.0.0",
    "pre-commit>=2.20.0"
]

docs_requires = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "myst-parser>=0.18.0"
]

test_requires = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.20.0", 
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0"
]

# Setup configuration
setup(
    name="abov3",
    version=get_version(),
    author="ABOV3 Team",
    author_email="contact@abov3.dev",
    description="Advanced Interactive AI Coding Assistant for Ollama",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abov3/abov3-ollama",
    project_urls={
        "Documentation": "https://abov3-ollama.readthedocs.io",
        "Repository": "https://github.com/abov3/abov3-ollama.git",
        "Bug Tracker": "https://github.com/abov3/abov3-ollama/issues",
        "Changelog": "https://github.com/abov3/abov3-ollama/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    package_data={
        "abov3": ["*.json", "*.yaml", "*.yml", "*.toml"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Shells",
        "Topic :: Terminals",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Utilities"
    ],
    keywords=[
        "ai", "coding", "assistant", "ollama", "cli", "interactive",
        "code-generation", "debugging", "refactoring"
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "docs": docs_requires,
        "test": test_requires,
        "all": dev_requires + docs_requires + test_requires,
    },
    entry_points={
        "console_scripts": [
            "abov3=abov3.cli:main",
        ],
    },
    zip_safe=False,
    platforms=["any"],
    license="MIT",
)