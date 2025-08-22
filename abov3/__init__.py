"""
ABOV3 4 Ollama - Advanced Interactive AI Coding Assistant

A powerful Python-based console application that provides an interactive CLI interface
for AI-powered code generation, debugging, and refactoring using local Ollama models.

Author: ABOV3 Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "ABOV3 Team"
__email__ = "contact@abov3.dev"
__description__ = "Advanced Interactive AI Coding Assistant for Ollama"

from .core.app import ABOV3App
from .core.config import Config
from .models.manager import ModelManager

__all__ = [
    "ABOV3App",
    "Config", 
    "ModelManager",
    "__version__",
    "__author__",
    "__description__"
]