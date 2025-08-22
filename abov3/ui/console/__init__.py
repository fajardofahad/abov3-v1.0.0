"""
ABOV3 Console UI Package

This package provides a sophisticated interactive console interface for ABOV3,
featuring advanced REPL capabilities, syntax highlighting, and rich formatting.
"""

from .repl import ABOV3REPL, REPLConfig
from .completers import (
    CommandCompleter,
    CodeCompleter,
    ContextAwareCompleter,
    MultiCompleter
)
from .formatters import (
    OutputFormatter,
    CodeFormatter,
    ErrorFormatter,
    StreamingFormatter
)
from .keybindings import create_keybindings, KeyBindingMode

__all__ = [
    'ABOV3REPL',
    'REPLConfig',
    'CommandCompleter',
    'CodeCompleter',
    'ContextAwareCompleter',
    'MultiCompleter',
    'OutputFormatter',
    'CodeFormatter',
    'ErrorFormatter',
    'StreamingFormatter',
    'create_keybindings',
    'KeyBindingMode'
]

__version__ = '1.0.0'