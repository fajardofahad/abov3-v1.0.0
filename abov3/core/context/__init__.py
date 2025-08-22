"""
Context management system for ABOV3 4 Ollama.

This module provides comprehensive context management capabilities including:
- Conversation state tracking and persistence
- Memory management (short-term and long-term)
- Context window management with token counting
- Context compression and truncation strategies
- Session management with save/restore functionality
- Semantic search and filtering
- Multi-conversation handling
- Export/import functionality

The system is designed to be thread-safe, async-compatible, and extensible
for future enhancements.
"""

from .manager import ContextManager
from .memory import MemoryManager, ContextMemory, MemoryType
from .session import SessionManager, ConversationSession

__all__ = [
    "ContextManager",
    "MemoryManager", 
    "ContextMemory",
    "MemoryType",
    "SessionManager",
    "ConversationSession",
]

# Version information
__version__ = "1.0.0"
__author__ = "ABOV3 Enterprise Team"
__description__ = "Advanced context management system for AI conversations"