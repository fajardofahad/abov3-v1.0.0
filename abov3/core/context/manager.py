"""
Context Manager for ABOV3 4 Ollama.

This module provides the core ContextManager class that handles conversation
context, token counting, context window management, and intelligent context
compression strategies.
"""

import asyncio
import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from ..config import get_config


logger = logging.getLogger(__name__)


@dataclass
class ContextEntry:
    """Represents a single context entry (message or interaction)."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    role: str = "user"  # user, assistant, system, tool
    content: str = ""
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0  # Higher values indicate higher importance
    compressed: bool = False
    
    def __post_init__(self):
        """Calculate token count if not provided."""
        if self.token_count == 0 and self.content:
            self.token_count = self._estimate_tokens(self.content)
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple approximation: ~4 characters per token for English text
        # This can be replaced with a more accurate tokenizer if needed
        return len(text) // 4 + len(text.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "role": self.role,
            "content": self.content,
            "token_count": self.token_count,
            "metadata": self.metadata,
            "priority": self.priority,
            "compressed": self.compressed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextEntry":
        """Create from dictionary."""
        data = data.copy()
        if "timestamp" in data:
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ContextWindow:
    """Represents a context window with entries and metadata."""
    
    entries: List[ContextEntry] = field(default_factory=list)
    max_tokens: int = 8192
    current_tokens: int = 0
    compressed_entries: List[ContextEntry] = field(default_factory=list)
    summary: str = ""
    
    def add_entry(self, entry: ContextEntry) -> bool:
        """Add an entry to the context window."""
        if self.current_tokens + entry.token_count <= self.max_tokens:
            self.entries.append(entry)
            self.current_tokens += entry.token_count
            return True
        return False
    
    def remove_oldest(self, count: int = 1) -> List[ContextEntry]:
        """Remove oldest entries and return them."""
        removed = []
        for _ in range(min(count, len(self.entries))):
            if self.entries:
                entry = self.entries.pop(0)
                self.current_tokens -= entry.token_count
                removed.append(entry)
        return removed
    
    def get_total_tokens(self) -> int:
        """Get total token count including compressed entries."""
        return self.current_tokens + sum(e.token_count for e in self.compressed_entries)


class ContextManager:
    """
    Advanced context manager for handling conversation state and memory.
    
    Features:
    - Token-aware context window management
    - Intelligent context compression
    - Priority-based entry retention
    - Semantic search and filtering
    - Thread-safe operations
    - Async support for heavy operations
    - Configurable truncation strategies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the context manager."""
        self.config = get_config()
        self._custom_config = config or {}
        
        # Context window settings
        self.max_tokens = self._get_setting("max_tokens", self.config.model.context_length)
        self.reserve_tokens = self._get_setting("reserve_tokens", 1024)  # Reserve for response
        self.compression_threshold = self._get_setting("compression_threshold", 0.8)
        
        # Sliding window settings
        self.sliding_window_size = self._get_setting("sliding_window_size", 10)
        self.min_priority_threshold = self._get_setting("min_priority_threshold", 0.3)
        
        # State management
        self._context_window = ContextWindow(max_tokens=self.max_tokens - self.reserve_tokens)
        self._conversation_history: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        self._session_id = str(uuid4())
        
        # Compression and summarization
        self._compression_queue: List[ContextEntry] = []
        self._last_compression = datetime.now()
        self._compression_interval = timedelta(minutes=5)
        
        # Search index for semantic search
        self._search_index: Dict[str, Set[str]] = {}
        self._keyword_cache: Dict[str, List[str]] = {}
        
        logger.info(f"ContextManager initialized with session {self._session_id}")
    
    def _get_setting(self, key: str, default: Any) -> Any:
        """Get setting from custom config or default."""
        return self._custom_config.get(key, default)
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 1.0
    ) -> str:
        """
        Add a message to the context.
        
        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            metadata: Optional metadata dictionary
            priority: Priority level (higher = more important)
            
        Returns:
            str: Entry ID
        """
        with self._lock:
            entry = ContextEntry(
                role=role,
                content=content,
                metadata=metadata or {},
                priority=priority
            )
            
            # Add to conversation history
            self._conversation_history.append(entry)
            
            # Try to add to current context window
            if not self._context_window.add_entry(entry):
                # Context window is full, need to manage space
                self._manage_context_space(entry)
            
            # Update search index
            self._update_search_index(entry)
            
            # Check if compression is needed
            self._check_compression_trigger()
            
            logger.debug(f"Added message {entry.id} (role: {role}, tokens: {entry.token_count})")
            return entry.id
    
    def _manage_context_space(self, new_entry: ContextEntry) -> None:
        """Manage context space when adding new entry."""
        tokens_needed = new_entry.token_count
        
        # Strategy 1: Remove low-priority entries
        self._remove_low_priority_entries(tokens_needed)
        
        # Strategy 2: Compress old entries if still not enough space
        if self._context_window.current_tokens + tokens_needed > self._context_window.max_tokens:
            self._compress_old_entries(tokens_needed)
        
        # Strategy 3: Remove oldest entries as last resort
        while (self._context_window.current_tokens + tokens_needed > self._context_window.max_tokens 
               and self._context_window.entries):
            removed = self._context_window.remove_oldest(1)
            if removed:
                self._compression_queue.extend(removed)
        
        # Now add the new entry
        self._context_window.add_entry(new_entry)
    
    def _remove_low_priority_entries(self, tokens_needed: int) -> None:
        """Remove low-priority entries to make space."""
        removed_tokens = 0
        entries_to_remove = []
        
        # Sort by priority (ascending) to remove lowest first
        sorted_entries = sorted(
            enumerate(self._context_window.entries),
            key=lambda x: x[1].priority
        )
        
        for idx, entry in sorted_entries:
            if (entry.priority < self.min_priority_threshold and 
                removed_tokens < tokens_needed):
                entries_to_remove.append(idx)
                removed_tokens += entry.token_count
        
        # Remove entries in reverse order to maintain indices
        for idx in sorted(entries_to_remove, reverse=True):
            removed_entry = self._context_window.entries.pop(idx)
            self._context_window.current_tokens -= removed_entry.token_count
            self._compression_queue.append(removed_entry)
    
    def _compress_old_entries(self, tokens_needed: int) -> None:
        """Compress old entries to make space."""
        if len(self._context_window.entries) <= self.sliding_window_size:
            return
        
        # Move entries beyond sliding window to compression queue
        entries_to_compress = len(self._context_window.entries) - self.sliding_window_size
        compressed_tokens = 0
        
        for _ in range(min(entries_to_compress, len(self._context_window.entries))):
            if compressed_tokens >= tokens_needed:
                break
            
            entry = self._context_window.entries.pop(0)
            self._context_window.current_tokens -= entry.token_count
            self._compression_queue.append(entry)
            compressed_tokens += entry.token_count
    
    def _update_search_index(self, entry: ContextEntry) -> None:
        """Update search index with entry keywords."""
        keywords = self._extract_keywords(entry.content)
        self._keyword_cache[entry.id] = keywords
        
        for keyword in keywords:
            if keyword not in self._search_index:
                self._search_index[keyword] = set()
            self._search_index[keyword].add(entry.id)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for search indexing."""
        # Simple keyword extraction - can be enhanced with NLP libraries
        text = text.lower()
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text)
        # Filter out common stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'out', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can'
        }
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return list(set(keywords))  # Remove duplicates
    
    def _check_compression_trigger(self) -> None:
        """Check if compression should be triggered."""
        current_usage = self._context_window.current_tokens / self._context_window.max_tokens
        time_since_compression = datetime.now() - self._last_compression
        
        if (current_usage > self.compression_threshold or 
            time_since_compression > self._compression_interval):
            self._trigger_compression()
    
    def _trigger_compression(self) -> None:
        """Trigger compression of queued entries."""
        if not self._compression_queue:
            return
        
        # Group entries by time windows for better compression
        compressed_summary = self._compress_entries(self._compression_queue)
        
        if compressed_summary:
            # Create a compressed entry
            compressed_entry = ContextEntry(
                role="system",
                content=f"[COMPRESSED]: {compressed_summary}",
                metadata={"type": "compression", "original_count": len(self._compression_queue)},
                priority=0.5,
                compressed=True
            )
            
            self._context_window.compressed_entries.append(compressed_entry)
            self._compression_queue.clear()
            self._last_compression = datetime.now()
            
            logger.info(f"Compressed {len(self._compression_queue)} entries into summary")
    
    def _compress_entries(self, entries: List[ContextEntry]) -> str:
        """Compress a list of entries into a summary."""
        if not entries:
            return ""
        
        # Simple compression strategy - can be enhanced with AI summarization
        key_points = []
        user_messages = []
        assistant_messages = []
        
        for entry in entries:
            if entry.role == "user":
                user_messages.append(entry.content[:100])  # Truncate for summary
            elif entry.role == "assistant":
                assistant_messages.append(entry.content[:100])
        
        summary_parts = []
        if user_messages:
            summary_parts.append(f"User discussed: {', '.join(user_messages[:3])}")
        if assistant_messages:
            summary_parts.append(f"Assistant provided: {', '.join(assistant_messages[:3])}")
        
        return " | ".join(summary_parts)
    
    def get_context_for_model(self, include_compressed: bool = True) -> List[Dict[str, str]]:
        """
        Get context formatted for model consumption.
        
        Args:
            include_compressed: Whether to include compressed entries
            
        Returns:
            List of messages formatted for the model
        """
        with self._lock:
            messages = []
            
            # Add compressed entries first if requested
            if include_compressed:
                for entry in self._context_window.compressed_entries:
                    messages.append({
                        "role": entry.role,
                        "content": entry.content
                    })
            
            # Add current context window entries
            for entry in self._context_window.entries:
                messages.append({
                    "role": entry.role,
                    "content": entry.content
                })
            
            return messages
    
    def search_context(
        self,
        query: str,
        max_results: int = 10,
        include_compressed: bool = False
    ) -> List[ContextEntry]:
        """
        Search context entries by keyword.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            include_compressed: Whether to search compressed entries
            
        Returns:
            List of matching context entries
        """
        with self._lock:
            query_keywords = self._extract_keywords(query.lower())
            
            # Find entries that match any query keyword
            matching_entry_ids = set()
            for keyword in query_keywords:
                if keyword in self._search_index:
                    matching_entry_ids.update(self._search_index[keyword])
            
            # Get actual entries and sort by relevance
            results = []
            all_entries = self._context_window.entries[:]
            
            if include_compressed:
                all_entries.extend(self._context_window.compressed_entries)
            
            for entry in all_entries:
                if entry.id in matching_entry_ids:
                    # Calculate relevance score
                    entry_keywords = self._keyword_cache.get(entry.id, [])
                    relevance = len(set(query_keywords) & set(entry_keywords))
                    results.append((entry, relevance))
            
            # Sort by relevance and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            return [entry for entry, _ in results[:max_results]]
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        with self._lock:
            return {
                "session_id": self._session_id,
                "current_entries": len(self._context_window.entries),
                "current_tokens": self._context_window.current_tokens,
                "max_tokens": self._context_window.max_tokens,
                "token_usage": self._context_window.current_tokens / self._context_window.max_tokens,
                "compressed_entries": len(self._context_window.compressed_entries),
                "total_history": len(self._conversation_history),
                "compression_queue": len(self._compression_queue),
                "search_keywords": len(self._search_index),
                "last_compression": self._last_compression.isoformat(),
            }
    
    def clear_context(self, keep_system: bool = True) -> None:
        """
        Clear the current context.
        
        Args:
            keep_system: Whether to keep system messages
        """
        with self._lock:
            if keep_system:
                # Keep system messages
                system_entries = [e for e in self._context_window.entries if e.role == "system"]
                self._context_window.entries = system_entries
                self._context_window.current_tokens = sum(e.token_count for e in system_entries)
            else:
                self._context_window.entries.clear()
                self._context_window.current_tokens = 0
            
            self._context_window.compressed_entries.clear()
            self._compression_queue.clear()
            self._search_index.clear()
            self._keyword_cache.clear()
            
            logger.info("Context cleared")
    
    def export_context(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export context to various formats.
        
        Args:
            format: Export format ("json", "dict", "text")
            
        Returns:
            Exported context data
        """
        with self._lock:
            data = {
                "session_id": self._session_id,
                "timestamp": datetime.now().isoformat(),
                "config": self._custom_config,
                "context_window": {
                    "entries": [entry.to_dict() for entry in self._context_window.entries],
                    "compressed_entries": [entry.to_dict() for entry in self._context_window.compressed_entries],
                    "max_tokens": self._context_window.max_tokens,
                    "current_tokens": self._context_window.current_tokens,
                },
                "stats": self.get_context_stats(),
            }
            
            if format == "json":
                return json.dumps(data, indent=2, ensure_ascii=False)
            elif format == "dict":
                return data
            elif format == "text":
                lines = [f"Context Export - Session {self._session_id}"]
                lines.append(f"Generated: {data['timestamp']}")
                lines.append(f"Entries: {len(data['context_window']['entries'])}")
                lines.append(f"Tokens: {data['context_window']['current_tokens']}")
                lines.append("\n--- Messages ---")
                
                for entry_data in data['context_window']['entries']:
                    entry = ContextEntry.from_dict(entry_data)
                    lines.append(f"\n[{entry.timestamp}] {entry.role.upper()}:")
                    lines.append(entry.content)
                
                return "\n".join(lines)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def import_context(self, data: Union[str, Dict[str, Any]]) -> None:
        """
        Import context from exported data.
        
        Args:
            data: Context data (JSON string or dictionary)
        """
        with self._lock:
            if isinstance(data, str):
                import_data = json.loads(data)
            else:
                import_data = data
            
            # Clear current context
            self.clear_context(keep_system=False)
            
            # Import entries
            if "context_window" in import_data:
                context_data = import_data["context_window"]
                
                # Import regular entries
                for entry_data in context_data.get("entries", []):
                    entry = ContextEntry.from_dict(entry_data)
                    self._context_window.entries.append(entry)
                    self._update_search_index(entry)
                
                # Import compressed entries
                for entry_data in context_data.get("compressed_entries", []):
                    entry = ContextEntry.from_dict(entry_data)
                    self._context_window.compressed_entries.append(entry)
                
                # Update token count
                self._context_window.current_tokens = sum(
                    e.token_count for e in self._context_window.entries
                )
            
            logger.info(f"Imported context with {len(self._context_window.entries)} entries")
    
    async def async_compress(self) -> None:
        """Async version of compression for heavy operations."""
        await asyncio.get_event_loop().run_in_executor(None, self._trigger_compression)
    
    def get_conversation_summary(self, max_length: int = 500) -> str:
        """
        Get a summary of the conversation.
        
        Args:
            max_length: Maximum length of summary
            
        Returns:
            Conversation summary
        """
        with self._lock:
            if not self._context_window.entries:
                return "No conversation yet."
            
            # Extract key points from conversation
            user_topics = []
            assistant_responses = []
            
            for entry in self._context_window.entries[-10:]:  # Last 10 entries
                if entry.role == "user":
                    # Extract main topic/question
                    content = entry.content.strip()
                    if len(content) > 100:
                        content = content[:97] + "..."
                    user_topics.append(content)
                elif entry.role == "assistant":
                    # Extract key points from response
                    content = entry.content.strip()
                    if len(content) > 150:
                        content = content[:147] + "..."
                    assistant_responses.append(content)
            
            summary_parts = []
            if user_topics:
                summary_parts.append(f"Recent topics: {'; '.join(user_topics[-3:])}")
            if assistant_responses:
                summary_parts.append(f"Key responses: {'; '.join(assistant_responses[-2:])}")
            
            summary = " | ".join(summary_parts)
            
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return summary or "Conversation in progress."