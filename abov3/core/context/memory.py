"""
Memory management system for ABOV3 4 Ollama.

This module provides comprehensive memory management capabilities including:
- Short-term and long-term memory storage
- Memory persistence and retrieval
- Memory indexing and search
- Memory compression and optimization
- Contextual memory associations
- Memory decay and cleanup
"""

import asyncio
import json
import logging
import pickle
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from ..config import get_config


logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"  # Recent conversations, temporary data
    LONG_TERM = "long_term"    # Important information, user preferences
    EPISODIC = "episodic"      # Specific conversation episodes
    SEMANTIC = "semantic"      # General knowledge and facts
    PROCEDURAL = "procedural"  # How-to information, procedures
    CONTEXTUAL = "contextual"  # Context-dependent information


@dataclass
class MemoryEntry:
    """Represents a single memory entry."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    memory_type: MemoryType = MemoryType.SHORT_TERM
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    associations: Set[str] = field(default_factory=set)  # IDs of related memories
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # Importance and decay
    importance: float = 1.0  # 0.0 to 1.0
    decay_rate: float = 0.1  # How fast importance decays
    min_importance: float = 0.1  # Minimum importance before deletion
    
    # Context information
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    context_window: Optional[str] = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
        if isinstance(self.associations, list):
            self.associations = set(self.associations)
    
    def access(self) -> None:
        """Record memory access and update importance."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        # Boost importance on access
        self.importance = min(1.0, self.importance + 0.1)
    
    def decay(self, time_factor: float = 1.0) -> None:
        """Apply decay to memory importance."""
        decay_amount = self.decay_rate * time_factor
        self.importance = max(self.min_importance, self.importance - decay_amount)
    
    def should_delete(self) -> bool:
        """Check if memory should be deleted due to low importance."""
        return self.importance <= self.min_importance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "associations": list(self.associations),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "decay_rate": self.decay_rate,
            "min_importance": self.min_importance,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "context_window": self.context_window,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
        data = data.copy()
        data["memory_type"] = MemoryType(data["memory_type"])
        data["tags"] = set(data.get("tags", []))
        data["associations"] = set(data.get("associations", []))
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "last_accessed" in data:
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        return cls(**data)


class ContextMemory:
    """
    Context-aware memory that maintains associations between related information.
    """
    
    def __init__(self, context_id: str, max_size: int = 1000):
        """Initialize context memory."""
        self.context_id = context_id
        self.max_size = max_size
        self.memories: Dict[str, MemoryEntry] = {}
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.content_index: Dict[str, Set[str]] = defaultdict(set)
        self.association_graph: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
    
    def add_memory(self, memory: MemoryEntry) -> None:
        """Add a memory to this context."""
        with self._lock:
            memory.context_window = self.context_id
            self.memories[memory.id] = memory
            
            # Update indices
            self._update_indices(memory)
            
            # Manage size
            if len(self.memories) > self.max_size:
                self._evict_memories()
    
    def _update_indices(self, memory: MemoryEntry) -> None:
        """Update search indices for a memory."""
        # Tag index
        for tag in memory.tags:
            self.tag_index[tag].add(memory.id)
        
        # Content index (simple keyword indexing)
        words = memory.content.lower().split()
        for word in words:
            if len(word) > 2:  # Skip short words
                self.content_index[word].add(memory.id)
        
        # Association graph
        for assoc_id in memory.associations:
            self.association_graph[memory.id].add(assoc_id)
            self.association_graph[assoc_id].add(memory.id)
    
    def _evict_memories(self) -> None:
        """Evict least important memories when at capacity."""
        # Sort by importance and last access
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: (m.importance, m.last_accessed)
        )
        
        # Remove lowest importance memories
        to_remove = len(self.memories) - self.max_size + 1
        for memory in sorted_memories[:to_remove]:
            self.remove_memory(memory.id)
    
    def remove_memory(self, memory_id: str) -> bool:
        """Remove a memory from this context."""
        with self._lock:
            if memory_id not in self.memories:
                return False
            
            memory = self.memories.pop(memory_id)
            
            # Update indices
            for tag in memory.tags:
                self.tag_index[tag].discard(memory_id)
                if not self.tag_index[tag]:
                    del self.tag_index[tag]
            
            words = memory.content.lower().split()
            for word in words:
                if word in self.content_index:
                    self.content_index[word].discard(memory_id)
                    if not self.content_index[word]:
                        del self.content_index[word]
            
            # Remove from association graph
            for assoc_id in self.association_graph[memory_id]:
                self.association_graph[assoc_id].discard(memory_id)
            del self.association_graph[memory_id]
            
            return True
    
    def search(
        self,
        query: str = "",
        tags: Optional[Set[str]] = None,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
        max_results: int = 10
    ) -> List[MemoryEntry]:
        """Search memories in this context."""
        with self._lock:
            candidates = set(self.memories.keys())
            
            # Filter by query
            if query:
                query_words = query.lower().split()
                query_matches = set()
                for word in query_words:
                    if word in self.content_index:
                        query_matches.update(self.content_index[word])
                candidates &= query_matches
            
            # Filter by tags
            if tags:
                tag_matches = set()
                for tag in tags:
                    if tag in self.tag_index:
                        tag_matches.update(self.tag_index[tag])
                candidates &= tag_matches
            
            # Filter by memory type and importance
            results = []
            for memory_id in candidates:
                memory = self.memories[memory_id]
                if (memory_type is None or memory.memory_type == memory_type) and \
                   memory.importance >= min_importance:
                    memory.access()  # Record access
                    results.append(memory)
            
            # Sort by relevance (importance + recency)
            results.sort(key=lambda m: (m.importance, m.last_accessed), reverse=True)
            return results[:max_results]
    
    def get_associated_memories(self, memory_id: str, depth: int = 1) -> List[MemoryEntry]:
        """Get memories associated with a given memory."""
        with self._lock:
            if memory_id not in self.memories:
                return []
            
            visited = set()
            to_visit = deque([(memory_id, 0)])
            results = []
            
            while to_visit:
                current_id, current_depth = to_visit.popleft()
                
                if current_id in visited or current_depth > depth:
                    continue
                
                visited.add(current_id)
                
                if current_id != memory_id and current_id in self.memories:
                    memory = self.memories[current_id]
                    memory.access()
                    results.append(memory)
                
                # Add associations for next level
                if current_depth < depth:
                    for assoc_id in self.association_graph.get(current_id, set()):
                        if assoc_id not in visited:
                            to_visit.append((assoc_id, current_depth + 1))
            
            return results


class MemoryManager:
    """
    Advanced memory management system for ABOV3 4 Ollama.
    
    Features:
    - Multiple memory types with different characteristics
    - Persistent storage with SQLite backend
    - Memory decay and cleanup
    - Context-aware memory associations
    - Efficient search and retrieval
    - Thread-safe operations
    - Async support for heavy operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the memory manager."""
        self.config = get_config()
        self._custom_config = config or {}
        
        # Memory settings
        self.max_short_term = self._get_setting("max_short_term", 1000)
        self.max_long_term = self._get_setting("max_long_term", 10000)
        self.decay_interval = self._get_setting("decay_interval", 3600)  # 1 hour
        self.cleanup_interval = self._get_setting("cleanup_interval", 86400)  # 24 hours
        
        # Storage
        self.storage_path = Path(self.config.get_data_dir()) / "memory"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "memory.db"
        
        # In-memory storage for performance
        self.short_term_memories: Dict[str, MemoryEntry] = {}
        self.context_memories: Dict[str, ContextMemory] = {}
        
        # Indices for fast search
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.type_index: Dict[MemoryType, Set[str]] = defaultdict(set)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self._stop_background = threading.Event()
        self._background_thread = threading.Thread(target=self._background_worker, daemon=True)
        
        # Initialize database
        self._init_database()
        
        # Load existing memories
        self._load_memories()
        
        # Start background worker
        self._background_thread.start()
        
        logger.info("MemoryManager initialized")
    
    def _get_setting(self, key: str, default: Any) -> Any:
        """Get setting from custom config or default."""
        return self._custom_config.get(key, default)
    
    def _init_database(self) -> None:
        """Initialize SQLite database for persistent storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    tags TEXT,
                    associations TEXT,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    importance REAL DEFAULT 1.0,
                    decay_rate REAL DEFAULT 0.1,
                    min_importance REAL DEFAULT 0.1,
                    session_id TEXT,
                    conversation_id TEXT,
                    context_window TEXT
                )
            """)
            
            # Create indices for better search performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation ON memories(conversation_id)")
            
            conn.commit()
    
    def _load_memories(self) -> None:
        """Load memories from database into memory."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM memories")
            
            for row in cursor.fetchall():
                memory_data = dict(row)
                
                # Parse JSON fields
                memory_data["metadata"] = json.loads(memory_data["metadata"] or "{}")
                memory_data["tags"] = set(json.loads(memory_data["tags"] or "[]"))
                memory_data["associations"] = set(json.loads(memory_data["associations"] or "[]"))
                
                memory = MemoryEntry.from_dict(memory_data)
                
                # Load into appropriate storage
                if memory.memory_type == MemoryType.SHORT_TERM:
                    self.short_term_memories[memory.id] = memory
                
                # Update indices
                self._update_indices(memory)
        
        logger.info(f"Loaded {len(self.short_term_memories)} memories from database")
    
    def _save_memory_to_db(self, memory: MemoryEntry) -> None:
        """Save a memory to the database."""
        with sqlite3.connect(self.db_path) as conn:
            data = memory.to_dict()
            conn.execute("""
                INSERT OR REPLACE INTO memories 
                (id, memory_type, content, metadata, tags, associations,
                 created_at, last_accessed, access_count, importance,
                 decay_rate, min_importance, session_id, conversation_id, context_window)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["id"], data["memory_type"], data["content"],
                json.dumps(data["metadata"]), json.dumps(data["tags"]),
                json.dumps(data["associations"]), data["created_at"],
                data["last_accessed"], data["access_count"], data["importance"],
                data["decay_rate"], data["min_importance"], data["session_id"],
                data["conversation_id"], data["context_window"]
            ))
            conn.commit()
    
    def _update_indices(self, memory: MemoryEntry) -> None:
        """Update search indices for a memory."""
        for tag in memory.tags:
            self.tag_index[tag].add(memory.id)
        
        self.type_index[memory.memory_type].add(memory.id)
    
    def store_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        associations: Optional[Set[str]] = None
    ) -> str:
        """
        Store a new memory.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            tags: Associated tags
            metadata: Additional metadata
            importance: Initial importance (0.0 to 1.0)
            session_id: Associated session ID
            conversation_id: Associated conversation ID
            associations: IDs of related memories
            
        Returns:
            str: Memory ID
        """
        with self._lock:
            memory = MemoryEntry(
                memory_type=memory_type,
                content=content,
                tags=tags or set(),
                metadata=metadata or {},
                importance=importance,
                session_id=session_id,
                conversation_id=conversation_id,
                associations=associations or set()
            )
            
            # Store in appropriate location
            if memory_type == MemoryType.SHORT_TERM:
                self.short_term_memories[memory.id] = memory
                
                # Manage size
                if len(self.short_term_memories) > self.max_short_term:
                    self._evict_short_term_memories()
            
            # Save to database for persistence
            self._save_memory_to_db(memory)
            
            # Update indices
            self._update_indices(memory)
            
            logger.debug(f"Stored {memory_type.value} memory {memory.id}")
            return memory.id
    
    def _evict_short_term_memories(self) -> None:
        """Evict least important short-term memories."""
        # Sort by importance and last access
        sorted_memories = sorted(
            self.short_term_memories.values(),
            key=lambda m: (m.importance, m.last_accessed)
        )
        
        # Remove lowest importance memories
        to_remove = len(self.short_term_memories) - self.max_short_term + 1
        for memory in sorted_memories[:to_remove]:
            self.forget_memory(memory.id)
    
    def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory by ID."""
        with self._lock:
            # Check short-term first
            if memory_id in self.short_term_memories:
                memory = self.short_term_memories[memory_id]
                memory.access()
                self._save_memory_to_db(memory)  # Update access info
                return memory
            
            # Check database for long-term memories
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
                row = cursor.fetchone()
                
                if row:
                    memory_data = dict(row)
                    memory_data["metadata"] = json.loads(memory_data["metadata"] or "{}")
                    memory_data["tags"] = set(json.loads(memory_data["tags"] or "[]"))
                    memory_data["associations"] = set(json.loads(memory_data["associations"] or "[]"))
                    
                    memory = MemoryEntry.from_dict(memory_data)
                    memory.access()
                    self._save_memory_to_db(memory)
                    return memory
            
            return None
    
    def search_memories(
        self,
        query: str = "",
        memory_type: Optional[MemoryType] = None,
        tags: Optional[Set[str]] = None,
        min_importance: float = 0.0,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        max_results: int = 10,
        include_associations: bool = False
    ) -> List[MemoryEntry]:
        """
        Search memories with various filters.
        
        Args:
            query: Text query to search in content
            memory_type: Filter by memory type
            tags: Filter by tags (AND operation)
            min_importance: Minimum importance threshold
            session_id: Filter by session ID
            conversation_id: Filter by conversation ID
            max_results: Maximum number of results
            include_associations: Whether to include associated memories
            
        Returns:
            List of matching memories
        """
        with self._lock:
            # Build SQL query
            conditions = ["importance >= ?"]
            params = [min_importance]
            
            if query:
                conditions.append("content LIKE ?")
                params.append(f"%{query}%")
            
            if memory_type:
                conditions.append("memory_type = ?")
                params.append(memory_type.value)
            
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)
            
            if conversation_id:
                conditions.append("conversation_id = ?")
                params.append(conversation_id)
            
            sql = f"""
                SELECT * FROM memories 
                WHERE {' AND '.join(conditions)}
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
            """
            params.append(max_results)
            
            results = []
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)
                
                for row in cursor.fetchall():
                    memory_data = dict(row)
                    memory_data["metadata"] = json.loads(memory_data["metadata"] or "{}")
                    memory_data["tags"] = set(json.loads(memory_data["tags"] or "[]"))
                    memory_data["associations"] = set(json.loads(memory_data["associations"] or "[]"))
                    
                    memory = MemoryEntry.from_dict(memory_data)
                    
                    # Filter by tags if specified
                    if tags and not tags.issubset(memory.tags):
                        continue
                    
                    memory.access()
                    results.append(memory)
            
            # Include associated memories if requested
            if include_associations:
                associated_ids = set()
                for memory in results:
                    associated_ids.update(memory.associations)
                
                for assoc_id in associated_ids:
                    if len(results) >= max_results:
                        break
                    assoc_memory = self.retrieve_memory(assoc_id)
                    if assoc_memory and assoc_memory not in results:
                        results.append(assoc_memory)
            
            # Update database with access info
            for memory in results:
                self._save_memory_to_db(memory)
            
            return results[:max_results]
    
    def forget_memory(self, memory_id: str) -> bool:
        """Remove a memory completely."""
        with self._lock:
            # Remove from in-memory storage
            removed = False
            if memory_id in self.short_term_memories:
                del self.short_term_memories[memory_id]
                removed = True
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                if cursor.rowcount > 0:
                    removed = True
                conn.commit()
            
            # Update indices
            for tag_set in self.tag_index.values():
                tag_set.discard(memory_id)
            
            for type_set in self.type_index.values():
                type_set.discard(memory_id)
            
            return removed
    
    def get_context_memory(self, context_id: str) -> ContextMemory:
        """Get or create a context memory instance."""
        with self._lock:
            if context_id not in self.context_memories:
                self.context_memories[context_id] = ContextMemory(context_id)
            return self.context_memories[context_id]
    
    def create_association(self, memory_id1: str, memory_id2: str) -> bool:
        """Create an association between two memories."""
        with self._lock:
            memory1 = self.retrieve_memory(memory_id1)
            memory2 = self.retrieve_memory(memory_id2)
            
            if not memory1 or not memory2:
                return False
            
            memory1.associations.add(memory_id2)
            memory2.associations.add(memory_id1)
            
            # Save updated memories
            self._save_memory_to_db(memory1)
            self._save_memory_to_db(memory2)
            
            return True
    
    def _background_worker(self) -> None:
        """Background worker for memory maintenance."""
        last_decay = time.time()
        last_cleanup = time.time()
        
        while not self._stop_background.wait(60):  # Check every minute
            current_time = time.time()
            
            # Apply decay
            if current_time - last_decay >= self.decay_interval:
                self._apply_decay()
                last_decay = current_time
            
            # Cleanup old memories
            if current_time - last_cleanup >= self.cleanup_interval:
                self._cleanup_memories()
                last_cleanup = current_time
    
    def _apply_decay(self) -> None:
        """Apply decay to all memories."""
        with self._lock:
            current_time = datetime.now()
            
            # Decay short-term memories
            to_remove = []
            for memory in self.short_term_memories.values():
                time_diff = (current_time - memory.last_accessed).total_seconds() / 3600  # hours
                memory.decay(time_diff)
                
                if memory.should_delete():
                    to_remove.append(memory.id)
            
            # Remove decayed memories
            for memory_id in to_remove:
                self.forget_memory(memory_id)
            
            logger.debug(f"Applied decay, removed {len(to_remove)} memories")
    
    def _cleanup_memories(self) -> None:
        """Clean up old and unimportant memories from database."""
        cutoff_date = datetime.now() - timedelta(days=30)  # Remove memories older than 30 days
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM memories 
                WHERE importance < 0.1 AND last_accessed < ?
            """, (cutoff_date.isoformat(),))
            
            removed_count = cursor.rowcount
            conn.commit()
        
        logger.info(f"Cleaned up {removed_count} old memories")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        with self._lock:
            stats = {
                "short_term_count": len(self.short_term_memories),
                "max_short_term": self.max_short_term,
                "context_memories": len(self.context_memories),
            }
            
            # Get database stats
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM memories")
                stats["total_memories"] = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT memory_type, COUNT(*) 
                    FROM memories 
                    GROUP BY memory_type
                """)
                stats["by_type"] = dict(cursor.fetchall())
                
                cursor = conn.execute("""
                    SELECT AVG(importance), MIN(importance), MAX(importance)
                    FROM memories
                """)
                avg_imp, min_imp, max_imp = cursor.fetchone()
                stats["importance"] = {
                    "average": avg_imp or 0,
                    "minimum": min_imp or 0,
                    "maximum": max_imp or 0
                }
            
            return stats
    
    def export_memories(
        self,
        format: str = "json",
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0
    ) -> Union[str, Dict[str, Any]]:
        """
        Export memories to various formats.
        
        Args:
            format: Export format ("json", "dict")
            memory_type: Filter by memory type
            min_importance: Minimum importance threshold
            
        Returns:
            Exported memory data
        """
        memories = self.search_memories(
            memory_type=memory_type,
            min_importance=min_importance,
            max_results=10000  # Large number to get all
        )
        
        data = {
            "exported_at": datetime.now().isoformat(),
            "total_memories": len(memories),
            "memories": [memory.to_dict() for memory in memories],
            "stats": self.get_memory_stats(),
        }
        
        if format == "json":
            return json.dumps(data, indent=2, ensure_ascii=False)
        elif format == "dict":
            return data
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_memories(self, data: Union[str, Dict[str, Any]]) -> int:
        """
        Import memories from exported data.
        
        Args:
            data: Memory data (JSON string or dictionary)
            
        Returns:
            Number of imported memories
        """
        if isinstance(data, str):
            import_data = json.loads(data)
        else:
            import_data = data
        
        imported_count = 0
        for memory_data in import_data.get("memories", []):
            memory = MemoryEntry.from_dict(memory_data)
            
            # Store the memory
            with self._lock:
                if memory.memory_type == MemoryType.SHORT_TERM:
                    self.short_term_memories[memory.id] = memory
                
                self._save_memory_to_db(memory)
                self._update_indices(memory)
                imported_count += 1
        
        logger.info(f"Imported {imported_count} memories")
        return imported_count
    
    def shutdown(self) -> None:
        """Shutdown the memory manager."""
        self._stop_background.set()
        if self._background_thread.is_alive():
            self._background_thread.join(timeout=5)
        
        # Final save of all memories
        with self._lock:
            for memory in self.short_term_memories.values():
                self._save_memory_to_db(memory)
        
        logger.info("MemoryManager shutdown complete")
    
    async def async_search(self, **kwargs) -> List[MemoryEntry]:
        """Async version of search_memories."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.search_memories(**kwargs)
        )