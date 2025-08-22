"""
Session management system for ABOV3 4 Ollama.

This module provides comprehensive session management capabilities including:
- Conversation session creation and management
- Session persistence and restoration
- Multi-session handling
- Session metadata and statistics
- Session export/import functionality
- Session search and filtering
- Automatic session cleanup and archiving
"""

import asyncio
import json
import logging
import shutil
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from .manager import ContextManager, ContextEntry
from .memory import MemoryManager, MemoryType
from ..config import get_config


logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    """Metadata for a conversation session."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    
    # Session statistics
    message_count: int = 0
    total_tokens: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    duration: timedelta = field(default_factory=timedelta)
    
    # Configuration
    model_config: Dict[str, Any] = field(default_factory=dict)
    context_config: Dict[str, Any] = field(default_factory=dict)
    
    # State
    is_active: bool = True
    is_archived: bool = False
    is_favorite: bool = False
    
    def __post_init__(self):
        """Initialize computed fields."""
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
        if not self.name:
            self.name = f"Session {self.id[:8]}"
    
    def update_stats(self, message: ContextEntry) -> None:
        """Update session statistics with a new message."""
        self.message_count += 1
        self.total_tokens += message.token_count
        self.last_modified = datetime.now()
        
        if message.role == "user":
            self.user_messages += 1
        elif message.role == "assistant":
            self.assistant_messages += 1
        
        # Update duration
        self.duration = self.last_modified - self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
            "user_messages": self.user_messages,
            "assistant_messages": self.assistant_messages,
            "duration": self.duration.total_seconds(),
            "model_config": self.model_config,
            "context_config": self.context_config,
            "is_active": self.is_active,
            "is_archived": self.is_archived,
            "is_favorite": self.is_favorite,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMetadata":
        """Create from dictionary."""
        data = data.copy()
        data["tags"] = set(data.get("tags", []))
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "last_accessed" in data:
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        if "last_modified" in data:
            data["last_modified"] = datetime.fromisoformat(data["last_modified"])
        if "duration" in data:
            data["duration"] = timedelta(seconds=data["duration"])
        return cls(**data)


class ConversationSession:
    """
    Represents a single conversation session with context and memory management.
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        context_config: Optional[Dict[str, Any]] = None,
        memory_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize a conversation session."""
        self.metadata = SessionMetadata(id=session_id or str(uuid4()))
        
        # Initialize context and memory managers
        self.context_manager = ContextManager(context_config)
        self.memory_manager = MemoryManager(memory_config)
        
        # Session state
        self._lock = threading.RLock()
        self._message_history: List[ContextEntry] = []
        self._is_restored = False
        
        logger.info(f"Created conversation session {self.metadata.id}")
    
    @property
    def id(self) -> str:
        """Get session ID."""
        return self.metadata.id
    
    @property
    def name(self) -> str:
        """Get session name."""
        return self.metadata.name
    
    @name.setter
    def name(self, value: str) -> None:
        """Set session name."""
        with self._lock:
            self.metadata.name = value
            self.metadata.last_modified = datetime.now()
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        store_memory: bool = True,
        memory_importance: float = 1.0
    ) -> str:
        """
        Add a message to the session.
        
        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            metadata: Optional metadata
            store_memory: Whether to store in memory
            memory_importance: Importance for memory storage
            
        Returns:
            Message ID
        """
        with self._lock:
            # Add to context manager
            message_id = self.context_manager.add_message(
                role=role,
                content=content,
                metadata=metadata
            )
            
            # Get the created entry for history tracking
            context_messages = self.context_manager.get_context_for_model()
            if context_messages:
                # Find the entry we just added
                for entry in self.context_manager._context_window.entries:
                    if entry.id == message_id:
                        self._message_history.append(entry)
                        self.metadata.update_stats(entry)
                        break
            
            # Store in memory if requested
            if store_memory and content.strip():
                memory_tags = {role, "conversation"}
                if metadata:
                    memory_tags.update(metadata.get("tags", []))
                
                self.memory_manager.store_memory(
                    content=content,
                    memory_type=MemoryType.EPISODIC,
                    tags=memory_tags,
                    metadata={
                        "role": role,
                        "session_id": self.id,
                        "message_id": message_id,
                        **(metadata or {})
                    },
                    importance=memory_importance,
                    session_id=self.id
                )
            
            self.metadata.last_accessed = datetime.now()
            return message_id
    
    def get_context_for_model(self, include_memory: bool = False) -> List[Dict[str, str]]:
        """
        Get context formatted for model consumption.
        
        Args:
            include_memory: Whether to include relevant memories
            
        Returns:
            List of messages for the model
        """
        with self._lock:
            messages = self.context_manager.get_context_for_model()
            
            if include_memory and messages:
                # Get relevant memories based on recent context
                recent_content = " ".join([msg["content"] for msg in messages[-3:]])
                relevant_memories = self.memory_manager.search_memories(
                    query=recent_content[:200],  # Use first 200 chars for search
                    session_id=self.id,
                    max_results=5,
                    min_importance=0.5
                )
                
                # Add memory context before conversation
                if relevant_memories:
                    memory_content = "Relevant context from previous conversations:\n"
                    for memory in relevant_memories:
                        memory_content += f"- {memory.content[:100]}...\n"
                    
                    messages.insert(0, {
                        "role": "system",
                        "content": memory_content
                    })
            
            return messages
    
    def search_messages(
        self,
        query: str,
        role: Optional[str] = None,
        max_results: int = 10
    ) -> List[ContextEntry]:
        """
        Search messages in this session.
        
        Args:
            query: Search query
            role: Filter by message role
            max_results: Maximum results
            
        Returns:
            List of matching messages
        """
        with self._lock:
            results = []
            query_lower = query.lower()
            
            for entry in self._message_history:
                if (not role or entry.role == role) and \
                   query_lower in entry.content.lower():
                    results.append(entry)
            
            # Sort by relevance and recency
            results.sort(key=lambda e: e.timestamp, reverse=True)
            return results[:max_results]
    
    def get_session_summary(self, max_length: int = 500) -> str:
        """Get a summary of the session."""
        with self._lock:
            if not self._message_history:
                return "No messages in this session."
            
            # Use context manager's summary feature
            return self.context_manager.get_conversation_summary(max_length)
    
    def clear_context(self, keep_system: bool = True) -> None:
        """Clear the session context."""
        with self._lock:
            self.context_manager.clear_context(keep_system)
            self.metadata.last_modified = datetime.now()
    
    def archive(self) -> None:
        """Archive this session."""
        with self._lock:
            self.metadata.is_archived = True
            self.metadata.is_active = False
            self.metadata.last_modified = datetime.now()
    
    def restore(self) -> None:
        """Restore this session from archive."""
        with self._lock:
            self.metadata.is_archived = False
            self.metadata.is_active = True
            self.metadata.last_accessed = datetime.now()
            self.metadata.last_modified = datetime.now()
    
    def export_session(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Export session data."""
        with self._lock:
            data = {
                "metadata": self.metadata.to_dict(),
                "context": self.context_manager.export_context("dict"),
                "message_history": [msg.to_dict() for msg in self._message_history],
                "memories": self.memory_manager.search_memories(
                    session_id=self.id,
                    max_results=1000
                ),
                "exported_at": datetime.now().isoformat(),
            }
            
            if format == "json":
                return json.dumps(data, indent=2, ensure_ascii=False)
            elif format == "dict":
                return data
            else:
                raise ValueError(f"Unsupported format: {format}")


class SessionManager:
    """
    Advanced session management system for ABOV3 4 Ollama.
    
    Features:
    - Multi-session handling
    - Session persistence and restoration
    - Session search and filtering
    - Automatic cleanup and archiving
    - Session templates and presets
    - Export/import functionality
    - Session analytics and statistics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the session manager."""
        self.config = get_config()
        self._custom_config = config or {}
        
        # Session settings
        self.max_active_sessions = self._get_setting("max_active_sessions", 10)
        self.auto_archive_days = self._get_setting("auto_archive_days", 30)
        self.cleanup_interval = self._get_setting("cleanup_interval", 86400)  # 24 hours
        
        # Storage
        self.storage_path = Path(self.config.get_data_dir()) / "sessions"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "sessions.db"
        
        # Session management
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.current_session_id: Optional[str] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self._stop_background = threading.Event()
        self._background_thread = threading.Thread(target=self._background_worker, daemon=True)
        
        # Initialize database
        self._init_database()
        
        # Load existing sessions
        self._load_sessions()
        
        # Start background worker
        self._background_thread.start()
        
        logger.info("SessionManager initialized")
    
    def _get_setting(self, key: str, default: Any) -> Any:
        """Get setting from custom config or default."""
        return self._custom_config.get(key, default)
    
    def _init_database(self) -> None:
        """Initialize SQLite database for session metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    user_messages INTEGER DEFAULT 0,
                    assistant_messages INTEGER DEFAULT 0,
                    duration REAL DEFAULT 0,
                    model_config TEXT,
                    context_config TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    is_archived BOOLEAN DEFAULT 0,
                    is_favorite BOOLEAN DEFAULT 0
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_name ON sessions(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_active ON sessions(is_active)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_archived ON sessions(is_archived)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_last_accessed ON sessions(last_accessed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_favorite ON sessions(is_favorite)")
            
            conn.commit()
    
    def _load_sessions(self) -> None:
        """Load active sessions from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM sessions 
                WHERE is_active = 1 AND is_archived = 0
                ORDER BY last_accessed DESC
                LIMIT ?
            """, (self.max_active_sessions,))
            
            loaded_count = 0
            for row in cursor.fetchall():
                session_data = dict(row)
                session_data["tags"] = set(json.loads(session_data["tags"] or "[]"))
                session_data["model_config"] = json.loads(session_data["model_config"] or "{}")
                session_data["context_config"] = json.loads(session_data["context_config"] or "{}")
                
                metadata = SessionMetadata.from_dict(session_data)
                
                # Create session object (will be lazy-loaded when accessed)
                # For now, just track the metadata
                loaded_count += 1
        
        logger.info(f"Session metadata loaded for {loaded_count} sessions")
    
    def _save_session_metadata(self, metadata: SessionMetadata) -> None:
        """Save session metadata to database."""
        with sqlite3.connect(self.db_path) as conn:
            data = metadata.to_dict()
            conn.execute("""
                INSERT OR REPLACE INTO sessions 
                (id, name, description, tags, created_at, last_accessed, last_modified,
                 message_count, total_tokens, user_messages, assistant_messages,
                 duration, model_config, context_config, is_active, is_archived, is_favorite)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["id"], data["name"], data["description"],
                json.dumps(data["tags"]), data["created_at"],
                data["last_accessed"], data["last_modified"],
                data["message_count"], data["total_tokens"],
                data["user_messages"], data["assistant_messages"],
                data["duration"], json.dumps(data["model_config"]),
                json.dumps(data["context_config"]), data["is_active"],
                data["is_archived"], data["is_favorite"]
            ))
            conn.commit()
    
    def create_session(
        self,
        name: Optional[str] = None,
        description: str = "",
        tags: Optional[Set[str]] = None,
        context_config: Optional[Dict[str, Any]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        make_current: bool = True
    ) -> str:
        """
        Create a new conversation session.
        
        Args:
            name: Session name
            description: Session description
            tags: Session tags
            context_config: Context manager configuration
            memory_config: Memory manager configuration
            make_current: Whether to make this the current session
            
        Returns:
            Session ID
        """
        with self._lock:
            session = ConversationSession(
                context_config=context_config,
                memory_config=memory_config
            )
            
            # Update metadata
            if name:
                session.metadata.name = name
            session.metadata.description = description
            session.metadata.tags = tags or set()
            session.metadata.context_config = context_config or {}
            session.metadata.model_config = self.config.get_model_params()
            
            # Add to active sessions
            self.active_sessions[session.id] = session
            
            # Make current if requested
            if make_current:
                self.current_session_id = session.id
            
            # Save metadata
            self._save_session_metadata(session.metadata)
            
            # Manage session limit
            self._manage_session_limit()
            
            logger.info(f"Created session {session.id} ({session.name})")
            return session.id
    
    def _manage_session_limit(self) -> None:
        """Manage the number of active sessions."""
        if len(self.active_sessions) <= self.max_active_sessions:
            return
        
        # Sort by last access time
        sessions_by_access = sorted(
            self.active_sessions.items(),
            key=lambda x: x[1].metadata.last_accessed
        )
        
        # Archive oldest sessions
        to_remove = len(self.active_sessions) - self.max_active_sessions
        for session_id, session in sessions_by_access[:to_remove]:
            if session_id != self.current_session_id:  # Don't remove current session
                self._archive_session(session_id)
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a session by ID."""
        with self._lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.metadata.last_accessed = datetime.now()
                self._save_session_metadata(session.metadata)
                return session
            
            # Try to load from storage
            return self._load_session_from_storage(session_id)
    
    def _load_session_from_storage(self, session_id: str) -> Optional[ConversationSession]:
        """Load a session from persistent storage."""
        session_file = self.storage_path / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Create session and restore state
            session = ConversationSession(session_id=session_id)
            
            # Restore metadata
            session.metadata = SessionMetadata.from_dict(data["metadata"])
            
            # Restore context if available
            if "context" in data:
                session.context_manager.import_context(data["context"])
            
            # Restore message history
            if "message_history" in data:
                for msg_data in data["message_history"]:
                    entry = ContextEntry.from_dict(msg_data)
                    session._message_history.append(entry)
            
            session._is_restored = True
            session.metadata.last_accessed = datetime.now()
            
            # Add to active sessions
            self.active_sessions[session_id] = session
            self._save_session_metadata(session.metadata)
            
            logger.info(f"Loaded session {session_id} from storage")
            return session
        
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def get_current_session(self) -> Optional[ConversationSession]:
        """Get the current active session."""
        if self.current_session_id:
            return self.get_session(self.current_session_id)
        return None
    
    def set_current_session(self, session_id: str) -> bool:
        """Set the current active session."""
        with self._lock:
            session = self.get_session(session_id)
            if session:
                self.current_session_id = session_id
                return True
            return False
    
    def list_sessions(
        self,
        include_archived: bool = False,
        tags: Optional[Set[str]] = None,
        name_filter: Optional[str] = None,
        max_results: int = 50
    ) -> List[SessionMetadata]:
        """
        List sessions with filtering options.
        
        Args:
            include_archived: Whether to include archived sessions
            tags: Filter by tags (AND operation)
            name_filter: Filter by name (substring match)
            max_results: Maximum number of results
            
        Returns:
            List of session metadata
        """
        with self._lock:
            # Build SQL query
            conditions = []
            params = []
            
            if not include_archived:
                conditions.append("is_archived = 0")
            
            if name_filter:
                conditions.append("name LIKE ?")
                params.append(f"%{name_filter}%")
            
            where_clause = ""
            if conditions:
                where_clause = f"WHERE {' AND '.join(conditions)}"
            
            sql = f"""
                SELECT * FROM sessions 
                {where_clause}
                ORDER BY is_favorite DESC, last_accessed DESC
                LIMIT ?
            """
            params.append(max_results)
            
            results = []
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)
                
                for row in cursor.fetchall():
                    session_data = dict(row)
                    session_data["tags"] = set(json.loads(session_data["tags"] or "[]"))
                    session_data["model_config"] = json.loads(session_data["model_config"] or "{}")
                    session_data["context_config"] = json.loads(session_data["context_config"] or "{}")
                    
                    metadata = SessionMetadata.from_dict(session_data)
                    
                    # Filter by tags if specified
                    if tags and not tags.issubset(metadata.tags):
                        continue
                    
                    results.append(metadata)
            
            return results
    
    def search_sessions(
        self,
        query: str,
        search_content: bool = True,
        max_results: int = 20
    ) -> List[Tuple[SessionMetadata, float]]:
        """
        Search sessions by name, description, or content.
        
        Args:
            query: Search query
            search_content: Whether to search message content
            max_results: Maximum number of results
            
        Returns:
            List of (metadata, relevance_score) tuples
        """
        results = []
        query_lower = query.lower()
        
        # Search session metadata
        sessions = self.list_sessions(include_archived=True, max_results=1000)
        for metadata in sessions:
            score = 0.0
            
            # Name match (highest weight)
            if query_lower in metadata.name.lower():
                score += 3.0
            
            # Description match
            if query_lower in metadata.description.lower():
                score += 2.0
            
            # Tag match
            for tag in metadata.tags:
                if query_lower in tag.lower():
                    score += 1.0
            
            if score > 0:
                results.append((metadata, score))
        
        # Search message content if requested
        if search_content:
            for session_id, session in self.active_sessions.items():
                content_matches = session.search_messages(query, max_results=5)
                if content_matches:
                    # Find metadata for this session
                    metadata = next((m for m in sessions if m.id == session_id), None)
                    if metadata:
                        # Add content relevance score
                        content_score = len(content_matches) * 0.5
                        existing_result = next((r for r in results if r[0].id == session_id), None)
                        if existing_result:
                            # Update existing score
                            idx = results.index(existing_result)
                            results[idx] = (metadata, existing_result[1] + content_score)
                        else:
                            results.append((metadata, content_score))
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def archive_session(self, session_id: str) -> bool:
        """Archive a session."""
        return self._archive_session(session_id)
    
    def _archive_session(self, session_id: str) -> bool:
        """Internal method to archive a session."""
        with self._lock:
            session = self.get_session(session_id)
            if not session:
                return False
            
            # Save session to storage before archiving
            self._save_session_to_storage(session)
            
            # Archive the session
            session.archive()
            self._save_session_metadata(session.metadata)
            
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Update current session if needed
            if self.current_session_id == session_id:
                self.current_session_id = None
                # Try to set a new current session
                if self.active_sessions:
                    self.current_session_id = list(self.active_sessions.keys())[0]
            
            logger.info(f"Archived session {session_id}")
            return True
    
    def restore_session(self, session_id: str) -> bool:
        """Restore an archived session."""
        with self._lock:
            # Load session from storage
            session = self._load_session_from_storage(session_id)
            if not session:
                return False
            
            session.restore()
            self._save_session_metadata(session.metadata)
            
            # Manage session limit
            self._manage_session_limit()
            
            logger.info(f"Restored session {session_id}")
            return True
    
    def delete_session(self, session_id: str) -> bool:
        """Permanently delete a session."""
        with self._lock:
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
            
            # Remove session file
            session_file = self.storage_path / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            # Update current session if needed
            if self.current_session_id == session_id:
                self.current_session_id = None
                if self.active_sessions:
                    self.current_session_id = list(self.active_sessions.keys())[0]
            
            logger.info(f"Deleted session {session_id}")
            return deleted
    
    def _save_session_to_storage(self, session: ConversationSession) -> None:
        """Save session data to persistent storage."""
        session_file = self.storage_path / f"{session.id}.json"
        
        try:
            data = session.export_session("dict")
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save session {session.id} to storage: {e}")
    
    def export_session(self, session_id: str, format: str = "json") -> Optional[str]:
        """Export a session to a file or string."""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return session.export_session(format)
    
    def import_session(self, data: Union[str, Dict[str, Any]]) -> Optional[str]:
        """
        Import a session from exported data.
        
        Args:
            data: Session data (JSON string or dictionary)
            
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            if isinstance(data, str):
                import_data = json.loads(data)
            else:
                import_data = data
            
            # Create new session
            session_id = self.create_session(make_current=False)
            session = self.get_session(session_id)
            
            if not session:
                return None
            
            # Import metadata
            if "metadata" in import_data:
                imported_metadata = SessionMetadata.from_dict(import_data["metadata"])
                # Keep the new ID but import other fields
                imported_metadata.id = session_id
                imported_metadata.created_at = datetime.now()
                session.metadata = imported_metadata
            
            # Import context
            if "context" in import_data:
                session.context_manager.import_context(import_data["context"])
            
            # Import message history
            if "message_history" in import_data:
                for msg_data in import_data["message_history"]:
                    entry = ContextEntry.from_dict(msg_data)
                    session._message_history.append(entry)
            
            # Save to storage
            self._save_session_to_storage(session)
            self._save_session_metadata(session.metadata)
            
            logger.info(f"Imported session as {session_id}")
            return session_id
        
        except Exception as e:
            logger.error(f"Failed to import session: {e}")
            return None
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session management statistics."""
        with self._lock:
            stats = {
                "active_sessions": len(self.active_sessions),
                "current_session": self.current_session_id,
                "max_active_sessions": self.max_active_sessions,
            }
            
            # Get database stats
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM sessions")
                stats["total_sessions"] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM sessions WHERE is_archived = 1")
                stats["archived_sessions"] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM sessions WHERE is_favorite = 1")
                stats["favorite_sessions"] = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT AVG(message_count), AVG(total_tokens), AVG(duration)
                    FROM sessions WHERE is_active = 1
                """)
                avg_msgs, avg_tokens, avg_duration = cursor.fetchone()
                stats["averages"] = {
                    "messages_per_session": avg_msgs or 0,
                    "tokens_per_session": avg_tokens or 0,
                    "duration_per_session": avg_duration or 0
                }
            
            return stats
    
    def _background_worker(self) -> None:
        """Background worker for session maintenance."""
        last_cleanup = time.time()
        last_save = time.time()
        
        while not self._stop_background.wait(300):  # Check every 5 minutes
            current_time = time.time()
            
            # Save active sessions periodically
            if current_time - last_save >= 1800:  # 30 minutes
                self._save_active_sessions()
                last_save = current_time
            
            # Cleanup old sessions
            if current_time - last_cleanup >= self.cleanup_interval:
                self._cleanup_old_sessions()
                last_cleanup = current_time
    
    def _save_active_sessions(self) -> None:
        """Save all active sessions to storage."""
        with self._lock:
            for session in self.active_sessions.values():
                self._save_session_to_storage(session)
                self._save_session_metadata(session.metadata)
        
        logger.debug(f"Saved {len(self.active_sessions)} active sessions")
    
    def _cleanup_old_sessions(self) -> None:
        """Clean up old and unused sessions."""
        cutoff_date = datetime.now() - timedelta(days=self.auto_archive_days)
        
        with self._lock:
            # Auto-archive old sessions
            sessions_to_archive = []
            for session in self.active_sessions.values():
                if (session.metadata.last_accessed < cutoff_date and 
                    session.id != self.current_session_id):
                    sessions_to_archive.append(session.id)
            
            for session_id in sessions_to_archive:
                self._archive_session(session_id)
            
            # Clean up very old archived sessions (optional)
            very_old_cutoff = datetime.now() - timedelta(days=365)  # 1 year
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id FROM sessions 
                    WHERE is_archived = 1 AND last_accessed < ?
                """, (very_old_cutoff.isoformat(),))
                
                old_sessions = cursor.fetchall()
                for (session_id,) in old_sessions:
                    # Just mark as deleted, don't actually delete
                    conn.execute("""
                        UPDATE sessions SET is_active = 0 
                        WHERE id = ?
                    """, (session_id,))
                
                conn.commit()
        
        logger.info(f"Archived {len(sessions_to_archive)} old sessions")
    
    def shutdown(self) -> None:
        """Shutdown the session manager."""
        self._stop_background.set()
        if self._background_thread.is_alive():
            self._background_thread.join(timeout=10)
        
        # Save all active sessions
        self._save_active_sessions()
        
        # Shutdown session components
        for session in self.active_sessions.values():
            session.memory_manager.shutdown()
        
        logger.info("SessionManager shutdown complete")