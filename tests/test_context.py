"""
Test suite for context management system.

This module tests conversation context, memory management,
session handling, and context persistence.
"""

import json
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from abov3.core.context.manager import ContextManager
from abov3.core.context.memory import (
    ConversationMemory,
    MemoryStore,
    ShortTermMemory,
    LongTermMemory,
    WorkingMemory
)
from abov3.core.context.session import Session, SessionManager


class TestConversationMemory:
    """Test cases for ConversationMemory class."""
    
    def test_memory_initialization(self):
        """Test memory initialization."""
        memory = ConversationMemory(max_size=10)
        
        assert memory.max_size == 10
        assert len(memory.messages) == 0
        assert memory.total_tokens == 0
    
    def test_add_message(self):
        """Test adding messages to memory."""
        memory = ConversationMemory()
        
        memory.add_message("user", "Hello")
        assert len(memory.messages) == 1
        assert memory.messages[0]["role"] == "user"
        assert memory.messages[0]["content"] == "Hello"
        
        memory.add_message("assistant", "Hi there!")
        assert len(memory.messages) == 2
    
    def test_max_size_enforcement(self):
        """Test maximum size enforcement."""
        memory = ConversationMemory(max_size=3)
        
        for i in range(5):
            memory.add_message("user", f"Message {i}")
        
        # Should only keep last 3 messages
        assert len(memory.messages) == 3
        assert memory.messages[0]["content"] == "Message 2"
        assert memory.messages[-1]["content"] == "Message 4"
    
    def test_token_counting(self):
        """Test token counting functionality."""
        memory = ConversationMemory()
        
        # Add messages with token counts
        memory.add_message("user", "Hello", tokens=2)
        memory.add_message("assistant", "Hi there!", tokens=3)
        
        assert memory.total_tokens == 5
    
    def test_clear_memory(self):
        """Test clearing memory."""
        memory = ConversationMemory()
        
        memory.add_message("user", "Test")
        memory.add_message("assistant", "Response")
        
        memory.clear()
        assert len(memory.messages) == 0
        assert memory.total_tokens == 0
    
    def test_get_context(self):
        """Test getting conversation context."""
        memory = ConversationMemory()
        
        memory.add_message("user", "Question 1")
        memory.add_message("assistant", "Answer 1")
        memory.add_message("user", "Question 2")
        
        context = memory.get_context(last_n=2)
        assert len(context) == 2
        assert context[0]["content"] == "Answer 1"
        assert context[1]["content"] == "Question 2"
    
    def test_memory_summary(self):
        """Test memory summarization."""
        memory = ConversationMemory()
        
        for i in range(10):
            memory.add_message("user", f"Long message {i} with lots of content")
        
        summary = memory.get_summary()
        assert "messages" in summary
        assert summary["message_count"] == 10
        assert "total_tokens" in summary
    
    def test_memory_serialization(self):
        """Test memory serialization."""
        memory = ConversationMemory()
        
        memory.add_message("user", "Test message")
        memory.add_message("assistant", "Test response")
        
        # Serialize
        serialized = memory.to_dict()
        assert "messages" in serialized
        assert "max_size" in serialized
        
        # Deserialize
        new_memory = ConversationMemory.from_dict(serialized)
        assert len(new_memory.messages) == 2
        assert new_memory.messages[0]["content"] == "Test message"


class TestShortTermMemory:
    """Test cases for ShortTermMemory class."""
    
    def test_short_term_initialization(self):
        """Test short-term memory initialization."""
        memory = ShortTermMemory(capacity=5, ttl_minutes=30)
        
        assert memory.capacity == 5
        assert memory.ttl_minutes == 30
        assert len(memory.items) == 0
    
    def test_add_item_with_ttl(self):
        """Test adding items with TTL."""
        memory = ShortTermMemory(ttl_minutes=1)
        
        memory.add("key1", "value1")
        assert memory.get("key1") == "value1"
        
        # Simulate time passing
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(minutes=2)
            # Item should be expired
            assert memory.get("key1") is None
    
    def test_capacity_limit(self):
        """Test capacity limit enforcement."""
        memory = ShortTermMemory(capacity=3)
        
        for i in range(5):
            memory.add(f"key{i}", f"value{i}")
        
        # Should only keep last 3 items
        assert len(memory.items) == 3
        assert memory.get("key0") is None
        assert memory.get("key4") == "value4"
    
    def test_update_access_time(self):
        """Test access time update on retrieval."""
        memory = ShortTermMemory()
        
        memory.add("key1", "value1")
        initial_time = memory.items["key1"]["accessed_at"]
        
        # Access the item
        memory.get("key1")
        updated_time = memory.items["key1"]["accessed_at"]
        
        assert updated_time >= initial_time
    
    def test_remove_item(self):
        """Test removing items."""
        memory = ShortTermMemory()
        
        memory.add("key1", "value1")
        memory.add("key2", "value2")
        
        memory.remove("key1")
        assert memory.get("key1") is None
        assert memory.get("key2") == "value2"
    
    def test_clear_expired(self):
        """Test clearing expired items."""
        memory = ShortTermMemory(ttl_minutes=1)
        
        memory.add("key1", "value1")
        memory.add("key2", "value2")
        
        # Expire only key1
        with patch('datetime.datetime') as mock_datetime:
            # Set key1 as expired
            memory.items["key1"]["created_at"] = datetime.now() - timedelta(minutes=2)
            
            memory.clear_expired()
            assert memory.get("key1") is None
            assert memory.get("key2") == "value2"


class TestLongTermMemory:
    """Test cases for LongTermMemory class."""
    
    def test_long_term_initialization(self):
        """Test long-term memory initialization."""
        memory = LongTermMemory(max_size=1000)
        
        assert memory.max_size == 1000
        assert len(memory.store) == 0
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving data."""
        memory = LongTermMemory()
        
        data = {"conversation": "important", "outcome": "success"}
        memory.store("session1", data)
        
        retrieved = memory.retrieve("session1")
        assert retrieved == data
    
    def test_search_functionality(self):
        """Test searching in long-term memory."""
        memory = LongTermMemory()
        
        memory.store("session1", {"topic": "python", "content": "coding"})
        memory.store("session2", {"topic": "java", "content": "programming"})
        memory.store("session3", {"topic": "python", "content": "scripting"})
        
        # Search by topic
        results = memory.search({"topic": "python"})
        assert len(results) == 2
        assert all(r["topic"] == "python" for r in results)
    
    def test_persistence(self, temp_dir):
        """Test memory persistence to disk."""
        memory = LongTermMemory()
        memory_file = temp_dir / "long_term.pkl"
        
        memory.store("key1", {"data": "important"})
        memory.store("key2", {"data": "critical"})
        
        # Save to disk
        memory.save(memory_file)
        assert memory_file.exists()
        
        # Load into new instance
        new_memory = LongTermMemory()
        new_memory.load(memory_file)
        
        assert new_memory.retrieve("key1") == {"data": "important"}
        assert new_memory.retrieve("key2") == {"data": "critical"}
    
    def test_compression(self):
        """Test memory compression."""
        memory = LongTermMemory(compress=True)
        
        large_data = {"content": "x" * 10000}
        memory.store("large", large_data)
        
        # Data should be compressed internally
        retrieved = memory.retrieve("large")
        assert retrieved == large_data
    
    def test_indexing(self):
        """Test memory indexing for fast search."""
        memory = LongTermMemory(enable_index=True)
        
        # Store many items
        for i in range(100):
            memory.store(f"key{i}", {
                "category": f"cat{i % 10}",
                "value": i
            })
        
        # Search should be fast with index
        import time
        start = time.perf_counter()
        results = memory.search({"category": "cat5"})
        end = time.perf_counter()
        
        assert len(results) == 10
        assert (end - start) < 0.01  # Should be very fast


class TestWorkingMemory:
    """Test cases for WorkingMemory class."""
    
    def test_working_memory_initialization(self):
        """Test working memory initialization."""
        memory = WorkingMemory(capacity=7)
        
        assert memory.capacity == 7
        assert len(memory.buffer) == 0
        assert memory.focus is None
    
    def test_add_to_buffer(self):
        """Test adding items to working memory."""
        memory = WorkingMemory(capacity=3)
        
        memory.add("item1")
        memory.add("item2")
        memory.add("item3")
        
        assert len(memory.buffer) == 3
        assert "item1" in memory.buffer
    
    def test_capacity_constraint(self):
        """Test Miller's law capacity constraint (7±2)."""
        memory = WorkingMemory(capacity=7)
        
        for i in range(10):
            memory.add(f"item{i}")
        
        # Should only keep 7 items
        assert len(memory.buffer) == 7
        # Oldest items should be removed
        assert "item0" not in memory.buffer
        assert "item9" in memory.buffer
    
    def test_focus_management(self):
        """Test focus/attention management."""
        memory = WorkingMemory()
        
        memory.add("item1")
        memory.add("item2")
        memory.add("item3")
        
        memory.set_focus("item2")
        assert memory.focus == "item2"
        assert memory.get_focused() == "item2"
    
    def test_rehearsal_mechanism(self):
        """Test rehearsal to prevent decay."""
        memory = WorkingMemory()
        
        memory.add("important", priority=10)
        memory.add("normal", priority=5)
        memory.add("low", priority=1)
        
        # Rehearse important item
        memory.rehearse("important")
        
        # Important item should have refreshed timestamp
        assert memory.get_priority("important") > 10
    
    def test_chunking(self):
        """Test chunking for better memory utilization."""
        memory = WorkingMemory()
        
        # Add related items
        memory.add_chunk("numbers", ["1", "2", "3", "4"])
        memory.add_chunk("letters", ["a", "b", "c", "d"])
        
        # Chunks count as single items
        assert len(memory.buffer) == 2
        assert memory.get_chunk("numbers") == ["1", "2", "3", "4"]


class TestContextManager:
    """Test cases for ContextManager class."""
    
    @pytest.fixture
    def context_manager(self):
        """Create a test context manager."""
        return ContextManager()
    
    def test_context_initialization(self, context_manager):
        """Test context manager initialization."""
        assert context_manager.current_context is not None
        assert len(context_manager.history) == 0
    
    def test_add_to_context(self, context_manager):
        """Test adding messages to context."""
        context_manager.add_message("user", "Hello")
        context_manager.add_message("assistant", "Hi!")
        
        messages = context_manager.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
    
    def test_context_window_management(self, context_manager):
        """Test context window size management."""
        context_manager.set_max_context_size(3)
        
        for i in range(5):
            context_manager.add_message("user", f"Message {i}")
        
        messages = context_manager.get_messages()
        assert len(messages) == 3
        assert messages[0]["content"] == "Message 2"
    
    def test_save_and_restore_context(self, context_manager, temp_dir):
        """Test saving and restoring context."""
        context_manager.add_message("user", "Test message")
        context_manager.add_message("assistant", "Test response")
        
        # Save context
        context_file = temp_dir / "context.json"
        context_manager.save_context(context_file)
        
        # Create new manager and restore
        new_manager = ContextManager()
        new_manager.load_context(context_file)
        
        messages = new_manager.get_messages()
        assert len(messages) == 2
        assert messages[0]["content"] == "Test message"
    
    def test_context_switching(self, context_manager):
        """Test switching between multiple contexts."""
        # Create first context
        context_manager.add_message("user", "Context 1")
        context_id_1 = context_manager.save_current_context()
        
        # Create second context
        context_manager.new_context()
        context_manager.add_message("user", "Context 2")
        context_id_2 = context_manager.save_current_context()
        
        # Switch back to first context
        context_manager.switch_context(context_id_1)
        messages = context_manager.get_messages()
        assert messages[0]["content"] == "Context 1"
        
        # Switch to second context
        context_manager.switch_context(context_id_2)
        messages = context_manager.get_messages()
        assert messages[0]["content"] == "Context 2"
    
    def test_context_metadata(self, context_manager):
        """Test context metadata management."""
        context_manager.set_metadata("model", "llama3.2")
        context_manager.set_metadata("temperature", 0.7)
        
        metadata = context_manager.get_metadata()
        assert metadata["model"] == "llama3.2"
        assert metadata["temperature"] == 0.7
    
    def test_context_search(self, context_manager):
        """Test searching through context history."""
        # Add multiple contexts
        for i in range(5):
            context_manager.new_context()
            context_manager.add_message("user", f"Question about {['python', 'java', 'rust', 'go', 'c++'][i]}")
            context_manager.save_current_context()
        
        # Search for python-related contexts
        results = context_manager.search_contexts("python")
        assert len(results) > 0
        assert "python" in results[0]["messages"][0]["content"].lower()
    
    def test_context_compression(self, context_manager):
        """Test context compression for long conversations."""
        # Add many messages
        for i in range(100):
            context_manager.add_message("user", f"Question {i}")
            context_manager.add_message("assistant", f"Answer {i}")
        
        # Compress context
        compressed = context_manager.compress_context(max_messages=10)
        assert len(compressed["messages"]) <= 10
        assert "summary" in compressed


class TestSession:
    """Test cases for Session class."""
    
    def test_session_creation(self):
        """Test session creation."""
        session = Session(user_id="user123")
        
        assert session.user_id == "user123"
        assert session.session_id is not None
        assert session.created_at is not None
        assert session.is_active is True
    
    def test_session_lifecycle(self):
        """Test session lifecycle management."""
        session = Session()
        
        # Session should be active initially
        assert session.is_active is True
        
        # End session
        session.end()
        assert session.is_active is False
        assert session.ended_at is not None
    
    def test_session_data_storage(self):
        """Test storing data in session."""
        session = Session()
        
        session.set("key1", "value1")
        session.set("key2", {"nested": "data"})
        
        assert session.get("key1") == "value1"
        assert session.get("key2")["nested"] == "data"
        assert session.get("nonexistent") is None
    
    def test_session_persistence(self, temp_dir):
        """Test session persistence."""
        session = Session(user_id="test_user")
        session.set("data", "important")
        
        # Save session
        session_file = temp_dir / "session.json"
        session.save(session_file)
        
        # Load into new session
        loaded_session = Session.load(session_file)
        assert loaded_session.user_id == "test_user"
        assert loaded_session.get("data") == "important"
    
    def test_session_expiration(self):
        """Test session expiration."""
        session = Session(ttl_minutes=1)
        
        # Session should be valid initially
        assert not session.is_expired()
        
        # Simulate time passing
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(minutes=2)
            assert session.is_expired()


class TestSessionManager:
    """Test cases for SessionManager class."""
    
    @pytest.fixture
    def session_manager(self):
        """Create a test session manager."""
        return SessionManager()
    
    def test_create_session(self, session_manager):
        """Test creating a new session."""
        session = session_manager.create_session("user123")
        
        assert session.user_id == "user123"
        assert session.session_id in session_manager.active_sessions
    
    def test_get_session(self, session_manager):
        """Test retrieving a session."""
        session = session_manager.create_session("user123")
        session_id = session.session_id
        
        retrieved = session_manager.get_session(session_id)
        assert retrieved is session
    
    def test_end_session(self, session_manager):
        """Test ending a session."""
        session = session_manager.create_session("user123")
        session_id = session.session_id
        
        session_manager.end_session(session_id)
        assert session.is_active is False
        assert session_id not in session_manager.active_sessions
    
    def test_cleanup_expired_sessions(self, session_manager):
        """Test cleaning up expired sessions."""
        # Create sessions with different expiration times
        session1 = session_manager.create_session("user1", ttl_minutes=1)
        session2 = session_manager.create_session("user2", ttl_minutes=60)
        
        # Expire session1
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(minutes=2)
            session_manager.cleanup_expired()
        
        assert session1.session_id not in session_manager.active_sessions
        assert session2.session_id in session_manager.active_sessions
    
    def test_get_user_sessions(self, session_manager):
        """Test getting all sessions for a user."""
        session1 = session_manager.create_session("user123")
        session2 = session_manager.create_session("user123")
        session3 = session_manager.create_session("user456")
        
        user_sessions = session_manager.get_user_sessions("user123")
        assert len(user_sessions) == 2
        assert session1 in user_sessions
        assert session2 in user_sessions
        assert session3 not in user_sessions


class TestMemoryIntegration:
    """Integration tests for memory systems."""
    
    def test_memory_hierarchy(self):
        """Test integration of memory hierarchy."""
        # Create memory hierarchy
        working = WorkingMemory(capacity=5)
        short_term = ShortTermMemory(capacity=20, ttl_minutes=30)
        long_term = LongTermMemory(max_size=1000)
        
        # Simulate information flow
        # Working -> Short-term -> Long-term
        
        # Add to working memory
        working.add("immediate_task")
        
        # Transfer to short-term
        if len(working.buffer) > 3:
            for item in working.buffer[:2]:
                short_term.add(f"st_{item}", item)
        
        # Transfer to long-term
        important_items = []
        for key in short_term.items:
            if short_term.items[key].get("importance", 0) > 5:
                important_items.append((key, short_term.items[key]))
        
        for key, value in important_items:
            long_term.store(key, value)
            short_term.remove(key)
    
    def test_context_with_memory(self):
        """Test context manager with memory integration."""
        context = ContextManager()
        memory = ConversationMemory(max_size=10)
        
        # Add messages to both
        messages = [
            ("user", "What is Python?"),
            ("assistant", "Python is a programming language."),
            ("user", "What are its uses?"),
            ("assistant", "Web development, data science, automation...")
        ]
        
        for role, content in messages:
            context.add_message(role, content)
            memory.add_message(role, content)
        
        # Context and memory should be synchronized
        context_messages = context.get_messages()
        memory_messages = memory.messages
        
        assert len(context_messages) == len(memory_messages)
        for c, m in zip(context_messages, memory_messages):
            assert c["role"] == m["role"]
            assert c["content"] == m["content"]


class TestPerformance:
    """Performance tests for context management."""
    
    @pytest.mark.performance
    def test_large_context_handling(self):
        """Test handling large conversation contexts."""
        manager = ContextManager()
        
        # Add many messages
        import time
        start = time.perf_counter()
        
        for i in range(1000):
            manager.add_message("user", f"Question {i}")
            manager.add_message("assistant", f"Answer {i}")
        
        end = time.perf_counter()
        
        # Should handle 2000 messages quickly
        assert (end - start) < 1.0  # Less than 1 second
        
        # Retrieval should also be fast
        start = time.perf_counter()
        messages = manager.get_messages(last_n=100)
        end = time.perf_counter()
        
        assert len(messages) == 100
        assert (end - start) < 0.01  # Less than 10ms
    
    @pytest.mark.performance
    def test_memory_search_performance(self):
        """Test search performance with large memory."""
        long_term = LongTermMemory(enable_index=True)
        
        # Store many items
        for i in range(10000):
            long_term.store(f"key{i}", {
                "category": f"cat{i % 100}",
                "content": f"Content for item {i}",
                "timestamp": datetime.now().isoformat()
            })
        
        import time
        
        # Search should be fast even with 10k items
        start = time.perf_counter()
        results = long_term.search({"category": "cat42"})
        end = time.perf_counter()
        
        assert len(results) == 100
        assert (end - start) < 0.1  # Less than 100ms