"""
Comprehensive Model Registry for ABOV3 4 Ollama.

This module provides persistent model metadata storage and management capabilities including:
- Persistent model metadata storage with SQLite backend
- Advanced model search and filtering capabilities
- Model ratings and reviews system
- Performance benchmarking data storage
- Model usage analytics and insights
- Import/export of model configurations
- Integration with online model repositories

Features:
- Thread-safe operations with connection pooling
- Advanced search with full-text search capabilities
- Model performance tracking and analytics
- User ratings and feedback system
- Automatic backup and recovery
- Integration with security framework
- Support for custom model categories and tags
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
import hashlib

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None

from ..core.config import Config, get_config
from ..utils.security import SecurityManager, SecurityEvent
from .info import ModelInfo, ModelMetadata, ModelSize, ModelType, ModelCapability


logger = logging.getLogger(__name__)


@dataclass
class ModelRating:
    """Model rating and review."""
    model_name: str
    user_id: str
    rating: float  # 1.0 to 5.0
    review: Optional[str] = None
    use_case: Optional[str] = None
    performance_rating: Optional[float] = None  # 1.0 to 5.0
    ease_of_use: Optional[float] = None  # 1.0 to 5.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    helpful_votes: int = 0
    verified_user: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRating":
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class ModelBenchmark:
    """Model performance benchmark data."""
    model_name: str
    benchmark_type: str  # e.g., 'coding', 'chat', 'reasoning'
    score: float
    metric_name: str  # e.g., 'accuracy', 'speed', 'quality'
    details: Dict[str, Any] = field(default_factory=dict)
    test_date: datetime = field(default_factory=datetime.now)
    test_environment: Optional[str] = None
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['test_date'] = self.test_date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelBenchmark":
        """Create from dictionary."""
        data = data.copy()
        data['test_date'] = datetime.fromisoformat(data['test_date'])
        return cls(**data)


@dataclass
class ModelUsageStats:
    """Model usage statistics."""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_response_time: float = 0.0
    average_response_time: float = 0.0
    first_used: Optional[datetime] = None
    last_used: Optional[datetime] = None
    unique_users: Set[str] = field(default_factory=set)
    popular_use_cases: Dict[str, int] = field(default_factory=dict)
    
    def update_usage(self, response_time: float, tokens: int, 
                    success: bool, user_id: str, use_case: str = None) -> None:
        """Update usage statistics."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_tokens += tokens
        self.total_response_time += response_time
        self.average_response_time = self.total_response_time / self.total_requests
        
        now = datetime.now()
        if self.first_used is None:
            self.first_used = now
        self.last_used = now
        
        self.unique_users.add(user_id)
        
        if use_case:
            self.popular_use_cases[use_case] = self.popular_use_cases.get(use_case, 0) + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['first_used'] = self.first_used.isoformat() if self.first_used else None
        data['last_used'] = self.last_used.isoformat() if self.last_used else None
        data['unique_users'] = list(self.unique_users)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelUsageStats":
        """Create from dictionary."""
        data = data.copy()
        data['first_used'] = datetime.fromisoformat(data['first_used']) if data['first_used'] else None
        data['last_used'] = datetime.fromisoformat(data['last_used']) if data['last_used'] else None
        data['unique_users'] = set(data.get('unique_users', []))
        return cls(**data)


@dataclass
class SearchFilter:
    """Search filter criteria."""
    query: Optional[str] = None
    model_types: Optional[List[ModelType]] = None
    capabilities: Optional[List[ModelCapability]] = None
    size_categories: Optional[List[ModelSize]] = None
    min_rating: Optional[float] = None
    tags: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    sort_by: str = "name"  # name, rating, usage, created_at, updated_at
    sort_order: str = "asc"  # asc, desc
    limit: int = 50
    offset: int = 0


class ModelRegistry:
    """
    Comprehensive model registry for ABOV3 4 Ollama.
    
    Features:
    - Persistent storage with SQLite backend
    - Advanced search and filtering
    - Model ratings and reviews
    - Performance benchmarking
    - Usage analytics
    - Import/export capabilities
    - Security integration
    """
    
    def __init__(self, config: Optional[Config] = None,
                 security_manager: Optional[SecurityManager] = None,
                 db_path: Optional[Path] = None):
        """
        Initialize the model registry.
        
        Args:
            config: Configuration object
            security_manager: Security manager for secure operations
            db_path: Custom database path
        """
        self.config = config or get_config()
        self.security_manager = security_manager or SecurityManager()
        
        # Database setup
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = self.config.get_data_dir() / "models_registry.db"
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Threading and async support
        self._lock = threading.Lock()
        self._connection_pool: Dict[int, sqlite3.Connection] = {}
        self._pool_lock = threading.Lock()
        
        # In-memory caches
        self._metadata_cache: Dict[str, ModelMetadata] = {}
        self._ratings_cache: Dict[str, List[ModelRating]] = {}
        self._stats_cache: Dict[str, ModelUsageStats] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_refresh = 0.0
        
        # Search index
        self._search_index: Dict[str, Set[str]] = defaultdict(set)
        self._index_dirty = True
        
        # Initialize database
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        thread_id = threading.get_ident()
        
        with self._pool_lock:
            if thread_id not in self._connection_pool:
                conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                self._connection_pool[thread_id] = conn
            
            return self._connection_pool[thread_id]
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        
        # Model metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                name TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                description TEXT,
                version TEXT,
                model_type TEXT,
                capabilities TEXT,  -- JSON array
                size_category TEXT,
                parameter_count TEXT,
                context_length INTEGER,
                architecture TEXT,
                quantization TEXT,
                training_data TEXT,
                license TEXT,
                performance_score REAL,
                speed_score REAL,
                quality_score REAL,
                download_count INTEGER DEFAULT 0,
                usage_count INTEGER DEFAULT 0,
                last_used TEXT,
                default_temperature REAL,
                recommended_params TEXT,  -- JSON
                created_at TEXT,
                updated_at TEXT,
                tags TEXT  -- JSON array
            )
        """)
        
        # Model ratings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                rating REAL NOT NULL,
                review TEXT,
                use_case TEXT,
                performance_rating REAL,
                ease_of_use REAL,
                created_at TEXT,
                updated_at TEXT,
                helpful_votes INTEGER DEFAULT 0,
                verified_user BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (model_name) REFERENCES model_metadata (name),
                UNIQUE (model_name, user_id)
            )
        """)
        
        # Model benchmarks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                benchmark_type TEXT NOT NULL,
                score REAL NOT NULL,
                metric_name TEXT NOT NULL,
                details TEXT,  -- JSON
                test_date TEXT,
                test_environment TEXT,
                version TEXT,
                FOREIGN KEY (model_name) REFERENCES model_metadata (name)
            )
        """)
        
        # Model usage statistics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_usage_stats (
                model_name TEXT PRIMARY KEY,
                total_requests INTEGER DEFAULT 0,
                successful_requests INTEGER DEFAULT 0,
                failed_requests INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                total_response_time REAL DEFAULT 0.0,
                average_response_time REAL DEFAULT 0.0,
                first_used TEXT,
                last_used TEXT,
                unique_users TEXT,  -- JSON array
                popular_use_cases TEXT,  -- JSON
                FOREIGN KEY (model_name) REFERENCES model_metadata (name)
            )
        """)
        
        # Create indexes for better performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_type ON model_metadata (model_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_size_category ON model_metadata (size_category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rating ON model_ratings (rating)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_type ON model_benchmarks (benchmark_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_last_used ON model_usage_stats (last_used)")
        
        # Create full-text search virtual table
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS model_search 
            USING fts5(name, display_name, description, tags, content='model_metadata')
        """)
        
        conn.commit()
        
        # Initialize search index
        self._rebuild_search_index()
    
    def _rebuild_search_index(self) -> None:
        """Rebuild the full-text search index."""
        conn = self._get_connection()
        
        # Clear existing index
        conn.execute("DELETE FROM model_search")
        
        # Rebuild from metadata
        cursor = conn.execute("""
            SELECT name, display_name, description, tags 
            FROM model_metadata
        """)
        
        for row in cursor.fetchall():
            conn.execute("""
                INSERT INTO model_search (name, display_name, description, tags)
                VALUES (?, ?, ?, ?)
            """, (row['name'], row['display_name'], row['description'], row['tags']))
        
        conn.commit()
        self._index_dirty = False
    
    async def register_model(self, metadata: ModelMetadata, 
                           user_id: Optional[str] = None) -> bool:
        """
        Register a new model or update existing metadata.
        
        Args:
            metadata: Model metadata to register
            user_id: User ID for security logging
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Security validation
            if not self._validate_metadata_security(metadata, user_id):
                logger.warning(f"Model registration blocked for security: {metadata.name}")
                return False
            
            conn = self._get_connection()
            
            # Check if model exists
            cursor = conn.execute("SELECT name FROM model_metadata WHERE name = ?", (metadata.name,))
            exists = cursor.fetchone() is not None
            
            # Prepare data for insertion/update
            metadata_dict = metadata.to_dict()
            
            if exists:
                # Update existing model
                conn.execute("""
                    UPDATE model_metadata SET
                        display_name = ?, description = ?, version = ?, model_type = ?,
                        capabilities = ?, size_category = ?, parameter_count = ?,
                        context_length = ?, architecture = ?, quantization = ?,
                        training_data = ?, license = ?, performance_score = ?,
                        speed_score = ?, quality_score = ?, default_temperature = ?,
                        recommended_params = ?, updated_at = ?, tags = ?
                    WHERE name = ?
                """, (
                    metadata.display_name, metadata.description, metadata.version,
                    metadata.model_type.value, json.dumps([cap.value for cap in metadata.capabilities]),
                    metadata.size_category.value, metadata.parameter_count,
                    metadata.context_length, metadata.architecture, metadata.quantization,
                    metadata.training_data, metadata.license, metadata.performance_score,
                    metadata.speed_score, metadata.quality_score, metadata.default_temperature,
                    json.dumps(metadata.recommended_params), datetime.now().isoformat(),
                    json.dumps(metadata.tags), metadata.name
                ))
            else:
                # Insert new model
                conn.execute("""
                    INSERT INTO model_metadata (
                        name, display_name, description, version, model_type,
                        capabilities, size_category, parameter_count, context_length,
                        architecture, quantization, training_data, license,
                        performance_score, speed_score, quality_score,
                        download_count, usage_count, last_used, default_temperature,
                        recommended_params, created_at, updated_at, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.name, metadata.display_name, metadata.description,
                    metadata.version, metadata.model_type.value,
                    json.dumps([cap.value for cap in metadata.capabilities]),
                    metadata.size_category.value, metadata.parameter_count,
                    metadata.context_length, metadata.architecture, metadata.quantization,
                    metadata.training_data, metadata.license, metadata.performance_score,
                    metadata.speed_score, metadata.quality_score, metadata.download_count,
                    metadata.usage_count, metadata.last_used.isoformat() if metadata.last_used else None,
                    metadata.default_temperature, json.dumps(metadata.recommended_params),
                    metadata.created_at.isoformat(), metadata.updated_at.isoformat(),
                    json.dumps(metadata.tags)
                ))
            
            conn.commit()
            
            # Update caches
            self._metadata_cache[metadata.name] = metadata
            self._index_dirty = True
            
            # Log the registration
            self.security_manager.logger.log_event(SecurityEvent(
                event_type='MODEL_REGISTRATION',
                severity='LOW',
                message=f"Model {'updated' if exists else 'registered'}: {metadata.name}",
                user_id=user_id,
                metadata={'model_name': metadata.name, 'action': 'update' if exists else 'register'}
            ))
            
            logger.info(f"Successfully {'updated' if exists else 'registered'} model: {metadata.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {metadata.name}: {e}")
            return False
    
    async def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """
        Get metadata for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelMetadata object or None if not found
        """
        # Check cache first
        if model_name in self._metadata_cache:
            return self._metadata_cache[model_name]
        
        try:
            conn = self._get_connection()
            cursor = conn.execute("SELECT * FROM model_metadata WHERE name = ?", (model_name,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Convert row to ModelMetadata
            metadata = self._row_to_metadata(row)
            
            # Cache the result
            self._metadata_cache[model_name] = metadata
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get metadata for {model_name}: {e}")
            return None
    
    async def search_models(self, search_filter: SearchFilter) -> List[ModelMetadata]:
        """
        Search models with advanced filtering.
        
        Args:
            search_filter: Search criteria
            
        Returns:
            List of matching ModelMetadata objects
        """
        try:
            conn = self._get_connection()
            
            # Build query
            query = "SELECT DISTINCT m.* FROM model_metadata m"
            conditions = []
            params = []
            
            # Full-text search
            if search_filter.query:
                query += " JOIN model_search s ON m.name = s.name"
                conditions.append("model_search MATCH ?")
                params.append(search_filter.query)
            
            # Type filter
            if search_filter.model_types:
                type_placeholders = ",".join(["?"] * len(search_filter.model_types))
                conditions.append(f"m.model_type IN ({type_placeholders})")
                params.extend([t.value for t in search_filter.model_types])
            
            # Size filter
            if search_filter.size_categories:
                size_placeholders = ",".join(["?"] * len(search_filter.size_categories))
                conditions.append(f"m.size_category IN ({size_placeholders})")
                params.extend([s.value for s in search_filter.size_categories])
            
            # Rating filter
            if search_filter.min_rating:
                query += " LEFT JOIN (SELECT model_name, AVG(rating) as avg_rating FROM model_ratings GROUP BY model_name) r ON m.name = r.model_name"
                conditions.append("(r.avg_rating >= ? OR r.avg_rating IS NULL)")
                params.append(search_filter.min_rating)
            
            # Date filters
            if search_filter.created_after:
                conditions.append("m.created_at >= ?")
                params.append(search_filter.created_after.isoformat())
            
            if search_filter.created_before:
                conditions.append("m.created_at <= ?")
                params.append(search_filter.created_before.isoformat())
            
            # Tags filter
            if search_filter.tags:
                for tag in search_filter.tags:
                    conditions.append("m.tags LIKE ?")
                    params.append(f'%"{tag}"%')
            
            # Add WHERE clause
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Sorting
            sort_column = self._get_sort_column(search_filter.sort_by)
            query += f" ORDER BY {sort_column} {search_filter.sort_order.upper()}"
            
            # Pagination
            query += " LIMIT ? OFFSET ?"
            params.extend([search_filter.limit, search_filter.offset])
            
            # Execute query
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to ModelMetadata objects
            results = []
            for row in rows:
                metadata = self._row_to_metadata(row)
                results.append(metadata)
                # Cache the result
                self._metadata_cache[metadata.name] = metadata
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            return []
    
    async def add_rating(self, rating: ModelRating, user_id: Optional[str] = None) -> bool:
        """
        Add or update a model rating.
        
        Args:
            rating: Model rating to add
            user_id: User ID for security validation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Security validation
            if user_id and rating.user_id != user_id:
                logger.warning(f"Rating user_id mismatch: {rating.user_id} != {user_id}")
                return False
            
            conn = self._get_connection()
            
            # Check if rating exists
            cursor = conn.execute(
                "SELECT id FROM model_ratings WHERE model_name = ? AND user_id = ?",
                (rating.model_name, rating.user_id)
            )
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update existing rating
                conn.execute("""
                    UPDATE model_ratings SET
                        rating = ?, review = ?, use_case = ?, performance_rating = ?,
                        ease_of_use = ?, updated_at = ?, helpful_votes = ?, verified_user = ?
                    WHERE model_name = ? AND user_id = ?
                """, (
                    rating.rating, rating.review, rating.use_case,
                    rating.performance_rating, rating.ease_of_use,
                    rating.updated_at.isoformat(), rating.helpful_votes,
                    rating.verified_user, rating.model_name, rating.user_id
                ))
            else:
                # Insert new rating
                conn.execute("""
                    INSERT INTO model_ratings (
                        model_name, user_id, rating, review, use_case,
                        performance_rating, ease_of_use, created_at, updated_at,
                        helpful_votes, verified_user
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rating.model_name, rating.user_id, rating.rating,
                    rating.review, rating.use_case, rating.performance_rating,
                    rating.ease_of_use, rating.created_at.isoformat(),
                    rating.updated_at.isoformat(), rating.helpful_votes,
                    rating.verified_user
                ))
            
            conn.commit()
            
            # Clear ratings cache for this model
            self._ratings_cache.pop(rating.model_name, None)
            
            logger.info(f"Successfully {'updated' if exists else 'added'} rating for {rating.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add rating for {rating.model_name}: {e}")
            return False
    
    async def get_model_ratings(self, model_name: str) -> List[ModelRating]:
        """
        Get all ratings for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of ModelRating objects
        """
        # Check cache first
        if model_name in self._ratings_cache:
            return self._ratings_cache[model_name]
        
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM model_ratings WHERE model_name = ? ORDER BY created_at DESC",
                (model_name,)
            )
            rows = cursor.fetchall()
            
            ratings = []
            for row in rows:
                rating = ModelRating(
                    model_name=row['model_name'],
                    user_id=row['user_id'],
                    rating=row['rating'],
                    review=row['review'],
                    use_case=row['use_case'],
                    performance_rating=row['performance_rating'],
                    ease_of_use=row['ease_of_use'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    helpful_votes=row['helpful_votes'],
                    verified_user=bool(row['verified_user'])
                )
                ratings.append(rating)
            
            # Cache the results
            self._ratings_cache[model_name] = ratings
            
            return ratings
            
        except Exception as e:
            logger.error(f"Failed to get ratings for {model_name}: {e}")
            return []
    
    async def add_benchmark(self, benchmark: ModelBenchmark) -> bool:
        """
        Add a performance benchmark for a model.
        
        Args:
            benchmark: Benchmark data to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            
            conn.execute("""
                INSERT INTO model_benchmarks (
                    model_name, benchmark_type, score, metric_name,
                    details, test_date, test_environment, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                benchmark.model_name, benchmark.benchmark_type,
                benchmark.score, benchmark.metric_name,
                json.dumps(benchmark.details), benchmark.test_date.isoformat(),
                benchmark.test_environment, benchmark.version
            ))
            
            conn.commit()
            
            logger.info(f"Added benchmark for {benchmark.model_name}: {benchmark.benchmark_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add benchmark: {e}")
            return False
    
    async def get_model_benchmarks(self, model_name: str,
                                 benchmark_type: Optional[str] = None) -> List[ModelBenchmark]:
        """
        Get benchmarks for a model.
        
        Args:
            model_name: Name of the model
            benchmark_type: Optional filter by benchmark type
            
        Returns:
            List of ModelBenchmark objects
        """
        try:
            conn = self._get_connection()
            
            if benchmark_type:
                cursor = conn.execute("""
                    SELECT * FROM model_benchmarks 
                    WHERE model_name = ? AND benchmark_type = ?
                    ORDER BY test_date DESC
                """, (model_name, benchmark_type))
            else:
                cursor = conn.execute("""
                    SELECT * FROM model_benchmarks 
                    WHERE model_name = ?
                    ORDER BY test_date DESC
                """, (model_name,))
            
            rows = cursor.fetchall()
            
            benchmarks = []
            for row in rows:
                benchmark = ModelBenchmark(
                    model_name=row['model_name'],
                    benchmark_type=row['benchmark_type'],
                    score=row['score'],
                    metric_name=row['metric_name'],
                    details=json.loads(row['details']) if row['details'] else {},
                    test_date=datetime.fromisoformat(row['test_date']),
                    test_environment=row['test_environment'],
                    version=row['version']
                )
                benchmarks.append(benchmark)
            
            return benchmarks
            
        except Exception as e:
            logger.error(f"Failed to get benchmarks for {model_name}: {e}")
            return []
    
    async def update_usage_stats(self, model_name: str, response_time: float,
                               tokens: int, success: bool, user_id: str,
                               use_case: Optional[str] = None) -> None:
        """
        Update usage statistics for a model.
        
        Args:
            model_name: Name of the model
            response_time: Response time in seconds
            tokens: Number of tokens processed
            success: Whether the operation was successful
            user_id: User ID
            use_case: Optional use case description
        """
        try:
            # Get or create stats
            if model_name in self._stats_cache:
                stats = self._stats_cache[model_name]
            else:
                stats = await self.get_usage_stats(model_name)
                if not stats:
                    stats = ModelUsageStats(model_name=model_name)
            
            # Update stats
            stats.update_usage(response_time, tokens, success, user_id, use_case)
            
            # Save to database
            conn = self._get_connection()
            
            conn.execute("""
                INSERT OR REPLACE INTO model_usage_stats (
                    model_name, total_requests, successful_requests, failed_requests,
                    total_tokens, total_response_time, average_response_time,
                    first_used, last_used, unique_users, popular_use_cases
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats.model_name, stats.total_requests, stats.successful_requests,
                stats.failed_requests, stats.total_tokens, stats.total_response_time,
                stats.average_response_time,
                stats.first_used.isoformat() if stats.first_used else None,
                stats.last_used.isoformat() if stats.last_used else None,
                json.dumps(list(stats.unique_users)),
                json.dumps(stats.popular_use_cases)
            ))
            
            conn.commit()
            
            # Update cache
            self._stats_cache[model_name] = stats
            
        except Exception as e:
            logger.error(f"Failed to update usage stats for {model_name}: {e}")
    
    async def get_usage_stats(self, model_name: str) -> Optional[ModelUsageStats]:
        """
        Get usage statistics for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelUsageStats object or None if not found
        """
        # Check cache first
        if model_name in self._stats_cache:
            return self._stats_cache[model_name]
        
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM model_usage_stats WHERE model_name = ?",
                (model_name,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            stats = ModelUsageStats(
                model_name=row['model_name'],
                total_requests=row['total_requests'],
                successful_requests=row['successful_requests'],
                failed_requests=row['failed_requests'],
                total_tokens=row['total_tokens'],
                total_response_time=row['total_response_time'],
                average_response_time=row['average_response_time'],
                first_used=datetime.fromisoformat(row['first_used']) if row['first_used'] else None,
                last_used=datetime.fromisoformat(row['last_used']) if row['last_used'] else None,
                unique_users=set(json.loads(row['unique_users']) if row['unique_users'] else []),
                popular_use_cases=json.loads(row['popular_use_cases']) if row['popular_use_cases'] else {}
            )
            
            # Cache the result
            self._stats_cache[model_name] = stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get usage stats for {model_name}: {e}")
            return None
    
    async def export_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Export model configuration and metadata.
        
        Args:
            model_name: Name of the model to export
            
        Returns:
            Dictionary containing model configuration
        """
        try:
            metadata = await self.get_model_metadata(model_name)
            if not metadata:
                return None
            
            ratings = await self.get_model_ratings(model_name)
            benchmarks = await self.get_model_benchmarks(model_name)
            stats = await self.get_usage_stats(model_name)
            
            config = {
                'metadata': metadata.to_dict(),
                'ratings': [rating.to_dict() for rating in ratings],
                'benchmarks': [benchmark.to_dict() for benchmark in benchmarks],
                'usage_stats': stats.to_dict() if stats else None,
                'export_date': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to export config for {model_name}: {e}")
            return None
    
    async def import_model_config(self, config: Dict[str, Any],
                                user_id: Optional[str] = None) -> bool:
        """
        Import model configuration and metadata.
        
        Args:
            config: Configuration dictionary to import
            user_id: User ID for security logging
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate config format
            if 'metadata' not in config:
                logger.error("Invalid config format: missing metadata")
                return False
            
            # Import metadata
            metadata = ModelMetadata.from_dict(config['metadata'])
            success = await self.register_model(metadata, user_id)
            
            if not success:
                return False
            
            # Import ratings
            if 'ratings' in config:
                for rating_data in config['ratings']:
                    rating = ModelRating.from_dict(rating_data)
                    await self.add_rating(rating, user_id)
            
            # Import benchmarks
            if 'benchmarks' in config:
                for benchmark_data in config['benchmarks']:
                    benchmark = ModelBenchmark.from_dict(benchmark_data)
                    await self.add_benchmark(benchmark)
            
            logger.info(f"Successfully imported config for {metadata.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import model config: {e}")
            return False
    
    async def get_popular_models(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get most popular models by usage.
        
        Args:
            limit: Maximum number of models to return
            
        Returns:
            List of (model_name, usage_count) tuples
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute("""
                SELECT m.name, COALESCE(s.total_requests, 0) as usage_count
                FROM model_metadata m
                LEFT JOIN model_usage_stats s ON m.name = s.model_name
                ORDER BY usage_count DESC
                LIMIT ?
            """, (limit,))
            
            return [(row['name'], row['usage_count']) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Failed to get popular models: {e}")
            return []
    
    async def get_top_rated_models(self, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Get top rated models.
        
        Args:
            limit: Maximum number of models to return
            
        Returns:
            List of (model_name, average_rating) tuples
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute("""
                SELECT r.model_name, AVG(r.rating) as avg_rating, COUNT(r.rating) as rating_count
                FROM model_ratings r
                GROUP BY r.model_name
                HAVING rating_count >= 3  -- At least 3 ratings
                ORDER BY avg_rating DESC
                LIMIT ?
            """, (limit,))
            
            return [(row['model_name'], row['avg_rating']) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Failed to get top rated models: {e}")
            return []
    
    def _validate_metadata_security(self, metadata: ModelMetadata,
                                  user_id: Optional[str] = None) -> bool:
        """Validate metadata for security issues."""
        # Check model name
        is_safe, issues = self.security_manager.is_content_safe(metadata.name, user_id)
        if not is_safe:
            logger.warning(f"Security validation failed for model name: {metadata.name} - {issues}")
            return False
        
        # Check description
        if metadata.description:
            is_safe, issues = self.security_manager.is_content_safe(metadata.description, user_id)
            if not is_safe:
                logger.warning(f"Security validation failed for description: {issues}")
                return False
        
        return True
    
    def _row_to_metadata(self, row: sqlite3.Row) -> ModelMetadata:
        """Convert database row to ModelMetadata object."""
        return ModelMetadata(
            name=row['name'],
            display_name=row['display_name'],
            description=row['description'] or "",
            version=row['version'],
            model_type=ModelType(row['model_type']),
            capabilities=[ModelCapability(cap) for cap in json.loads(row['capabilities'] or '[]')],
            size_category=ModelSize(row['size_category']),
            parameter_count=row['parameter_count'],
            context_length=row['context_length'],
            architecture=row['architecture'],
            quantization=row['quantization'],
            training_data=row['training_data'],
            license=row['license'],
            performance_score=row['performance_score'],
            speed_score=row['speed_score'],
            quality_score=row['quality_score'],
            download_count=row['download_count'],
            usage_count=row['usage_count'],
            last_used=datetime.fromisoformat(row['last_used']) if row['last_used'] else None,
            default_temperature=row['default_temperature'],
            recommended_params=json.loads(row['recommended_params'] or '{}'),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            tags=json.loads(row['tags'] or '[]')
        )
    
    def _get_sort_column(self, sort_by: str) -> str:
        """Get database column for sorting."""
        sort_mapping = {
            'name': 'm.name',
            'rating': 'r.avg_rating',
            'usage': 'm.usage_count',
            'created_at': 'm.created_at',
            'updated_at': 'm.updated_at',
            'size': 'm.size_category'
        }
        return sort_mapping.get(sort_by, 'm.name')
    
    def close(self) -> None:
        """Close all database connections."""
        with self._pool_lock:
            for conn in self._connection_pool.values():
                conn.close()
            self._connection_pool.clear()


# Convenience functions
async def get_model_registry(config: Optional[Config] = None,
                           security_manager: Optional[SecurityManager] = None,
                           db_path: Optional[Path] = None) -> ModelRegistry:
    """
    Get a configured model registry instance.
    
    Args:
        config: Optional configuration
        security_manager: Optional security manager
        db_path: Optional database path
        
    Returns:
        ModelRegistry instance
    """
    return ModelRegistry(config=config, security_manager=security_manager, db_path=db_path)


async def quick_register_model(metadata: ModelMetadata, 
                             config: Optional[Config] = None) -> bool:
    """
    Quick model registration for convenience.
    
    Args:
        metadata: Model metadata to register
        config: Optional configuration
        
    Returns:
        True if successful, False otherwise
    """
    registry = await get_model_registry(config=config)
    try:
        return await registry.register_model(metadata)
    finally:
        registry.close()


async def quick_search_models(query: str, config: Optional[Config] = None) -> List[ModelMetadata]:
    """
    Quick model search for convenience.
    
    Args:
        query: Search query
        config: Optional configuration
        
    Returns:
        List of matching models
    """
    registry = await get_model_registry(config=config)
    try:
        search_filter = SearchFilter(query=query, limit=20)
        return await registry.search_models(search_filter)
    finally:
        registry.close()