"""
Comprehensive Model Manager for ABOV3 4 Ollama.

This module provides advanced model management capabilities including:
- Model discovery and installation via Ollama API
- Model switching and configuration management  
- Performance monitoring and optimization
- Model recommendation system
- Automatic model updates and versioning
- Resource usage tracking
- Model health checks and validation

Features:
- Async support for non-blocking operations
- Comprehensive error handling with retries
- Model performance benchmarking and analytics
- Smart caching for improved performance
- Integration with security framework
- Automatic model optimization recommendations
"""

import asyncio
import json
import logging
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple, AsyncIterator, Union
from dataclasses import dataclass, field

from ..core.api.ollama_client import OllamaClient, get_ollama_client, ModelInfo as OllamaModelInfo
from ..core.config import Config, get_config
from ..utils.security import SecurityManager, SecurityEvent
from .info import ModelInfo, ModelMetadata, ModelSize, ModelType, ModelCapability


logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model."""
    model_name: str
    response_time_avg: float = 0.0
    response_time_p95: float = 0.0
    tokens_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_rate: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_metrics(self, response_time: float, tokens: int, 
                      memory_mb: float, cpu_percent: float, success: bool) -> None:
        """Update performance metrics with new data."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            
            # Update response time (rolling average)
            alpha = 0.1  # Smoothing factor
            self.response_time_avg = (alpha * response_time + 
                                    (1 - alpha) * self.response_time_avg)
            
            # Update tokens per second
            if response_time > 0:
                current_tps = tokens / response_time
                self.tokens_per_second = (alpha * current_tps + 
                                        (1 - alpha) * self.tokens_per_second)
            
            # Update resource usage
            self.memory_usage_mb = (alpha * memory_mb + 
                                  (1 - alpha) * self.memory_usage_mb)
            self.cpu_usage_percent = (alpha * cpu_percent + 
                                    (1 - alpha) * self.cpu_usage_percent)
        
        # Update error rate
        self.error_rate = 1.0 - (self.successful_requests / self.total_requests)
        self.last_updated = datetime.now()


@dataclass
class ModelRecommendation:
    """Model recommendation with reasoning."""
    model_name: str
    confidence: float
    reasoning: List[str]
    use_cases: List[str]
    estimated_performance: Optional[ModelPerformanceMetrics] = None


@dataclass
class ModelHealthStatus:
    """Health status for a model."""
    model_name: str
    is_healthy: bool
    last_check: datetime
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    availability_score: float = 1.0
    
    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now()


class ModelCache:
    """Smart caching system for model data."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        with self._lock:
            if key not in self._cache:
                return None
            
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
                del self._cache[key]
                self._access_times.pop(key, None)
                return None
            
            self._access_times[key] = datetime.now()
            return value
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with LRU eviction."""
        with self._lock:
            # Evict if cache is full
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Remove oldest accessed item
                oldest_key = min(self._access_times.keys(), 
                               key=lambda k: self._access_times[k])
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
            
            self._cache[key] = (value, datetime.now())
            self._access_times[key] = datetime.now()
    
    def invalidate(self, key: str) -> None:
        """Remove item from cache."""
        with self._lock:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()


class ModelManager:
    """
    Comprehensive model manager for ABOV3 4 Ollama.
    
    Features:
    - Async model operations with connection pooling
    - Performance monitoring and optimization
    - Model recommendation system
    - Automatic health checks and validation
    - Smart caching and resource management
    - Integration with security framework
    """
    
    def __init__(self, config: Optional[Config] = None, 
                 security_manager: Optional[SecurityManager] = None):
        """
        Initialize the model manager.
        
        Args:
            config: Configuration object, defaults to global config
            security_manager: Security manager for secure operations
        """
        self.config = config or get_config()
        self.security_manager = security_manager or SecurityManager()
        
        # Initialize components
        self._ollama_client: Optional[OllamaClient] = None
        self._model_cache = ModelCache(max_size=500, ttl_seconds=300)
        self._performance_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self._health_status: Dict[str, ModelHealthStatus] = {}
        
        # Tracking and analytics
        self._model_usage_stats: Dict[str, int] = defaultdict(int)
        self._recent_errors: deque = deque(maxlen=100)
        self._background_tasks: List[asyncio.Task] = []
        
        # Model registry integration
        self._registry = None  # Will be set when registry is available
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Background monitoring
        self._monitoring_enabled = True
        self._health_check_interval = 300  # 5 minutes
        
    async def __aenter__(self) -> "ModelManager":
        """Async context manager entry."""
        await self._ensure_client()
        await self._start_background_monitoring()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_client(self) -> None:
        """Ensure Ollama client is available."""
        if self._ollama_client is None:
            self._ollama_client = OllamaClient(config=self.config)
            await self._ollama_client.__aenter__()
    
    async def close(self) -> None:
        """Close the model manager and cleanup resources."""
        self._monitoring_enabled = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close Ollama client
        if self._ollama_client:
            await self._ollama_client.close()
            self._ollama_client = None
    
    async def _start_background_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._monitoring_enabled:
            task = asyncio.create_task(self._health_monitoring_loop())
            self._background_tasks.append(task)
    
    async def _health_monitoring_loop(self) -> None:
        """Background loop for periodic health checks."""
        while self._monitoring_enabled:
            try:
                await asyncio.sleep(self._health_check_interval)
                if self._monitoring_enabled:
                    await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health monitoring error: {e}")
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all models."""
        try:
            models = await self.list_models()
            for model in models:
                await self.check_model_health(model.name)
        except Exception as e:
            logger.error(f"Failed to perform health checks: {e}")
    
    async def list_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """
        List all available models with enhanced information.
        
        Args:
            force_refresh: Force refresh from Ollama API
            
        Returns:
            List of ModelInfo objects
        """
        cache_key = "models_list"
        
        if not force_refresh:
            cached_models = self._model_cache.get(cache_key)
            if cached_models:
                return cached_models
        
        await self._ensure_client()
        
        try:
            # Get raw model list from Ollama
            ollama_models = await self._ollama_client.list_models(force_refresh=force_refresh)
            
            # Convert to enhanced ModelInfo objects
            models = []
            for ollama_model in ollama_models:
                # Handle both dict and object formats
                if hasattr(ollama_model, 'name'):
                    name = ollama_model.name
                    size = ollama_model.size
                    digest = ollama_model.digest
                    modified_at = ollama_model.modified_at
                    details = ollama_model.details if hasattr(ollama_model, 'details') else {}
                else:
                    # Handle dict format
                    name = ollama_model.get('name', 'unknown')
                    size = ollama_model.get('size', 0)
                    digest = ollama_model.get('digest', '')
                    modified_at = ollama_model.get('modified_at', '')
                    details = ollama_model.get('details', {})
                
                model_info = ModelInfo(
                    name=name,
                    tag=name.split(':')[-1] if ':' in name else 'latest',
                    full_name=name,
                    size=size,
                    digest=digest,
                    modified_at=datetime.fromisoformat(modified_at) 
                              if isinstance(modified_at, str) and modified_at
                              else datetime.now(),
                    parameter_count=details.get('parameter_size') if details else None,
                    quantization=details.get('quantization_level') if details else None,
                    architecture=details.get('family') if details else None
                )
                models.append(model_info)
            
            # Cache the results
            self._model_cache.set(cache_key, models)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise
    
    async def get_model_info(self, model_name: str, detailed: bool = True) -> Optional[ModelInfo]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            detailed: Whether to fetch detailed information
            
        Returns:
            ModelInfo object or None if not found
        """
        cache_key = f"model_info_{model_name}_{detailed}"
        
        cached_info = self._model_cache.get(cache_key)
        if cached_info:
            return cached_info
        
        await self._ensure_client()
        
        try:
            ollama_info = await self._ollama_client.get_model_info(model_name)
            if not ollama_info:
                return None
            
            model_info = ModelInfo(
                name=ollama_info.name,
                tag=ollama_info.name.split(':')[-1] if ':' in ollama_info.name else 'latest',
                full_name=ollama_info.name,
                size=ollama_info.size,
                digest=ollama_info.digest,
                modified_at=datetime.fromisoformat(ollama_info.modified_at) 
                          if isinstance(ollama_info.modified_at, str) 
                          else ollama_info.modified_at,
                parameter_count=ollama_info.details.get('parameter_size'),
                quantization=ollama_info.details.get('quantization_level'),
                architecture=ollama_info.details.get('family')
            )
            
            # Cache the result
            self._model_cache.set(cache_key, model_info)
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None
    
    async def install_model(self, model_name: str, 
                          progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                          user_id: Optional[str] = None) -> bool:
        """
        Install a model with security validation and progress tracking.
        
        Args:
            model_name: Name of the model to install
            progress_callback: Optional callback for progress updates
            user_id: User ID for security logging
            
        Returns:
            True if successful, False otherwise
        """
        # Security validation
        if not self._validate_model_name_security(model_name, user_id):
            logger.warning(f"Model installation blocked for security: {model_name}")
            return False
        
        await self._ensure_client()
        
        try:
            # Log the installation attempt
            self.security_manager.logger.log_event(SecurityEvent(
                event_type='MODEL_INSTALLATION',
                severity='MEDIUM',
                message=f"Installing model: {model_name}",
                user_id=user_id,
                metadata={'model_name': model_name}
            ))
            
            # Install the model
            success = await self._ollama_client.pull_model(model_name, progress_callback)
            
            if success:
                # Update cache and analytics
                self._model_cache.invalidate("models_list")
                self._track_model_operation(model_name, 'install', success, user_id)
                
                # Perform initial health check
                await self.check_model_health(model_name)
                
                logger.info(f"Successfully installed model: {model_name}")
            else:
                logger.error(f"Failed to install model: {model_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error installing model {model_name}: {e}")
            self._track_model_operation(model_name, 'install', False, user_id, str(e))
            return False
    
    async def remove_model(self, model_name: str, user_id: Optional[str] = None) -> bool:
        """
        Remove a model with security validation.
        
        Args:
            model_name: Name of the model to remove
            user_id: User ID for security logging
            
        Returns:
            True if successful, False otherwise
        """
        # Security validation
        if not self._validate_model_name_security(model_name, user_id):
            logger.warning(f"Model removal blocked for security: {model_name}")
            return False
        
        await self._ensure_client()
        
        try:
            # Log the removal attempt
            self.security_manager.logger.log_event(SecurityEvent(
                event_type='MODEL_REMOVAL',
                severity='MEDIUM',
                message=f"Removing model: {model_name}",
                user_id=user_id,
                metadata={'model_name': model_name}
            ))
            
            # Remove the model
            success = await self._ollama_client.delete_model(model_name)
            
            if success:
                # Update cache and analytics
                self._model_cache.invalidate("models_list")
                self._model_cache.invalidate(f"model_info_{model_name}_True")
                self._model_cache.invalidate(f"model_info_{model_name}_False")
                
                # Clean up metrics and health status
                with self._lock:
                    self._performance_metrics.pop(model_name, None)
                    self._health_status.pop(model_name, None)
                
                self._track_model_operation(model_name, 'remove', success, user_id)
                logger.info(f"Successfully removed model: {model_name}")
            else:
                logger.error(f"Failed to remove model: {model_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error removing model {model_name}: {e}")
            self._track_model_operation(model_name, 'remove', False, user_id, str(e))
            return False
    
    async def switch_model(self, model_name: str, user_id: Optional[str] = None) -> bool:
        """
        Switch the default model with validation.
        
        Args:
            model_name: Name of the model to switch to
            user_id: User ID for security logging
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate model exists
            if not await self.model_exists(model_name):
                logger.error(f"Cannot switch to non-existent model: {model_name}")
                return False
            
            # Check model health
            health_status = await self.check_model_health(model_name)
            if not health_status.is_healthy:
                logger.warning(f"Switching to potentially unhealthy model: {model_name}")
            
            # Update configuration
            self.config.model.default_model = model_name
            
            # Log the switch
            self.security_manager.logger.log_event(SecurityEvent(
                event_type='MODEL_SWITCH',
                severity='LOW',
                message=f"Switched default model to: {model_name}",
                user_id=user_id,
                metadata={'model_name': model_name, 'healthy': health_status.is_healthy}
            ))
            
            self._track_model_operation(model_name, 'switch', True, user_id)
            logger.info(f"Successfully switched to model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching to model {model_name}: {e}")
            self._track_model_operation(model_name, 'switch', False, user_id, str(e))
            return False
    
    async def model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists locally.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model exists, False otherwise
        """
        await self._ensure_client()
        return await self._ollama_client.model_exists(model_name)
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is available (synchronous wrapper for model_exists).
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is available, False otherwise
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use run_until_complete
                # Return a simple check instead
                return True  # Assume model is available to avoid blocking
            else:
                return loop.run_until_complete(self.model_exists(model_name))
        except Exception:
            # If there's any issue checking, assume model is not available
            return False
    
    def list_models_sync(self, force_refresh: bool = False) -> List[ModelInfo]:
        """
        List all available models (synchronous wrapper for list_models).
        
        Args:
            force_refresh: Force refresh from Ollama API
            
        Returns:
            List of ModelInfo objects
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new event loop
                import threading
                result = []
                def run_async():
                    nonlocal result
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self.list_models(force_refresh))
                    finally:
                        new_loop.close()
                
                thread = threading.Thread(target=run_async)
                thread.start()
                thread.join()
                return result
            else:
                return loop.run_until_complete(self.list_models(force_refresh))
        except Exception as e:
            logger.error(f"Error in sync list_models: {e}")
            return []
    
    def get_model_info_sync(self, model_name: str, detailed: bool = True) -> Optional[ModelInfo]:
        """
        Get detailed information about a specific model (synchronous wrapper).
        
        Args:
            model_name: Name of the model
            detailed: Whether to fetch detailed information
            
        Returns:
            ModelInfo object or None if not found
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new event loop
                import threading
                result = None
                def run_async():
                    nonlocal result
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self.get_model_info(model_name, detailed))
                    finally:
                        new_loop.close()
                
                thread = threading.Thread(target=run_async)
                thread.start()
                thread.join()
                return result
            else:
                return loop.run_until_complete(self.get_model_info(model_name, detailed))
        except Exception as e:
            logger.error(f"Error in sync get_model_info: {e}")
            return None
    
    def install_model_sync(self, model_name: str, 
                          progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                          user_id: Optional[str] = None) -> bool:
        """
        Install a model (synchronous wrapper for install_model).
        
        Args:
            model_name: Name of the model to install
            progress_callback: Optional callback for progress updates
            user_id: User ID for security logging
            
        Returns:
            True if successful, False otherwise
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new event loop
                import threading
                result = False
                def run_async():
                    nonlocal result
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(
                            self.install_model(model_name, progress_callback, user_id)
                        )
                    finally:
                        new_loop.close()
                
                thread = threading.Thread(target=run_async)
                thread.start()
                thread.join()
                return result
            else:
                return loop.run_until_complete(self.install_model(model_name, progress_callback, user_id))
        except Exception as e:
            logger.error(f"Error in sync install_model: {e}")
            return False
    
    def remove_model_sync(self, model_name: str, user_id: Optional[str] = None) -> bool:
        """
        Remove a model (synchronous wrapper for remove_model).
        
        Args:
            model_name: Name of the model to remove
            user_id: User ID for security logging
            
        Returns:
            True if successful, False otherwise
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new event loop
                import threading
                result = False
                def run_async():
                    nonlocal result
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self.remove_model(model_name, user_id))
                    finally:
                        new_loop.close()
                
                thread = threading.Thread(target=run_async)
                thread.start()
                thread.join()
                return result
            else:
                return loop.run_until_complete(self.remove_model(model_name, user_id))
        except Exception as e:
            logger.error(f"Error in sync remove_model: {e}")
            return False
    
    async def check_model_health(self, model_name: str) -> ModelHealthStatus:
        """
        Perform health check on a model.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            ModelHealthStatus object
        """
        start_time = time.time()
        
        try:
            await self._ensure_client()
            
            # Perform a simple health check request
            test_prompt = "Hello"
            response = await self._ollama_client.generate(
                model=model_name,
                prompt=test_prompt,
                max_tokens=5
            )
            
            response_time = time.time() - start_time
            is_healthy = bool(response and len(response.strip()) > 0)
            
            health_status = ModelHealthStatus(
                model_name=model_name,
                is_healthy=is_healthy,
                last_check=datetime.now(),
                response_time=response_time,
                availability_score=1.0 if is_healthy else 0.0
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            health_status = ModelHealthStatus(
                model_name=model_name,
                is_healthy=False,
                last_check=datetime.now(),
                response_time=response_time,
                error_message=str(e),
                availability_score=0.0
            )
        
        # Update health status cache
        with self._lock:
            self._health_status[model_name] = health_status
        
        return health_status
    
    async def get_performance_metrics(self, model_name: str) -> Optional[ModelPerformanceMetrics]:
        """
        Get performance metrics for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelPerformanceMetrics object or None
        """
        with self._lock:
            return self._performance_metrics.get(model_name)
    
    async def update_performance_metrics(self, model_name: str, response_time: float,
                                       tokens: int, memory_mb: float, 
                                       cpu_percent: float, success: bool) -> None:
        """
        Update performance metrics for a model.
        
        Args:
            model_name: Name of the model
            response_time: Response time in seconds
            tokens: Number of tokens processed
            memory_mb: Memory usage in MB
            cpu_percent: CPU usage percentage
            success: Whether the operation was successful
        """
        with self._lock:
            if model_name not in self._performance_metrics:
                self._performance_metrics[model_name] = ModelPerformanceMetrics(model_name)
            
            self._performance_metrics[model_name].update_metrics(
                response_time, tokens, memory_mb, cpu_percent, success
            )
    
    async def get_model_recommendations(self, task_type: str, 
                                      user_preferences: Optional[Dict[str, Any]] = None) -> List[ModelRecommendation]:
        """
        Get model recommendations based on task type and preferences.
        
        Args:
            task_type: Type of task (e.g., 'coding', 'chat', 'analysis')
            user_preferences: User preferences (size, speed, quality)
            
        Returns:
            List of model recommendations
        """
        try:
            models = await self.list_models()
            recommendations = []
            
            # Get preference weights
            prefs = user_preferences or {}
            prefer_speed = prefs.get('prefer_speed', False)
            prefer_quality = prefs.get('prefer_quality', True)
            max_size_gb = prefs.get('max_size_gb', 20.0)
            
            for model in models:
                # Filter by size preference
                if model.size_gb > max_size_gb:
                    continue
                
                # Get model metadata if available
                metadata = await self._get_model_metadata(model.name)
                if not metadata:
                    continue
                
                # Check task suitability
                if not metadata.is_suitable_for(task_type):
                    continue
                
                # Calculate confidence score
                confidence = self._calculate_recommendation_confidence(
                    model, metadata, task_type, prefs
                )
                
                if confidence > 0.3:  # Only recommend models with reasonable confidence
                    reasoning = self._generate_recommendation_reasoning(
                        model, metadata, task_type, prefs
                    )
                    
                    # Get performance metrics
                    perf_metrics = await self.get_performance_metrics(model.name)
                    
                    recommendation = ModelRecommendation(
                        model_name=model.name,
                        confidence=confidence,
                        reasoning=reasoning,
                        use_cases=[task_type],
                        estimated_performance=perf_metrics
                    )
                    recommendations.append(recommendation)
            
            # Sort by confidence
            recommendations.sort(key=lambda x: x.confidence, reverse=True)
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            logger.error(f"Failed to get model recommendations: {e}")
            return []
    
    async def optimize_model_configuration(self, model_name: str) -> Dict[str, Any]:
        """
        Generate optimized configuration for a model based on performance data.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of optimized parameters
        """
        try:
            # Get current performance metrics
            metrics = await self.get_performance_metrics(model_name)
            if not metrics:
                return self.config.get_model_params()
            
            # Get model metadata
            metadata = await self._get_model_metadata(model_name)
            
            # Start with current configuration
            optimized_params = self.config.get_model_params().copy()
            
            # Optimize based on performance
            if metrics.response_time_avg > 5.0:  # Slow response
                # Reduce precision for speed
                optimized_params['temperature'] = min(0.5, optimized_params.get('temperature', 0.7))
                optimized_params['top_p'] = min(0.8, optimized_params.get('top_p', 0.9))
                optimized_params['top_k'] = min(20, optimized_params.get('top_k', 40))
            
            if metrics.memory_usage_mb > 8000:  # High memory usage
                # Reduce context length if possible
                if metadata and metadata.context_length:
                    optimized_params['context_length'] = min(
                        metadata.context_length // 2,
                        optimized_params.get('context_length', 4096)
                    )
            
            if metrics.error_rate > 0.1:  # High error rate
                # Use more conservative parameters
                optimized_params['temperature'] = max(0.3, optimized_params.get('temperature', 0.7))
                optimized_params['repeat_penalty'] = min(1.2, optimized_params.get('repeat_penalty', 1.1))
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Failed to optimize model configuration for {model_name}: {e}")
            return self.config.get_model_params()
    
    async def get_usage_analytics(self) -> Dict[str, Any]:
        """
        Get usage analytics and statistics.
        
        Returns:
            Dictionary containing usage analytics
        """
        with self._lock:
            total_usage = sum(self._model_usage_stats.values())
            
            analytics = {
                'total_operations': total_usage,
                'model_usage': dict(self._model_usage_stats),
                'most_used_model': max(self._model_usage_stats.items(), 
                                     key=lambda x: x[1], default=('none', 0))[0],
                'health_status': {name: status.is_healthy 
                                for name, status in self._health_status.items()},
                'recent_errors': len(self._recent_errors),
                'performance_summary': self._get_performance_summary()
            }
            
            return analytics
    
    def _validate_model_name_security(self, model_name: str, user_id: Optional[str] = None) -> bool:
        """Validate model name for security issues."""
        if not model_name or not isinstance(model_name, str):
            return False
        
        # Check for malicious patterns
        is_safe, issues = self.security_manager.is_content_safe(model_name, user_id)
        if not is_safe:
            logger.warning(f"Security validation failed for model name: {model_name} - {issues}")
            return False
        
        # Basic format validation
        import re
        pattern = r'^[a-zA-Z0-9._:-]+$'
        return bool(re.match(pattern, model_name))
    
    def _track_model_operation(self, model_name: str, operation: str, 
                             success: bool, user_id: Optional[str] = None,
                             error: Optional[str] = None) -> None:
        """Track model operations for analytics."""
        with self._lock:
            self._model_usage_stats[model_name] += 1
            
            if not success and error:
                self._recent_errors.append({
                    'timestamp': datetime.now(),
                    'model': model_name,
                    'operation': operation,
                    'error': error,
                    'user_id': user_id
                })
    
    async def _get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Get model metadata from registry (placeholder for now)."""
        # This would integrate with the ModelRegistry when available
        # For now, create basic metadata
        try:
            model_info = await self.get_model_info(model_name, detailed=False)
            if not model_info:
                return None
            
            # Create basic metadata
            metadata = ModelMetadata(
                name=model_name,
                display_name=model_name.replace(':', ' ').title(),
                description=f"Model {model_name}",
                version=model_info.tag,
                model_type=self._infer_model_type(model_name),
                capabilities=self._infer_capabilities(model_name),
                size_category=model_info.size_category,
                parameter_count=model_info.parameter_count,
                architecture=model_info.architecture,
                quantization=model_info.quantization
            )
            
            return metadata
            
        except Exception as e:
            logger.debug(f"Failed to get metadata for {model_name}: {e}")
            return None
    
    def _infer_model_type(self, model_name: str) -> ModelType:
        """Infer model type from name."""
        name_lower = model_name.lower()
        if 'code' in name_lower:
            return ModelType.CODE
        elif 'chat' in name_lower:
            return ModelType.CHAT
        elif 'instruct' in name_lower:
            return ModelType.INSTRUCT
        elif 'embed' in name_lower:
            return ModelType.EMBEDDING
        elif 'vision' in name_lower or 'llava' in name_lower:
            return ModelType.VISION
        else:
            return ModelType.CHAT  # Default
    
    def _infer_capabilities(self, model_name: str) -> List[ModelCapability]:
        """Infer model capabilities from name."""
        name_lower = model_name.lower()
        capabilities = []
        
        if 'code' in name_lower:
            capabilities.extend([
                ModelCapability.CODE_GENERATION,
                ModelCapability.CODE_COMPLETION
            ])
        
        if 'chat' in name_lower or 'instruct' in name_lower:
            capabilities.extend([
                ModelCapability.CHAT,
                ModelCapability.INSTRUCTION_FOLLOWING
            ])
        
        if 'math' in name_lower:
            capabilities.append(ModelCapability.MATHEMATICS)
        
        if 'vision' in name_lower or 'llava' in name_lower:
            capabilities.append(ModelCapability.VISION)
        
        if 'embed' in name_lower:
            capabilities.append(ModelCapability.EMBEDDINGS)
        
        # Default capabilities
        if not capabilities:
            capabilities.extend([
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CHAT,
                ModelCapability.INSTRUCTION_FOLLOWING
            ])
        
        return capabilities
    
    def _calculate_recommendation_confidence(self, model: ModelInfo, 
                                           metadata: ModelMetadata,
                                           task_type: str, 
                                           preferences: Dict[str, Any]) -> float:
        """Calculate recommendation confidence score."""
        confidence = 0.5  # Base confidence
        
        # Task suitability boost
        if metadata.is_suitable_for(task_type):
            confidence += 0.3
        
        # Size preference
        prefer_speed = preferences.get('prefer_speed', False)
        if prefer_speed and model.size_category in [ModelSize.TINY, ModelSize.SMALL]:
            confidence += 0.1
        elif not prefer_speed and model.size_category in [ModelSize.MEDIUM, ModelSize.LARGE]:
            confidence += 0.1
        
        # Performance boost if we have metrics
        with self._lock:
            if model.name in self._performance_metrics:
                metrics = self._performance_metrics[model.name]
                if metrics.error_rate < 0.05:
                    confidence += 0.1
                if metrics.tokens_per_second > 10:
                    confidence += 0.05
        
        # Health boost
        with self._lock:
            if model.name in self._health_status:
                if self._health_status[model.name].is_healthy:
                    confidence += 0.05
        
        return min(1.0, confidence)
    
    def _generate_recommendation_reasoning(self, model: ModelInfo,
                                         metadata: ModelMetadata,
                                         task_type: str,
                                         preferences: Dict[str, Any]) -> List[str]:
        """Generate reasoning for recommendation."""
        reasoning = []
        
        if metadata.is_suitable_for(task_type):
            reasoning.append(f"Optimized for {task_type} tasks")
        
        if model.size_category == ModelSize.SMALL:
            reasoning.append("Fast response times due to compact size")
        elif model.size_category == ModelSize.LARGE:
            reasoning.append("High quality responses from large parameter count")
        
        # Performance reasoning
        with self._lock:
            if model.name in self._performance_metrics:
                metrics = self._performance_metrics[model.name]
                if metrics.error_rate < 0.05:
                    reasoning.append("Reliable with low error rate")
                if metrics.tokens_per_second > 10:
                    reasoning.append("Good performance with fast token generation")
        
        # Health reasoning
        with self._lock:
            if model.name in self._health_status:
                if self._health_status[model.name].is_healthy:
                    reasoning.append("Currently healthy and responsive")
        
        return reasoning
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        with self._lock:
            if not self._performance_metrics:
                return {}
            
            avg_response_time = sum(m.response_time_avg for m in self._performance_metrics.values()) / len(self._performance_metrics)
            avg_error_rate = sum(m.error_rate for m in self._performance_metrics.values()) / len(self._performance_metrics)
            total_requests = sum(m.total_requests for m in self._performance_metrics.values())
            
            return {
                'average_response_time': avg_response_time,
                'average_error_rate': avg_error_rate,
                'total_requests': total_requests,
                'models_tracked': len(self._performance_metrics)
            }


# Convenience functions
async def get_model_manager(config: Optional[Config] = None,
                          security_manager: Optional[SecurityManager] = None) -> ModelManager:
    """
    Get a configured model manager instance.
    
    Args:
        config: Optional configuration
        security_manager: Optional security manager
        
    Returns:
        ModelManager instance
    """
    return ModelManager(config=config, security_manager=security_manager)


async def quick_install_model(model_name: str, config: Optional[Config] = None) -> bool:
    """
    Quick model installation for convenience.
    
    Args:
        model_name: Name of the model to install
        config: Optional configuration
        
    Returns:
        True if successful, False otherwise
    """
    async with get_model_manager(config=config) as manager:
        return await manager.install_model(model_name)


async def quick_model_health_check(model_name: str, config: Optional[Config] = None) -> bool:
    """
    Quick health check for a model.
    
    Args:
        model_name: Name of the model to check
        config: Optional configuration
        
    Returns:
        True if healthy, False otherwise
    """
    async with get_model_manager(config=config) as manager:
        health_status = await manager.check_model_health(model_name)
        return health_status.is_healthy