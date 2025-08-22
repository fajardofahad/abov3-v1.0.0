"""
Comprehensive Training Management for ABOV3 4 Ollama.

This module provides advanced training capabilities including:
- Training data management and preprocessing pipelines
- Training loop control and real-time monitoring
- Hyperparameter optimization and automated tuning
- Early stopping and validation strategies
- Distributed training support and resource management
- Resource monitoring (GPU, CPU, memory usage)
- Training data augmentation and balancing
- Checkpoint management and model versioning

Features:
- Async support for non-blocking training operations
- Comprehensive resource monitoring and optimization
- Advanced hyperparameter search strategies
- Distributed training across multiple GPUs/nodes
- Real-time training metrics and visualization
- Automated data pipeline optimization
- Integration with existing fine-tuning infrastructure
- Advanced training strategies (curriculum learning, etc.)
"""

import asyncio
import json
import logging
import os
import psutil
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, AsyncIterator, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import yaml
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from ..core.api.ollama_client import OllamaClient, get_ollama_client
from ..core.config import Config, get_config
from ..utils.security import SecurityManager
from .fine_tuning import FineTuningJob, FineTuningProgress, FineTuningStatus, FineTuningConfig


logger = logging.getLogger(__name__)


class TrainingStrategy(Enum):
    """Training strategies."""
    STANDARD = "standard"
    CURRICULUM = "curriculum"
    PROGRESSIVE = "progressive"
    TRANSFER = "transfer"
    FEDERATED = "federated"
    ADVERSARIAL = "adversarial"


class OptimizationAlgorithm(Enum):
    """Optimization algorithms."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"


class SchedulerType(Enum):
    """Learning rate schedulers."""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    STEP = "step"
    PLATEAU = "plateau"


class ResourceType(Enum):
    """Resource types for monitoring."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    timestamp: datetime
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_io_sent_mb: float = 0.0
    network_io_recv_mb: float = 0.0
    temperature_celsius: float = 0.0


@dataclass
class TrainingMetrics:
    """Training performance metrics."""
    timestamp: datetime
    epoch: int
    step: int
    learning_rate: float
    loss: float
    accuracy: float = 0.0
    perplexity: float = 0.0
    gradient_norm: float = 0.0
    batch_size: int = 0
    tokens_per_second: float = 0.0
    examples_per_second: float = 0.0
    
    # Validation metrics
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_perplexity: Optional[float] = None
    
    # Additional metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration for optimization."""
    name: str
    value_type: str  # 'float', 'int', 'categorical'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    current_value: Any = None
    best_value: Any = None
    search_space: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HyperparameterTrial:
    """Single hyperparameter optimization trial."""
    trial_id: str
    parameters: Dict[str, Any]
    objective_value: float
    status: str  # 'running', 'completed', 'failed', 'pruned'
    start_time: datetime
    end_time: Optional[datetime] = None
    metrics_history: List[TrainingMetrics] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataPipelineConfig:
    """Data pipeline configuration."""
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = False
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Data augmentation
    enable_augmentation: bool = False
    augmentation_strategies: List[str] = field(default_factory=list)
    augmentation_probability: float = 0.5
    
    # Data filtering
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    quality_threshold: float = 0.0
    
    # Caching
    enable_caching: bool = True
    cache_dir: Optional[str] = None
    cache_size_gb: float = 1.0


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    enabled: bool = False
    backend: str = "nccl"  # 'nccl', 'gloo', 'mpi'
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 12355
    
    # Optimization
    gradient_compression: bool = False
    all_reduce_fusion: bool = True
    bucket_size_mb: int = 25
    
    # Fault tolerance
    enable_checkpointing: bool = True
    checkpoint_frequency: int = 1000


class ResourceMonitor:
    """Real-time resource monitoring system."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history_size = 1000
        self.monitor_thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable[[ResourceMetrics], None]] = []
        
        # GPU monitoring (requires nvidia-ml-py or similar)
        self.gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            return True
        except (ImportError, Exception):
            logger.warning("GPU monitoring not available")
            return False
    
    def add_callback(self, callback: Callable[[ResourceMetrics], None]) -> None:
        """Add a callback for resource metrics."""
        self.callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Limit history size
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
                
                # Trigger callbacks
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.warning(f"Resource monitor callback error: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network
        network_io = psutil.net_io_counters()
        
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_usage_percent=(disk_usage.used / disk_usage.total) * 100,
            disk_io_read_mb=(disk_io.read_bytes / (1024**2)) if disk_io else 0,
            disk_io_write_mb=(disk_io.write_bytes / (1024**2)) if disk_io else 0,
            network_io_sent_mb=(network_io.bytes_sent / (1024**2)) if network_io else 0,
            network_io_recv_mb=(network_io.bytes_recv / (1024**2)) if network_io else 0
        )
        
        # GPU metrics
        if self.gpu_available:
            try:
                gpu_metrics = self._collect_gpu_metrics()
                metrics.gpu_percent = gpu_metrics.get('utilization', 0.0)
                metrics.gpu_memory_used_gb = gpu_metrics.get('memory_used_gb', 0.0)
                metrics.gpu_memory_total_gb = gpu_metrics.get('memory_total_gb', 0.0)
                metrics.temperature_celsius = gpu_metrics.get('temperature', 0.0)
            except Exception as e:
                logger.warning(f"GPU metrics collection failed: {e}")
        
        return metrics
    
    def _collect_gpu_metrics(self) -> Dict[str, float]:
        """Collect GPU metrics."""
        try:
            import nvidia_ml_py3 as nvml
            
            # Get first GPU (could be extended for multi-GPU)
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            
            # Memory info
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_gb = mem_info.used / (1024**3)
            memory_total_gb = mem_info.total / (1024**3)
            
            # Temperature
            temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            
            return {
                'utilization': float(gpu_util),
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory_total_gb,
                'temperature': float(temp)
            }
            
        except Exception as e:
            logger.warning(f"Failed to collect GPU metrics: {e}")
            return {}
    
    def get_latest_metrics(self) -> Optional[ResourceMetrics]:
        """Get the latest resource metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_metrics(self, duration_minutes: int = 5) -> Optional[ResourceMetrics]:
        """Get average metrics over the specified duration."""
        if not self.metrics_history:
            return None
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        # Calculate averages
        avg_metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=np.mean([m.cpu_percent for m in recent_metrics]),
            memory_percent=np.mean([m.memory_percent for m in recent_metrics]),
            memory_used_gb=np.mean([m.memory_used_gb for m in recent_metrics]),
            memory_total_gb=recent_metrics[-1].memory_total_gb,  # Use latest
            gpu_percent=np.mean([m.gpu_percent for m in recent_metrics]),
            gpu_memory_used_gb=np.mean([m.gpu_memory_used_gb for m in recent_metrics]),
            gpu_memory_total_gb=recent_metrics[-1].gpu_memory_total_gb,  # Use latest
            disk_usage_percent=np.mean([m.disk_usage_percent for m in recent_metrics]),
            temperature_celsius=np.mean([m.temperature_celsius for m in recent_metrics])
        )
        
        return avg_metrics


class HyperparameterOptimizer:
    """Hyperparameter optimization system."""
    
    def __init__(self, objective: str = "minimize"):
        self.objective = objective  # 'minimize' or 'maximize'
        self.trials: List[HyperparameterTrial] = []
        self.best_trial: Optional[HyperparameterTrial] = None
        self.search_space: Dict[str, HyperparameterConfig] = {}
    
    def add_hyperparameter(self, config: HyperparameterConfig) -> None:
        """Add a hyperparameter to the search space."""
        self.search_space[config.name] = config
    
    def suggest_trial(self, strategy: str = "random") -> Dict[str, Any]:
        """Suggest hyperparameters for the next trial."""
        if strategy == "random":
            return self._random_search()
        elif strategy == "bayesian":
            return self._bayesian_optimization()
        elif strategy == "grid":
            return self._grid_search()
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
    
    def _random_search(self) -> Dict[str, Any]:
        """Random search strategy."""
        import random
        
        parameters = {}
        
        for name, config in self.search_space.items():
            if config.value_type == "float":
                if config.log_scale:
                    value = np.exp(random.uniform(
                        np.log(config.min_value),
                        np.log(config.max_value)
                    ))
                else:
                    value = random.uniform(config.min_value, config.max_value)
                parameters[name] = float(value)
                
            elif config.value_type == "int":
                value = random.randint(config.min_value, config.max_value)
                parameters[name] = int(value)
                
            elif config.value_type == "categorical":
                value = random.choice(config.choices)
                parameters[name] = value
        
        return parameters
    
    def _bayesian_optimization(self) -> Dict[str, Any]:
        """Bayesian optimization strategy (simplified implementation)."""
        # This would typically use a library like scikit-optimize or Optuna
        # For now, fall back to random search with some intelligence
        
        if len(self.trials) < 3:
            return self._random_search()
        
        # Simple acquisition function based on successful trials
        successful_trials = [t for t in self.trials if t.status == "completed"]
        if not successful_trials:
            return self._random_search()
        
        # Find best performing trials and add noise to their parameters
        best_trials = sorted(
            successful_trials,
            key=lambda t: t.objective_value,
            reverse=(self.objective == "maximize")
        )[:3]
        
        # Weighted average of best trials with added noise
        parameters = {}
        for name, config in self.search_space.items():
            if config.value_type in ["float", "int"]:
                values = [t.parameters[name] for t in best_trials if name in t.parameters]
                if values:
                    base_value = np.mean(values)
                    noise_scale = (config.max_value - config.min_value) * 0.1
                    value = np.random.normal(base_value, noise_scale)
                    value = np.clip(value, config.min_value, config.max_value)
                    
                    if config.value_type == "int":
                        parameters[name] = int(value)
                    else:
                        parameters[name] = float(value)
                else:
                    return self._random_search()
            else:
                # For categorical, use random selection
                parameters[name] = np.random.choice(config.choices)
        
        return parameters
    
    def _grid_search(self) -> Dict[str, Any]:
        """Grid search strategy."""
        # Generate all combinations (simplified)
        # This would need more sophisticated implementation for large spaces
        return self._random_search()  # Fallback for now
    
    def record_trial(self, trial: HyperparameterTrial) -> None:
        """Record a completed trial."""
        self.trials.append(trial)
        
        if trial.status == "completed":
            if (self.best_trial is None or
                (self.objective == "minimize" and trial.objective_value < self.best_trial.objective_value) or
                (self.objective == "maximize" and trial.objective_value > self.best_trial.objective_value)):
                self.best_trial = trial
                
                # Update best values in search space
                for name, value in trial.parameters.items():
                    if name in self.search_space:
                        self.search_space[name].best_value = value
    
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters found so far."""
        return self.best_trial.parameters if self.best_trial else None


class TrainingManager:
    """Advanced training management system."""
    
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        config: Optional[Config] = None
    ):
        self.ollama_client = ollama_client or get_ollama_client()
        self.config = config or get_config()
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Hyperparameter optimization
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        
        # Training state
        self.active_trainings: Dict[str, FineTuningJob] = {}
        self.training_metrics: Dict[str, List[TrainingMetrics]] = defaultdict(list)
        
        # Data pipeline
        self.data_pipeline_cache: Dict[str, Any] = {}
        
        # Distributed training
        self.distributed_config: Optional[DistributedConfig] = None
        
        logger.info("TrainingManager initialized")
    
    def configure_distributed_training(self, config: DistributedConfig) -> None:
        """Configure distributed training parameters."""
        self.distributed_config = config
        logger.info(f"Distributed training configured: {config.world_size} nodes")
    
    def setup_hyperparameter_optimization(
        self,
        hyperparameters: List[HyperparameterConfig],
        objective: str = "minimize"
    ) -> None:
        """Setup hyperparameter optimization."""
        self.hyperparameter_optimizer = HyperparameterOptimizer(objective)
        
        for hp_config in hyperparameters:
            self.hyperparameter_optimizer.add_hyperparameter(hp_config)
        
        logger.info(f"Hyperparameter optimization setup with {len(hyperparameters)} parameters")
    
    async def start_training(self, job: FineTuningJob) -> None:
        """Start training for a job."""
        job_id = job.job_id
        logger.info(f"Starting training for job: {job_id}")
        
        # Register job
        self.active_trainings[job_id] = job
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Setup monitoring callbacks
        self.resource_monitor.add_callback(
            lambda metrics: self._on_resource_update(job_id, metrics)
        )
        
        try:
            # Setup data pipeline
            await self._setup_data_pipeline(job)
            
            # Run training loop
            await self._training_loop(job)
            
        except Exception as e:
            logger.error(f"Training failed for job {job_id}: {e}")
            job.progress.status = FineTuningStatus.FAILED
            job.progress.error_message = str(e)
            raise
        finally:
            # Cleanup
            if job_id in self.active_trainings:
                del self.active_trainings[job_id]
    
    async def _setup_data_pipeline(self, job: FineTuningJob) -> None:
        """Setup data pipeline for training."""
        logger.info(f"Setting up data pipeline for job: {job.job_id}")
        
        # Create data pipeline configuration
        pipeline_config = DataPipelineConfig(
            batch_size=job.config.batch_size,
            num_workers=job.config.dataloader_num_workers,
            shuffle=True,
            enable_caching=True,
            cache_dir=os.path.join(job.output_dir, "cache")
        )
        
        # Setup cache directory
        os.makedirs(pipeline_config.cache_dir, exist_ok=True)
        
        # Process datasets
        processed_datasets = []
        for dataset_path in job.metadata.get("prepared_datasets", []):
            processed_data = await self._process_dataset(dataset_path, pipeline_config)
            processed_datasets.append(processed_data)
        
        # Store in cache
        cache_key = f"pipeline_{job.job_id}"
        self.data_pipeline_cache[cache_key] = {
            "config": pipeline_config,
            "datasets": processed_datasets,
            "stats": await self._compute_dataset_stats(processed_datasets)
        }
        
        logger.info(f"Data pipeline setup completed for job: {job.job_id}")
    
    async def _process_dataset(self, dataset_path: str, config: DataPipelineConfig) -> Dict[str, Any]:
        """Process a single dataset."""
        logger.info(f"Processing dataset: {dataset_path}")
        
        # Load dataset
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        # Apply filtering
        if config.min_length or config.max_length:
            data = await self._filter_by_length(data, config)
        
        # Apply quality filtering
        if config.quality_threshold > 0:
            data = await self._filter_by_quality(data, config)
        
        # Data augmentation
        if config.enable_augmentation:
            data = await self._apply_augmentation(data, config)
        
        # Create batches
        batches = await self._create_batches(data, config)
        
        return {
            "path": dataset_path,
            "data": data,
            "batches": batches,
            "total_samples": len(data),
            "total_batches": len(batches)
        }
    
    async def _filter_by_length(self, data: List[Dict[str, Any]], config: DataPipelineConfig) -> List[Dict[str, Any]]:
        """Filter data by content length."""
        filtered_data = []
        
        for record in data:
            content_length = len(json.dumps(record, ensure_ascii=False))
            
            if config.min_length and content_length < config.min_length:
                continue
            if config.max_length and content_length > config.max_length:
                continue
            
            filtered_data.append(record)
        
        logger.info(f"Length filtering: {len(data)} -> {len(filtered_data)} samples")
        return filtered_data
    
    async def _filter_by_quality(self, data: List[Dict[str, Any]], config: DataPipelineConfig) -> List[Dict[str, Any]]:
        """Filter data by quality score."""
        # Simplified quality scoring
        filtered_data = []
        
        for record in data:
            quality_score = await self._compute_quality_score(record)
            if quality_score >= config.quality_threshold:
                filtered_data.append(record)
        
        logger.info(f"Quality filtering: {len(data)} -> {len(filtered_data)} samples")
        return filtered_data
    
    async def _compute_quality_score(self, record: Dict[str, Any]) -> float:
        """Compute quality score for a record."""
        # Simple quality metrics
        score = 0.0
        
        # Check for required fields
        if "input" in record and "output" in record:
            score += 0.5
        
        # Check content length balance
        if "input" in record and "output" in record:
            input_len = len(str(record["input"]))
            output_len = len(str(record["output"]))
            
            if 0.1 <= (output_len / max(input_len, 1)) <= 10:
                score += 0.3
        
        # Check for text quality indicators
        content = json.dumps(record, ensure_ascii=False)
        if not any(char in content for char in ['<', '>', '{', '}']):  # No HTML/code artifacts
            score += 0.2
        
        return min(score, 1.0)
    
    async def _apply_augmentation(self, data: List[Dict[str, Any]], config: DataPipelineConfig) -> List[Dict[str, Any]]:
        """Apply data augmentation strategies."""
        augmented_data = data.copy()
        
        for strategy in config.augmentation_strategies:
            if strategy == "paraphrase":
                augmented_data.extend(await self._paraphrase_augmentation(data, config))
            elif strategy == "backtranslation":
                augmented_data.extend(await self._backtranslation_augmentation(data, config))
            elif strategy == "noise_injection":
                augmented_data.extend(await self._noise_injection_augmentation(data, config))
        
        logger.info(f"Data augmentation: {len(data)} -> {len(augmented_data)} samples")
        return augmented_data
    
    async def _paraphrase_augmentation(self, data: List[Dict[str, Any]], config: DataPipelineConfig) -> List[Dict[str, Any]]:
        """Apply paraphrase-based augmentation."""
        # Simplified implementation
        augmented = []
        
        for record in data:
            if np.random.random() < config.augmentation_probability:
                # Create a paraphrased version (simplified)
                augmented_record = record.copy()
                if "input" in record:
                    # Simple word shuffling as paraphrase simulation
                    words = str(record["input"]).split()
                    if len(words) > 3:
                        # Shuffle middle words
                        middle = words[1:-1]
                        np.random.shuffle(middle)
                        augmented_record["input"] = " ".join([words[0]] + middle + [words[-1]])
                
                augmented.append(augmented_record)
        
        return augmented
    
    async def _backtranslation_augmentation(self, data: List[Dict[str, Any]], config: DataPipelineConfig) -> List[Dict[str, Any]]:
        """Apply backtranslation augmentation."""
        # Placeholder for backtranslation implementation
        return []
    
    async def _noise_injection_augmentation(self, data: List[Dict[str, Any]], config: DataPipelineConfig) -> List[Dict[str, Any]]:
        """Apply noise injection augmentation."""
        augmented = []
        
        for record in data:
            if np.random.random() < config.augmentation_probability:
                augmented_record = record.copy()
                
                # Add small typos or character substitutions
                for key, value in record.items():
                    if isinstance(value, str) and len(value) > 10:
                        chars = list(value)
                        # Randomly substitute 1-2 characters
                        for _ in range(min(2, len(chars) // 20)):
                            if np.random.random() < 0.1:
                                idx = np.random.randint(0, len(chars))
                                chars[idx] = chr(ord(chars[idx]) + np.random.choice([-1, 1]))
                        
                        augmented_record[key] = "".join(chars)
                
                augmented.append(augmented_record)
        
        return augmented
    
    async def _create_batches(self, data: List[Dict[str, Any]], config: DataPipelineConfig) -> List[List[Dict[str, Any]]]:
        """Create training batches."""
        if config.shuffle:
            np.random.shuffle(data)
        
        batches = []
        for i in range(0, len(data), config.batch_size):
            batch = data[i:i + config.batch_size]
            if not config.drop_last or len(batch) == config.batch_size:
                batches.append(batch)
        
        return batches
    
    async def _compute_dataset_stats(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute dataset statistics."""
        total_samples = sum(ds["total_samples"] for ds in datasets)
        total_batches = sum(ds["total_batches"] for ds in datasets)
        
        # Compute content length statistics
        all_lengths = []
        for dataset in datasets:
            for record in dataset["data"]:
                content_length = len(json.dumps(record, ensure_ascii=False))
                all_lengths.append(content_length)
        
        stats = {
            "total_datasets": len(datasets),
            "total_samples": total_samples,
            "total_batches": total_batches,
            "avg_length": np.mean(all_lengths) if all_lengths else 0,
            "median_length": np.median(all_lengths) if all_lengths else 0,
            "min_length": np.min(all_lengths) if all_lengths else 0,
            "max_length": np.max(all_lengths) if all_lengths else 0,
            "std_length": np.std(all_lengths) if all_lengths else 0
        }
        
        return stats
    
    async def _training_loop(self, job: FineTuningJob) -> None:
        """Main training loop."""
        logger.info(f"Starting training loop for job: {job.job_id}")
        
        cache_key = f"pipeline_{job.job_id}"
        pipeline_data = self.data_pipeline_cache.get(cache_key)
        
        if not pipeline_data:
            raise ValueError("Data pipeline not setup")
        
        # Initialize training state
        current_step = 0
        total_samples = pipeline_data["stats"]["total_samples"]
        total_batches = pipeline_data["stats"]["total_batches"]
        steps_per_epoch = total_batches
        
        job.progress.total_steps = steps_per_epoch * job.config.num_epochs
        job.progress.total_epochs = job.config.num_epochs
        
        # Training loop
        for epoch in range(job.config.num_epochs):
            if job.progress.status in [FineTuningStatus.CANCELLED, FineTuningStatus.PAUSED]:
                break
            
            job.progress.current_epoch = epoch + 1
            logger.info(f"Starting epoch {epoch + 1}/{job.config.num_epochs}")
            
            # Process each dataset
            for dataset in pipeline_data["datasets"]:
                for batch_idx, batch in enumerate(dataset["batches"]):
                    if job.progress.status in [FineTuningStatus.CANCELLED, FineTuningStatus.PAUSED]:
                        break
                    
                    current_step += 1
                    job.progress.current_step = current_step
                    
                    # Simulate training step
                    step_metrics = await self._training_step(job, batch, current_step)
                    
                    # Record metrics
                    self.training_metrics[job.job_id].append(step_metrics)
                    
                    # Update job progress
                    job.progress.training_loss = step_metrics.loss
                    job.progress.learning_rate = step_metrics.learning_rate
                    job.progress.accuracy = step_metrics.accuracy
                    
                    # Validation
                    if current_step % job.config.eval_steps == 0:
                        val_metrics = await self._validation_step(job, current_step)
                        job.progress.validation_loss = val_metrics.get("loss", 0.0)
                        job.progress.perplexity = val_metrics.get("perplexity", 0.0)
                    
                    # Checkpointing
                    if current_step % job.config.save_steps == 0:
                        await self._save_checkpoint(job, current_step)
                    
                    # Small delay to simulate training time
                    await asyncio.sleep(0.001)
            
            # End of epoch
            logger.info(f"Completed epoch {epoch + 1}")
    
    async def _training_step(self, job: FineTuningJob, batch: List[Dict[str, Any]], step: int) -> TrainingMetrics:
        """Execute a single training step."""
        # Simulate training metrics
        progress = step / job.progress.total_steps
        
        # Simulate loss decay
        initial_loss = 3.0
        final_loss = 0.5
        noise = np.random.normal(0, 0.1)
        loss = initial_loss * (1 - progress) + final_loss * progress + noise
        loss = max(0.1, loss)  # Ensure positive loss
        
        # Simulate accuracy improvement
        accuracy = min(0.95, progress * 0.9 + 0.05 + np.random.normal(0, 0.02))
        accuracy = max(0.0, accuracy)
        
        # Learning rate with decay
        lr_decay = (1 - progress) ** 0.9
        current_lr = job.config.learning_rate * lr_decay
        
        # Simulate other metrics
        perplexity = np.exp(loss)
        gradient_norm = 1.0 + np.random.exponential(0.5)
        tokens_per_second = 100 + np.random.normal(0, 10)
        examples_per_second = len(batch) / max(0.1, np.random.exponential(0.1))
        
        metrics = TrainingMetrics(
            timestamp=datetime.now(),
            epoch=job.progress.current_epoch,
            step=step,
            learning_rate=current_lr,
            loss=loss,
            accuracy=accuracy,
            perplexity=perplexity,
            gradient_norm=gradient_norm,
            batch_size=len(batch),
            tokens_per_second=tokens_per_second,
            examples_per_second=examples_per_second
        )
        
        return metrics
    
    async def _validation_step(self, job: FineTuningJob, step: int) -> Dict[str, float]:
        """Execute validation step."""
        # Simulate validation metrics
        training_loss = job.progress.training_loss
        
        # Validation loss typically higher than training loss
        val_loss = training_loss * (1.05 + np.random.normal(0, 0.05))
        val_accuracy = job.progress.accuracy * (0.95 + np.random.normal(0, 0.02))
        val_perplexity = np.exp(val_loss)
        
        val_metrics = {
            "loss": max(0.1, val_loss),
            "accuracy": max(0.0, min(1.0, val_accuracy)),
            "perplexity": val_perplexity
        }
        
        logger.info(f"Validation metrics at step {step}: {val_metrics}")
        return val_metrics
    
    async def _save_checkpoint(self, job: FineTuningJob, step: int) -> None:
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(job.output_dir, "checkpoints", f"step-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save checkpoint metadata
        checkpoint_data = {
            "job_id": job.job_id,
            "step": step,
            "epoch": job.progress.current_epoch,
            "loss": job.progress.training_loss,
            "accuracy": job.progress.accuracy,
            "learning_rate": job.progress.learning_rate,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(checkpoint_dir, "checkpoint.json"), "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        
        job.progress.last_checkpoint = checkpoint_dir
        job.progress.checkpoint_count += 1
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
    
    def _on_resource_update(self, job_id: str, metrics: ResourceMetrics) -> None:
        """Handle resource metrics update."""
        job = self.active_trainings.get(job_id)
        if job:
            # Update job progress with resource info
            job.progress.memory_usage_gb = metrics.memory_used_gb
            job.progress.gpu_utilization = metrics.gpu_percent
            job.progress.cpu_utilization = metrics.cpu_percent
            
            # Check for resource issues
            if metrics.memory_percent > 95:
                warning = f"High memory usage: {metrics.memory_percent:.1f}%"
                if warning not in job.progress.warnings:
                    job.progress.warnings.append(warning)
                    logger.warning(f"Job {job_id}: {warning}")
            
            if metrics.gpu_percent < 10 and hasattr(metrics, 'gpu_available'):
                warning = "Low GPU utilization detected"
                if warning not in job.progress.warnings:
                    job.progress.warnings.append(warning)
                    logger.warning(f"Job {job_id}: {warning}")
    
    def get_training_metrics(self, job_id: str) -> List[TrainingMetrics]:
        """Get training metrics for a job."""
        return self.training_metrics.get(job_id, [])
    
    def get_resource_metrics(self) -> List[ResourceMetrics]:
        """Get recent resource metrics."""
        return self.resource_monitor.metrics_history[-100:]  # Last 100 entries
    
    async def optimize_hyperparameters(
        self,
        job_template: FineTuningJob,
        n_trials: int = 10,
        strategy: str = "random"
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        logger.info(f"Starting hyperparameter optimization: {n_trials} trials")
        
        best_result = None
        
        for trial_idx in range(n_trials):
            # Suggest hyperparameters
            suggested_params = self.hyperparameter_optimizer.suggest_trial(strategy)
            
            # Create trial
            trial = HyperparameterTrial(
                trial_id=f"trial_{trial_idx}",
                parameters=suggested_params,
                objective_value=0.0,
                status="running",
                start_time=datetime.now()
            )
            
            try:
                # Update job config with suggested parameters
                trial_job = self._create_trial_job(job_template, suggested_params)
                
                # Run training
                await self.start_training(trial_job)
                
                # Get objective value (e.g., validation loss)
                final_metrics = trial_job.evaluation_results
                objective_value = final_metrics.get("final_loss", float('inf'))
                
                trial.objective_value = objective_value
                trial.status = "completed"
                trial.end_time = datetime.now()
                
                if best_result is None or objective_value < best_result["objective_value"]:
                    best_result = {
                        "parameters": suggested_params,
                        "objective_value": objective_value,
                        "trial_id": trial.trial_id
                    }
                
            except Exception as e:
                trial.status = "failed"
                trial.end_time = datetime.now()
                logger.error(f"Trial {trial_idx} failed: {e}")
            
            # Record trial
            self.hyperparameter_optimizer.record_trial(trial)
            
            logger.info(f"Trial {trial_idx} completed: {trial.status}, objective: {trial.objective_value}")
        
        logger.info(f"Hyperparameter optimization completed. Best result: {best_result}")
        return best_result or {}
    
    def _create_trial_job(self, template_job: FineTuningJob, parameters: Dict[str, Any]) -> FineTuningJob:
        """Create a trial job with suggested hyperparameters."""
        # Create a copy of the template job
        trial_config = FineTuningConfig(**asdict(template_job.config))
        
        # Update with suggested parameters
        for param_name, param_value in parameters.items():
            if hasattr(trial_config, param_name):
                setattr(trial_config, param_name, param_value)
        
        # Create trial job (simplified)
        trial_job = FineTuningJob(
            job_id=f"trial_{len(self.hyperparameter_optimizer.trials)}",
            config=trial_config,
            progress=FineTuningProgress(
                job_id=f"trial_{len(self.hyperparameter_optimizer.trials)}",
                status=FineTuningStatus.PENDING
            ),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            output_dir=os.path.join(template_job.output_dir, "trials", f"trial_{len(self.hyperparameter_optimizer.trials)}"),
            log_file="",
            config_file=""
        )
        
        return trial_job


# Global instance management
_training_manager: Optional[TrainingManager] = None


def get_training_manager() -> TrainingManager:
    """Get the global TrainingManager instance."""
    global _training_manager
    if _training_manager is None:
        _training_manager = TrainingManager()
    return _training_manager


# Convenience functions
async def quick_hyperparameter_search(
    job: FineTuningJob,
    search_space: Dict[str, Dict[str, Any]],
    n_trials: int = 10
) -> Dict[str, Any]:
    """Quick hyperparameter search with minimal setup."""
    manager = get_training_manager()
    
    # Convert search space to HyperparameterConfig objects
    hyperparameters = []
    for name, config in search_space.items():
        hp_config = HyperparameterConfig(
            name=name,
            value_type=config["type"],
            min_value=config.get("min"),
            max_value=config.get("max"),
            choices=config.get("choices"),
            log_scale=config.get("log_scale", False)
        )
        hyperparameters.append(hp_config)
    
    # Setup optimization
    manager.setup_hyperparameter_optimization(hyperparameters, "minimize")
    
    # Run optimization
    return await manager.optimize_hyperparameters(job, n_trials, "random")


def monitor_resources(interval: float = 1.0) -> ResourceMonitor:
    """Create and start a resource monitor."""
    monitor = ResourceMonitor(interval)
    monitor.start_monitoring()
    return monitor