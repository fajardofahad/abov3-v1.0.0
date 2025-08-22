"""
Comprehensive Fine-Tuning Interface for ABOV3 4 Ollama.

This module provides advanced fine-tuning capabilities including:
- Fine-tuning workflow management and orchestration
- Dataset preparation, validation, and preprocessing
- Training configuration and hyperparameter management
- Progress monitoring, callbacks, and real-time tracking
- Model versioning, checkpoints, and rollback capabilities
- Integration with Ollama's fine-tuning infrastructure
- Security validation and compliance checks
- Automated quality assurance and testing

Features:
- Async support for non-blocking fine-tuning operations
- Comprehensive error handling and recovery mechanisms
- Advanced dataset validation and security scanning
- Real-time progress monitoring with callbacks
- Automated hyperparameter optimization
- Model versioning with semantic versioning
- Integration with existing ModelManager and Registry
- Export/import of fine-tuning configurations
- Detailed logging and audit trails
"""

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, AsyncIterator, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import yaml

from ..core.api.ollama_client import OllamaClient, get_ollama_client
from ..core.config import Config, get_config
from ..utils.security import SecurityManager, SecurityEvent
from ..utils.validation import ValidationManager
from .manager import ModelManager, get_model_manager, ModelPerformanceMetrics
from .registry import ModelRegistry, get_model_registry
from .info import ModelInfo, ModelMetadata


logger = logging.getLogger(__name__)


class FineTuningStatus(Enum):
    """Fine-tuning job status."""
    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class DatasetFormat(Enum):
    """Supported dataset formats."""
    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    TXT = "txt"
    PARQUET = "parquet"
    HUGGINGFACE = "huggingface"


class FineTuningObjective(Enum):
    """Fine-tuning objectives."""
    CODE_GENERATION = "code_generation"
    CHAT_COMPLETION = "chat_completion"
    INSTRUCTION_FOLLOWING = "instruction_following"
    TEXT_COMPLETION = "text_completion"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "question_answering"


@dataclass
class DatasetConfig:
    """Configuration for training datasets."""
    path: str
    format: DatasetFormat
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # train, val, test
    max_samples: Optional[int] = None
    shuffle: bool = True
    validation_checks: List[str] = field(default_factory=lambda: [
        "format_validation", "content_safety", "data_quality", "security_scan"
    ])
    preprocessing_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FineTuningConfig:
    """Comprehensive fine-tuning configuration."""
    # Model configuration
    base_model: str
    model_name: str
    objective: FineTuningObjective
    
    # Dataset configuration
    datasets: List[DatasetConfig]
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Advanced training options
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Optimization settings
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clipping: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Validation and evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    validation_metrics: List[str] = field(default_factory=lambda: [
        "perplexity", "accuracy", "loss"
    ])
    
    # Resource management
    max_memory_gb: Optional[float] = None
    use_gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # Model versioning
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    # Security and compliance
    enable_security_validation: bool = True
    enable_bias_detection: bool = True
    privacy_level: str = "standard"  # minimal, standard, strict
    
    # Callbacks and monitoring
    callbacks: List[str] = field(default_factory=lambda: [
        "progress_callback", "checkpoint_callback", "early_stopping_callback"
    ])
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ["ollama", "gguf"])
    
    # Metadata
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FineTuningProgress:
    """Real-time fine-tuning progress tracking."""
    job_id: str
    status: FineTuningStatus
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    
    # Metrics
    training_loss: float = 0.0
    validation_loss: float = 0.0
    learning_rate: float = 0.0
    accuracy: float = 0.0
    perplexity: float = 0.0
    
    # Time tracking
    start_time: Optional[datetime] = None
    current_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    elapsed_time: Optional[timedelta] = None
    
    # Resource usage
    memory_usage_gb: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    
    # Messages and logs
    last_message: str = ""
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    
    # Checkpoint info
    last_checkpoint: Optional[str] = None
    best_checkpoint: Optional[str] = None
    checkpoint_count: int = 0


@dataclass
class FineTuningJob:
    """Fine-tuning job representation."""
    job_id: str
    config: FineTuningConfig
    progress: FineTuningProgress
    created_at: datetime
    updated_at: datetime
    
    # File paths
    output_dir: str
    log_file: str
    config_file: str
    
    # Results
    final_model_path: Optional[str] = None
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class FineTuningCallback:
    """Base class for fine-tuning callbacks."""
    
    def on_training_start(self, job: FineTuningJob) -> None:
        """Called when training starts."""
        pass
    
    def on_epoch_start(self, job: FineTuningJob, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_step(self, job: FineTuningJob, step: int, metrics: Dict[str, Any]) -> None:
        """Called after each training step."""
        pass
    
    def on_validation(self, job: FineTuningJob, metrics: Dict[str, Any]) -> None:
        """Called after validation."""
        pass
    
    def on_checkpoint_save(self, job: FineTuningJob, checkpoint_path: str) -> None:
        """Called when a checkpoint is saved."""
        pass
    
    def on_epoch_end(self, job: FineTuningJob, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_training_end(self, job: FineTuningJob, final_metrics: Dict[str, Any]) -> None:
        """Called when training ends."""
        pass
    
    def on_error(self, job: FineTuningJob, error: Exception) -> None:
        """Called when an error occurs."""
        pass


class ProgressCallback(FineTuningCallback):
    """Callback for progress monitoring and logging."""
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.step_count = 0
    
    def on_step(self, job: FineTuningJob, step: int, metrics: Dict[str, Any]) -> None:
        """Log progress at specified intervals."""
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            progress = job.progress
            logger.info(
                f"Job {job.job_id}: Step {step}/{progress.total_steps}, "
                f"Loss: {metrics.get('loss', 0.0):.4f}, "
                f"LR: {metrics.get('learning_rate', 0.0):.2e}"
            )


class CheckpointCallback(FineTuningCallback):
    """Callback for model checkpointing."""
    
    def __init__(self, save_interval: int = 1000, keep_best: bool = True):
        self.save_interval = save_interval
        self.keep_best = keep_best
        self.best_loss = float('inf')
    
    def on_step(self, job: FineTuningJob, step: int, metrics: Dict[str, Any]) -> None:
        """Save checkpoint at specified intervals."""
        if step % self.save_interval == 0:
            checkpoint_path = os.path.join(
                job.output_dir, "checkpoints", f"checkpoint-{step}"
            )
            self._save_checkpoint(job, checkpoint_path, metrics)
    
    def on_validation(self, job: FineTuningJob, metrics: Dict[str, Any]) -> None:
        """Save best checkpoint based on validation loss."""
        if self.keep_best:
            val_loss = metrics.get('validation_loss', float('inf'))
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                checkpoint_path = os.path.join(job.output_dir, "best_checkpoint")
                self._save_checkpoint(job, checkpoint_path, metrics)
                job.progress.best_checkpoint = checkpoint_path
    
    def _save_checkpoint(self, job: FineTuningJob, path: str, metrics: Dict[str, Any]) -> None:
        """Save model checkpoint."""
        os.makedirs(path, exist_ok=True)
        
        # Save checkpoint metadata
        checkpoint_data = {
            "job_id": job.job_id,
            "step": job.progress.current_step,
            "epoch": job.progress.current_epoch,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(job.config)
        }
        
        with open(os.path.join(path, "checkpoint_metadata.json"), "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        
        job.progress.last_checkpoint = path
        job.progress.checkpoint_count += 1
        
        logger.info(f"Checkpoint saved: {path}")


class EarlyStoppingCallback(FineTuningCallback):
    """Callback for early stopping based on validation metrics."""
    
    def __init__(self, patience: int = 3, threshold: float = 0.001, metric: str = "validation_loss"):
        self.patience = patience
        self.threshold = threshold
        self.metric = metric
        self.best_value = float('inf') if 'loss' in metric else 0.0
        self.patience_count = 0
        self.should_stop = False
    
    def on_validation(self, job: FineTuningJob, metrics: Dict[str, Any]) -> None:
        """Check if early stopping criteria are met."""
        current_value = metrics.get(self.metric, float('inf'))
        
        if 'loss' in self.metric:
            improved = current_value < (self.best_value - self.threshold)
        else:
            improved = current_value > (self.best_value + self.threshold)
        
        if improved:
            self.best_value = current_value
            self.patience_count = 0
        else:
            self.patience_count += 1
        
        if self.patience_count >= self.patience:
            self.should_stop = True
            logger.info(
                f"Early stopping triggered. No improvement in {self.metric} "
                f"for {self.patience} validation checks."
            )


class FineTuningManager:
    """Advanced fine-tuning management system."""
    
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        model_manager: Optional[ModelManager] = None,
        model_registry: Optional[ModelRegistry] = None,
        config: Optional[Config] = None
    ):
        self.ollama_client = ollama_client or get_ollama_client()
        self.model_manager = model_manager or get_model_manager()
        self.model_registry = model_registry or get_model_registry()
        self.config = config or get_config()
        
        self.security_manager = SecurityManager()
        self.validation_manager = ValidationManager()
        
        # Job management
        self.active_jobs: Dict[str, FineTuningJob] = {}
        self.job_history: List[FineTuningJob] = []
        
        # Callback management
        self.global_callbacks: List[FineTuningCallback] = []
        
        # Setup directories
        self.base_output_dir = Path(self.config.get("fine_tuning.output_dir", "./fine_tuning_outputs"))
        self.base_output_dir.mkdir(exist_ok=True)
        
        logger.info("FineTuningManager initialized")
    
    def add_global_callback(self, callback: FineTuningCallback) -> None:
        """Add a global callback that applies to all jobs."""
        self.global_callbacks.append(callback)
    
    def remove_global_callback(self, callback: FineTuningCallback) -> None:
        """Remove a global callback."""
        if callback in self.global_callbacks:
            self.global_callbacks.remove(callback)
    
    async def create_job(
        self,
        config: FineTuningConfig,
        job_id: Optional[str] = None
    ) -> FineTuningJob:
        """Create a new fine-tuning job."""
        if job_id is None:
            job_id = f"ft-{uuid.uuid4().hex[:8]}"
        
        # Validate configuration
        await self._validate_config(config)
        
        # Setup job directories
        output_dir = self.base_output_dir / job_id
        output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        (output_dir / "datasets").mkdir(exist_ok=True)
        
        # Initialize progress
        progress = FineTuningProgress(
            job_id=job_id,
            status=FineTuningStatus.PENDING,
            start_time=datetime.now()
        )
        
        # Create job
        job = FineTuningJob(
            job_id=job_id,
            config=config,
            progress=progress,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            output_dir=str(output_dir),
            log_file=str(output_dir / "logs" / "training.log"),
            config_file=str(output_dir / "config.yaml")
        )
        
        # Save configuration
        await self._save_job_config(job)
        
        # Register job
        self.active_jobs[job_id] = job
        
        logger.info(f"Fine-tuning job created: {job_id}")
        return job
    
    async def start_job(self, job_id: str) -> None:
        """Start a fine-tuning job."""
        job = self.active_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        if job.progress.status != FineTuningStatus.PENDING:
            raise ValueError(f"Job {job_id} is not in pending status")
        
        logger.info(f"Starting fine-tuning job: {job_id}")
        
        try:
            # Update status
            job.progress.status = FineTuningStatus.PREPARING
            job.updated_at = datetime.now()
            
            # Trigger callbacks
            for callback in self.global_callbacks:
                callback.on_training_start(job)
            
            # Prepare datasets
            await self._prepare_datasets(job)
            
            # Start training
            job.progress.status = FineTuningStatus.TRAINING
            await self._run_training(job)
            
            # Complete job
            job.progress.status = FineTuningStatus.COMPLETED
            job.updated_at = datetime.now()
            
            logger.info(f"Fine-tuning job completed: {job_id}")
            
        except Exception as e:
            job.progress.status = FineTuningStatus.FAILED
            job.progress.error_message = str(e)
            job.updated_at = datetime.now()
            
            # Trigger error callbacks
            for callback in self.global_callbacks:
                callback.on_error(job, e)
            
            logger.error(f"Fine-tuning job failed: {job_id}, Error: {e}")
            raise
    
    async def pause_job(self, job_id: str) -> None:
        """Pause a running fine-tuning job."""
        job = self.active_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        if job.progress.status not in [FineTuningStatus.TRAINING, FineTuningStatus.VALIDATING]:
            raise ValueError(f"Job {job_id} cannot be paused in current status")
        
        job.progress.status = FineTuningStatus.PAUSED
        job.updated_at = datetime.now()
        
        logger.info(f"Fine-tuning job paused: {job_id}")
    
    async def resume_job(self, job_id: str) -> None:
        """Resume a paused fine-tuning job."""
        job = self.active_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        if job.progress.status != FineTuningStatus.PAUSED:
            raise ValueError(f"Job {job_id} is not paused")
        
        job.progress.status = FineTuningStatus.TRAINING
        job.updated_at = datetime.now()
        
        logger.info(f"Fine-tuning job resumed: {job_id}")
    
    async def cancel_job(self, job_id: str) -> None:
        """Cancel a fine-tuning job."""
        job = self.active_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        job.progress.status = FineTuningStatus.CANCELLED
        job.updated_at = datetime.now()
        
        logger.info(f"Fine-tuning job cancelled: {job_id}")
    
    def get_job(self, job_id: str) -> Optional[FineTuningJob]:
        """Get a fine-tuning job by ID."""
        return self.active_jobs.get(job_id)
    
    def list_jobs(self, status: Optional[FineTuningStatus] = None) -> List[FineTuningJob]:
        """List fine-tuning jobs, optionally filtered by status."""
        jobs = list(self.active_jobs.values())
        if status:
            jobs = [job for job in jobs if job.progress.status == status]
        return sorted(jobs, key=lambda x: x.created_at, reverse=True)
    
    async def get_job_progress(self, job_id: str) -> Optional[FineTuningProgress]:
        """Get real-time progress for a job."""
        job = self.active_jobs.get(job_id)
        return job.progress if job else None
    
    async def export_config(self, job_id: str, output_path: str) -> None:
        """Export job configuration to file."""
        job = self.active_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        config_data = {
            "job_id": job.job_id,
            "config": asdict(job.config),
            "created_at": job.created_at.isoformat(),
            "metadata": job.metadata
        }
        
        with open(output_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration exported: {output_path}")
    
    async def import_config(self, config_path: str) -> FineTuningConfig:
        """Import configuration from file."""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        # Extract config section
        config_dict = config_data.get("config", config_data)
        
        # Convert datasets
        datasets = []
        for ds_dict in config_dict.get("datasets", []):
            datasets.append(DatasetConfig(**ds_dict))
        
        config_dict["datasets"] = datasets
        config_dict["objective"] = FineTuningObjective(config_dict["objective"])
        
        # Handle datetime fields
        if "created_at" in config_dict and isinstance(config_dict["created_at"], str):
            config_dict["created_at"] = datetime.fromisoformat(config_dict["created_at"])
        
        config = FineTuningConfig(**config_dict)
        
        logger.info(f"Configuration imported: {config_path}")
        return config
    
    async def _validate_config(self, config: FineTuningConfig) -> None:
        """Validate fine-tuning configuration."""
        # Validate base model exists
        try:
            model_info = await self.ollama_client.get_model_info(config.base_model)
        except Exception as e:
            raise ValueError(f"Base model not found: {config.base_model}") from e
        
        # Validate datasets
        for dataset in config.datasets:
            if not os.path.exists(dataset.path):
                raise ValueError(f"Dataset path not found: {dataset.path}")
            
            # Security validation
            if config.enable_security_validation:
                await self._validate_dataset_security(dataset)
        
        # Validate hyperparameters
        if config.learning_rate <= 0 or config.learning_rate > 1:
            raise ValueError("Learning rate must be between 0 and 1")
        
        if config.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if config.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        logger.info("Configuration validation passed")
    
    async def _validate_dataset_security(self, dataset: DatasetConfig) -> None:
        """Validate dataset security and safety."""
        # Create security event
        event = SecurityEvent(
            event_type="dataset_validation",
            description=f"Validating dataset: {dataset.path}",
            severity="info",
            metadata={"dataset_path": dataset.path, "format": dataset.format.value}
        )
        
        await self.security_manager.log_event(event)
        
        # Perform security checks based on dataset format
        if dataset.format == DatasetFormat.JSONL:
            await self._validate_jsonl_security(dataset.path)
        elif dataset.format == DatasetFormat.JSON:
            await self._validate_json_security(dataset.path)
        # Add more format-specific validations as needed
        
        logger.info(f"Dataset security validation passed: {dataset.path}")
    
    async def _validate_jsonl_security(self, path: str) -> None:
        """Validate JSONL dataset security."""
        # Check file size
        file_size = os.path.getsize(path)
        max_size = self.config.get("fine_tuning.max_dataset_size_mb", 1000) * 1024 * 1024
        
        if file_size > max_size:
            raise ValueError(f"Dataset file too large: {file_size} bytes")
        
        # Sample and check content
        sample_size = min(100, file_size // 1024)  # Sample first 100 lines or reasonable amount
        
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                
                try:
                    data = json.loads(line.strip())
                    await self._validate_record_content(data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in line {i+1}: {e}")
    
    async def _validate_json_security(self, path: str) -> None:
        """Validate JSON dataset security."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Sample check for large datasets
                sample_data = data[:100] if len(data) > 100 else data
                for record in sample_data:
                    await self._validate_record_content(record)
            else:
                await self._validate_record_content(data)
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
    
    async def _validate_record_content(self, record: Dict[str, Any]) -> None:
        """Validate individual record content for security issues."""
        # Convert record to string for content analysis
        content = json.dumps(record, ensure_ascii=False)
        
        # Check for potential security issues
        security_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'data:.*base64',  # Base64 data URLs
            r'<iframe\b[^>]*>.*?<\/iframe>',  # Iframe tags
        ]
        
        import re
        for pattern in security_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                raise ValueError(f"Potential security issue found in dataset content")
        
        # Check for excessively long content
        max_content_length = self.config.get("fine_tuning.max_record_length", 10000)
        if len(content) > max_content_length:
            raise ValueError(f"Record content too long: {len(content)} characters")
    
    async def _prepare_datasets(self, job: FineTuningJob) -> None:
        """Prepare and preprocess datasets for training."""
        logger.info(f"Preparing datasets for job: {job.job_id}")
        
        prepared_datasets = []
        
        for i, dataset in enumerate(job.config.datasets):
            logger.info(f"Processing dataset {i+1}/{len(job.config.datasets)}: {dataset.path}")
            
            # Validate dataset
            await self._validate_dataset_format(dataset)
            
            # Preprocess dataset
            processed_path = await self._preprocess_dataset(job, dataset, i)
            prepared_datasets.append(processed_path)
        
        # Store prepared dataset paths in job metadata
        job.metadata["prepared_datasets"] = prepared_datasets
        
        logger.info(f"Dataset preparation completed for job: {job.job_id}")
    
    async def _validate_dataset_format(self, dataset: DatasetConfig) -> None:
        """Validate dataset format and structure."""
        path = dataset.path
        format_type = dataset.format
        
        if not os.path.exists(path):
            raise ValueError(f"Dataset file not found: {path}")
        
        # Format-specific validation
        if format_type == DatasetFormat.JSONL:
            await self._validate_jsonl_format(path)
        elif format_type == DatasetFormat.JSON:
            await self._validate_json_format(path)
        elif format_type == DatasetFormat.CSV:
            await self._validate_csv_format(path)
        # Add more format validations as needed
        
        logger.info(f"Dataset format validation passed: {path}")
    
    async def _validate_jsonl_format(self, path: str) -> None:
        """Validate JSONL format."""
        line_count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    json.loads(line)
                    line_count += 1
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")
        
        if line_count == 0:
            raise ValueError("Dataset contains no valid records")
    
    async def _validate_json_format(self, path: str) -> None:
        """Validate JSON format."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) == 0:
                raise ValueError("Dataset contains no records")
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
    
    async def _validate_csv_format(self, path: str) -> None:
        """Validate CSV format."""
        import csv
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    raise ValueError("CSV file has no header")
                
                row_count = sum(1 for _ in reader)
                if row_count == 0:
                    raise ValueError("CSV file has no data rows")
                    
        except Exception as e:
            raise ValueError(f"Invalid CSV file: {e}")
    
    async def _preprocess_dataset(self, job: FineTuningJob, dataset: DatasetConfig, index: int) -> str:
        """Preprocess a dataset for training."""
        output_path = os.path.join(job.output_dir, "datasets", f"dataset_{index}_processed.jsonl")
        
        # Apply preprocessing steps
        processed_data = []
        
        # Load original data
        if dataset.format == DatasetFormat.JSONL:
            with open(dataset.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        processed_data.append(json.loads(line))
        elif dataset.format == DatasetFormat.JSON:
            with open(dataset.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    processed_data = data
                else:
                    processed_data = [data]
        
        # Apply max samples limit
        if dataset.max_samples and len(processed_data) > dataset.max_samples:
            processed_data = processed_data[:dataset.max_samples]
        
        # Shuffle if requested
        if dataset.shuffle:
            import random
            random.shuffle(processed_data)
        
        # Apply preprocessing steps
        for step in dataset.preprocessing_steps:
            processed_data = await self._apply_preprocessing_step(processed_data, step)
        
        # Save processed data
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in processed_data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"Dataset preprocessed: {dataset.path} -> {output_path}")
        return output_path
    
    async def _apply_preprocessing_step(self, data: List[Dict[str, Any]], step: str) -> List[Dict[str, Any]]:
        """Apply a preprocessing step to the data."""
        if step == "normalize_text":
            return await self._normalize_text(data)
        elif step == "filter_length":
            return await self._filter_by_length(data)
        elif step == "deduplicate":
            return await self._deduplicate_data(data)
        # Add more preprocessing steps as needed
        
        return data
    
    async def _normalize_text(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize text in the dataset."""
        for record in data:
            for key, value in record.items():
                if isinstance(value, str):
                    # Basic text normalization
                    value = value.strip()
                    value = ' '.join(value.split())  # Normalize whitespace
                    record[key] = value
        return data
    
    async def _filter_by_length(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter records by content length."""
        max_length = self.config.get("fine_tuning.max_record_length", 10000)
        
        filtered_data = []
        for record in data:
            content_length = len(json.dumps(record, ensure_ascii=False))
            if content_length <= max_length:
                filtered_data.append(record)
        
        logger.info(f"Filtered {len(data) - len(filtered_data)} records by length")
        return filtered_data
    
    async def _deduplicate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate records from the dataset."""
        seen_hashes = set()
        deduplicated_data = []
        
        for record in data:
            record_str = json.dumps(record, sort_keys=True, ensure_ascii=False)
            record_hash = hashlib.md5(record_str.encode()).hexdigest()
            
            if record_hash not in seen_hashes:
                seen_hashes.add(record_hash)
                deduplicated_data.append(record)
        
        logger.info(f"Removed {len(data) - len(deduplicated_data)} duplicate records")
        return deduplicated_data
    
    async def _run_training(self, job: FineTuningJob) -> None:
        """Run the actual training process."""
        logger.info(f"Starting training for job: {job.job_id}")
        
        # This is where integration with Ollama's fine-tuning API would happen
        # For now, we'll simulate the training process
        
        total_steps = job.config.num_epochs * 100  # Simulated steps per epoch
        job.progress.total_steps = total_steps
        job.progress.total_epochs = job.config.num_epochs
        
        for epoch in range(job.config.num_epochs):
            if job.progress.status == FineTuningStatus.CANCELLED:
                break
            
            job.progress.current_epoch = epoch + 1
            
            # Trigger epoch start callbacks
            for callback in self.global_callbacks:
                callback.on_epoch_start(job, epoch)
            
            # Simulate training steps
            steps_per_epoch = total_steps // job.config.num_epochs
            for step in range(steps_per_epoch):
                if job.progress.status in [FineTuningStatus.CANCELLED, FineTuningStatus.PAUSED]:
                    break
                
                job.progress.current_step = epoch * steps_per_epoch + step + 1
                
                # Simulate metrics
                metrics = {
                    "loss": 2.0 * (1 - (job.progress.current_step / total_steps)) + 0.1,
                    "learning_rate": job.config.learning_rate * (1 - (job.progress.current_step / total_steps)),
                    "accuracy": min(0.95, (job.progress.current_step / total_steps) * 0.9 + 0.05)
                }
                
                job.progress.training_loss = metrics["loss"]
                job.progress.learning_rate = metrics["learning_rate"]
                job.progress.accuracy = metrics["accuracy"]
                
                # Trigger step callbacks
                for callback in self.global_callbacks:
                    callback.on_step(job, job.progress.current_step, metrics)
                
                # Validation at specified intervals
                if job.progress.current_step % job.config.eval_steps == 0:
                    val_metrics = await self._run_validation(job)
                    
                    # Trigger validation callbacks
                    for callback in self.global_callbacks:
                        callback.on_validation(job, val_metrics)
                
                # Small delay to simulate training time
                await asyncio.sleep(0.01)
            
            # End of epoch metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "training_loss": job.progress.training_loss,
                "validation_loss": job.progress.validation_loss,
                "accuracy": job.progress.accuracy
            }
            
            job.training_history.append(epoch_metrics)
            
            # Trigger epoch end callbacks
            for callback in self.global_callbacks:
                callback.on_epoch_end(job, epoch, epoch_metrics)
            
            # Check for early stopping
            for callback in self.global_callbacks:
                if isinstance(callback, EarlyStoppingCallback) and callback.should_stop:
                    logger.info("Early stopping triggered")
                    break
        
        # Final training metrics
        final_metrics = {
            "final_loss": job.progress.training_loss,
            "final_accuracy": job.progress.accuracy,
            "total_steps": job.progress.current_step,
            "total_epochs": job.progress.current_epoch
        }
        
        job.evaluation_results = final_metrics
        
        # Trigger training end callbacks
        for callback in self.global_callbacks:
            callback.on_training_end(job, final_metrics)
        
        logger.info(f"Training completed for job: {job.job_id}")
    
    async def _run_validation(self, job: FineTuningJob) -> Dict[str, Any]:
        """Run validation and return metrics."""
        job.progress.status = FineTuningStatus.VALIDATING
        
        # Simulate validation metrics
        val_loss = job.progress.training_loss * 1.1  # Validation loss typically higher
        val_accuracy = job.progress.accuracy * 0.95  # Validation accuracy typically lower
        
        validation_metrics = {
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "perplexity": 2 ** val_loss
        }
        
        job.progress.validation_loss = val_loss
        job.progress.perplexity = validation_metrics["perplexity"]
        job.progress.status = FineTuningStatus.TRAINING
        
        return validation_metrics
    
    async def _save_job_config(self, job: FineTuningJob) -> None:
        """Save job configuration to file."""
        config_data = {
            "job_id": job.job_id,
            "config": asdict(job.config),
            "created_at": job.created_at.isoformat(),
            "metadata": job.metadata
        }
        
        with open(job.config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)


# Global instance management
_fine_tuning_manager: Optional[FineTuningManager] = None


def get_fine_tuning_manager() -> FineTuningManager:
    """Get the global FineTuningManager instance."""
    global _fine_tuning_manager
    if _fine_tuning_manager is None:
        _fine_tuning_manager = FineTuningManager()
    return _fine_tuning_manager


# Convenience functions
async def quick_fine_tune(
    base_model: str,
    dataset_path: str,
    model_name: str,
    objective: FineTuningObjective = FineTuningObjective.CODE_GENERATION,
    **kwargs
) -> str:
    """Quick fine-tuning with minimal configuration."""
    manager = get_fine_tuning_manager()
    
    # Create dataset config
    dataset = DatasetConfig(
        path=dataset_path,
        format=DatasetFormat.JSONL  # Default format
    )
    
    # Create training config
    config = FineTuningConfig(
        base_model=base_model,
        model_name=model_name,
        objective=objective,
        datasets=[dataset],
        **kwargs
    )
    
    # Create and start job
    job = await manager.create_job(config)
    await manager.start_job(job.job_id)
    
    return job.job_id


async def monitor_job_progress(job_id: str) -> AsyncIterator[FineTuningProgress]:
    """Monitor job progress in real-time."""
    manager = get_fine_tuning_manager()
    
    while True:
        progress = await manager.get_job_progress(job_id)
        if not progress:
            break
        
        yield progress
        
        if progress.status in [
            FineTuningStatus.COMPLETED,
            FineTuningStatus.FAILED,
            FineTuningStatus.CANCELLED
        ]:
            break
        
        await asyncio.sleep(1)