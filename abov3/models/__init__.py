"""
Comprehensive Model Management for ABOV3 4 Ollama.

This package provides advanced model management functionality including:
- Model discovery and installation via Ollama API
- Model switching and configuration management
- Performance monitoring and optimization
- Model recommendation system
- Automatic model updates and versioning
- Resource usage tracking
- Model health checks and validation
- Persistent model metadata storage
- Model search and filtering capabilities
- Model ratings and reviews system
- Performance benchmarking data
- Model usage analytics
- Import/export of model configurations
- Integration with online model repositories
- Advanced model fine-tuning and training
- Comprehensive model evaluation and testing
- A/B testing and model comparison

Core Components:
    ModelManager: Advanced model operations with async support, performance monitoring, 
                 and health checks
    ModelRegistry: Persistent storage for model metadata, ratings, and analytics
    ModelInfo: Basic model information from Ollama API
    ModelMetadata: Extended model metadata with capabilities and performance data
    FineTuningManager: Comprehensive fine-tuning workflow management
    TrainingManager: Advanced training data management and monitoring
    EvaluationEngine: Model evaluation, benchmarking, and comparison
    
Additional Classes:
    ModelPerformanceMetrics: Performance tracking and analytics
    ModelRecommendation: AI-powered model recommendations
    ModelHealthStatus: Model health monitoring
    ModelRating: User ratings and reviews
    ModelBenchmark: Performance benchmark data
    ModelUsageStats: Usage analytics and statistics
    SearchFilter: Advanced search and filtering criteria
    FineTuningJob: Fine-tuning job management and tracking
    TrainingMetrics: Training performance monitoring
    EvaluationResult: Comprehensive evaluation results
"""

# Core components
from .manager import (
    ModelManager,
    ModelPerformanceMetrics,
    ModelRecommendation, 
    ModelHealthStatus,
    ModelCache,
    get_model_manager,
    quick_install_model,
    quick_model_health_check
)

from .registry import (
    ModelRegistry,
    ModelRating,
    ModelBenchmark,
    ModelUsageStats,
    SearchFilter,
    get_model_registry,
    quick_register_model,
    quick_search_models
)

from .info import (
    ModelInfo,
    ModelMetadata,
    ModelSize,
    ModelType,
    ModelCapability
)

# Fine-tuning components
from .fine_tuning import (
    FineTuningManager,
    FineTuningJob,
    FineTuningConfig,
    FineTuningStatus,
    FineTuningProgress,
    DatasetConfig,
    FineTuningCallback,
    get_fine_tuning_manager,
    quick_fine_tune,
    monitor_job_progress
)

# Training components  
from .training import (
    TrainingManager,
    TrainingMetrics,
    ResourceMonitor,
    HyperparameterOptimizer,
    HyperparameterConfig,
    ResourceMetrics,
    get_training_manager,
    quick_hyperparameter_search,
    monitor_resources
)

# Evaluation components
from .evaluation import (
    EvaluationEngine,
    EvaluationResult,
    EvaluationMetric,
    BenchmarkTask,
    ComparisonResult,
    ABTestConfig,
    EvaluationMetricType,
    get_evaluation_engine,
    quick_evaluate,
    quick_compare,
    create_code_generation_benchmark
)

__all__ = [
    # Core classes
    "ModelManager",
    "ModelRegistry", 
    "ModelInfo",
    "ModelMetadata",
    
    # Manager components
    "ModelPerformanceMetrics",
    "ModelRecommendation",
    "ModelHealthStatus", 
    "ModelCache",
    
    # Registry components
    "ModelRating",
    "ModelBenchmark",
    "ModelUsageStats",
    "SearchFilter",
    
    # Fine-tuning classes
    "FineTuningManager",
    "FineTuningJob",
    "FineTuningConfig",
    "FineTuningStatus",
    "FineTuningProgress",
    "DatasetConfig",
    "FineTuningCallback",
    
    # Training classes
    "TrainingManager",
    "TrainingMetrics",
    "ResourceMonitor",
    "HyperparameterOptimizer",
    "HyperparameterConfig",
    "ResourceMetrics",
    
    # Evaluation classes
    "EvaluationEngine",
    "EvaluationResult",
    "EvaluationMetric",
    "BenchmarkTask",
    "ComparisonResult",
    "ABTestConfig",
    "EvaluationMetricType",
    
    # Enums and types
    "ModelSize",
    "ModelType", 
    "ModelCapability",
    
    # Convenience functions
    "get_model_manager",
    "get_model_registry",
    "quick_install_model",
    "quick_model_health_check",
    "quick_register_model",
    "quick_search_models",
    
    # Fine-tuning functions
    "get_fine_tuning_manager",
    "quick_fine_tune",
    "monitor_job_progress",
    
    # Training functions
    "get_training_manager",
    "quick_hyperparameter_search",
    "monitor_resources",
    
    # Evaluation functions
    "get_evaluation_engine",
    "quick_evaluate",
    "quick_compare",
    "create_code_generation_benchmark"
]

# Version info
__version__ = "1.0.0"
__author__ = "ABOV3 Enterprise AI/ML Expert Agent"