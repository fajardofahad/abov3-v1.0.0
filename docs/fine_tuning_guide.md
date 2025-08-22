# ABOV3 4 Ollama Fine-Tuning Interface Guide

This guide provides comprehensive documentation for the ABOV3 4 Ollama fine-tuning interface, covering all aspects from basic usage to advanced configurations.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Fine-Tuning Manager](#fine-tuning-manager)
4. [Training Manager](#training-manager)
5. [Evaluation Engine](#evaluation-engine)
6. [Configuration Reference](#configuration-reference)
7. [Examples](#examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

The ABOV3 fine-tuning interface provides enterprise-grade capabilities for:

- **Fine-Tuning Workflow Management**: Complete lifecycle management from dataset preparation to model deployment
- **Training Data Management**: Advanced preprocessing, validation, and augmentation
- **Real-time Monitoring**: Resource usage, training metrics, and progress tracking
- **Model Evaluation**: Comprehensive benchmarking, A/B testing, and quality assessment
- **Hyperparameter Optimization**: Automated search strategies for optimal configurations
- **Security & Compliance**: Built-in security validation and bias detection

### Key Features

- Async support for non-blocking operations
- Integration with Ollama's fine-tuning infrastructure
- Advanced dataset validation and security scanning
- Real-time progress monitoring with callbacks
- Automated hyperparameter optimization
- Model versioning with semantic versioning
- Export/import of fine-tuning configurations
- Comprehensive evaluation and benchmarking

## Quick Start

### Basic Fine-Tuning

```python
import asyncio
from abov3.models.fine_tuning import quick_fine_tune, FineTuningObjective

async def basic_example():
    # Quick fine-tuning with minimal configuration
    job_id = await quick_fine_tune(
        base_model="llama3.2:1b",
        dataset_path="./my_dataset.jsonl",
        model_name="my-fine-tuned-model",
        objective=FineTuningObjective.CODE_GENERATION
    )
    print(f"Fine-tuning started: {job_id}")

asyncio.run(basic_example())
```

### Monitor Progress

```python
from abov3.models.fine_tuning import monitor_job_progress

async def monitor_example():
    async for progress in monitor_job_progress("your_job_id"):
        print(f"Epoch {progress.current_epoch}, Loss: {progress.training_loss:.4f}")
        if progress.status.value in ["completed", "failed"]:
            break

asyncio.run(monitor_example())
```

## Fine-Tuning Manager

The `FineTuningManager` is the central component for managing fine-tuning workflows.

### Basic Usage

```python
from abov3.models.fine_tuning import (
    FineTuningManager, 
    FineTuningConfig, 
    DatasetConfig,
    FineTuningObjective,
    DatasetFormat
)

# Get manager instance
ft_manager = get_fine_tuning_manager()

# Create dataset configuration
dataset_config = DatasetConfig(
    path="./training_data.jsonl",
    format=DatasetFormat.JSONL,
    shuffle=True,
    max_samples=1000,
    validation_checks=["format_validation", "security_scan"]
)

# Create fine-tuning configuration
config = FineTuningConfig(
    base_model="llama3.2:3b",
    model_name="my-specialized-model",
    objective=FineTuningObjective.CODE_GENERATION,
    datasets=[dataset_config],
    learning_rate=2e-5,
    batch_size=4,
    num_epochs=3
)

# Create and start job
job = await ft_manager.create_job(config)
await ft_manager.start_job(job.job_id)
```

### Advanced Configuration

```python
# Advanced configuration with LoRA and early stopping
config = FineTuningConfig(
    base_model="llama3.2:7b",
    model_name="advanced-model",
    objective=FineTuningObjective.INSTRUCTION_FOLLOWING,
    datasets=[dataset_config],
    
    # Training hyperparameters
    learning_rate=1e-5,
    batch_size=8,
    num_epochs=5,
    max_seq_length=2048,
    gradient_accumulation_steps=2,
    
    # LoRA configuration
    use_lora=True,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    
    # Optimization settings
    optimizer="adamw",
    scheduler="cosine",
    weight_decay=0.01,
    gradient_clipping=1.0,
    
    # Early stopping
    early_stopping_patience=3,
    early_stopping_threshold=0.001,
    
    # Validation and checkpointing
    eval_steps=500,
    save_steps=1000,
    logging_steps=50,
    
    # Security and compliance
    enable_security_validation=True,
    enable_bias_detection=True,
    privacy_level="strict"
)
```

### Callbacks

```python
from abov3.models.fine_tuning import (
    ProgressCallback,
    CheckpointCallback,
    EarlyStoppingCallback
)

# Add monitoring callbacks
ft_manager.add_global_callback(ProgressCallback(log_interval=10))
ft_manager.add_global_callback(CheckpointCallback(save_interval=500))
ft_manager.add_global_callback(EarlyStoppingCallback(patience=3))
```

### Job Management

```python
# List all jobs
jobs = ft_manager.list_jobs()
for job in jobs:
    print(f"Job {job.job_id}: {job.progress.status.value}")

# Get specific job
job = ft_manager.get_job("job_id")
if job:
    print(f"Progress: {job.progress.current_step}/{job.progress.total_steps}")

# Control job execution
await ft_manager.pause_job("job_id")
await ft_manager.resume_job("job_id")
await ft_manager.cancel_job("job_id")
```

## Training Manager

The `TrainingManager` handles training data processing and resource monitoring.

### Resource Monitoring

```python
from abov3.models.training import get_training_manager, monitor_resources

# Start resource monitoring
monitor = monitor_resources(interval=1.0)

# Add custom callback
def resource_alert(metrics):
    if metrics.memory_percent > 90:
        print(f"WARNING: High memory usage: {metrics.memory_percent:.1f}%")

monitor.add_callback(resource_alert)

# Get metrics
latest_metrics = monitor.get_latest_metrics()
avg_metrics = monitor.get_average_metrics(duration_minutes=5)
```

### Hyperparameter Optimization

```python
from abov3.models.training import (
    HyperparameterConfig,
    quick_hyperparameter_search
)

# Define search space
search_space = {
    "learning_rate": {
        "type": "float",
        "min": 1e-6,
        "max": 1e-3,
        "log_scale": True
    },
    "batch_size": {
        "type": "int",
        "min": 1,
        "max": 16
    },
    "lora_rank": {
        "type": "categorical",
        "choices": [8, 16, 32, 64]
    }
}

# Run optimization
best_params = await quick_hyperparameter_search(
    job=your_job,
    search_space=search_space,
    n_trials=20
)
```

### Data Pipeline Configuration

```python
from abov3.models.training import DataPipelineConfig

pipeline_config = DataPipelineConfig(
    batch_size=8,
    num_workers=4,
    shuffle=True,
    
    # Data augmentation
    enable_augmentation=True,
    augmentation_strategies=["paraphrase", "noise_injection"],
    augmentation_probability=0.3,
    
    # Quality filtering
    min_length=10,
    max_length=4096,
    quality_threshold=0.7,
    
    # Caching
    enable_caching=True,
    cache_size_gb=2.0
)
```

## Evaluation Engine

The `EvaluationEngine` provides comprehensive model evaluation and comparison capabilities.

### Basic Evaluation

```python
from abov3.models.evaluation import (
    get_evaluation_engine,
    EvaluationMetricType,
    quick_evaluate
)

# Quick evaluation
result = await quick_evaluate(
    model_name="my-model",
    dataset_path="./test_data.jsonl",
    metrics=["accuracy", "bleu", "latency", "code_quality"]
)

print(f"Overall quality: {result.overall_quality_score:.3f}")
print(f"Code quality: {result.code_quality_score:.3f}")
```

### Model Comparison

```python
from abov3.models.evaluation import quick_compare, ComparisonMethod

# Compare two models
comparison = await quick_compare(
    model_a="model-v1",
    model_b="model-v2", 
    dataset_path="./benchmark.jsonl",
    metrics=["accuracy", "bleu", "latency"]
)

if comparison.significant:
    print(f"Winner: {comparison.winner}")
    print(f"Improvement: {comparison.improvement_percentage:.2f}%")
```

### Custom Benchmarks

```python
from abov3.models.evaluation import BenchmarkTask, EvaluationMetricType

# Create custom benchmark
benchmark = BenchmarkTask(
    task_id="my_benchmark",
    name="Custom Code Benchmark",
    description="Test specific coding scenarios",
    task_type="code_generation",
    inputs=[
        {"prompt": "Write a function to sort a list"},
        {"prompt": "Create a binary tree class"}
    ],
    expected_outputs=[
        {"response": "def sort_list(lst):\n    return sorted(lst)"},
        {"response": "class BinaryTree:\n    def __init__(self, value):\n        self.value = value"}
    ],
    metrics=[
        EvaluationMetricType.ACCURACY,
        EvaluationMetricType.CODE_QUALITY
    ]
)

# Add to evaluation engine
eval_engine = get_evaluation_engine()
eval_engine.add_benchmark_task(benchmark)

# Run benchmark
result = await eval_engine.run_benchmark("my-model", "my_benchmark")
```

### A/B Testing

```python
from abov3.models.evaluation import ABTestConfig

# Configure A/B test
ab_config = ABTestConfig(
    test_id="model_comparison_test",
    model_a="baseline-model",
    model_b="new-model",
    sample_size=200,
    confidence_level=0.95,
    test_dataset="./ab_test_data.jsonl",
    evaluation_metrics=[
        EvaluationMetricType.ACCURACY,
        EvaluationMetricType.LATENCY
    ]
)

# Start A/B test
test_id = await eval_engine.start_ab_test(ab_config)
result = eval_engine.get_ab_test_result(test_id)
```

## Configuration Reference

### Dataset Formats

Supported dataset formats:

- **JSONL**: JSON Lines format (recommended)
- **JSON**: Standard JSON format
- **CSV**: Comma-separated values
- **TXT**: Plain text format
- **Parquet**: Columnar storage format

### Dataset Structure

For code generation tasks, use this structure:

```json
{
  "input": "Write a Python function to calculate factorial",
  "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
}
```

For instruction following:

```json
{
  "instruction": "Sort the following list",
  "input": "[3, 1, 4, 1, 5, 9]",
  "output": "[1, 1, 3, 4, 5, 9]"
}
```

### Fine-Tuning Objectives

- `CODE_GENERATION`: Code generation tasks
- `CHAT_COMPLETION`: Conversational AI
- `INSTRUCTION_FOLLOWING`: Following specific instructions
- `TEXT_COMPLETION`: General text completion
- `CLASSIFICATION`: Text classification
- `SUMMARIZATION`: Text summarization
- `TRANSLATION`: Language translation
- `QUESTION_ANSWERING`: Q&A tasks

### Hyperparameters

Key hyperparameters and their typical ranges:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `learning_rate` | 1e-6 to 1e-3 | Learning rate for optimization |
| `batch_size` | 1 to 32 | Number of samples per batch |
| `num_epochs` | 1 to 10 | Number of training epochs |
| `lora_rank` | 4 to 64 | LoRA adaptation rank |
| `lora_alpha` | 8 to 64 | LoRA scaling parameter |
| `max_seq_length` | 512 to 4096 | Maximum sequence length |

## Examples

### Complete Fine-Tuning Workflow

```python
import asyncio
from abov3.models import *

async def complete_workflow():
    # 1. Prepare dataset
    dataset_config = DatasetConfig(
        path="./my_dataset.jsonl",
        format=DatasetFormat.JSONL,
        validation_checks=["format_validation", "security_scan"],
        preprocessing_steps=["normalize_text", "filter_length"]
    )
    
    # 2. Configure fine-tuning
    ft_config = FineTuningConfig(
        base_model="llama3.2:3b",
        model_name="my-specialized-model",
        objective=FineTuningObjective.CODE_GENERATION,
        datasets=[dataset_config],
        learning_rate=2e-5,
        batch_size=4,
        num_epochs=3,
        use_lora=True,
        lora_rank=16
    )
    
    # 3. Start fine-tuning
    ft_manager = get_fine_tuning_manager()
    job = await ft_manager.create_job(ft_config)
    await ft_manager.start_job(job.job_id)
    
    # 4. Monitor progress
    async for progress in monitor_job_progress(job.job_id):
        print(f"Progress: {progress.current_step}/{progress.total_steps}")
        if progress.status.value == "completed":
            break
    
    # 5. Evaluate results
    eval_result = await quick_evaluate(
        model_name=ft_config.model_name,
        dataset_path="./test_data.jsonl",
        metrics=["accuracy", "code_quality", "latency"]
    )
    
    print(f"Evaluation completed: {eval_result.overall_quality_score:.3f}")
    
    # 6. Compare with baseline
    comparison = await quick_compare(
        model_a=ft_config.base_model,
        model_b=ft_config.model_name,
        dataset_path="./test_data.jsonl"
    )
    
    if comparison.significant:
        print(f"Improvement: {comparison.improvement_percentage:.2f}%")

# Run the workflow
asyncio.run(complete_workflow())
```

## Best Practices

### Dataset Preparation

1. **Quality over Quantity**: Use high-quality, diverse datasets
2. **Security Validation**: Always enable security validation
3. **Data Preprocessing**: Apply appropriate preprocessing steps
4. **Balanced Datasets**: Ensure balanced representation
5. **Validation Split**: Reserve data for validation and testing

### Training Configuration

1. **Start Small**: Begin with smaller models and datasets
2. **Use LoRA**: Enable LoRA for efficient fine-tuning
3. **Monitor Resources**: Track GPU/CPU usage during training
4. **Early Stopping**: Prevent overfitting with early stopping
5. **Regular Checkpoints**: Save checkpoints frequently

### Evaluation Strategy

1. **Multiple Metrics**: Use diverse evaluation metrics
2. **Separate Test Data**: Use unseen data for final evaluation
3. **Statistical Significance**: Ensure statistically significant results
4. **Benchmark Comparison**: Compare against established benchmarks
5. **Human Evaluation**: Include human evaluation for quality assessment

### Performance Optimization

1. **Batch Size Tuning**: Optimize batch size for your hardware
2. **Gradient Accumulation**: Use for effective larger batch sizes
3. **Mixed Precision**: Enable for faster training (if supported)
4. **Data Pipeline**: Optimize data loading and preprocessing
5. **Resource Monitoring**: Monitor and optimize resource usage

## Troubleshooting

### Common Issues

#### Out of Memory Errors

```python
# Reduce batch size
config.batch_size = 1
config.gradient_accumulation_steps = 8  # Effective batch size = 8

# Enable gradient checkpointing
config.use_gradient_checkpointing = True

# Reduce sequence length
config.max_seq_length = 1024
```

#### Training Instability

```python
# Lower learning rate
config.learning_rate = 1e-6

# Add gradient clipping
config.gradient_clipping = 0.5

# Increase warmup steps
config.warmup_steps = 500
```

#### Poor Performance

```python
# Increase training data
dataset_config.max_samples = None  # Use all data

# Adjust hyperparameters
config.learning_rate = 5e-5
config.num_epochs = 5

# Use larger LoRA rank
config.lora_rank = 32
```

#### Dataset Issues

```python
# Enable comprehensive validation
dataset_config.validation_checks = [
    "format_validation",
    "content_safety", 
    "data_quality",
    "security_scan"
]

# Add preprocessing
dataset_config.preprocessing_steps = [
    "normalize_text",
    "filter_length",
    "deduplicate"
]
```

### Error Messages

| Error | Solution |
|-------|----------|
| "Dataset path not found" | Check file path and permissions |
| "Invalid JSON in dataset" | Validate JSON format |
| "Security validation failed" | Review dataset content for security issues |
| "Model not found" | Ensure base model is available in Ollama |
| "Insufficient GPU memory" | Reduce batch size or use LoRA |

### Performance Tips

1. **Use SSD Storage**: Store datasets and checkpoints on fast storage
2. **Optimize CPU Usage**: Set appropriate number of data workers
3. **Monitor Memory**: Track memory usage and adjust accordingly
4. **Use Caching**: Enable dataset caching for repeated training
5. **Parallel Processing**: Utilize multiple GPUs if available

### Logging and Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Monitor specific components
logger = logging.getLogger("abov3.models.fine_tuning")
logger.setLevel(logging.DEBUG)

# Add custom logging to callbacks
class CustomCallback(FineTuningCallback):
    def on_step(self, job, step, metrics):
        logger.info(f"Step {step}: Loss={metrics['loss']:.4f}")
```

This comprehensive guide covers all aspects of the ABOV3 fine-tuning interface. For additional help, refer to the example scripts and API documentation.