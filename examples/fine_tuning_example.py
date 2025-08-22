"""
Example: Fine-Tuning Interface Usage for ABOV3 4 Ollama

This example demonstrates how to use the comprehensive fine-tuning interface
to fine-tune models, monitor training progress, and evaluate results.
"""

import asyncio
import json
import os
from pathlib import Path

# Import ABOV3 fine-tuning components
from abov3.models.fine_tuning import (
    FineTuningManager,
    FineTuningConfig,
    FineTuningObjective,
    DatasetConfig,
    DatasetFormat,
    ProgressCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    get_fine_tuning_manager,
    quick_fine_tune,
    monitor_job_progress
)

from abov3.models.training import (
    TrainingManager,
    HyperparameterConfig,
    ResourceMonitor,
    get_training_manager,
    quick_hyperparameter_search,
    monitor_resources
)

from abov3.models.evaluation import (
    EvaluationEngine,
    EvaluationMetricType,
    get_evaluation_engine,
    quick_evaluate,
    quick_compare,
    create_code_generation_benchmark
)


async def create_sample_dataset():
    """Create a sample dataset for fine-tuning."""
    # Create sample training data for code generation
    sample_data = [
        {
            "input": "Write a Python function to calculate the factorial of a number",
            "output": """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""
        },
        {
            "input": "Create a Python class for a simple calculator",
            "output": """class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b"""
        },
        {
            "input": "Implement a binary search algorithm in Python",
            "output": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1"""
        },
        {
            "input": "Write a Python function to reverse a string",
            "output": """def reverse_string(s):
    return s[::-1]
    
# Alternative implementation
def reverse_string_iterative(s):
    result = ""
    for char in reversed(s):
        result += char
    return result"""
        },
        {
            "input": "Create a Python function to check if a number is prime",
            "output": """def is_prime(n):
    if n < 2:
        return False
    
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    
    return True"""
        }
    ]
    
    # Create dataset directory
    dataset_dir = Path("./sample_datasets")
    dataset_dir.mkdir(exist_ok=True)
    
    # Save as JSONL file
    dataset_path = dataset_dir / "code_generation_sample.jsonl"
    with open(dataset_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Sample dataset created: {dataset_path}")
    return str(dataset_path)


async def basic_fine_tuning_example():
    """Basic fine-tuning example using the quick interface."""
    print("\n=== Basic Fine-Tuning Example ===")
    
    # Create sample dataset
    dataset_path = await create_sample_dataset()
    
    # Quick fine-tuning
    try:
        job_id = await quick_fine_tune(
            base_model="llama3.2:1b",  # Use a smaller model for example
            dataset_path=dataset_path,
            model_name="llama3.2-code-tuned",
            objective=FineTuningObjective.CODE_GENERATION,
            learning_rate=1e-5,
            batch_size=2,
            num_epochs=2
        )
        
        print(f"Fine-tuning job started: {job_id}")
        
        # Monitor progress
        async for progress in monitor_job_progress(job_id):
            print(f"Progress: Epoch {progress.current_epoch}/{progress.total_epochs}, "
                  f"Step {progress.current_step}/{progress.total_steps}, "
                  f"Loss: {progress.training_loss:.4f}")
            
            if progress.status.value in ["completed", "failed", "cancelled"]:
                break
        
        print(f"Fine-tuning {progress.status.value}")
        
    except Exception as e:
        print(f"Fine-tuning failed: {e}")


async def advanced_fine_tuning_example():
    """Advanced fine-tuning example with custom configuration."""
    print("\n=== Advanced Fine-Tuning Example ===")
    
    # Get fine-tuning manager
    ft_manager = get_fine_tuning_manager()
    
    # Add callbacks for monitoring
    ft_manager.add_global_callback(ProgressCallback(log_interval=5))
    ft_manager.add_global_callback(CheckpointCallback(save_interval=500))
    ft_manager.add_global_callback(EarlyStoppingCallback(patience=2))
    
    # Create dataset configuration
    dataset_path = await create_sample_dataset()
    dataset_config = DatasetConfig(
        path=dataset_path,
        format=DatasetFormat.JSONL,
        max_samples=5,  # Limit for example
        shuffle=True,
        validation_checks=["format_validation", "content_safety"],
        preprocessing_steps=["normalize_text", "filter_length"]
    )
    
    # Create fine-tuning configuration
    config = FineTuningConfig(
        base_model="llama3.2:1b",
        model_name="llama3.2-advanced-code",
        objective=FineTuningObjective.CODE_GENERATION,
        datasets=[dataset_config],
        
        # Training parameters
        learning_rate=2e-5,
        batch_size=1,
        num_epochs=2,
        max_seq_length=1024,
        
        # LoRA settings
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        
        # Validation settings
        eval_steps=250,
        save_steps=500,
        early_stopping_patience=2,
        
        # Security and compliance
        enable_security_validation=True,
        enable_bias_detection=True,
        
        # Metadata
        description="Advanced code generation fine-tuning example",
        tags=["code", "python", "example"]
    )
    
    try:
        # Create job
        job = await ft_manager.create_job(config)
        print(f"Created fine-tuning job: {job.job_id}")
        
        # Start training
        await ft_manager.start_job(job.job_id)
        
        # Export configuration
        config_export_path = f"./fine_tuning_config_{job.job_id}.yaml"
        await ft_manager.export_config(job.job_id, config_export_path)
        print(f"Configuration exported: {config_export_path}")
        
    except Exception as e:
        print(f"Advanced fine-tuning failed: {e}")


async def hyperparameter_optimization_example():
    """Example of hyperparameter optimization."""
    print("\n=== Hyperparameter Optimization Example ===")
    
    # Get training manager
    training_manager = get_training_manager()
    
    # Define hyperparameter search space
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
            "max": 8
        },
        "lora_rank": {
            "type": "categorical",
            "choices": [4, 8, 16, 32]
        }
    }
    
    # Create a template job for optimization
    dataset_path = await create_sample_dataset()
    dataset_config = DatasetConfig(path=dataset_path, format=DatasetFormat.JSONL)
    
    template_config = FineTuningConfig(
        base_model="llama3.2:1b",
        model_name="llama3.2-optimized",
        objective=FineTuningObjective.CODE_GENERATION,
        datasets=[dataset_config],
        num_epochs=1  # Short for example
    )
    
    # Note: This would create a template job in a real implementation
    print("Hyperparameter optimization would run multiple trials here...")
    print(f"Search space: {search_space}")


async def model_evaluation_example():
    """Example of model evaluation and comparison."""
    print("\n=== Model Evaluation Example ===")
    
    # Get evaluation engine
    eval_engine = get_evaluation_engine()
    
    # Create evaluation dataset
    dataset_path = await create_sample_dataset()
    
    # Add a code generation benchmark
    benchmark = create_code_generation_benchmark()
    eval_engine.add_benchmark_task(benchmark)
    
    try:
        # Quick evaluation
        result = await quick_evaluate(
            model_name="llama3.2:1b",
            dataset_path=dataset_path,
            metrics=["accuracy", "bleu", "latency", "code_quality"]
        )
        
        print(f"Evaluation completed: {result.evaluation_id}")
        print(f"Overall quality score: {result.overall_quality_score:.3f}")
        print(f"Code quality score: {result.code_quality_score:.3f}")
        print(f"Average latency: {result.latency_ms:.2f}ms")
        
        # Print detailed metrics
        for metric in result.metrics:
            print(f"{metric.name}: {metric.value:.3f}")
        
        # Generate report
        report_path = f"./evaluation_report_{result.evaluation_id}.json"
        await eval_engine.generate_report(result.evaluation_id, report_path)
        print(f"Evaluation report saved: {report_path}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")


async def resource_monitoring_example():
    """Example of resource monitoring during training."""
    print("\n=== Resource Monitoring Example ===")
    
    # Create resource monitor
    monitor = monitor_resources(interval=2.0)
    
    # Add custom callback
    def resource_callback(metrics):
        print(f"Resource Update - CPU: {metrics.cpu_percent:.1f}%, "
              f"Memory: {metrics.memory_percent:.1f}%, "
              f"GPU: {metrics.gpu_percent:.1f}%")
    
    monitor.add_callback(resource_callback)
    
    # Simulate some work
    print("Monitoring resources for 10 seconds...")
    await asyncio.sleep(10)
    
    # Get latest metrics
    latest = monitor.get_latest_metrics()
    if latest:
        print(f"Latest metrics: CPU {latest.cpu_percent:.1f}%, "
              f"Memory {latest.memory_used_gb:.1f}GB")
    
    # Stop monitoring
    monitor.stop_monitoring()


async def configuration_management_example():
    """Example of configuration management."""
    print("\n=== Configuration Management Example ===")
    
    # Create configuration
    dataset_path = await create_sample_dataset()
    config = FineTuningConfig(
        base_model="llama3.2:1b",
        model_name="llama3.2-config-example",
        objective=FineTuningObjective.CODE_GENERATION,
        datasets=[DatasetConfig(path=dataset_path, format=DatasetFormat.JSONL)],
        learning_rate=1e-5,
        description="Configuration management example"
    )
    
    ft_manager = get_fine_tuning_manager()
    
    # Export configuration
    export_path = "./example_config.yaml"
    job = await ft_manager.create_job(config)
    await ft_manager.export_config(job.job_id, export_path)
    print(f"Configuration exported to: {export_path}")
    
    # Import configuration
    imported_config = await ft_manager.import_config(export_path)
    print(f"Configuration imported: {imported_config.model_name}")
    
    # Clean up
    if os.path.exists(export_path):
        os.remove(export_path)


async def main():
    """Run all examples."""
    print("ABOV3 4 Ollama Fine-Tuning Interface Examples")
    print("=" * 50)
    
    try:
        # Run examples
        await basic_fine_tuning_example()
        await advanced_fine_tuning_example()
        await hyperparameter_optimization_example()
        await model_evaluation_example()
        await resource_monitoring_example()
        await configuration_management_example()
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Example failed with error: {e}")
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())