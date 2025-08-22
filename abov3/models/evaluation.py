"""
Comprehensive Model Evaluation System for ABOV3 4 Ollama.

This module provides advanced model evaluation capabilities including:
- Model evaluation metrics and comprehensive benchmarking
- Performance comparison between different models
- Automated testing suites and quality assurance
- Quality assessment tools and code analysis
- A/B testing framework for model comparison
- Regression testing for model updates
- Real-time evaluation monitoring and reporting
- Integration with existing model management infrastructure

Features:
- Async support for non-blocking evaluation operations
- Multi-dimensional evaluation metrics (accuracy, performance, safety)
- Automated benchmark generation and execution
- Statistical significance testing for comparisons
- Custom evaluation metric definitions
- Integration with fine-tuning and training systems
- Comprehensive reporting and visualization
- Continuous evaluation monitoring
"""

import asyncio
import json
import logging
import os
import statistics
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, AsyncIterator, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import yaml
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import re

from ..core.api.ollama_client import OllamaClient, get_ollama_client
from ..core.config import Config, get_config
from ..utils.security import SecurityManager
from .fine_tuning import FineTuningJob, FineTuningConfig
from .manager import ModelManager, get_model_manager
from .registry import ModelRegistry, get_model_registry


logger = logging.getLogger(__name__)


class EvaluationMetricType(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    BLEU = "bleu"
    ROUGE = "rouge"
    PERPLEXITY = "perplexity"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CODE_QUALITY = "code_quality"
    SECURITY_SCORE = "security_score"
    BIAS_SCORE = "bias_score"
    CUSTOM = "custom"


class EvaluationCategory(Enum):
    """Evaluation categories."""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    FAIRNESS = "fairness"
    USABILITY = "usability"


class TestType(Enum):
    """Types of tests."""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    STRESS_TEST = "stress_test"
    SECURITY_TEST = "security_test"
    BIAS_TEST = "bias_test"
    REGRESSION_TEST = "regression_test"
    A_B_TEST = "a_b_test"
    BENCHMARK_TEST = "benchmark_test"


class ComparisonMethod(Enum):
    """Statistical comparison methods."""
    T_TEST = "t_test"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"


@dataclass
class EvaluationMetric:
    """Single evaluation metric result."""
    name: str
    metric_type: EvaluationMetricType
    category: EvaluationCategory
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    standard_deviation: Optional[float] = None
    sample_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model."""
    model_name: str
    model_version: str
    evaluation_id: str
    timestamp: datetime
    
    # Metrics
    metrics: List[EvaluationMetric] = field(default_factory=list)
    
    # Test results
    test_results: Dict[str, Any] = field(default_factory=dict)
    
    # Performance data
    latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Quality scores
    overall_quality_score: float = 0.0
    code_quality_score: float = 0.0
    safety_score: float = 0.0
    
    # Metadata
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Additional data
    sample_outputs: List[Dict[str, Any]] = field(default_factory=list)
    detailed_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkTask:
    """Definition of a benchmark task."""
    task_id: str
    name: str
    description: str
    task_type: str  # 'code_generation', 'text_completion', 'classification', etc.
    
    # Test data
    inputs: List[Dict[str, Any]]
    expected_outputs: List[Dict[str, Any]]
    
    # Evaluation criteria
    metrics: List[EvaluationMetricType]
    scoring_function: Optional[Callable] = None
    
    # Configuration
    max_tokens: int = 1000
    temperature: float = 0.0
    timeout_seconds: int = 30
    
    # Metadata
    difficulty_level: str = "medium"  # 'easy', 'medium', 'hard', 'expert'
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing two models."""
    model_a: str
    model_b: str
    comparison_id: str
    timestamp: datetime
    
    # Statistical results
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    significant: bool
    
    # Method and configuration
    method: ComparisonMethod
    
    # Detailed results
    model_a_metrics: List[EvaluationMetric]
    model_b_metrics: List[EvaluationMetric]
    
    # Configuration with defaults
    alpha: float = 0.05
    
    # Summary
    winner: Optional[str] = None
    improvement_percentage: float = 0.0
    
    # Metadata
    comparison_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    # Required fields first
    test_id: str
    model_a: str
    model_b: str
    test_dataset: str
    evaluation_metrics: List[EvaluationMetricType]
    
    # Optional fields with defaults
    # Test parameters
    sample_size: int = 100
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.05
    power: float = 0.8
    
    # Traffic allocation  
    traffic_split: Tuple[float, float] = (0.5, 0.5)
    
    # Duration
    max_duration_hours: int = 24
    early_stopping: bool = True
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvaluationEngine:
    """Core evaluation engine."""
    
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        config: Optional[Config] = None
    ):
        self.ollama_client = ollama_client or get_ollama_client()
        self.config = config or get_config()
        
        # Evaluation state
        self.active_evaluations: Dict[str, EvaluationResult] = {}
        self.evaluation_history: List[EvaluationResult] = []
        
        # Benchmarks
        self.benchmark_tasks: Dict[str, BenchmarkTask] = {}
        self.benchmark_results: Dict[str, Dict[str, EvaluationResult]] = defaultdict(dict)
        
        # A/B tests
        self.active_ab_tests: Dict[str, ABTestConfig] = {}
        self.ab_test_results: Dict[str, ComparisonResult] = {}
        
        # Custom metrics
        self.custom_metrics: Dict[str, Callable] = {}
        
        logger.info("EvaluationEngine initialized")
    
    def register_custom_metric(
        self,
        name: str,
        metric_function: Callable[[str, str], float],
        category: EvaluationCategory = EvaluationCategory.PERFORMANCE
    ) -> None:
        """Register a custom evaluation metric."""
        self.custom_metrics[name] = {
            "function": metric_function,
            "category": category
        }
        logger.info(f"Custom metric registered: {name}")
    
    def add_benchmark_task(self, task: BenchmarkTask) -> None:
        """Add a benchmark task."""
        self.benchmark_tasks[task.task_id] = task
        logger.info(f"Benchmark task added: {task.task_id}")
    
    async def evaluate_model(
        self,
        model_name: str,
        dataset_path: str,
        metrics: List[EvaluationMetricType],
        sample_size: Optional[int] = None
    ) -> EvaluationResult:
        """Evaluate a model on a dataset."""
        evaluation_id = f"eval_{int(time.time())}"
        logger.info(f"Starting evaluation {evaluation_id} for model {model_name}")
        
        # Load dataset
        dataset = await self._load_dataset(dataset_path)
        
        # Sample if requested
        if sample_size and len(dataset) > sample_size:
            np.random.shuffle(dataset)
            dataset = dataset[:sample_size]
        
        # Initialize result
        result = EvaluationResult(
            model_name=model_name,
            model_version="1.0.0",  # Could be extracted from model info
            evaluation_id=evaluation_id,
            timestamp=datetime.now(),
            dataset_info={
                "path": dataset_path,
                "size": len(dataset),
                "sample_size": len(dataset)
            }
        )
        
        # Register active evaluation
        self.active_evaluations[evaluation_id] = result
        
        try:
            # Run evaluation
            await self._run_evaluation(result, dataset, metrics)
            
            # Calculate overall scores
            await self._calculate_overall_scores(result)
            
            # Store result
            self.evaluation_history.append(result)
            
            logger.info(f"Evaluation completed: {evaluation_id}")
            
        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Evaluation failed: {evaluation_id}, Error: {e}")
            raise
        finally:
            # Cleanup
            if evaluation_id in self.active_evaluations:
                del self.active_evaluations[evaluation_id]
        
        return result
    
    async def _load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load evaluation dataset."""
        dataset = []
        
        if dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        dataset.append(json.loads(line))
        elif dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    dataset = data
                else:
                    dataset = [data]
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
        
        return dataset
    
    async def _run_evaluation(
        self,
        result: EvaluationResult,
        dataset: List[Dict[str, Any]],
        metrics: List[EvaluationMetricType]
    ) -> None:
        """Run the actual evaluation."""
        logger.info(f"Running evaluation for {len(dataset)} samples")
        
        # Prepare for metrics collection
        predictions = []
        ground_truths = []
        latencies = []
        
        # Process each sample
        for i, sample in enumerate(dataset):
            try:
                # Extract input and expected output
                input_text = sample.get("input", sample.get("prompt", ""))
                expected_output = sample.get("output", sample.get("expected", ""))
                
                # Generate prediction
                start_time = time.time()
                prediction = await self._generate_prediction(result.model_name, input_text)
                end_time = time.time()
                
                # Record results
                predictions.append(prediction)
                ground_truths.append(expected_output)
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
                
                # Store sample results
                if len(result.sample_outputs) < 10:  # Store first 10 samples
                    result.sample_outputs.append({
                        "input": input_text,
                        "predicted": prediction,
                        "expected": expected_output,
                        "latency_ms": latencies[-1]
                    })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(dataset)} samples")
                
            except Exception as e:
                result.errors.append(f"Sample {i}: {str(e)}")
                logger.warning(f"Error processing sample {i}: {e}")
        
        # Calculate metrics
        result.latency_ms = np.mean(latencies) if latencies else 0.0
        result.throughput_tokens_per_sec = await self._calculate_throughput(predictions, latencies)
        
        # Calculate evaluation metrics
        for metric_type in metrics:
            metric_result = await self._calculate_metric(
                metric_type, predictions, ground_truths, latencies
            )
            if metric_result:
                result.metrics.append(metric_result)
    
    async def _generate_prediction(self, model_name: str, input_text: str) -> str:
        """Generate prediction from model."""
        try:
            # Use Ollama client to generate response
            response = await self.ollama_client.generate(
                model=model_name,
                prompt=input_text,
                options={"temperature": 0.0}  # Deterministic for evaluation
            )
            return response.get("response", "")
        except Exception as e:
            logger.warning(f"Prediction generation failed: {e}")
            return ""
    
    async def _calculate_throughput(self, predictions: List[str], latencies: List[float]) -> float:
        """Calculate throughput in tokens per second."""
        if not predictions or not latencies:
            return 0.0
        
        total_tokens = sum(len(pred.split()) for pred in predictions)
        total_time_seconds = sum(latencies) / 1000.0  # Convert ms to seconds
        
        return total_tokens / total_time_seconds if total_time_seconds > 0 else 0.0
    
    async def _calculate_metric(
        self,
        metric_type: EvaluationMetricType,
        predictions: List[str],
        ground_truths: List[str],
        latencies: List[float]
    ) -> Optional[EvaluationMetric]:
        """Calculate a specific metric."""
        try:
            if metric_type == EvaluationMetricType.ACCURACY:
                return await self._calculate_accuracy(predictions, ground_truths)
            elif metric_type == EvaluationMetricType.BLEU:
                return await self._calculate_bleu(predictions, ground_truths)
            elif metric_type == EvaluationMetricType.ROUGE:
                return await self._calculate_rouge(predictions, ground_truths)
            elif metric_type == EvaluationMetricType.PERPLEXITY:
                return await self._calculate_perplexity(predictions, ground_truths)
            elif metric_type == EvaluationMetricType.LATENCY:
                return await self._calculate_latency_metric(latencies)
            elif metric_type == EvaluationMetricType.CODE_QUALITY:
                return await self._calculate_code_quality(predictions)
            elif metric_type == EvaluationMetricType.SECURITY_SCORE:
                return await self._calculate_security_score(predictions)
            elif metric_type == EvaluationMetricType.CUSTOM:
                return await self._calculate_custom_metrics(predictions, ground_truths)
            
        except Exception as e:
            logger.error(f"Error calculating {metric_type}: {e}")
        
        return None
    
    async def _calculate_accuracy(self, predictions: List[str], ground_truths: List[str]) -> EvaluationMetric:
        """Calculate accuracy metric."""
        if not predictions or not ground_truths:
            return EvaluationMetric(
                name="accuracy",
                metric_type=EvaluationMetricType.ACCURACY,
                category=EvaluationCategory.PERFORMANCE,
                value=0.0
            )
        
        # Exact match accuracy
        exact_matches = sum(1 for p, g in zip(predictions, ground_truths) if p.strip() == g.strip())
        accuracy = exact_matches / len(predictions)
        
        # Token-level accuracy
        token_matches = []
        for p, g in zip(predictions, ground_truths):
            p_tokens = p.split()
            g_tokens = g.split()
            if len(p_tokens) == len(g_tokens):
                matches = sum(1 for pt, gt in zip(p_tokens, g_tokens) if pt == gt)
                token_matches.append(matches / max(len(p_tokens), 1))
            else:
                # Calculate approximate token accuracy using longest common subsequence
                token_matches.append(self._lcs_similarity(p_tokens, g_tokens))
        
        token_accuracy = np.mean(token_matches) if token_matches else 0.0
        
        return EvaluationMetric(
            name="accuracy",
            metric_type=EvaluationMetricType.ACCURACY,
            category=EvaluationCategory.PERFORMANCE,
            value=accuracy,
            standard_deviation=np.std([1 if p.strip() == g.strip() else 0 
                                     for p, g in zip(predictions, ground_truths)]),
            sample_size=len(predictions),
            metadata={
                "exact_match_accuracy": accuracy,
                "token_accuracy": token_accuracy,
                "exact_matches": exact_matches,
                "total_samples": len(predictions)
            }
        )
    
    def _lcs_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity using longest common subsequence."""
        if not seq1 or not seq2:
            return 0.0
        
        # Dynamic programming LCS
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        return lcs_length / max(m, n)
    
    async def _calculate_bleu(self, predictions: List[str], ground_truths: List[str]) -> EvaluationMetric:
        """Calculate BLEU score."""
        # Simplified BLEU implementation
        bleu_scores = []
        
        for pred, truth in zip(predictions, ground_truths):
            pred_tokens = pred.lower().split()
            truth_tokens = truth.lower().split()
            
            if not pred_tokens or not truth_tokens:
                bleu_scores.append(0.0)
                continue
            
            # Calculate n-gram precision for n=1,2,3,4
            precisions = []
            for n in range(1, 5):
                pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)]
                truth_ngrams = [tuple(truth_tokens[i:i+n]) for i in range(len(truth_tokens)-n+1)]
                
                if not pred_ngrams:
                    precisions.append(0.0)
                    continue
                
                pred_count = Counter(pred_ngrams)
                truth_count = Counter(truth_ngrams)
                
                matched = sum(min(pred_count[ng], truth_count[ng]) for ng in pred_count)
                precision = matched / len(pred_ngrams) if pred_ngrams else 0.0
                precisions.append(precision)
            
            # Brevity penalty
            bp = 1.0
            if len(pred_tokens) < len(truth_tokens):
                bp = np.exp(1 - len(truth_tokens) / len(pred_tokens))
            
            # BLEU score
            if all(p > 0 for p in precisions):
                bleu = bp * np.exp(np.mean([np.log(p) for p in precisions]))
            else:
                bleu = 0.0
            
            bleu_scores.append(bleu)
        
        mean_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        
        return EvaluationMetric(
            name="bleu",
            metric_type=EvaluationMetricType.BLEU,
            category=EvaluationCategory.QUALITY,
            value=mean_bleu,
            standard_deviation=np.std(bleu_scores) if bleu_scores else 0.0,
            sample_size=len(bleu_scores),
            metadata={
                "individual_scores": bleu_scores[:10],  # Store first 10 scores
                "min_score": min(bleu_scores) if bleu_scores else 0.0,
                "max_score": max(bleu_scores) if bleu_scores else 0.0
            }
        )
    
    async def _calculate_rouge(self, predictions: List[str], ground_truths: List[str]) -> EvaluationMetric:
        """Calculate ROUGE score."""
        rouge_scores = []
        
        for pred, truth in zip(predictions, ground_truths):
            pred_tokens = set(pred.lower().split())
            truth_tokens = set(truth.lower().split())
            
            if not truth_tokens:
                rouge_scores.append(0.0)
                continue
            
            # ROUGE-L (Longest Common Subsequence)
            pred_list = pred.lower().split()
            truth_list = truth.lower().split()
            
            lcs_length = self._lcs_length(pred_list, truth_list)
            
            if len(pred_list) == 0 or len(truth_list) == 0:
                rouge_scores.append(0.0)
            else:
                precision = lcs_length / len(pred_list)
                recall = lcs_length / len(truth_list)
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                
                rouge_scores.append(f1)
        
        mean_rouge = np.mean(rouge_scores) if rouge_scores else 0.0
        
        return EvaluationMetric(
            name="rouge",
            metric_type=EvaluationMetricType.ROUGE,
            category=EvaluationCategory.QUALITY,
            value=mean_rouge,
            standard_deviation=np.std(rouge_scores) if rouge_scores else 0.0,
            sample_size=len(rouge_scores),
            metadata={
                "individual_scores": rouge_scores[:10],
                "min_score": min(rouge_scores) if rouge_scores else 0.0,
                "max_score": max(rouge_scores) if rouge_scores else 0.0
            }
        )
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence."""
        if not seq1 or not seq2:
            return 0
        
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    async def _calculate_perplexity(self, predictions: List[str], ground_truths: List[str]) -> EvaluationMetric:
        """Calculate perplexity metric."""
        # Simplified perplexity calculation
        # In practice, this would require access to model probabilities
        
        perplexities = []
        
        for pred, truth in zip(predictions, ground_truths):
            # Approximate perplexity based on string similarity
            # Lower similarity = higher perplexity
            similarity = self._calculate_string_similarity(pred, truth)
            
            # Convert similarity to perplexity-like score
            # Higher similarity = lower perplexity
            perplexity = 1.0 / max(similarity, 0.01)  # Avoid division by zero
            perplexities.append(perplexity)
        
        mean_perplexity = np.mean(perplexities) if perplexities else float('inf')
        
        return EvaluationMetric(
            name="perplexity",
            metric_type=EvaluationMetricType.PERPLEXITY,
            category=EvaluationCategory.PERFORMANCE,
            value=mean_perplexity,
            standard_deviation=np.std(perplexities) if perplexities else 0.0,
            sample_size=len(perplexities),
            metadata={
                "note": "Approximated perplexity based on string similarity",
                "min_perplexity": min(perplexities) if perplexities else 0.0,
                "max_perplexity": max(perplexities) if perplexities else 0.0
            }
        )
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Jaccard similarity."""
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _calculate_latency_metric(self, latencies: List[float]) -> EvaluationMetric:
        """Calculate latency metric."""
        if not latencies:
            return EvaluationMetric(
                name="latency",
                metric_type=EvaluationMetricType.LATENCY,
                category=EvaluationCategory.EFFICIENCY,
                value=0.0
            )
        
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        return EvaluationMetric(
            name="latency",
            metric_type=EvaluationMetricType.LATENCY,
            category=EvaluationCategory.EFFICIENCY,
            value=mean_latency,
            standard_deviation=np.std(latencies),
            sample_size=len(latencies),
            metadata={
                "mean_latency_ms": mean_latency,
                "p95_latency_ms": p95_latency,
                "p99_latency_ms": p99_latency,
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies)
            }
        )
    
    async def _calculate_code_quality(self, predictions: List[str]) -> EvaluationMetric:
        """Calculate code quality score."""
        quality_scores = []
        
        for prediction in predictions:
            score = await self._analyze_code_quality(prediction)
            quality_scores.append(score)
        
        mean_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return EvaluationMetric(
            name="code_quality",
            metric_type=EvaluationMetricType.CODE_QUALITY,
            category=EvaluationCategory.QUALITY,
            value=mean_quality,
            standard_deviation=np.std(quality_scores) if quality_scores else 0.0,
            sample_size=len(quality_scores),
            metadata={
                "individual_scores": quality_scores[:10],
                "min_score": min(quality_scores) if quality_scores else 0.0,
                "max_score": max(quality_scores) if quality_scores else 0.0
            }
        )
    
    async def _analyze_code_quality(self, code: str) -> float:
        """Analyze code quality."""
        score = 0.0
        max_score = 10.0
        
        # Check for basic code structure
        if re.search(r'def\s+\w+\s*\(', code):  # Function definition
            score += 2.0
        
        if re.search(r'class\s+\w+', code):  # Class definition
            score += 1.5
        
        # Check for proper indentation
        lines = code.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
        if indented_lines > 0:
            score += 1.0
        
        # Check for comments
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        if comment_lines > 0:
            score += 1.0
        
        # Check for docstrings
        if '"""' in code or "'''" in code:
            score += 1.0
        
        # Check for imports
        if re.search(r'^(import|from)\s+', code, re.MULTILINE):
            score += 0.5
        
        # Check for error handling
        if 'try:' in code and 'except' in code:
            score += 1.0
        
        # Penalty for very short code
        if len(code.strip()) < 20:
            score *= 0.5
        
        # Penalty for syntax issues (simplified check)
        if code.count('(') != code.count(')'):
            score *= 0.7
        
        if code.count('[') != code.count(']'):
            score *= 0.7
        
        if code.count('{') != code.count('}'):
            score *= 0.7
        
        return min(score / max_score, 1.0)
    
    async def _calculate_security_score(self, predictions: List[str]) -> EvaluationMetric:
        """Calculate security score."""
        security_scores = []
        
        for prediction in predictions:
            score = await self._analyze_security(prediction)
            security_scores.append(score)
        
        mean_security = np.mean(security_scores) if security_scores else 0.0
        
        return EvaluationMetric(
            name="security_score",
            metric_type=EvaluationMetricType.SECURITY_SCORE,
            category=EvaluationCategory.SAFETY,
            value=mean_security,
            standard_deviation=np.std(security_scores) if security_scores else 0.0,
            sample_size=len(security_scores),
            metadata={
                "individual_scores": security_scores[:10],
                "min_score": min(security_scores) if security_scores else 0.0,
                "max_score": max(security_scores) if security_scores else 0.0
            }
        )
    
    async def _analyze_security(self, code: str) -> float:
        """Analyze security issues in code."""
        security_score = 1.0  # Start with perfect score
        
        # Security anti-patterns (reduce score for each found)
        security_issues = [
            (r'eval\s*\(', 0.3),  # eval() usage
            (r'exec\s*\(', 0.3),  # exec() usage
            (r'os\.system\s*\(', 0.2),  # os.system() usage
            (r'subprocess\.call\s*\(.*shell\s*=\s*True', 0.2),  # shell=True
            (r'input\s*\([^)]*\)', 0.1),  # raw input (potential injection)
            (r'pickle\.loads?\s*\(', 0.1),  # pickle usage
            (r'yaml\.load\s*\((?!.*Loader)', 0.1),  # unsafe yaml.load
            (r'sql.*\+.*%', 0.2),  # potential SQL injection
            (r'<script', 0.2),  # XSS potential
            (r'password\s*=\s*["\'][^"\']*["\']', 0.1),  # hardcoded passwords
        ]
        
        for pattern, penalty in security_issues:
            if re.search(pattern, code, re.IGNORECASE):
                security_score -= penalty
        
        # Positive security patterns (increase score)
        positive_patterns = [
            (r'import\s+hashlib', 0.05),  # Using secure hashing
            (r'import\s+secrets', 0.05),  # Using secure random
            (r'try:.*except', 0.05),  # Error handling
            (r'assert\s+', 0.02),  # Input validation
        ]
        
        for pattern, bonus in positive_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                security_score += bonus
        
        return max(0.0, min(1.0, security_score))
    
    async def _calculate_custom_metrics(self, predictions: List[str], ground_truths: List[str]) -> Optional[EvaluationMetric]:
        """Calculate custom metrics."""
        if not self.custom_metrics:
            return None
        
        # For now, calculate the first custom metric
        metric_name = list(self.custom_metrics.keys())[0]
        metric_info = self.custom_metrics[metric_name]
        
        scores = []
        for pred, truth in zip(predictions, ground_truths):
            try:
                score = metric_info["function"](pred, truth)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Custom metric {metric_name} failed: {e}")
                scores.append(0.0)
        
        mean_score = np.mean(scores) if scores else 0.0
        
        return EvaluationMetric(
            name=metric_name,
            metric_type=EvaluationMetricType.CUSTOM,
            category=metric_info["category"],
            value=mean_score,
            standard_deviation=np.std(scores) if scores else 0.0,
            sample_size=len(scores)
        )
    
    async def _calculate_overall_scores(self, result: EvaluationResult) -> None:
        """Calculate overall quality and safety scores."""
        # Overall quality score (weighted average of quality metrics)
        quality_metrics = [m for m in result.metrics 
                          if m.category in [EvaluationCategory.QUALITY, EvaluationCategory.PERFORMANCE]]
        
        if quality_metrics:
            weights = {
                EvaluationMetricType.ACCURACY: 0.3,
                EvaluationMetricType.BLEU: 0.2,
                EvaluationMetricType.ROUGE: 0.2,
                EvaluationMetricType.CODE_QUALITY: 0.3
            }
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for metric in quality_metrics:
                weight = weights.get(metric.metric_type, 0.1)
                weighted_sum += metric.value * weight
                total_weight += weight
            
            result.overall_quality_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Code quality score
        code_quality_metrics = [m for m in result.metrics 
                               if m.metric_type == EvaluationMetricType.CODE_QUALITY]
        if code_quality_metrics:
            result.code_quality_score = code_quality_metrics[0].value
        
        # Safety score
        safety_metrics = [m for m in result.metrics 
                         if m.category == EvaluationCategory.SAFETY]
        if safety_metrics:
            result.safety_score = np.mean([m.value for m in safety_metrics])
    
    async def compare_models(
        self,
        model_a: str,
        model_b: str,
        dataset_path: str,
        metrics: List[EvaluationMetricType],
        method: ComparisonMethod = ComparisonMethod.T_TEST
    ) -> ComparisonResult:
        """Compare two models statistically."""
        logger.info(f"Comparing models: {model_a} vs {model_b}")
        
        # Evaluate both models
        result_a = await self.evaluate_model(model_a, dataset_path, metrics)
        result_b = await self.evaluate_model(model_b, dataset_path, metrics)
        
        # Perform statistical comparison
        comparison_id = f"comp_{int(time.time())}"
        comparison = ComparisonResult(
            model_a=model_a,
            model_b=model_b,
            comparison_id=comparison_id,
            timestamp=datetime.now(),
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            significant=False,
            method=method,
            model_a_metrics=result_a.metrics,
            model_b_metrics=result_b.metrics
        )
        
        # Find matching metrics
        for metric_a in result_a.metrics:
            metric_b = next((m for m in result_b.metrics if m.metric_type == metric_a.metric_type), None)
            
            if metric_b:
                stat_result = await self._statistical_test(
                    [metric_a.value], [metric_b.value], method
                )
                
                # Use the first significant result or the most important metric
                if stat_result["significant"] or comparison.statistic == 0.0:
                    comparison.statistic = stat_result["statistic"]
                    comparison.p_value = stat_result["p_value"]
                    comparison.effect_size = stat_result["effect_size"]
                    comparison.confidence_interval = stat_result["confidence_interval"]
                    comparison.significant = stat_result["significant"]
                    
                    # Determine winner
                    if stat_result["significant"]:
                        if metric_a.value > metric_b.value:
                            comparison.winner = model_a
                            comparison.improvement_percentage = ((metric_a.value - metric_b.value) / metric_b.value) * 100
                        else:
                            comparison.winner = model_b
                            comparison.improvement_percentage = ((metric_b.value - metric_a.value) / metric_a.value) * 100
        
        return comparison
    
    async def _statistical_test(
        self,
        sample_a: List[float],
        sample_b: List[float],
        method: ComparisonMethod
    ) -> Dict[str, Any]:
        """Perform statistical test."""
        # Simplified statistical testing
        # In practice, would use scipy.stats
        
        if not sample_a or not sample_b:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "effect_size": 0.0,
                "confidence_interval": (0.0, 0.0),
                "significant": False
            }
        
        mean_a = np.mean(sample_a)
        mean_b = np.mean(sample_b)
        
        # Simple effect size (Cohen's d approximation)
        pooled_std = np.sqrt(((np.std(sample_a) ** 2) + (np.std(sample_b) ** 2)) / 2)
        effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
        
        # Simplified t-test
        if method == ComparisonMethod.T_TEST:
            # Very simplified t-statistic
            se = pooled_std / np.sqrt(len(sample_a) + len(sample_b))
            t_stat = (mean_a - mean_b) / se if se > 0 else 0.0
            
            # Approximate p-value (very simplified)
            p_value = 2 * (1 - 0.5 * (1 + np.tanh(abs(t_stat) / 2)))
            
            significant = p_value < 0.05
            
            # Confidence interval (simplified)
            margin = 1.96 * se  # 95% CI
            ci = (mean_a - mean_b - margin, mean_a - mean_b + margin)
            
            return {
                "statistic": t_stat,
                "p_value": p_value,
                "effect_size": effect_size,
                "confidence_interval": ci,
                "significant": significant
            }
        
        # Default return for other methods
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size": effect_size,
            "confidence_interval": (0.0, 0.0),
            "significant": False
        }
    
    async def run_benchmark(self, model_name: str, benchmark_id: str) -> EvaluationResult:
        """Run a specific benchmark on a model."""
        if benchmark_id not in self.benchmark_tasks:
            raise ValueError(f"Benchmark not found: {benchmark_id}")
        
        task = self.benchmark_tasks[benchmark_id]
        logger.info(f"Running benchmark {benchmark_id} on model {model_name}")
        
        # Create temporary dataset from benchmark task
        temp_dataset = []
        for input_data, expected in zip(task.inputs, task.expected_outputs):
            temp_dataset.append({
                "input": input_data.get("prompt", str(input_data)),
                "output": expected.get("response", str(expected))
            })
        
        # Save temporary dataset
        temp_file = f"/tmp/benchmark_{benchmark_id}_{int(time.time())}.jsonl"
        with open(temp_file, 'w') as f:
            for item in temp_dataset:
                f.write(json.dumps(item) + '\n')
        
        try:
            # Run evaluation
            result = await self.evaluate_model(model_name, temp_file, task.metrics)
            
            # Store benchmark result
            if model_name not in self.benchmark_results:
                self.benchmark_results[model_name] = {}
            self.benchmark_results[model_name][benchmark_id] = result
            
            return result
            
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    async def start_ab_test(self, config: ABTestConfig) -> str:
        """Start an A/B test."""
        logger.info(f"Starting A/B test: {config.test_id}")
        
        # Register test
        self.active_ab_tests[config.test_id] = config
        
        # Run evaluation on both models
        result_a = await self.evaluate_model(
            config.model_a, 
            config.test_dataset, 
            config.evaluation_metrics,
            sample_size=int(config.sample_size * config.traffic_split[0])
        )
        
        result_b = await self.evaluate_model(
            config.model_b,
            config.test_dataset,
            config.evaluation_metrics,
            sample_size=int(config.sample_size * config.traffic_split[1])
        )
        
        # Perform comparison
        comparison = await self.compare_models(
            config.model_a,
            config.model_b,
            config.test_dataset,
            config.evaluation_metrics
        )
        
        # Store result
        self.ab_test_results[config.test_id] = comparison
        
        # Cleanup active test
        if config.test_id in self.active_ab_tests:
            del self.active_ab_tests[config.test_id]
        
        logger.info(f"A/B test completed: {config.test_id}")
        return config.test_id
    
    def get_evaluation_result(self, evaluation_id: str) -> Optional[EvaluationResult]:
        """Get evaluation result by ID."""
        return next((r for r in self.evaluation_history if r.evaluation_id == evaluation_id), None)
    
    def list_evaluations(self, model_name: Optional[str] = None) -> List[EvaluationResult]:
        """List evaluation results, optionally filtered by model."""
        results = self.evaluation_history
        if model_name:
            results = [r for r in results if r.model_name == model_name]
        return sorted(results, key=lambda x: x.timestamp, reverse=True)
    
    def get_benchmark_results(self, model_name: str) -> Dict[str, EvaluationResult]:
        """Get all benchmark results for a model."""
        return self.benchmark_results.get(model_name, {})
    
    def get_ab_test_result(self, test_id: str) -> Optional[ComparisonResult]:
        """Get A/B test result by ID."""
        return self.ab_test_results.get(test_id)
    
    async def generate_report(self, evaluation_id: str, output_path: str) -> None:
        """Generate evaluation report."""
        result = self.get_evaluation_result(evaluation_id)
        if not result:
            raise ValueError(f"Evaluation not found: {evaluation_id}")
        
        report = {
            "evaluation_summary": {
                "evaluation_id": result.evaluation_id,
                "model_name": result.model_name,
                "model_version": result.model_version,
                "timestamp": result.timestamp.isoformat(),
                "overall_quality_score": result.overall_quality_score,
                "code_quality_score": result.code_quality_score,
                "safety_score": result.safety_score
            },
            "performance_metrics": {
                "latency_ms": result.latency_ms,
                "throughput_tokens_per_sec": result.throughput_tokens_per_sec,
                "memory_usage_mb": result.memory_usage_mb
            },
            "evaluation_metrics": [
                {
                    "name": m.name,
                    "type": m.metric_type.value,
                    "category": m.category.value,
                    "value": m.value,
                    "standard_deviation": m.standard_deviation,
                    "sample_size": m.sample_size,
                    "metadata": m.metadata
                }
                for m in result.metrics
            ],
            "dataset_info": result.dataset_info,
            "sample_outputs": result.sample_outputs,
            "errors": result.errors,
            "warnings": result.warnings
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report generated: {output_path}")


# Global instance management
_evaluation_engine: Optional[EvaluationEngine] = None


def get_evaluation_engine() -> EvaluationEngine:
    """Get the global EvaluationEngine instance."""
    global _evaluation_engine
    if _evaluation_engine is None:
        _evaluation_engine = EvaluationEngine()
    return _evaluation_engine


# Convenience functions
async def quick_evaluate(
    model_name: str,
    dataset_path: str,
    metrics: Optional[List[str]] = None
) -> EvaluationResult:
    """Quick model evaluation with default settings."""
    engine = get_evaluation_engine()
    
    # Default metrics
    if metrics is None:
        metrics = ["accuracy", "bleu", "latency", "code_quality"]
    
    # Convert string metrics to enum
    metric_types = []
    for metric in metrics:
        try:
            metric_types.append(EvaluationMetricType(metric))
        except ValueError:
            logger.warning(f"Unknown metric: {metric}")
    
    return await engine.evaluate_model(model_name, dataset_path, metric_types)


async def quick_compare(
    model_a: str,
    model_b: str,
    dataset_path: str,
    metrics: Optional[List[str]] = None
) -> ComparisonResult:
    """Quick model comparison."""
    engine = get_evaluation_engine()
    
    # Default metrics
    if metrics is None:
        metrics = ["accuracy", "bleu", "latency"]
    
    # Convert string metrics to enum
    metric_types = []
    for metric in metrics:
        try:
            metric_types.append(EvaluationMetricType(metric))
        except ValueError:
            logger.warning(f"Unknown metric: {metric}")
    
    return await engine.compare_models(model_a, model_b, dataset_path, metric_types)


def create_code_generation_benchmark() -> BenchmarkTask:
    """Create a standard code generation benchmark."""
    inputs = [
        {"prompt": "Write a Python function to calculate factorial"},
        {"prompt": "Create a class for a simple calculator"},
        {"prompt": "Implement bubble sort algorithm"},
        {"prompt": "Write a function to reverse a string"},
        {"prompt": "Create a binary search implementation"}
    ]
    
    expected_outputs = [
        {"response": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"},
        {"response": "class Calculator:\n    def add(self, a, b):\n        return a + b\n    def subtract(self, a, b):\n        return a - b"},
        {"response": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"},
        {"response": "def reverse_string(s):\n    return s[::-1]"},
        {"response": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"}
    ]
    
    return BenchmarkTask(
        task_id="code_generation_basic",
        name="Basic Code Generation",
        description="Test basic code generation capabilities",
        task_type="code_generation",
        inputs=inputs,
        expected_outputs=expected_outputs,
        metrics=[
            EvaluationMetricType.ACCURACY,
            EvaluationMetricType.CODE_QUALITY,
            EvaluationMetricType.LATENCY
        ],
        difficulty_level="easy",
        tags=["code", "basic", "algorithms"]
    )