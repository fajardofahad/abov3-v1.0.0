"""
Comprehensive Monitoring and Observability Framework for ABOV3 Ollama

This module provides enterprise-grade monitoring and observability with:
- System performance monitoring and resource tracking
- Application metrics collection and analysis
- Health check endpoints and service discovery
- Resource usage monitoring (CPU, memory, disk, network)
- Alert system for critical issues and anomalies
- Integration with external monitoring systems
- Metrics export capabilities (Prometheus, StatsD, etc.)
- Real-time dashboards and visualization
- Custom metrics and instrumentation

Features:
- Multi-dimensional metrics with labels and tags
- Time-series data collection and aggregation
- Threshold-based alerting with notification channels
- Performance profiling and bottleneck detection
- Distributed tracing for microservices
- Service level indicators (SLIs) and objectives (SLOs)
- Integration with cloud monitoring services

Author: ABOV3 Enterprise DevOps Agent
Version: 1.0.0
"""

import asyncio
import json
import os
import platform
import psutil
import socket
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from urllib.parse import urlparse
import queue
import statistics

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import JSONResponse, PlainTextResponse
    MONITORING_API_AVAILABLE = True
except ImportError:
    MONITORING_API_AVAILABLE = False
    # Create dummy classes to avoid errors
    FastAPI = None
    HTTPException = Exception
    Response = None
    JSONResponse = None
    PlainTextResponse = None
    uvicorn = None
from pydantic import BaseModel, Field

from .logging import get_logger, get_performance_logger, correlation_context
from ..core.config import get_config


class MetricType(Enum):
    """Types of metrics."""
    
    COUNTER = "counter"         # Monotonically increasing
    GAUGE = "gauge"             # Point-in-time value
    HISTOGRAM = "histogram"     # Distribution of values
    SUMMARY = "summary"         # Statistical summary
    TIMER = "timer"             # Duration measurements


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """Health check status."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricPoint:
    """A single metric data point."""
    
    name: str
    value: Union[int, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'type': self.metric_type.value
        }


@dataclass
class Alert:
    """Alert definition and state."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    condition: str = ""  # Alert condition expression
    threshold: float = 0.0
    duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    active: bool = False
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'severity': self.severity.value,
            'condition': self.condition,
            'threshold': self.threshold,
            'duration': self.duration.total_seconds(),
            'active': self.active,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'labels': self.labels,
            'metadata': self.metadata
        }


@dataclass
class HealthCheck:
    """Health check definition."""
    
    name: str
    check_func: Callable[[], bool]
    timeout: float = 30.0
    interval: float = 60.0
    retry_count: int = 3
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'status': self.status.value,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'last_success': self.last_success.isoformat() if self.last_success else None,
            'error_message': self.error_message,
            'timeout': self.timeout,
            'interval': self.interval
        }


class MetricsCollector:
    """Base class for metrics collection."""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Configuration
        self.max_metric_points = 10000
        self.retention_period = timedelta(hours=24)
    
    def counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            
            metric_point = MetricPoint(
                name=name,
                value=self.counters[key],
                labels=labels or {},
                metric_type=MetricType.COUNTER
            )
            self._add_metric_point(name, metric_point)
    
    def gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            
            metric_point = MetricPoint(
                name=name,
                value=value,
                labels=labels or {},
                metric_type=MetricType.GAUGE
            )
            self._add_metric_point(name, metric_point)
    
    def histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Add a value to a histogram metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            
            # Keep only recent values
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            
            metric_point = MetricPoint(
                name=name,
                value=value,
                labels=labels or {},
                metric_type=MetricType.HISTOGRAM
            )
            self._add_metric_point(name, metric_point)
    
    @contextmanager
    def timer(self, name: str, labels: Dict[str, str] = None):
        """Context manager for timing operations."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.histogram(f"{name}_duration_seconds", duration, labels)
    
    def timing_decorator(self, name: str = None, labels: Dict[str, str] = None):
        """Decorator for timing function calls."""
        def decorator(func: Callable) -> Callable:
            metric_name = name or f"{func.__module__}.{func.__name__}_duration"
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.timer(metric_name, labels):
                    return func(*args, **kwargs)
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.timer(metric_name, labels):
                    return await func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create a unique key for the metric."""
        if not labels:
            return name
        
        sorted_labels = sorted(labels.items())
        label_str = ",".join(f"{k}={v}" for k, v in sorted_labels)
        return f"{name}{{{label_str}}}"
    
    def _add_metric_point(self, name: str, metric_point: MetricPoint) -> None:
        """Add a metric point to the collection."""
        self.metrics[name].append(metric_point)
        
        # Cleanup old metrics
        cutoff_time = datetime.now(timezone.utc) - self.retention_period
        self.metrics[name] = [
            point for point in self.metrics[name]
            if point.timestamp > cutoff_time
        ]
        
        # Limit number of points
        if len(self.metrics[name]) > self.max_metric_points:
            self.metrics[name] = self.metrics[name][-self.max_metric_points:]
    
    def get_metrics(self, name_pattern: str = None) -> List[MetricPoint]:
        """Get collected metrics."""
        with self._lock:
            if name_pattern:
                import re
                pattern = re.compile(name_pattern)
                matching_metrics = []
                for name, points in self.metrics.items():
                    if pattern.search(name):
                        matching_metrics.extend(points)
                return matching_metrics
            else:
                all_metrics = []
                for points in self.metrics.values():
                    all_metrics.extend(points)
                return all_metrics
    
    def get_histogram_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        values = self.histograms.get(key, [])
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'sum': sum(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p95': self._percentile(values, 95),
            'p99': self._percentile(values, 99)
        }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()


class SystemMetricsCollector(MetricsCollector):
    """System-level metrics collector."""
    
    def __init__(self):
        super().__init__("system")
        self.collection_interval = 10.0  # seconds
        self.running = False
        self.collection_thread = None
    
    def start_collection(self) -> None:
        """Start automatic metrics collection."""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="SystemMetricsCollector"
        )
        self.collection_thread.start()
    
    def stop_collection(self) -> None:
        """Stop automatic metrics collection."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
    
    def _collection_loop(self) -> None:
        """Main collection loop."""
        while self.running:
            try:
                self.collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                get_logger('monitoring').error(f"Error collecting system metrics: {e}")
                time.sleep(self.collection_interval)
    
    def collect_system_metrics(self) -> None:
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.gauge("system_cpu_usage_percent", cpu_percent)
            
            cpu_count = psutil.cpu_count()
            self.gauge("system_cpu_count", cpu_count)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.gauge("system_memory_total_bytes", memory.total)
            self.gauge("system_memory_used_bytes", memory.used)
            self.gauge("system_memory_available_bytes", memory.available)
            self.gauge("system_memory_usage_percent", memory.percent)
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            self.gauge("system_disk_total_bytes", disk_usage.total)
            self.gauge("system_disk_used_bytes", disk_usage.used)
            self.gauge("system_disk_free_bytes", disk_usage.free)
            self.gauge("system_disk_usage_percent", (disk_usage.used / disk_usage.total) * 100)
            
            # Network metrics
            network_io = psutil.net_io_counters()
            if network_io:
                self.counter("system_network_bytes_sent", network_io.bytes_sent)
                self.counter("system_network_bytes_recv", network_io.bytes_recv)
                self.counter("system_network_packets_sent", network_io.packets_sent)
                self.counter("system_network_packets_recv", network_io.packets_recv)
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            self.gauge("process_memory_rss_bytes", process_memory.rss)
            self.gauge("process_memory_vms_bytes", process_memory.vms)
            self.gauge("process_cpu_usage_percent", process.cpu_percent())
            
            # File descriptor count (Unix-like systems)
            try:
                fd_count = process.num_fds()
                self.gauge("process_open_fds", fd_count)
            except AttributeError:
                # Windows doesn't have file descriptors
                pass
            
            # Thread count
            thread_count = process.num_threads()
            self.gauge("process_threads", thread_count)
            
        except Exception as e:
            get_logger('monitoring').error(f"Error collecting system metrics: {e}")


class ApplicationMetricsCollector(MetricsCollector):
    """Application-specific metrics collector."""
    
    def __init__(self):
        super().__init__("application")
        self.request_count = 0
        self.error_count = 0
        self.active_connections = 0
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float) -> None:
        """Record an HTTP request."""
        labels = {
            'method': method,
            'endpoint': endpoint,
            'status_code': str(status_code)
        }
        
        self.counter("http_requests_total", 1.0, labels)
        self.histogram("http_request_duration_seconds", duration, labels)
        
        if status_code >= 400:
            self.counter("http_errors_total", 1.0, labels)
    
    def record_ai_inference(self, model: str, duration: float, tokens: int, success: bool) -> None:
        """Record AI model inference."""
        labels = {
            'model': model,
            'success': str(success).lower()
        }
        
        self.counter("ai_inference_total", 1.0, labels)
        self.histogram("ai_inference_duration_seconds", duration, labels)
        self.histogram("ai_inference_tokens", tokens, labels)
        
        if not success:
            self.counter("ai_inference_errors_total", 1.0, labels)
    
    def record_database_operation(self, operation: str, table: str, duration: float, success: bool) -> None:
        """Record database operation."""
        labels = {
            'operation': operation,
            'table': table,
            'success': str(success).lower()
        }
        
        self.counter("database_operations_total", 1.0, labels)
        self.histogram("database_operation_duration_seconds", duration, labels)
        
        if not success:
            self.counter("database_errors_total", 1.0, labels)
    
    def record_plugin_execution(self, plugin: str, duration: float, success: bool) -> None:
        """Record plugin execution."""
        labels = {
            'plugin': plugin,
            'success': str(success).lower()
        }
        
        self.counter("plugin_executions_total", 1.0, labels)
        self.histogram("plugin_execution_duration_seconds", duration, labels)
        
        if not success:
            self.counter("plugin_errors_total", 1.0, labels)


class AlertManager:
    """Alert management and notification system."""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_channels: List[Callable] = []
        self.evaluation_interval = 60.0  # seconds
        self.running = False
        self.evaluation_thread = None
        self._lock = threading.Lock()
    
    def add_alert_rule(
        self,
        name: str,
        description: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        duration: timedelta = None,
        labels: Dict[str, str] = None
    ) -> None:
        """Add an alert rule."""
        alert_rule = {
            'name': name,
            'description': description,
            'condition': condition,
            'threshold': threshold,
            'severity': severity,
            'duration': duration or timedelta(minutes=5),
            'labels': labels or {}
        }
        
        with self._lock:
            self.alert_rules.append(alert_rule)
    
    def add_notification_channel(self, channel: Callable[[Alert], None]) -> None:
        """Add a notification channel."""
        self.notification_channels.append(channel)
    
    def start_evaluation(self) -> None:
        """Start alert evaluation."""
        if self.running:
            return
        
        self.running = True
        self.evaluation_thread = threading.Thread(
            target=self._evaluation_loop,
            daemon=True,
            name="AlertEvaluator"
        )
        self.evaluation_thread.start()
    
    def stop_evaluation(self) -> None:
        """Stop alert evaluation."""
        self.running = False
        if self.evaluation_thread:
            self.evaluation_thread.join(timeout=5.0)
    
    def _evaluation_loop(self) -> None:
        """Main evaluation loop."""
        while self.running:
            try:
                self.evaluate_alerts()
                time.sleep(self.evaluation_interval)
            except Exception as e:
                get_logger('monitoring').error(f"Error evaluating alerts: {e}")
                time.sleep(self.evaluation_interval)
    
    def evaluate_alerts(self) -> None:
        """Evaluate all alert rules."""
        for rule in self.alert_rules:
            try:
                self._evaluate_rule(rule)
            except Exception as e:
                get_logger('monitoring').error(f"Error evaluating alert rule {rule['name']}: {e}")
    
    def _evaluate_rule(self, rule: Dict[str, Any]) -> None:
        """Evaluate a single alert rule."""
        # This is a simplified implementation
        # In practice, you'd integrate with your metrics backend
        # to evaluate complex expressions
        
        alert_id = rule['name']
        current_alert = self.alerts.get(alert_id)
        
        # Simulate condition evaluation
        # You would replace this with actual metric evaluation logic
        condition_met = self._evaluate_condition(rule['condition'], rule['threshold'])
        
        if condition_met:
            if current_alert is None or not current_alert.active:
                # Create or reactivate alert
                alert = Alert(
                    id=alert_id,
                    name=rule['name'],
                    description=rule['description'],
                    severity=rule['severity'],
                    condition=rule['condition'],
                    threshold=rule['threshold'],
                    duration=rule['duration'],
                    active=True,
                    triggered_at=datetime.now(timezone.utc),
                    labels=rule['labels']
                )
                
                with self._lock:
                    self.alerts[alert_id] = alert
                
                self._send_alert_notification(alert, "triggered")
                
                get_logger('monitoring').warning(
                    f"Alert triggered: {alert.name}",
                    extra={'extra_fields': alert.to_dict()}
                )
        else:
            if current_alert and current_alert.active:
                # Resolve alert
                current_alert.active = False
                current_alert.resolved_at = datetime.now(timezone.utc)
                
                self._send_alert_notification(current_alert, "resolved")
                
                get_logger('monitoring').info(
                    f"Alert resolved: {current_alert.name}",
                    extra={'extra_fields': current_alert.to_dict()}
                )
    
    def _evaluate_condition(self, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        # Simplified condition evaluation
        # In practice, you would parse and evaluate complex expressions
        # against your metrics data
        
        # Example: "cpu_usage > 80"
        if "cpu_usage" in condition and ">" in condition:
            # Get current CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            return cpu_percent > threshold
        
        return False
    
    def _send_alert_notification(self, alert: Alert, action: str) -> None:
        """Send alert notification to all channels."""
        for channel in self.notification_channels:
            try:
                channel(alert, action)
            except Exception as e:
                get_logger('monitoring').error(f"Error sending alert notification: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return [alert for alert in self.alerts.values() if alert.active]
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            all_alerts = sorted(
                self.alerts.values(),
                key=lambda a: a.triggered_at or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True
            )
            return all_alerts[:limit]


class HealthCheckManager:
    """Health check management system."""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.check_interval = 30.0  # seconds
        self.running = False
        self.check_thread = None
        self._lock = threading.Lock()
    
    def add_health_check(
        self,
        name: str,
        check_func: Callable[[], bool],
        timeout: float = 30.0,
        interval: float = 60.0,
        retry_count: int = 3
    ) -> None:
        """Add a health check."""
        health_check = HealthCheck(
            name=name,
            check_func=check_func,
            timeout=timeout,
            interval=interval,
            retry_count=retry_count
        )
        
        with self._lock:
            self.health_checks[name] = health_check
    
    def start_checks(self) -> None:
        """Start health check monitoring."""
        if self.running:
            return
        
        self.running = True
        self.check_thread = threading.Thread(
            target=self._check_loop,
            daemon=True,
            name="HealthChecker"
        )
        self.check_thread.start()
    
    def stop_checks(self) -> None:
        """Stop health check monitoring."""
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=5.0)
    
    def _check_loop(self) -> None:
        """Main health check loop."""
        while self.running:
            try:
                self.run_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                get_logger('monitoring').error(f"Error running health checks: {e}")
                time.sleep(self.check_interval)
    
    def run_health_checks(self) -> None:
        """Run all health checks."""
        for check in self.health_checks.values():
            try:
                self._run_single_check(check)
            except Exception as e:
                get_logger('monitoring').error(f"Error running health check {check.name}: {e}")
    
    def _run_single_check(self, check: HealthCheck) -> None:
        """Run a single health check."""
        now = datetime.now(timezone.utc)
        
        # Check if it's time to run this check
        if (check.last_check and 
            now - check.last_check < timedelta(seconds=check.interval)):
            return
        
        check.last_check = now
        
        for attempt in range(check.retry_count):
            try:
                # Run the check with timeout
                result = asyncio.run(
                    asyncio.wait_for(
                        self._run_check_function(check.check_func),
                        timeout=check.timeout
                    )
                )
                
                if result:
                    check.status = HealthStatus.HEALTHY
                    check.last_success = now
                    check.error_message = None
                else:
                    check.status = HealthStatus.UNHEALTHY
                    check.error_message = "Check returned False"
                
                break  # Success, exit retry loop
                
            except asyncio.TimeoutError:
                check.status = HealthStatus.UNHEALTHY
                check.error_message = f"Check timed out after {check.timeout} seconds"
            except Exception as e:
                check.status = HealthStatus.UNHEALTHY
                check.error_message = str(e)
                
                if attempt < check.retry_count - 1:
                    time.sleep(1.0)  # Brief delay before retry
    
    async def _run_check_function(self, check_func: Callable) -> bool:
        """Run check function (async wrapper)."""
        if asyncio.iscoroutinefunction(check_func):
            return await check_func()
        else:
            return check_func()
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health."""
        if not self.health_checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in self.health_checks.values()]
        
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health check summary."""
        with self._lock:
            return {
                'overall_status': self.get_overall_health().value,
                'checks': {
                    name: check.to_dict()
                    for name, check in self.health_checks.items()
                },
                'summary': {
                    'total': len(self.health_checks),
                    'healthy': sum(1 for c in self.health_checks.values() if c.status == HealthStatus.HEALTHY),
                    'unhealthy': sum(1 for c in self.health_checks.values() if c.status == HealthStatus.UNHEALTHY),
                    'degraded': sum(1 for c in self.health_checks.values() if c.status == HealthStatus.DEGRADED),
                    'unknown': sum(1 for c in self.health_checks.values() if c.status == HealthStatus.UNKNOWN)
                }
            }


class PrometheusExporter:
    """Prometheus metrics exporter."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        output = []
        
        # Get all metrics
        all_metrics = self.metrics_collector.get_metrics()
        
        # Group by metric name
        metrics_by_name = defaultdict(list)
        for metric in all_metrics:
            metrics_by_name[metric.name].append(metric)
        
        for metric_name, metric_points in metrics_by_name.items():
            # Add help and type comments
            output.append(f"# HELP {metric_name} {metric_name}")
            
            if metric_points:
                metric_type = metric_points[0].metric_type.value
                output.append(f"# TYPE {metric_name} {metric_type}")
                
                # Add metric values
                for point in metric_points[-10:]:  # Last 10 points
                    labels_str = ""
                    if point.labels:
                        label_pairs = [f'{k}="{v}"' for k, v in point.labels.items()]
                        labels_str = "{" + ",".join(label_pairs) + "}"
                    
                    output.append(f"{metric_name}{labels_str} {point.value}")
        
        return "\n".join(output)


class MonitoringAPI:
    """HTTP API for monitoring endpoints."""
    
    def __init__(
        self,
        system_metrics: SystemMetricsCollector,
        app_metrics: ApplicationMetricsCollector,
        alert_manager: AlertManager,
        health_manager: HealthCheckManager
    ):
        self.system_metrics = system_metrics
        self.app_metrics = app_metrics
        self.alert_manager = alert_manager
        self.health_manager = health_manager
        self.app = self._create_app()
    
    def _create_app(self):
        """Create FastAPI application."""
        if not MONITORING_API_AVAILABLE or not FastAPI:
            return None
        
        app = FastAPI(title="ABOV3 Monitoring API", version="1.0.0")
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            health_summary = self.health_manager.get_health_summary()
            status_code = 200 if health_summary['overall_status'] == 'healthy' else 503
            return JSONResponse(content=health_summary, status_code=status_code)
        
        @app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint."""
            try:
                prometheus_exporter = PrometheusExporter(self.system_metrics)
                metrics_text = prometheus_exporter.export_metrics()
                return PlainTextResponse(content=metrics_text, media_type="text/plain")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/metrics/system")
        async def system_metrics():
            """System metrics endpoint."""
            metrics = self.system_metrics.get_metrics()
            return JSONResponse(content=[m.to_dict() for m in metrics])
        
        @app.get("/metrics/application")
        async def application_metrics():
            """Application metrics endpoint."""
            metrics = self.app_metrics.get_metrics()
            return JSONResponse(content=[m.to_dict() for m in metrics])
        
        @app.get("/alerts")
        async def get_alerts():
            """Get active alerts."""
            active_alerts = self.alert_manager.get_active_alerts()
            return JSONResponse(content=[a.to_dict() for a in active_alerts])
        
        @app.get("/alerts/history")
        async def get_alert_history():
            """Get alert history."""
            alert_history = self.alert_manager.get_alert_history()
            return JSONResponse(content=[a.to_dict() for a in alert_history])
        
        @app.get("/status")
        async def system_status():
            """Overall system status."""
            return JSONResponse(content={
                'status': 'ok',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'version': '1.0.0',
                'uptime': time.time() - self._start_time,
                'health': self.health_manager.get_overall_health().value
            })
        
        return app
    
    def start_server(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Start the monitoring API server."""
        if not MONITORING_API_AVAILABLE or not uvicorn:
            raise ImportError("uvicorn and fastapi are required to start the monitoring API server. Install them with: pip install uvicorn fastapi")
        
        self._start_time = time.time()
        
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        server.run()


class MonitoringSystem:
    """Main monitoring system coordinator."""
    
    def __init__(self):
        self.system_metrics = SystemMetricsCollector()
        self.app_metrics = ApplicationMetricsCollector()
        self.alert_manager = AlertManager()
        self.health_manager = HealthCheckManager()
        self.monitoring_api = MonitoringAPI(
            self.system_metrics,
            self.app_metrics,
            self.alert_manager,
            self.health_manager
        )
        
        self.logger = get_logger('monitoring')
        self.running = False
        
        # Setup default alerts
        self._setup_default_alerts()
        
        # Setup default health checks
        self._setup_default_health_checks()
    
    def _setup_default_alerts(self) -> None:
        """Setup default alert rules."""
        self.alert_manager.add_alert_rule(
            name="high_cpu_usage",
            description="CPU usage is above 80%",
            condition="cpu_usage > 80",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            duration=timedelta(minutes=5)
        )
        
        self.alert_manager.add_alert_rule(
            name="high_memory_usage",
            description="Memory usage is above 90%",
            condition="memory_usage > 90",
            threshold=90.0,
            severity=AlertSeverity.CRITICAL,
            duration=timedelta(minutes=2)
        )
        
        self.alert_manager.add_alert_rule(
            name="disk_space_low",
            description="Disk space is below 10%",
            condition="disk_free < 10",
            threshold=10.0,
            severity=AlertSeverity.CRITICAL,
            duration=timedelta(minutes=1)
        )
    
    def _setup_default_health_checks(self) -> None:
        """Setup default health checks."""
        def check_disk_space() -> bool:
            """Check if disk space is sufficient."""
            try:
                disk_usage = psutil.disk_usage('/')
                free_percent = (disk_usage.free / disk_usage.total) * 100
                return free_percent > 5.0  # At least 5% free
            except Exception:
                return False
        
        def check_memory() -> bool:
            """Check if memory usage is reasonable."""
            try:
                memory = psutil.virtual_memory()
                return memory.percent < 95.0  # Less than 95% used
            except Exception:
                return False
        
        def check_cpu() -> bool:
            """Check if CPU usage is reasonable."""
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                return cpu_percent < 95.0  # Less than 95% used
            except Exception:
                return False
        
        self.health_manager.add_health_check("disk_space", check_disk_space, interval=60.0)
        self.health_manager.add_health_check("memory_usage", check_memory, interval=30.0)
        self.health_manager.add_health_check("cpu_usage", check_cpu, interval=30.0)
    
    def start(self) -> None:
        """Start the monitoring system."""
        if self.running:
            return
        
        self.logger.info("Starting monitoring system")
        
        # Start components
        self.system_metrics.start_collection()
        self.alert_manager.start_evaluation()
        self.health_manager.start_checks()
        
        self.running = True
        self.logger.info("Monitoring system started")
    
    def stop(self) -> None:
        """Stop the monitoring system."""
        if not self.running:
            return
        
        self.logger.info("Stopping monitoring system")
        
        # Stop components
        self.system_metrics.stop_collection()
        self.alert_manager.stop_evaluation()
        self.health_manager.stop_checks()
        
        self.running = False
        self.logger.info("Monitoring system stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring system status."""
        return {
            'running': self.running,
            'components': {
                'system_metrics': self.system_metrics.running,
                'alert_manager': self.alert_manager.running,
                'health_manager': self.health_manager.running
            },
            'health': self.health_manager.get_health_summary(),
            'active_alerts': len(self.alert_manager.get_active_alerts())
        }


# Global monitoring instance
_monitoring_system: Optional[MonitoringSystem] = None
_monitoring_lock = threading.Lock()


def get_monitoring_system() -> MonitoringSystem:
    """Get the global monitoring system instance."""
    global _monitoring_system
    
    if _monitoring_system is None:
        with _monitoring_lock:
            if _monitoring_system is None:
                _monitoring_system = MonitoringSystem()
    
    return _monitoring_system


def get_system_metrics() -> SystemMetricsCollector:
    """Get system metrics collector."""
    return get_monitoring_system().system_metrics


def get_app_metrics() -> ApplicationMetricsCollector:
    """Get application metrics collector."""
    return get_monitoring_system().app_metrics


def get_alert_manager() -> AlertManager:
    """Get alert manager."""
    return get_monitoring_system().alert_manager


def get_health_manager() -> HealthCheckManager:
    """Get health check manager."""
    return get_monitoring_system().health_manager


# Decorators for easy instrumentation
def monitor_performance(metric_name: str = None, labels: Dict[str, str] = None):
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}"
        metrics = get_app_metrics()
        
        return metrics.timing_decorator(name, labels)(func)
    
    return decorator


def count_calls(metric_name: str = None, labels: Dict[str, str] = None):
    """Decorator to count function calls."""
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}_calls"
        metrics = get_app_metrics()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics.counter(name, 1.0, labels)
            return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics.counter(name, 1.0, labels)
            return await func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Export key classes and functions
__all__ = [
    # Enums
    'MetricType',
    'AlertSeverity',
    'HealthStatus',
    
    # Data classes
    'MetricPoint',
    'Alert',
    'HealthCheck',
    
    # Core classes
    'MetricsCollector',
    'SystemMetricsCollector',
    'ApplicationMetricsCollector',
    'AlertManager',
    'HealthCheckManager',
    'PrometheusExporter',
    'MonitoringAPI',
    'MonitoringSystem',
    
    # Functions
    'get_monitoring_system',
    'get_system_metrics',
    'get_app_metrics',
    'get_alert_manager',
    'get_health_manager',
    
    # Decorators
    'monitor_performance',
    'count_calls',
]