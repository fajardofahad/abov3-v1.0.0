#!/usr/bin/env python3
"""
Example demonstrating the integrated logging, error handling, and monitoring framework.

This example shows how to use the comprehensive logging, error handling, and 
monitoring systems together in a production-ready ABOV3 application.

Usage:
    python examples/logging_error_monitoring_example.py

Author: ABOV3 Enterprise DevOps Agent
Version: 1.0.0
"""

import asyncio
import random
import time
from datetime import datetime, timezone

# Import the ABOV3 utilities
from abov3.utils import (
    # Logging
    get_logger,
    get_performance_logger,
    get_security_logger,
    correlation_context,
    log_context,
    log_function_call,
    
    # Error handling
    BaseError,
    ModelError,
    NetworkError,
    ValidationError,
    ErrorCategory,
    ErrorSeverity,
    error_handler,
    retry_on_error,
    with_error_context,
    handle_error,
    
    # Monitoring
    get_monitoring_system,
    get_app_metrics,
    get_system_metrics,
    get_alert_manager,
    get_health_manager,
    monitor_performance,
    count_calls,
    HealthStatus,
    AlertSeverity,
)


class AIModelService:
    """Example AI model service with integrated logging, error handling, and monitoring."""
    
    def __init__(self):
        self.logger = get_logger('ai_model_service')
        self.perf_logger = get_performance_logger('ai_model_service')
        self.security_logger = get_security_logger('ai_model_service')
        self.metrics = get_app_metrics()
        
        # Setup health checks
        health_manager = get_health_manager()
        health_manager.add_health_check(
            "model_availability",
            self._check_model_health,
            interval=30.0
        )
        
        # Setup custom alerts
        alert_manager = get_alert_manager()
        alert_manager.add_alert_rule(
            name="high_inference_error_rate",
            description="AI inference error rate is above 10%",
            condition="inference_error_rate > 0.1",
            threshold=0.1,
            severity=AlertSeverity.WARNING
        )
    
    def _check_model_health(self) -> bool:
        """Health check for model availability."""
        try:
            # Simulate model health check
            return random.random() > 0.1  # 10% chance of failure
        except Exception:
            return False
    
    @log_function_call()
    @monitor_performance()
    @count_calls()
    @with_error_context(component="ai_model_service", operation="generate_text")
    async def generate_text(self, prompt: str, model: str = "llama3.2") -> str:
        """Generate text using AI model with comprehensive monitoring."""
        
        with correlation_context() as corr_id:
            self.logger.info(
                f"Starting text generation",
                extra={'extra_fields': {
                    'prompt_length': len(prompt),
                    'model': model,
                    'correlation_id': corr_id
                }}
            )
            
            try:
                # Validate input
                if not prompt or len(prompt) > 10000:
                    raise ValidationError(
                        "Invalid prompt length",
                        details={'prompt_length': len(prompt), 'max_length': 10000}
                    )
                
                # Simulate AI inference with monitoring
                with self.perf_logger.timer("ai_inference", labels={'model': model}):
                    await self._simulate_inference(prompt, model)
                
                # Generate response
                response = f"Generated response for: {prompt[:50]}..."
                
                # Record metrics
                self.metrics.record_ai_inference(
                    model=model,
                    duration=random.uniform(0.5, 2.0),
                    tokens=len(response),
                    success=True
                )
                
                self.logger.info(
                    "Text generation completed successfully",
                    extra={'extra_fields': {
                        'response_length': len(response),
                        'model': model
                    }}
                )
                
                return response
                
            except Exception as e:
                # Handle and log error
                handled_error = handle_error(e)
                
                # Record error metrics
                self.metrics.record_ai_inference(
                    model=model,
                    duration=0.0,
                    tokens=0,
                    success=False
                )
                
                self.logger.error(
                    f"Text generation failed: {handled_error.message}",
                    extra={'extra_fields': handled_error.to_dict()}
                )
                
                raise handled_error
    
    @retry_on_error(max_retries=3, exceptions=(NetworkError,))
    async def _simulate_inference(self, prompt: str, model: str) -> None:
        """Simulate AI model inference with retry logic."""
        
        # Simulate random failures
        failure_rate = 0.2 if model == "unstable_model" else 0.05
        
        if random.random() < failure_rate:
            if random.random() < 0.5:
                raise NetworkError(
                    "Failed to connect to model server",
                    details={'model': model, 'prompt_length': len(prompt)}
                )
            else:
                raise ModelError(
                    f"Model '{model}' inference failed",
                    details={'model': model, 'error_type': 'inference_error'}
                )
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.1, 1.0))


class UserService:
    """Example user service with security logging."""
    
    def __init__(self):
        self.logger = get_logger('user_service')
        self.security_logger = get_security_logger('user_service')
        self.metrics = get_app_metrics()
    
    @error_handler()
    @monitor_performance()
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user with security logging."""
        
        with log_context(operation="user_authentication", username=username):
            try:
                # Simulate authentication
                success = username == "admin" and password == "secret"
                
                # Log authentication attempt
                self.security_logger.log_authentication(
                    success=success,
                    user_id=username,
                    ip_address="127.0.0.1"
                )
                
                if not success:
                    self.security_logger.log_security_violation(
                        violation_type="authentication_failure",
                        description=f"Failed login attempt for user: {username}",
                        severity="medium",
                        user_id=username
                    )
                
                return success
                
            except Exception as e:
                self.logger.error(f"Authentication error: {e}")
                raise


async def run_example():
    """Run the comprehensive example."""
    
    logger = get_logger('example')
    
    # Start monitoring system
    monitoring = get_monitoring_system()
    monitoring.start()
    
    logger.info("Starting comprehensive logging, error handling, and monitoring example")
    
    try:
        # Initialize services
        ai_service = AIModelService()
        user_service = UserService()
        
        # Simulate user authentication
        logger.info("Testing user authentication")
        user_service.authenticate_user("admin", "secret")
        user_service.authenticate_user("hacker", "wrongpassword")
        
        # Simulate AI model operations
        logger.info("Testing AI model service")
        
        tasks = []
        for i in range(10):
            prompt = f"Generate a creative story about {['AI', 'robots', 'space', 'magic'][i % 4]}"
            model = "llama3.2" if i % 3 != 0 else "unstable_model"
            
            task = ai_service.generate_text(prompt, model)
            tasks.append(task)
        
        # Execute tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        logger.info(
            f"AI operations completed",
            extra={'extra_fields': {
                'total': len(results),
                'successful': successful,
                'failed': failed,
                'success_rate': successful / len(results) if results else 0
            }}
        )
        
        # Display monitoring summary
        await display_monitoring_summary()
        
    except Exception as e:
        logger.exception(f"Example execution failed: {e}")
    
    finally:
        # Stop monitoring system
        monitoring.stop()
        logger.info("Example completed")


async def display_monitoring_summary():
    """Display monitoring and metrics summary."""
    
    logger = get_logger('example')
    
    # Get system status
    monitoring = get_monitoring_system()
    status = monitoring.get_status()
    
    # Get health summary
    health_manager = get_health_manager()
    health_summary = health_manager.get_health_summary()
    
    # Get active alerts
    alert_manager = get_alert_manager()
    active_alerts = alert_manager.get_active_alerts()
    
    # Get system metrics
    system_metrics = get_system_metrics()
    recent_metrics = system_metrics.get_metrics()[-5:]  # Last 5 metrics
    
    logger.info("=== MONITORING SUMMARY ===")
    logger.info(f"System Status: {status}")
    logger.info(f"Health Status: {health_summary['overall_status']}")
    logger.info(f"Active Alerts: {len(active_alerts)}")
    logger.info(f"Recent Metrics Count: {len(recent_metrics)}")
    
    if active_alerts:
        logger.warning("Active Alerts:")
        for alert in active_alerts:
            logger.warning(f"  - {alert.name}: {alert.description}")
    
    # Display error statistics
    from abov3.utils.errors import get_error_handler
    error_handler = get_error_handler()
    error_stats = error_handler.get_error_stats()
    
    logger.info(f"Error Statistics: {error_stats}")


if __name__ == "__main__":
    print("ABOV3 Comprehensive Logging, Error Handling, and Monitoring Example")
    print("=" * 70)
    
    # Run the example
    asyncio.run(run_example())