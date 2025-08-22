"""
Comprehensive Model Management Example for ABOV3 4 Ollama.

This example demonstrates the advanced model management capabilities including:
- Model discovery and installation
- Performance monitoring and optimization
- Model recommendations
- Health checks and validation
- Model registry integration
- Search and filtering
- Ratings and reviews
- Usage analytics

Author: ABOV3 Enterprise AI/ML Expert Agent
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ABOV3 components
from abov3.core.config import get_config
from abov3.utils.security import SecurityManager
from abov3.models import (
    ModelManager,
    ModelRegistry,
    ModelMetadata,
    ModelRating,
    ModelBenchmark,
    SearchFilter,
    ModelType,
    ModelCapability,
    ModelSize
)


async def demonstrate_model_management():
    """Demonstrate comprehensive model management capabilities."""
    
    print("🚀 ABOV3 Model Management Demo")
    print("=" * 50)
    
    # Initialize components
    config = get_config()
    security_manager = SecurityManager()
    
    async with ModelManager(config, security_manager) as manager:
        registry = ModelRegistry(config, security_manager)
        
        try:
            # 1. Model Discovery and Health Checks
            print("\n1. 📋 Model Discovery and Health Checks")
            print("-" * 40)
            
            models = await manager.list_models()
            print(f"Found {len(models)} local models:")
            
            for model in models[:3]:  # Show first 3 models
                print(f"  📦 {model.name}")
                print(f"     Size: {model.size_gb:.1f} GB")
                print(f"     Category: {model.size_category.value}")
                
                # Health check
                health = await manager.check_model_health(model.name)
                status = "✅ Healthy" if health.is_healthy else "❌ Unhealthy"
                print(f"     Health: {status}")
                
                if health.response_time:
                    print(f"     Response Time: {health.response_time:.2f}s")
                print()
            
            # 2. Model Installation (example)
            print("\n2. 🔽 Model Installation Demo")
            print("-" * 40)
            
            example_model = "llama3.2:1b"  # Small model for demo
            print(f"Checking if {example_model} is available...")
            
            exists = await manager.model_exists(example_model)
            if not exists:
                print(f"Installing {example_model}...")
                
                def progress_callback(progress):
                    if 'status' in progress:
                        print(f"  Status: {progress['status']}")
                
                success = await manager.install_model(
                    example_model, 
                    progress_callback=progress_callback,
                    user_id="demo_user"
                )
                
                if success:
                    print(f"✅ Successfully installed {example_model}")
                else:
                    print(f"❌ Failed to install {example_model}")
            else:
                print(f"✅ {example_model} is already available")
            
            # 3. Model Registry Integration
            print("\n3. 📚 Model Registry Integration")
            print("-" * 40)
            
            # Register model metadata
            if models:
                first_model = models[0]
                metadata = ModelMetadata(
                    name=first_model.name,
                    display_name=f"Enhanced {first_model.name}",
                    description=f"Advanced language model with {first_model.size_category.value} size",
                    version="1.0",
                    model_type=ModelType.CHAT,
                    capabilities=[
                        ModelCapability.TEXT_GENERATION,
                        ModelCapability.CHAT,
                        ModelCapability.INSTRUCTION_FOLLOWING
                    ],
                    size_category=first_model.size_category,
                    parameter_count=first_model.parameter_count,
                    architecture=first_model.architecture,
                    tags=["demo", "example", "chat"]
                )
                
                success = await registry.register_model(metadata, "demo_user")
                if success:
                    print(f"✅ Registered metadata for {first_model.name}")
                else:
                    print(f"❌ Failed to register metadata for {first_model.name}")
                
                # Add a rating
                rating = ModelRating(
                    model_name=first_model.name,
                    user_id="demo_user",
                    rating=4.5,
                    review="Excellent model for general chat tasks. Fast and reliable.",
                    use_case="general_chat",
                    performance_rating=4.0,
                    ease_of_use=5.0,
                    verified_user=True
                )
                
                await registry.add_rating(rating, "demo_user")
                print(f"✅ Added rating for {first_model.name}")
                
                # Add benchmark data
                benchmark = ModelBenchmark(
                    model_name=first_model.name,
                    benchmark_type="chat_quality",
                    score=85.5,
                    metric_name="overall_score",
                    details={
                        "coherence": 88.0,
                        "relevance": 84.0,
                        "helpfulness": 84.5
                    },
                    test_environment="ABOV3_benchmark_suite"
                )
                
                await registry.add_benchmark(benchmark)
                print(f"✅ Added benchmark for {first_model.name}")
            
            # 4. Advanced Search and Filtering
            print("\n4. 🔍 Advanced Search and Filtering")
            print("-" * 40)
            
            # Search for chat models
            search_filter = SearchFilter(
                query="chat",
                model_types=[ModelType.CHAT],
                capabilities=[ModelCapability.CHAT],
                sort_by="name",
                limit=5
            )
            
            search_results = await registry.search_models(search_filter)
            print(f"Found {len(search_results)} chat models:")
            
            for result in search_results:
                print(f"  💬 {result.display_name}")
                print(f"     Type: {result.model_type.value}")
                print(f"     Capabilities: {[cap.value for cap in result.capabilities]}")
                print()
            
            # 5. Model Recommendations
            print("\n5. 🎯 Model Recommendations")
            print("-" * 40)
            
            recommendations = await manager.get_model_recommendations(
                task_type="chat",
                user_preferences={
                    "prefer_speed": True,
                    "max_size_gb": 10.0
                }
            )
            
            print(f"Top {len(recommendations)} recommendations for chat tasks:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. 🏆 {rec.model_name}")
                print(f"     Confidence: {rec.confidence:.2f}")
                print(f"     Reasoning: {'; '.join(rec.reasoning)}")
                
                if rec.estimated_performance:
                    perf = rec.estimated_performance
                    print(f"     Performance: {perf.tokens_per_second:.1f} tokens/sec")
                print()
            
            # 6. Performance Monitoring
            print("\n6. 📊 Performance Monitoring")
            print("-" * 40)
            
            if models:
                model_name = models[0].name
                
                # Simulate performance data
                await manager.update_performance_metrics(
                    model_name=model_name,
                    response_time=2.5,
                    tokens=150,
                    memory_mb=2048.0,
                    cpu_percent=45.0,
                    success=True
                )
                
                # Update usage stats in registry
                await registry.update_usage_stats(
                    model_name=model_name,
                    response_time=2.5,
                    tokens=150,
                    success=True,
                    user_id="demo_user",
                    use_case="chat"
                )
                
                # Get performance metrics
                metrics = await manager.get_performance_metrics(model_name)
                if metrics:
                    print(f"Performance metrics for {model_name}:")
                    print(f"  ⏱️  Average response time: {metrics.response_time_avg:.2f}s")
                    print(f"  🚀 Tokens per second: {metrics.tokens_per_second:.1f}")
                    print(f"  💾 Memory usage: {metrics.memory_usage_mb:.0f} MB")
                    print(f"  🎯 Success rate: {(1 - metrics.error_rate) * 100:.1f}%")
                    print(f"  📈 Total requests: {metrics.total_requests}")
                
                # Get usage stats
                stats = await registry.get_usage_stats(model_name)
                if stats:
                    print(f"\nUsage statistics for {model_name}:")
                    print(f"  📊 Total requests: {stats.total_requests}")
                    print(f"  ✅ Successful: {stats.successful_requests}")
                    print(f"  ❌ Failed: {stats.failed_requests}")
                    print(f"  👥 Unique users: {len(stats.unique_users)}")
                    if stats.popular_use_cases:
                        print(f"  🎯 Popular use cases: {list(stats.popular_use_cases.keys())}")
            
            # 7. Analytics and Insights
            print("\n7. 📈 Analytics and Insights")
            print("-" * 40)
            
            # Get analytics from manager
            analytics = await manager.get_usage_analytics()
            print("System Analytics:")
            print(f"  📊 Total operations: {analytics['total_operations']}")
            print(f"  🏆 Most used model: {analytics['most_used_model']}")
            print(f"  ⚠️  Recent errors: {analytics['recent_errors']}")
            
            if analytics['performance_summary']:
                perf = analytics['performance_summary']
                print(f"  ⏱️  Average response time: {perf.get('average_response_time', 0):.2f}s")
                print(f"  📈 Models tracked: {perf.get('models_tracked', 0)}")
            
            # Get popular models from registry
            popular_models = await registry.get_popular_models(limit=5)
            if popular_models:
                print("\nMost Popular Models:")
                for name, usage in popular_models:
                    print(f"  🔥 {name}: {usage} requests")
            
            # Get top rated models
            top_rated = await registry.get_top_rated_models(limit=5)
            if top_rated:
                print("\nTop Rated Models:")
                for name, rating in top_rated:
                    print(f"  ⭐ {name}: {rating:.1f}/5.0")
            
            # 8. Model Configuration Optimization
            print("\n8. ⚙️  Model Configuration Optimization")
            print("-" * 40)
            
            if models:
                model_name = models[0].name
                optimized_config = await manager.optimize_model_configuration(model_name)
                
                print(f"Optimized configuration for {model_name}:")
                for param, value in optimized_config.items():
                    print(f"  {param}: {value}")
            
            # 9. Export/Import Configuration
            print("\n9. 💾 Export/Import Configuration")
            print("-" * 40)
            
            if models:
                model_name = models[0].name
                
                # Export model configuration
                config_data = await registry.export_model_config(model_name)
                if config_data:
                    print(f"✅ Exported configuration for {model_name}")
                    print(f"  📦 Includes: metadata, ratings, benchmarks, usage stats")
                    print(f"  📅 Export date: {config_data['export_date']}")
                    
                    # Could save to file or send to another system
                    # import json
                    # with open(f"{model_name}_config.json", "w") as f:
                    #     json.dump(config_data, f, indent=2)
                else:
                    print(f"❌ Failed to export configuration for {model_name}")
            
            print("\n🎉 Model Management Demo Complete!")
            print("=" * 50)
            
        finally:
            # Clean up
            registry.close()


async def demonstrate_security_features():
    """Demonstrate security features of model management."""
    
    print("\n🔒 Security Features Demo")
    print("=" * 30)
    
    config = get_config()
    security_manager = SecurityManager()
    
    async with ModelManager(config, security_manager) as manager:
        # Test security validation
        print("\n1. Security Validation")
        print("-" * 20)
        
        # Valid model name
        valid_name = "llama3.2:latest"
        print(f"Testing valid model name: {valid_name}")
        success = await manager.install_model(valid_name, user_id="test_user")
        print(f"Result: {'✅ Allowed' if success else '❌ Blocked'}")
        
        # Invalid model name (would be blocked by security)
        invalid_name = "../../../etc/passwd"
        print(f"Testing invalid model name: {invalid_name}")
        success = await manager.install_model(invalid_name, user_id="test_user")
        print(f"Result: {'❌ Security Issue - Should be blocked' if success else '✅ Properly blocked'}")
        
        print("\n2. Security Logging")
        print("-" * 20)
        
        # Get recent security events
        recent_events = security_manager.logger.get_recent_events(hours=1)
        print(f"Recent security events: {len(recent_events)}")
        
        for event in recent_events[-3:]:  # Show last 3 events
            print(f"  🛡️  {event.event_type}: {event.message}")


async def main():
    """Main demo function."""
    try:
        await demonstrate_model_management()
        await demonstrate_security_features()
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(main())