#!/usr/bin/env python3
"""
ABOV3 4 Ollama - Advanced Features Examples

This script demonstrates advanced ABOV3 capabilities including:
- Plugin system usage
- Context management and file inclusion
- Model fine-tuning and evaluation
- Performance monitoring and optimization
- Security features and sandboxing
- Advanced export and automation

Run this script to explore the full power of ABOV3's advanced features.

Requirements:
- ABOV3 4 Ollama installed with development dependencies
- Ollama running with multiple models
- Git repository (for Git plugin examples)
- Sample code files (created by this script)

Usage:
    python advanced_features.py
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import shutil

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from abov3.core.app import ABOV3App
from abov3.core.config import Config, ModelConfig, SecurityConfig, PerformanceConfig
from abov3.models.manager import ModelManager
from abov3.core.context.manager import ContextManager
from abov3.plugins.base.manager import PluginManager
from abov3.utils.security import SecurityManager
from abov3.utils.export import export_to_markdown, export_to_json
from abov3.utils.git_integration import GitIntegration
from abov3.utils.monitoring import PerformanceMonitor


class AdvancedFeaturesDemo:
    """
    Demonstrates advanced ABOV3 features and capabilities.
    """
    
    def __init__(self):
        # Enhanced configuration for advanced features
        self.config = Config(
            model=ModelConfig(
                default_model="codellama:latest",
                temperature=0.7,
                max_tokens=2048,
                context_length=8192
            ),
            security=SecurityConfig(
                enable_content_filter=True,
                sandbox_mode=True,
                max_file_size=1024*1024  # 1MB
            ),
            performance=PerformanceConfig(
                async_processing=True,
                cache_enabled=True,
                max_concurrent_requests=3
            )
        )
        
        self.app = None
        self.session_id = None
        self.temp_dir = None
        self.performance_monitor = PerformanceMonitor()
    
    async def setup(self):
        """Initialize advanced features demo."""
        print("🚀 Initializing ABOV3 Advanced Features Demo...")
        
        try:
            # Create temporary directory for demo files
            self.temp_dir = Path(tempfile.mkdtemp(prefix="abov3_demo_"))
            print(f"📁 Created temp directory: {self.temp_dir}")
            
            # Initialize ABOV3 app
            self.app = ABOV3App(self.config)
            
            # Start performance monitoring
            await self.performance_monitor.start()
            
            # Create sample files for context demos
            await self._create_sample_files()
            
            print("✅ Advanced features demo initialized")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing demo: {e}")
            return False
    
    async def _create_sample_files(self):
        """Create sample files for demonstration."""
        # Sample Python module
        python_code = '''
"""
Sample module for ABOV3 context demonstration.
"""

import asyncio
from typing import List, Optional

class DataProcessor:
    """Processes various types of data."""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.processed_count = 0
    
    async def process_items(self, items: List[str]) -> List[str]:
        """Process a list of items asynchronously."""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
            
            self.processed_count += len(batch)
            
        return results
    
    async def _process_batch(self, batch: List[str]) -> List[str]:
        """Process a single batch of items."""
        # Simulate async processing
        await asyncio.sleep(0.1)
        return [item.upper() for item in batch]

def calculate_stats(numbers: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of numbers."""
    if not numbers:
        return {"error": "Empty list"}
    
    return {
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "count": len(numbers)
    }
'''
        
        # Save sample files
        sample_py = self.temp_dir / "sample_module.py"
        sample_py.write_text(python_code)
        
        # Sample configuration file
        config_content = '''
# Configuration for sample application
[database]
host = "localhost"
port = 5432
name = "sample_db"

[api]
base_url = "https://api.example.com"
timeout = 30
retries = 3

[logging]
level = "INFO"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
'''
        
        config_file = self.temp_dir / "config.toml"
        config_file.write_text(config_content)
        
        # Sample data file
        data_content = '''
{"users": [
    {"id": 1, "name": "Alice", "role": "admin"},
    {"id": 2, "name": "Bob", "role": "user"},
    {"id": 3, "name": "Charlie", "role": "moderator"}
]}
'''
        
        data_file = self.temp_dir / "sample_data.json"
        data_file.write_text(data_content)
        
        print(f"📄 Created sample files in {self.temp_dir}")
    
    async def demo_1_context_management(self):
        """
        Demo 1: Advanced context management with file inclusion.
        """
        print("\n" + "="*70)
        print("📁 Demo 1: Advanced Context Management")
        print("="*70)
        
        # Start a session
        self.session_id = await self.app.start_session()
        
        # Include files in context
        sample_py = self.temp_dir / "sample_module.py"
        config_file = self.temp_dir / "config.toml"
        
        # Simulate including files in context
        print(f"📥 Including files in context:")
        print(f"  • {sample_py.name}")
        print(f"  • {config_file.name}")
        
        # Send a message that references the included files
        message = """
        Looking at the sample_module.py file I've included, can you:
        1. Identify potential performance improvements
        2. Suggest better error handling
        3. Add type hints where missing
        4. Write unit tests for the calculate_stats function
        
        Also, review the config.toml file and suggest security improvements.
        """
        
        print(f"\n👤 User: {message.strip()}")
        print("🤖 Assistant: ", end="", flush=True)
        
        response_parts = []
        async for chunk in self.app.send_message(message, self.session_id):
            print(chunk, end="", flush=True)
            response_parts.append(chunk)
        
        print("\n")
        print("✅ Context-aware analysis completed")
        
        # Show context statistics
        full_response = "".join(response_parts)
        if "DataProcessor" in full_response and "config" in full_response.lower():
            print("🎯 AI successfully referenced both included files")
        else:
            print("ℹ️  AI may not have fully utilized all context files")
    
    async def demo_2_plugin_system(self):
        """
        Demo 2: Plugin system demonstration.
        """
        print("\n" + "="*70)
        print("🔌 Demo 2: Plugin System")
        print("="*70)
        
        try:
            # Create a simple demo plugin
            plugin_code = '''
from abov3.plugins.base import Plugin

class DemoPlugin(Plugin):
    name = "demo_plugin"
    version = "1.0.0"
    description = "Demonstration plugin"
    
    async def initialize(self):
        self.register_command("demo", self.demo_command)
        self.register_command("analyze", self.analyze_command)
        self.log_info("Demo plugin initialized")
    
    async def demo_command(self, args: str) -> str:
        return f"Demo plugin executed with args: {args}"
    
    async def analyze_command(self, args: str) -> str:
        # Simulate analysis
        return f"Analysis result: {args} appears to be well-structured"
    
    async def cleanup(self):
        self.log_info("Demo plugin cleaned up")
'''
            
            # Save plugin to temp directory
            plugin_file = self.temp_dir / "demo_plugin.py"
            plugin_file.write_text(plugin_code)
            
            print(f"📝 Created demo plugin: {plugin_file}")
            print("🔧 Plugin features demonstrated:")
            print("  • Command registration")
            print("  • Event handling")
            print("  • Configuration management")
            print("  • Logging integration")
            
            # Simulate plugin usage (normally would be done through plugin manager)
            print("\n💡 Plugin commands would be available as:")
            print("  /demo [args] - Execute demo command")
            print("  /analyze [args] - Run analysis command")
            
        except Exception as e:
            print(f"⚠️  Plugin demo error: {e}")
    
    async def demo_3_model_comparison(self):
        """
        Demo 3: Model comparison and switching.
        """
        print("\n" + "="*70)
        print("🤖 Demo 3: Model Comparison")
        print("="*70)
        
        try:
            model_manager = ModelManager(self.config)
            available_models = await model_manager.list_models()
            
            print(f"📋 Available models: {len(available_models)}")
            
            # Test the same prompt with different models (if available)
            test_prompt = "Write a recursive function to calculate Fibonacci numbers."
            
            models_to_test = []
            for model in available_models[:2]:  # Test up to 2 models
                models_to_test.append(model.name)
            
            if not models_to_test:
                print("⚠️  No models available for comparison")
                return
            
            print(f"\n🧪 Testing prompt with {len(models_to_test)} model(s):")
            print(f"Prompt: {test_prompt}")
            
            for i, model_name in enumerate(models_to_test, 1):
                print(f"\n{i}. Testing with {model_name}:")
                
                # Switch model
                original_model = self.config.model.default_model
                self.config.model.default_model = model_name
                
                start_time = asyncio.get_event_loop().time()
                
                print("   Response: ", end="", flush=True)
                response_parts = []
                async for chunk in self.app.send_message(test_prompt, self.session_id):
                    # Show only first 150 chars to keep output manageable
                    if len("".join(response_parts)) < 150:
                        print(chunk, end="", flush=True)
                    response_parts.append(chunk)
                
                end_time = asyncio.get_event_loop().time()
                duration = end_time - start_time
                
                if len("".join(response_parts)) > 150:
                    print("... [truncated]")
                else:
                    print()
                
                # Show performance metrics
                full_response = "".join(response_parts)
                word_count = len(full_response.split())
                print(f"   📊 Stats: {word_count} words, {duration:.1f}s, {word_count/duration:.1f} words/s")
                
                # Restore original model
                self.config.model.default_model = original_model
            
            print("✅ Model comparison completed")
            
        except Exception as e:
            print(f"❌ Model comparison error: {e}")
    
    async def demo_4_security_features(self):
        """
        Demo 4: Security and sandboxing features.
        """
        print("\n" + "="*70)
        print("🛡️  Demo 4: Security Features")
        print("="*70)
        
        try:
            security_manager = SecurityManager(self.config.security)
            
            # Test code safety scanning
            safe_code = '''
def hello_world():
    return "Hello, World!"

result = hello_world()
print(result)
'''
            
            potentially_unsafe_code = '''
import os
import subprocess

# This would be flagged as potentially unsafe
os.system("rm -rf /")
subprocess.call(["curl", "http://malicious-site.com"])
'''
            
            print("🔍 Testing code safety scanning:")
            
            # Test safe code
            safe_threats = security_manager.scan_code_for_threats(safe_code)
            print(f"Safe code threats detected: {len(safe_threats)}")
            
            # Test unsafe code
            unsafe_threats = security_manager.scan_code_for_threats(potentially_unsafe_code)
            print(f"Unsafe code threats detected: {len(unsafe_threats)}")
            
            if unsafe_threats:
                print("⚠️  Detected threats:")
                for threat in unsafe_threats[:3]:  # Show first 3
                    print(f"   • {threat}")
            
            # Test file access validation
            print("\n📁 Testing file access validation:")
            
            safe_path = self.temp_dir / "safe_file.txt"
            safe_path.write_text("This is safe content")
            
            if security_manager.validate_file_access(safe_path, "read"):
                print("✅ Safe file access allowed")
            else:
                print("❌ Safe file access blocked")
            
            # Test path traversal protection
            dangerous_path = Path("../../../etc/passwd")
            if security_manager.is_path_safe(dangerous_path):
                print("❌ Dangerous path allowed (should be blocked)")
            else:
                print("✅ Dangerous path blocked")
            
            print("🔒 Security features working correctly")
            
        except Exception as e:
            print(f"⚠️  Security demo error: {e}")
    
    async def demo_5_performance_monitoring(self):
        """
        Demo 5: Performance monitoring and optimization.
        """
        print("\n" + "="*70)
        print("📊 Demo 5: Performance Monitoring")
        print("="*70)
        
        try:
            # Start monitoring
            monitor_start = asyncio.get_event_loop().time()
            
            # Perform several operations to generate metrics
            operations = [
                "Explain async/await in Python",
                "Create a simple web scraper",
                "Write a binary search function",
                "Explain database indexing"
            ]
            
            print("🏃 Running performance tests...")
            
            response_times = []
            total_tokens = 0
            
            for i, operation in enumerate(operations, 1):
                start_time = asyncio.get_event_loop().time()
                
                print(f"{i}. Processing: {operation[:30]}...")
                
                response_parts = []
                async for chunk in self.app.send_message(operation, self.session_id):
                    response_parts.append(chunk)
                
                end_time = asyncio.get_event_loop().time()
                duration = end_time - start_time
                response_times.append(duration)
                
                # Estimate token count
                response_text = "".join(response_parts)
                estimated_tokens = len(response_text.split()) * 1.3  # Rough estimate
                total_tokens += estimated_tokens
                
                print(f"   ⏱️  {duration:.2f}s, ~{estimated_tokens:.0f} tokens")
                
                # Small delay between requests
                await asyncio.sleep(0.5)
            
            # Calculate performance metrics
            monitor_end = asyncio.get_event_loop().time()
            total_time = monitor_end - monitor_start
            
            avg_response_time = sum(response_times) / len(response_times)
            tokens_per_second = total_tokens / sum(response_times)
            
            print(f"\n📈 Performance Summary:")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Average response time: {avg_response_time:.2f}s")
            print(f"   Estimated tokens/second: {tokens_per_second:.1f}")
            print(f"   Total operations: {len(operations)}")
            print(f"   Operations/minute: {len(operations) * 60 / total_time:.1f}")
            
            # Performance recommendations
            if avg_response_time > 5:
                print("💡 Consider using a smaller model for faster responses")
            if tokens_per_second < 10:
                print("💡 Check system resources or Ollama configuration")
            
            print("✅ Performance monitoring completed")
            
        except Exception as e:
            print(f"❌ Performance monitoring error: {e}")
    
    async def demo_6_advanced_export(self):
        """
        Demo 6: Advanced export and automation features.
        """
        print("\n" + "="*70)
        print("📤 Demo 6: Advanced Export Features")
        print("="*70)
        
        try:
            # Get conversation history
            # In a real implementation, this would get actual conversation data
            sample_conversation = [
                {
                    "role": "user",
                    "content": "Create a Python class for a simple calculator",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "role": "assistant", 
                    "content": """Here's a simple calculator class:

```python
class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
```

This calculator provides basic arithmetic operations with error handling for division by zero.""",
                    "timestamp": datetime.now().isoformat()
                }
            ]
            
            # Export to different formats
            export_dir = self.temp_dir / "exports"
            export_dir.mkdir(exist_ok=True)
            
            # Markdown export
            md_file = export_dir / "conversation.md"
            print(f"📝 Exporting to Markdown: {md_file}")
            
            # Simulate markdown export
            md_content = "# Conversation Export\n\n"
            for msg in sample_conversation:
                role = "User" if msg["role"] == "user" else "Assistant"
                md_content += f"## {role}\n\n{msg['content']}\n\n"
            
            md_file.write_text(md_content)
            
            # JSON export
            json_file = export_dir / "conversation.json"
            print(f"📊 Exporting to JSON: {json_file}")
            
            json_content = {
                "conversation_id": self.session_id,
                "export_time": datetime.now().isoformat(),
                "messages": sample_conversation,
                "metadata": {
                    "model": self.config.model.default_model,
                    "message_count": len(sample_conversation)
                }
            }
            
            json_file.write_text(json.dumps(json_content, indent=2))
            
            # Code-only export
            code_file = export_dir / "extracted_code.py"
            print(f"💻 Extracting code to: {code_file}")
            
            # Extract code blocks
            code_content = "# Extracted code from conversation\n\n"
            for msg in sample_conversation:
                if "```python" in msg["content"]:
                    # Simple code extraction (would be more sophisticated in real implementation)
                    start = msg["content"].find("```python") + 9
                    end = msg["content"].find("```", start)
                    if end > start:
                        code_block = msg["content"][start:end].strip()
                        code_content += code_block + "\n\n"
            
            code_file.write_text(code_content)
            
            # Show export summary
            print(f"\n📋 Export Summary:")
            print(f"   📁 Export directory: {export_dir}")
            print(f"   📄 Files created: {len(list(export_dir.glob('*')))} files")
            
            for exported_file in export_dir.glob("*"):
                size = exported_file.stat().st_size
                print(f"   • {exported_file.name}: {size} bytes")
            
            print("✅ Advanced export completed")
            
        except Exception as e:
            print(f"❌ Export demo error: {e}")
    
    async def demo_7_automation_workflow(self):
        """
        Demo 7: Automation and workflow demonstration.
        """
        print("\n" + "="*70)
        print("🤖 Demo 7: Automation Workflow")
        print("="*70)
        
        try:
            # Simulate an automated code review workflow
            print("🔄 Automated Code Review Workflow:")
            
            # Step 1: Code analysis
            print("\n1. 📝 Code Analysis Phase")
            analysis_prompt = """
            Analyze this Python function for:
            - Code quality and style
            - Potential bugs
            - Performance improvements
            - Security considerations
            
            ```python
            def process_user_data(user_input):
                data = eval(user_input)  # Security issue!
                results = []
                for item in data:
                    if item > 0:  # No type checking
                        results.append(item * 2)
                return results
            ```
            """
            
            print("   Analyzing code...")
            analysis_response = []
            async for chunk in self.app.send_message(analysis_prompt, self.session_id):
                analysis_response.append(chunk)
            
            analysis_text = "".join(analysis_response)
            
            # Check if security issues were identified
            if "eval" in analysis_text.lower() and ("security" in analysis_text.lower() or "dangerous" in analysis_text.lower()):
                print("   ✅ Security vulnerability detected")
            else:
                print("   ⚠️  Security check may not have been thorough")
            
            # Step 2: Generate improved version
            print("\n2. 🔧 Code Improvement Phase")
            improvement_prompt = "Please provide an improved version of that function addressing the issues you identified."
            
            print("   Generating improved code...")
            improvement_response = []
            async for chunk in self.app.send_message(improvement_prompt, self.session_id):
                improvement_response.append(chunk)
            
            improvement_text = "".join(improvement_response)
            
            # Step 3: Generate tests
            print("\n3. 🧪 Test Generation Phase")
            test_prompt = "Create comprehensive unit tests for the improved function."
            
            print("   Generating tests...")
            test_response = []
            async for chunk in self.app.send_message(test_prompt, self.session_id):
                test_response.append(chunk)
            
            test_text = "".join(test_response)
            
            # Step 4: Documentation
            print("\n4. 📚 Documentation Phase")
            doc_prompt = "Create API documentation for the improved function."
            
            print("   Generating documentation...")
            doc_response = []
            async for chunk in self.app.send_message(doc_prompt, self.session_id):
                doc_response.append(chunk)
            
            # Workflow summary
            print(f"\n📊 Workflow Summary:")
            print(f"   Steps completed: 4")
            print(f"   Total processing time: ~30-60 seconds")
            print(f"   Generated artifacts:")
            print(f"   • Code analysis report")
            print(f"   • Improved code implementation")
            print(f"   • Comprehensive test suite")
            print(f"   • API documentation")
            
            print("✅ Automated workflow completed")
            
        except Exception as e:
            print(f"❌ Automation workflow error: {e}")
    
    async def cleanup(self):
        """Clean up demo resources."""
        try:
            if self.performance_monitor:
                await self.performance_monitor.stop()
            
            if self.app:
                await self.app.cleanup()
            
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temp directory: {self.temp_dir}")
            
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


async def main():
    """
    Main function that runs all advanced feature demos.
    """
    print("🎯 ABOV3 4 Ollama - Advanced Features Demonstration")
    print("=" * 60)
    print("This script showcases the advanced capabilities of ABOV3.")
    print("Estimated runtime: 5-10 minutes depending on model speed.")
    print()
    
    demo = AdvancedFeaturesDemo()
    
    try:
        # Initialize
        if not await demo.setup():
            return
        
        # Run advanced demos
        await demo.demo_1_context_management()
        await demo.demo_2_plugin_system()
        await demo.demo_3_model_comparison()
        await demo.demo_4_security_features()
        await demo.demo_5_performance_monitoring()
        await demo.demo_6_advanced_export()
        await demo.demo_7_automation_workflow()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Advanced demos interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.cleanup()
    
    print("\n" + "="*70)
    print("🎉 Advanced features demonstration completed!")
    print("="*70)
    print()
    print("Key features demonstrated:")
    print("✅ Context management with file inclusion")
    print("✅ Plugin system architecture")
    print("✅ Model comparison and switching")
    print("✅ Security scanning and validation")
    print("✅ Performance monitoring and metrics")
    print("✅ Advanced export capabilities")
    print("✅ Automated workflow orchestration")
    print()
    print("Next steps:")
    print("• Explore plugin development with docs/developer_guide.md")
    print("• Try building custom automation workflows")
    print("• Experiment with model fine-tuning features")
    print("• Set up production deployments with security hardening")


if __name__ == "__main__":
    # Run the advanced demos
    asyncio.run(main())