#!/usr/bin/env python3
"""
ABOV3 4 Ollama - Basic Usage Examples

This script demonstrates the fundamental features of ABOV3 including:
- Basic chat interactions
- Model management
- Configuration handling
- Simple automation tasks

Run this script to see ABOV3 in action with common use cases.

Requirements:
- ABOV3 4 Ollama installed
- Ollama running with at least one model available
- Python 3.8+

Usage:
    python basic_usage.py
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from abov3.core.app import ABOV3App
from abov3.core.config import Config, ModelConfig, OllamaConfig
from abov3.models.manager import ModelManager
from abov3.core.api.ollama_client import ChatMessage, get_ollama_client


class BasicUsageExamples:
    """
    Demonstrates basic ABOV3 usage patterns.
    """
    
    def __init__(self):
        # Create a configuration for examples
        self.config = Config(
            model=ModelConfig(
                default_model="llama3.2:latest",
                temperature=0.7,
                max_tokens=1024
            ),
            ollama=OllamaConfig(
                host="http://localhost:11434",
                timeout=60
            )
        )
        
        self.app = None
        self.session_id = None
    
    async def setup(self):
        """Initialize the ABOV3 application."""
        print("🚀 Initializing ABOV3...")
        
        try:
            self.app = ABOV3App(self.config)
            
            # Check if Ollama is accessible
            client = get_ollama_client(self.config)
            if not await client.health_check():
                print("❌ Error: Cannot connect to Ollama server")
                print("   Please ensure Ollama is running on http://localhost:11434")
                return False
            
            print("✅ ABOV3 initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing ABOV3: {e}")
            return False
    
    async def example_1_simple_chat(self):
        """
        Example 1: Simple chat interaction
        
        Demonstrates how to send a message and receive a response.
        """
        print("\n" + "="*60)
        print("📝 Example 1: Simple Chat Interaction")
        print("="*60)
        
        # Start a new session
        self.session_id = await self.app.start_session()
        print(f"Started session: {self.session_id}")
        
        # Send a simple message
        message = "Hello! Can you explain what Python decorators are in simple terms?"
        print(f"\n👤 User: {message}")
        print("🤖 Assistant: ", end="", flush=True)
        
        # Stream the response
        response_parts = []
        async for chunk in self.app.send_message(message, self.session_id):
            print(chunk, end="", flush=True)
            response_parts.append(chunk)
        
        print("\n")
        
        # Show some metadata
        full_response = "".join(response_parts)
        word_count = len(full_response.split())
        print(f"📊 Response stats: {word_count} words, {len(full_response)} characters")
    
    async def example_2_code_generation(self):
        """
        Example 2: Code generation
        
        Shows how to request code generation and handle the response.
        """
        print("\n" + "="*60)
        print("💻 Example 2: Code Generation")
        print("="*60)
        
        # Request code generation
        message = """
        Create a Python function that:
        1. Takes a list of numbers as input
        2. Returns a dictionary with statistics (mean, median, mode)
        3. Includes proper error handling
        4. Has type hints and docstring
        """
        
        print(f"👤 User: {message.strip()}")
        print("🤖 Assistant: ", end="", flush=True)
        
        response_parts = []
        async for chunk in self.app.send_message(message, self.session_id):
            print(chunk, end="", flush=True)
            response_parts.append(chunk)
        
        print("\n")
        
        # Check if response contains code
        full_response = "".join(response_parts)
        if "def " in full_response and "```" in full_response:
            print("✅ Code block detected in response")
        else:
            print("ℹ️  No code block detected")
    
    async def example_3_follow_up_questions(self):
        """
        Example 3: Follow-up questions
        
        Demonstrates context retention in conversations.
        """
        print("\n" + "="*60)
        print("🔄 Example 3: Follow-up Questions")
        print("="*60)
        
        # Ask a follow-up question that relies on context
        message = "Can you add unit tests for that function you just created?"
        print(f"👤 User: {message}")
        print("🤖 Assistant: ", end="", flush=True)
        
        response_parts = []
        async for chunk in self.app.send_message(message, self.session_id):
            print(chunk, end="", flush=True)
            response_parts.append(chunk)
        
        print("\n")
        
        # Check if the AI understood the context
        full_response = "".join(response_parts)
        if any(keyword in full_response.lower() for keyword in ["test", "assert", "pytest"]):
            print("✅ Context maintained - AI understood the follow-up")
        else:
            print("⚠️  Context may not have been fully maintained")
    
    async def example_4_model_information(self):
        """
        Example 4: Model management
        
        Shows how to get information about available models.
        """
        print("\n" + "="*60)
        print("🤖 Example 4: Model Information")
        print("="*60)
        
        try:
            model_manager = ModelManager(self.config)
            
            # List available models
            models = await model_manager.list_models()
            print(f"📋 Available models ({len(models)}):")
            
            for i, model in enumerate(models[:5], 1):  # Show first 5 models
                print(f"  {i}. {model.name}")
                if hasattr(model, 'size') and model.size:
                    size_gb = model.size / (1024**3) if model.size > 1024**3 else model.size / (1024**2)
                    unit = "GB" if model.size > 1024**3 else "MB"
                    print(f"     Size: {size_gb:.1f} {unit}")
            
            if len(models) > 5:
                print(f"  ... and {len(models) - 5} more")
            
            # Show current model info
            current_model = self.config.model.default_model
            print(f"\n🎯 Current model: {current_model}")
            
            # Check if the model is available
            if model_manager.is_model_available(current_model):
                print("✅ Model is available and ready")
            else:
                print("⚠️  Model may not be available")
                
        except Exception as e:
            print(f"❌ Error accessing model information: {e}")
    
    async def example_5_configuration_demo(self):
        """
        Example 5: Configuration management
        
        Demonstrates how to work with configuration settings.
        """
        print("\n" + "="*60)
        print("⚙️  Example 5: Configuration Management")
        print("="*60)
        
        # Show current configuration
        print("📋 Current configuration:")
        print(f"  Model: {self.config.model.default_model}")
        print(f"  Temperature: {self.config.model.temperature}")
        print(f"  Max tokens: {self.config.model.max_tokens}")
        print(f"  Ollama host: {self.config.ollama.host}")
        
        # Demonstrate configuration modification
        print("\n🔧 Modifying temperature for next request...")
        original_temp = self.config.model.temperature
        self.config.model.temperature = 0.3  # More focused responses
        
        # Send a message with the new setting
        message = "Write a haiku about artificial intelligence."
        print(f"👤 User: {message}")
        print("🤖 Assistant (temp=0.3): ", end="", flush=True)
        
        response_parts = []
        async for chunk in self.app.send_message(message, self.session_id):
            print(chunk, end="", flush=True)
            response_parts.append(chunk)
        
        print("\n")
        
        # Restore original temperature
        self.config.model.temperature = original_temp
        print(f"🔄 Temperature restored to {original_temp}")
    
    async def example_6_error_handling(self):
        """
        Example 6: Error handling
        
        Shows how ABOV3 handles various error conditions.
        """
        print("\n" + "="*60)
        print("🛡️  Example 6: Error Handling")
        print("="*60)
        
        # Test with a very long message (potential token limit issue)
        long_message = "Please analyze this: " + "word " * 1000  # Very long message
        print("👤 User: [Sending very long message to test limits...]")
        
        try:
            print("🤖 Assistant: ", end="", flush=True)
            response_parts = []
            async for chunk in self.app.send_message(long_message, self.session_id):
                print(chunk, end="", flush=True)
                response_parts.append(chunk)
            
            print("\n✅ Long message handled successfully")
            
        except Exception as e:
            print(f"\n⚠️  Error handled: {type(e).__name__}: {e}")
            print("💡 This demonstrates ABOV3's error handling capabilities")
    
    async def example_7_batch_processing(self):
        """
        Example 7: Batch processing
        
        Demonstrates processing multiple related requests.
        """
        print("\n" + "="*60)
        print("📦 Example 7: Batch Processing")
        print("="*60)
        
        # List of programming concepts to explain
        concepts = [
            "list comprehensions",
            "lambda functions", 
            "context managers"
        ]
        
        print("Processing multiple requests:")
        
        for i, concept in enumerate(concepts, 1):
            message = f"Explain {concept} in Python with a simple example."
            print(f"\n{i}. 👤 User: {message}")
            print(f"   🤖 Assistant: ", end="", flush=True)
            
            response_parts = []
            async for chunk in self.app.send_message(message, self.session_id):
                # Only show first 100 characters to keep output manageable
                if len("".join(response_parts)) < 100:
                    print(chunk, end="", flush=True)
                response_parts.append(chunk)
            
            if len("".join(response_parts)) > 100:
                print("... [truncated for brevity]")
            else:
                print()
            
            # Small delay between requests
            await asyncio.sleep(0.5)
        
        print("\n✅ Batch processing completed")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.app:
            try:
                await self.app.cleanup()
                print("🧹 Cleanup completed")
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}")


async def main():
    """
    Main function that runs all examples.
    """
    print("🎯 ABOV3 4 Ollama - Basic Usage Examples")
    print("=" * 50)
    print("This script demonstrates basic ABOV3 functionality.")
    print("Make sure Ollama is running before proceeding.")
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ required")
        return
    
    examples = BasicUsageExamples()
    
    try:
        # Initialize ABOV3
        if not await examples.setup():
            return
        
        # Run examples
        await examples.example_1_simple_chat()
        await examples.example_2_code_generation()
        await examples.example_3_follow_up_questions()
        await examples.example_4_model_information()
        await examples.example_5_configuration_demo()
        await examples.example_6_error_handling()
        await examples.example_7_batch_processing()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await examples.cleanup()
    
    print("\n" + "="*60)
    print("🎉 Basic usage examples completed!")
    print("="*60)
    print()
    print("Next steps:")
    print("• Try 'abov3 chat' for interactive sessions")
    print("• Run 'python advanced_features.py' for advanced examples")
    print("• Read the documentation at docs/user_guide.md")
    print("• Explore plugins and customization options")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())