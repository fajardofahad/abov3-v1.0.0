"""
Example usage of the Ollama API client.

This script demonstrates various features of the OllamaClient including
model management, chat completion, streaming, and error handling.
"""

import asyncio
import logging
from typing import List

from abov3.core.api.ollama_client import (
    OllamaClient,
    ChatMessage,
    RetryConfig,
    get_ollama_client,
    quick_chat,
    quick_generate,
)
from abov3.core.config import get_config
from abov3.core.api.exceptions import APIError, ModelNotFoundError


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_example():
    """Basic usage example."""
    print("=== Basic Example ===")
    
    # Get configuration
    config = get_config()
    
    # Use context manager for automatic cleanup
    async with get_ollama_client(config) as client:
        # Check if server is healthy
        if not await client.health_check():
            print("Error: Ollama server is not responding")
            return
        
        # List available models
        models = await client.list_models()
        print(f"Available models: {[model.name for model in models]}")
        
        if not models:
            print("No models available. Please pull a model first.")
            return
        
        # Use the first available model or default
        model_name = models[0].name
        print(f"Using model: {model_name}")
        
        # Simple chat completion
        messages = [
            ChatMessage(role="user", content="Hello! How are you?")
        ]
        
        response = await client.chat(model_name, messages)
        print(f"Response: {response.message.content}")
        
        # Text generation
        generated_text = await client.generate(
            model_name,
            "Write a short poem about programming:",
            temperature=0.8
        )
        print(f"Generated text: {generated_text}")


async def streaming_example():
    """Streaming response example."""
    print("\n=== Streaming Example ===")
    
    async with get_ollama_client() as client:
        models = await client.list_models()
        if not models:
            print("No models available.")
            return
        
        model_name = models[0].name
        
        # Streaming chat
        messages = [
            ChatMessage(
                role="user",
                content="Tell me a story about a robot learning to code. Keep it short."
            )
        ]
        
        print("Streaming chat response:")
        async for chunk in await client.chat(model_name, messages, stream=True):
            if chunk.message.content:
                print(chunk.message.content, end="", flush=True)
        print("\n")
        
        # Streaming generation
        print("Streaming generation:")
        async for chunk in await client.generate(
            model_name,
            "Explain machine learning in simple terms:",
            stream=True,
            temperature=0.7
        ):
            print(chunk, end="", flush=True)
        print("\n")


async def model_management_example():
    """Model management example."""
    print("\n=== Model Management Example ===")
    
    async with get_ollama_client() as client:
        # List models with details
        models = await client.list_models()
        print("Detailed model information:")
        for model in models:
            print(f"  Name: {model.name}")
            print(f"  Size: {model.size / (1024**3):.2f} GB")
            print(f"  Modified: {model.modified_at}")
            print(f"  Digest: {model.digest[:16]}...")
            print()
        
        # Check if specific model exists
        test_model = "llama3.2:latest"
        exists = await client.model_exists(test_model)
        print(f"Model '{test_model}' exists: {exists}")
        
        # Get model info
        if exists:
            info = await client.get_model_info(test_model)
            if info:
                print(f"Model info: {info.name} ({info.size} bytes)")


async def error_handling_example():
    """Error handling and retry example."""
    print("\n=== Error Handling Example ===")
    
    # Custom retry configuration
    retry_config = RetryConfig(
        max_retries=2,
        base_delay=0.5,
        max_delay=5.0
    )
    
    async with get_ollama_client(retry_config=retry_config) as client:
        try:
            # Try to use a non-existent model
            messages = [ChatMessage(role="user", content="Hello")]
            await client.chat("non-existent-model", messages)
        except ModelNotFoundError as e:
            print(f"Expected error: {e}")
        
        try:
            # Try invalid parameters
            await client.generate(
                "any-model",
                "test",
                temperature=5.0  # Invalid temperature
            )
        except Exception as e:
            print(f"Validation error: {e}")


async def parameter_validation_example():
    """Parameter validation example."""
    print("\n=== Parameter Validation Example ===")
    
    async with get_ollama_client() as client:
        # Get current model parameters
        params = client.get_model_params()
        print(f"Current model parameters: {params}")
        
        # Update parameters
        client.update_model_params(
            temperature=0.9,
            top_p=0.95,
            max_tokens=2048
        )
        
        new_params = client.get_model_params()
        print(f"Updated parameters: {new_params}")
        
        models = await client.list_models()
        if models:
            model_name = models[0].name
            
            # Use custom parameters
            response = await client.generate(
                model_name,
                "Write a creative short story opening:",
                temperature=1.2,  # High creativity
                top_p=0.9,
                max_tokens=150
            )
            print(f"Creative response: {response[:200]}...")


async def convenience_functions_example():
    """Convenience functions example."""
    print("\n=== Convenience Functions Example ===")
    
    try:
        # Quick chat
        response = await quick_chat("What is Python?")
        print(f"Quick chat response: {response[:100]}...")
        
        # Quick generate
        generated = await quick_generate("Complete this sentence: Artificial intelligence is")
        print(f"Quick generation: {generated[:100]}...")
        
    except APIError as e:
        print(f"API error: {e}")


async def embedding_example():
    """Embedding generation example."""
    print("\n=== Embedding Example ===")
    
    async with get_ollama_client() as client:
        models = await client.list_models()
        if not models:
            print("No models available.")
            return
        
        # Use first available model (note: not all models support embeddings)
        model_name = models[0].name
        
        try:
            texts = [
                "Machine learning is a subset of artificial intelligence.",
                "Python is a popular programming language.",
                "Data science involves analyzing large datasets."
            ]
            
            embeddings = await client.embed(model_name, texts)
            print(f"Generated embeddings for {len(texts)} texts")
            print(f"Embedding dimensions: {len(embeddings[0]) if embeddings and embeddings[0] else 'N/A'}")
            
        except APIError as e:
            print(f"Embedding error (model may not support embeddings): {e}")


async def main():
    """Run all examples."""
    try:
        await basic_example()
        await streaming_example()
        await model_management_example()
        await error_handling_example()
        await parameter_validation_example()
        await convenience_functions_example()
        await embedding_example()
        
        print("\n=== All Examples Completed ===")
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())