# Ollama API Client Usage Guide

This guide provides comprehensive documentation for using the ABOV3 Ollama API client, a production-ready async client for interacting with Ollama models.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Client Configuration](#client-configuration)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Error Handling](#error-handling)
6. [Performance Optimization](#performance-optimization)
7. [Examples](#examples)
8. [API Reference](#api-reference)

## Quick Start

### Installation

The Ollama client is included with ABOV3. Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

### Basic Example

```python
import asyncio
from abov3.core.api import OllamaClient, ChatMessage, get_ollama_client

async def main():
    # Using context manager (recommended)
    async with get_ollama_client() as client:
        # Simple chat
        messages = [ChatMessage(role="user", content="Hello!")]
        response = await client.chat("llama3.2:latest", messages)
        print(response.message.content)

asyncio.run(main())
```

### Convenience Functions

For simple use cases, use the convenience functions:

```python
import asyncio
from abov3.core.api import quick_chat, quick_generate

async def main():
    # Quick chat
    response = await quick_chat("What is Python?")
    print(response)
    
    # Quick generation
    text = await quick_generate("Complete this sentence: Machine learning is")
    print(text)

asyncio.run(main())
```

## Client Configuration

### Using Default Configuration

The client automatically loads configuration from the global config:

```python
from abov3.core.api import get_ollama_client

async with get_ollama_client() as client:
    # Uses default configuration
    pass
```

### Custom Configuration

```python
from abov3.core.config import Config, OllamaConfig, ModelConfig
from abov3.core.api import OllamaClient

# Create custom configuration
config = Config(
    ollama=OllamaConfig(
        host="http://localhost:11434",
        timeout=120,
        max_retries=3
    ),
    model=ModelConfig(
        default_model="llama3.2:latest",
        temperature=0.7,
        max_tokens=2048
    )
)

client = OllamaClient(config=config)
```

### Retry Configuration

```python
from abov3.core.api import OllamaClient, RetryConfig

retry_config = RetryConfig(
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True
)

client = OllamaClient(retry_config=retry_config)
```

## Basic Usage

### Model Management

#### List Available Models

```python
async with get_ollama_client() as client:
    models = await client.list_models()
    for model in models:
        print(f"Model: {model.name}")
        print(f"Size: {model.size / (1024**3):.2f} GB")
        print(f"Modified: {model.modified_at}")
        print()
```

#### Check Model Existence

```python
async with get_ollama_client() as client:
    exists = await client.model_exists("llama3.2:latest")
    if not exists:
        print("Model not found, pulling...")
        await client.pull_model("llama3.2:latest")
```

#### Pull Models

```python
async with get_ollama_client() as client:
    def progress_callback(chunk):
        if "status" in chunk:
            print(f"Status: {chunk['status']}")
        if "completed" in chunk and "total" in chunk:
            percent = (chunk["completed"] / chunk["total"]) * 100
            print(f"Progress: {percent:.1f}%")
    
    success = await client.pull_model("llama3.2:latest", progress_callback)
    print(f"Pull successful: {success}")
```

#### Delete Models

```python
async with get_ollama_client() as client:
    success = await client.delete_model("old-model:latest")
    print(f"Delete successful: {success}")
```

### Chat Completion

#### Simple Chat

```python
from abov3.core.api import ChatMessage

async with get_ollama_client() as client:
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Explain quantum computing.")
    ]
    
    response = await client.chat("llama3.2:latest", messages)
    print(response.message.content)
    print(f"Tokens used: {response.eval_count}")
```

#### Chat with Custom Parameters

```python
async with get_ollama_client() as client:
    messages = [ChatMessage(role="user", content="Write a creative story.")]
    
    response = await client.chat(
        "llama3.2:latest",
        messages,
        temperature=1.2,  # High creativity
        top_p=0.9,
        max_tokens=1000
    )
    print(response.message.content)
```

#### Streaming Chat

```python
async with get_ollama_client() as client:
    messages = [ChatMessage(role="user", content="Tell me a long story.")]
    
    print("Streaming response:")
    async for chunk in await client.chat("llama3.2:latest", messages, stream=True):
        if chunk.message.content:
            print(chunk.message.content, end="", flush=True)
        if chunk.done:
            print(f"\nTotal duration: {chunk.total_duration}ns")
```

### Text Generation

#### Simple Generation

```python
async with get_ollama_client() as client:
    text = await client.generate(
        "llama3.2:latest",
        "Complete this code:\ndef fibonacci(n):"
    )
    print(text)
```

#### Streaming Generation

```python
async with get_ollama_client() as client:
    prompt = "Explain the benefits of async programming:"
    
    async for chunk in await client.generate("llama3.2:latest", prompt, stream=True):
        print(chunk, end="", flush=True)
    print()
```

### Embeddings

```python
async with get_ollama_client() as client:
    # Single text
    embeddings = await client.embed("llama3.2:latest", "Hello world")
    print(f"Embedding dimensions: {len(embeddings[0])}")
    
    # Multiple texts
    texts = ["Text 1", "Text 2", "Text 3"]
    embeddings = await client.embed("llama3.2:latest", texts)
    print(f"Generated {len(embeddings)} embeddings")
```

## Advanced Features

### Connection Pooling

The client automatically manages connection pooling:

```python
import aiohttp
from abov3.core.api import OllamaClient

# Custom session with specific settings
connector = aiohttp.TCPConnector(
    limit=200,  # Total connection pool size
    limit_per_host=20,  # Per-host limit
    keepalive_timeout=60
)

session = aiohttp.ClientSession(connector=connector)
client = OllamaClient(session=session)

# Use client...
await client.close()  # Don't forget to close
await session.close()
```

### Parameter Validation

The client automatically validates model parameters:

```python
async with get_ollama_client() as client:
    try:
        await client.chat(
            "llama3.2:latest",
            messages,
            temperature=5.0  # Invalid - will raise ValidationError
        )
    except ValidationError as e:
        print(f"Parameter error: {e}")
```

### Model Parameter Management

```python
async with get_ollama_client() as client:
    # Get current parameters
    params = client.get_model_params()
    print(f"Current parameters: {params}")
    
    # Update parameters
    client.update_model_params(
        temperature=0.9,
        top_p=0.95,
        max_tokens=2048
    )
    
    # Parameters will be used in subsequent calls
    response = await client.chat("llama3.2:latest", messages)
```

### Health Monitoring

```python
async with get_ollama_client() as client:
    if await client.health_check():
        print("Ollama server is healthy")
    else:
        print("Ollama server is not responding")
```

## Error Handling

### Exception Types

The client provides specific exception types:

```python
from abov3.core.api.exceptions import (
    APIError,
    ConnectionError,
    ModelNotFoundError,
    ValidationError,
    TimeoutError,
    RateLimitError
)

async with get_ollama_client() as client:
    try:
        response = await client.chat("non-existent-model", messages)
    except ModelNotFoundError:
        print("Model not found, trying to pull...")
        await client.pull_model("llama3.2:latest")
    except ConnectionError:
        print("Cannot connect to Ollama server")
    except TimeoutError:
        print("Request timed out")
    except ValidationError as e:
        print(f"Invalid parameters: {e}")
    except RateLimitError:
        print("Rate limit exceeded, waiting...")
        await asyncio.sleep(60)
    except APIError as e:
        print(f"General API error: {e}")
```

### Retry Logic

The client includes automatic retry with exponential backoff:

```python
from abov3.core.api import RetryConfig

# Custom retry behavior
retry_config = RetryConfig(
    max_retries=5,
    base_delay=2.0,
    max_delay=120.0,
    exponential_base=2.0,
    jitter=True  # Add randomness to prevent thundering herd
)

async with get_ollama_client(retry_config=retry_config) as client:
    # Requests will be retried automatically on transient failures
    response = await client.chat("llama3.2:latest", messages)
```

## Performance Optimization

### Connection Reuse

Always use the context manager or properly manage client lifecycle:

```python
# Good - automatic cleanup
async with get_ollama_client() as client:
    for i in range(100):
        response = await client.chat("llama3.2:latest", messages)

# Also good - manual management
client = OllamaClient()
try:
    async with client:
        for i in range(100):
            response = await client.chat("llama3.2:latest", messages)
finally:
    await client.close()
```

### Model Caching

The client caches model information to reduce API calls:

```python
async with get_ollama_client() as client:
    # First call fetches from API
    models = await client.list_models()
    
    # Subsequent calls use cache (within 5 minutes)
    models = await client.list_models()
    
    # Force refresh
    models = await client.list_models(force_refresh=True)
```

### Streaming for Large Responses

Use streaming for long responses to improve perceived performance:

```python
async with get_ollama_client() as client:
    # For long responses, streaming provides better UX
    async for chunk in await client.chat(model, messages, stream=True):
        if chunk.message.content:
            # Process chunk immediately
            process_text_chunk(chunk.message.content)
```

### Batch Operations

For multiple operations, reuse the client:

```python
async with get_ollama_client() as client:
    tasks = []
    for prompt in prompts:
        task = client.generate("llama3.2:latest", prompt)
        tasks.append(task)
    
    # Execute concurrently
    results = await asyncio.gather(*tasks)
```

## Examples

### Complete Chatbot Example

```python
import asyncio
from abov3.core.api import get_ollama_client, ChatMessage

class SimpleChatbot:
    def __init__(self, model_name="llama3.2:latest"):
        self.model_name = model_name
        self.conversation_history = []
    
    async def chat(self, user_message: str) -> str:
        # Add user message to history
        self.conversation_history.append(
            ChatMessage(role="user", content=user_message)
        )
        
        async with get_ollama_client() as client:
            # Ensure model exists
            if not await client.model_exists(self.model_name):
                print(f"Pulling model {self.model_name}...")
                await client.pull_model(self.model_name)
            
            # Get response
            response = await client.chat(
                self.model_name,
                self.conversation_history,
                temperature=0.7
            )
            
            # Add assistant response to history
            self.conversation_history.append(response.message)
            
            return response.message.content
    
    def clear_history(self):
        self.conversation_history = []

async def main():
    chatbot = SimpleChatbot()
    
    print("Chatbot started. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        try:
            response = await chatbot.chat(user_input)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Code Generation Assistant

```python
import asyncio
from abov3.core.api import get_ollama_client

class CodeAssistant:
    def __init__(self, model_name="llama3.2:latest"):
        self.model_name = model_name
    
    async def generate_code(self, description: str, language: str = "python") -> str:
        prompt = f"""
Generate {language} code for the following requirement:
{description}

Please provide clean, well-commented code that follows best practices.
Include error handling where appropriate.

Code:
"""
        
        async with get_ollama_client() as client:
            return await client.generate(
                self.model_name,
                prompt,
                temperature=0.3,  # Lower temperature for more consistent code
                max_tokens=2048
            )
    
    async def explain_code(self, code: str) -> str:
        prompt = f"""
Explain the following code in detail:

```
{code}
```

Please explain:
1. What the code does
2. How it works
3. Any important concepts or patterns used
4. Potential improvements or considerations

Explanation:
"""
        
        async with get_ollama_client() as client:
            return await client.generate(
                self.model_name,
                prompt,
                temperature=0.5
            )
    
    async def review_code(self, code: str) -> str:
        prompt = f"""
Review the following code and provide feedback:

```
{code}
```

Please check for:
1. Potential bugs or issues
2. Code quality and style
3. Performance considerations
4. Security concerns
5. Suggestions for improvement

Review:
"""
        
        async with get_ollama_client() as client:
            return await client.generate(
                self.model_name,
                prompt,
                temperature=0.4
            )

# Usage example
async def demo_code_assistant():
    assistant = CodeAssistant()
    
    # Generate code
    description = "A function that calculates factorial using recursion"
    code = await assistant.generate_code(description)
    print("Generated code:")
    print(code)
    print("\n" + "="*50 + "\n")
    
    # Explain the code
    explanation = await assistant.explain_code(code)
    print("Code explanation:")
    print(explanation)
    print("\n" + "="*50 + "\n")
    
    # Review the code
    review = await assistant.review_code(code)
    print("Code review:")
    print(review)

# asyncio.run(demo_code_assistant())
```

## API Reference

### OllamaClient

Main client class for interacting with Ollama API.

#### Constructor

```python
OllamaClient(
    config: Optional[Config] = None,
    retry_config: Optional[RetryConfig] = None,
    session: Optional[aiohttp.ClientSession] = None
)
```

#### Methods

##### Model Management

- `list_models(force_refresh: bool = False) -> List[ModelInfo]`
- `model_exists(model_name: str) -> bool`
- `pull_model(model_name: str, progress_callback: Optional[Callable] = None) -> bool`
- `delete_model(model_name: str) -> bool`
- `get_model_info(model_name: str) -> Optional[ModelInfo]`

##### Chat and Generation

- `chat(model: str, messages: List[ChatMessage], stream: bool = False, **kwargs) -> Union[ChatResponse, AsyncIterator[ChatResponse]]`
- `generate(model: str, prompt: str, stream: bool = False, **kwargs) -> Union[str, AsyncIterator[str]]`
- `embed(model: str, input_text: Union[str, List[str]]) -> List[List[float]]`

##### Utility Methods

- `health_check() -> bool`
- `get_model_params() -> Dict[str, Any]`
- `update_model_params(**kwargs) -> None`
- `close() -> None`

### Data Classes

#### ChatMessage

```python
@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str
    images: Optional[List[str]] = None
```

#### ChatResponse

```python
@dataclass
class ChatResponse:
    message: ChatMessage
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
```

#### ModelInfo

```python
@dataclass
class ModelInfo:
    name: str
    size: int
    modified_at: str
    digest: str
    details: Dict[str, Any] = field(default_factory=dict)
```

### Exception Hierarchy

```
APIError
├── ConnectionError
├── ModelNotFoundError
├── AuthenticationError
├── RateLimitError
├── TimeoutError
└── ValidationError
```

### Configuration

See the main configuration documentation for details on configuring the Ollama client through the Config system.

---

This client provides a robust, production-ready interface to Ollama with comprehensive error handling, automatic retries, connection pooling, and streaming support. Use it to build sophisticated AI-powered applications with confidence.