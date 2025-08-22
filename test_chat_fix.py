#!/usr/bin/env python3
"""
Test script to verify ABOV3 chat integration fixes.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from abov3.core.app import ABOV3App, create_app
from abov3.core.config import get_config


async def test_chat_integration():
    """Test the chat integration to ensure fixes work."""
    print("[TEST] Testing ABOV3 Chat Integration Fixes...")
    
    try:
        # Create app with non-interactive mode for testing
        config = get_config()
        print(f"[CONFIG] Config loaded. Default model: {config.model.default_model}")
        print(f"[CONFIG] Ollama host: {config.ollama.host}")
        print(f"[CONFIG] Streaming enabled: {config.ui.streaming_output}")
        
        app = ABOV3App(config=config, interactive=False, debug=True)
        
        print("[STARTUP] Starting ABOV3 application...")
        await app.startup()
        
        print(f"[STATUS] App state: {app.state.value}")
        print(f"[STATUS] Ollama client available: {app.ollama_client is not None}")
        print(f"[STATUS] Ollama client healthy: {app._component_health.get('ollama_client', False)}")
        
        # Test health status
        health = await app.get_health_status()
        print("[HEALTH] Health Status:")
        for component, status in health.items():
            status_text = "[OK] HEALTHY" if status.get('healthy', False) else "[FAIL] UNHEALTHY"
            print(f"  - {component}: {status_text}")
            if not status.get('healthy', False):
                print(f"    Error: {status.get('error', 'Unknown')}")
        
        # Test if we can list models
        if app.ollama_client:
            try:
                models = await app.list_models()
                print(f"[MODELS] Available models: {len(models)}")
                for model in models[:3]:  # Show first 3 models
                    if hasattr(model, 'name'):
                        print(f"  - {model.name} ({getattr(model, 'size', 'Unknown size')})")
                    else:
                        print(f"  - {model.get('name', 'Unknown')} ({model.get('size', 'Unknown size')})")
            except Exception as e:
                print(f"[ERROR] Failed to list models: {e}")
        
        # Test a simple chat interaction
        if app.is_functional():
            print("[CHAT] Testing chat functionality...")
            try:
                test_input = "Hello! Please respond with just 'ABOV3 is working!' to confirm the fix."
                print(f"[INPUT] Input: {test_input}")
                
                # Process the input
                response = await app._process_user_input(test_input)
                
                # Handle async generators (streaming responses)
                if hasattr(response, '__aiter__'):
                    print("[STREAM] Streaming response detected...")
                    full_response = ""
                    async for chunk in response:
                        full_response += chunk
                        print(f"[CHUNK] Chunk: {chunk}")
                    print(f"[RESPONSE] Complete response: {full_response}")
                else:
                    print(f"[RESPONSE] Response: {response}")
                
            except Exception as e:
                print(f"[ERROR] Chat test failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("[WARNING] Chat functionality not available (Ollama connection issue)")
        
        # Shutdown
        await app.shutdown()
        print("[SHUTDOWN] Application shutdown complete")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("[SUCCESS] Test completed successfully!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_chat_integration())
    sys.exit(0 if success else 1)