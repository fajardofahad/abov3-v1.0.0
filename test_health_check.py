#!/usr/bin/env python3
"""Test script to verify health check fixes."""

import asyncio
import sys
import logging
from pathlib import Path

# Add the parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from abov3.core.app import ABOV3App
from abov3.core.config import get_config

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

async def test_health_checks():
    """Test the health check functionality."""
    print("Testing ABOV3 health checks...")
    
    # Create app instance
    config = get_config()
    app = ABOV3App(config=config, interactive=False, debug=True)
    
    try:
        print("\n1. Testing startup health checks...")
        # Set to STARTING state to simulate startup
        from abov3.core.app import AppState
        app.state = AppState.STARTING
        
        # Initialize minimal components for testing
        print("   - Initializing components...")
        await app._initialize_context_manager()
        await app._initialize_model_manager()
        
        # Run health checks in startup mode
        print("   - Running health checks in startup mode...")
        health_status = await app.get_health_status(startup_mode=True)
        
        print("\n   Health Status (Startup Mode):")
        for component, status in health_status.items():
            healthy = "✓" if status.get("healthy", False) else "✗"
            critical = " [CRITICAL]" if status.get("critical", False) else ""
            error = f" - {status.get('error', '')}" if status.get('error') else ""
            print(f"   {healthy} {component}{critical}{error}")
        
        # Check for critical issues
        critical_issues = [
            component for component, status in health_status.items()
            if not status.get("healthy", False) and status.get("critical", False)
        ]
        
        if critical_issues:
            print(f"\n   ⚠ Critical issues found: {critical_issues}")
        else:
            print("\n   ✓ No critical issues during startup")
        
        print("\n2. Testing runtime health checks...")
        # Set to RUNNING state
        app.state = AppState.RUNNING
        
        # Run health checks in normal mode
        print("   - Running health checks in normal mode...")
        health_status = await app.get_health_status(startup_mode=False)
        
        print("\n   Health Status (Runtime Mode):")
        for component, status in health_status.items():
            healthy = "✓" if status.get("healthy", False) else "✗"
            critical = " [CRITICAL]" if status.get("critical", False) else ""
            error = f" - {status.get('error', '')}" if status.get('error') else ""
            print(f"   {healthy} {component}{critical}{error}")
        
        print("\n3. Testing async list_models method...")
        try:
            models = await app.list_models()
            print(f"   ✓ list_models() returned {len(models)} models")
        except Exception as e:
            print(f"   ✗ list_models() failed: {e}")
        
        print("\n✓ Health check tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if hasattr(app, 'ollama_client') and app.ollama_client:
            await app.ollama_client.close()
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_health_checks())
    sys.exit(0 if success else 1)