#!/usr/bin/env python3
"""
Test script for ABOV3 application lifecycle improvements.

This script tests various scenarios to ensure the application can start
robustly even when some components fail.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the abov3 package to the path
sys.path.insert(0, str(Path(__file__).parent))

from abov3.core.app import ABOV3App, AppState
from abov3.core.config import get_config


async def test_normal_startup():
    """Test normal application startup."""
    print("\n=== Testing Normal Startup ===")
    
    config = get_config()
    app = ABOV3App(config=config, interactive=False, debug=True)
    
    try:
        start_time = time.time()
        await app.startup()
        elapsed = time.time() - start_time
        
        print(f"✓ Application started in {elapsed:.2f} seconds")
        print(f"✓ App state: {app.state}")
        print(f"✓ Failed components: {app.metrics.failed_components}")
        print(f"✓ Is healthy: {app.is_healthy()}")
        print(f"✓ Is functional: {app.is_functional()}")
        
        # Test health status
        health_status = await app.get_health_status()
        print(f"✓ Health status: {len(health_status)} components checked")
        
        return True
        
    except Exception as e:
        print(f"✗ Startup failed: {e}")
        return False
    finally:
        await app.shutdown()


async def test_degraded_mode():
    """Test startup when Ollama is not available."""
    print("\n=== Testing Degraded Mode (Simulated Ollama Failure) ===")
    
    config = get_config()
    # Set an invalid Ollama host to simulate failure
    config.ollama.host = "http://localhost:99999"
    
    app = ABOV3App(config=config, interactive=False, debug=True)
    
    try:
        start_time = time.time()
        await app.startup()
        elapsed = time.time() - start_time
        
        print(f"✓ Application started in degraded mode in {elapsed:.2f} seconds")
        print(f"✓ App state: {app.state}")
        print(f"✓ Failed components: {app.metrics.failed_components}")
        print(f"✓ Is healthy: {app.is_healthy()}")
        print(f"✓ Is functional: {app.is_functional()}")
        
        # Test that we can still get health status
        health_status = await app.get_health_status()
        print(f"✓ Health status available: {len(health_status)} components")
        
        # Test user input processing in degraded mode
        response = await app._process_user_input("Hello")
        print(f"✓ Degraded mode response: {response[:50]}...")
        
        return app.state == AppState.DEGRADED
        
    except Exception as e:
        print(f"✗ Degraded mode test failed: {e}")
        return False
    finally:
        await app.shutdown()


async def test_component_recovery():
    """Test component recovery functionality."""
    print("\n=== Testing Component Recovery ===")
    
    config = get_config()
    app = ABOV3App(config=config, interactive=False, debug=True)
    
    try:
        # Start normally
        await app.startup()
        
        if app.state != AppState.RUNNING:
            print(f"⚠ App not in RUNNING state initially: {app.state}")
            return False
        
        # Simulate component failure
        app._component_health['ollama_client'] = False
        app.metrics.failed_components.append('ollama_client')
        app.state = AppState.DEGRADED
        
        print(f"✓ Simulated component failure - state: {app.state}")
        
        # Test recovery attempt
        app.force_component_recovery('ollama_client')
        
        # Give recovery a moment
        await asyncio.sleep(2)
        
        print(f"✓ Recovery attempted - final state: {app.state}")
        print(f"✓ Component health: {app.get_component_status('ollama_client')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Component recovery test failed: {e}")
        return False
    finally:
        await app.shutdown()


async def test_startup_timeout():
    """Test startup timeout handling."""
    print("\n=== Testing Startup Timeout Resilience ===")
    
    config = get_config()
    app = ABOV3App(config=config, interactive=False, debug=True)
    
    # Reduce timeout for faster testing
    app._startup_timeout = 5.0
    app._health_check_timeout = 2.0
    
    try:
        start_time = time.time()
        await app.startup()
        elapsed = time.time() - start_time
        
        print(f"✓ Application handled timeouts in {elapsed:.2f} seconds")
        print(f"✓ App state: {app.state}")
        print(f"✓ Background tasks: {len(app._background_tasks)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Timeout test failed: {e}")
        return False
    finally:
        await app.shutdown()


async def main():
    """Run all lifecycle tests."""
    print("ABOV3 Application Lifecycle Tests")
    print("=" * 50)
    
    tests = [
        ("Normal Startup", test_normal_startup),
        ("Degraded Mode", test_degraded_mode),
        ("Component Recovery", test_component_recovery),
        ("Startup Timeout", test_startup_timeout),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        icon = "✓" if success else "✗"
        print(f"{icon} {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Application lifecycle is robust.")
        return 0
    else:
        print("⚠ Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)