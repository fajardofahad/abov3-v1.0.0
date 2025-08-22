#!/usr/bin/env python3
"""
ABOV3 Entry Points Test Script

This script tests all the entry points created for ABOV3 to ensure they work correctly.
It validates imports, CLI availability, and basic functionality.

Author: ABOV3 Team
Version: 1.0.0
License: MIT
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path

def test_module_import():
    """Test that the abov3 module can be imported correctly."""
    print("Testing module import...")
    
    try:
        import abov3
        print(f"[OK] Module imported successfully. Version: {abov3.__version__}")
        return True
    except ImportError as e:
        print(f"[FAIL] Module import failed: {e}")
        return False

def test_cli_import():
    """Test that the CLI can be imported."""
    print("Testing CLI import...")
    
    try:
        from abov3.cli import main
        print("[OK] CLI imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] CLI import failed: {e}")
        return False

def test_main_module():
    """Test running as module with python -m abov3."""
    print("Testing module execution...")
    
    try:
        # Test with --version flag to avoid starting interactive session
        result = subprocess.run(
            [sys.executable, "-m", "abov3", "--version"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and "ABOV3" in result.stdout:
            print(f"[OK] Module execution successful: {result.stdout.strip()}")
            return True
        else:
            print(f"[FAIL] Module execution failed. Return code: {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[FAIL] Module execution timed out")
        return False
    except Exception as e:
        print(f"[FAIL] Module execution error: {e}")
        return False

def test_direct_script():
    """Test running the direct abov3.py script."""
    print("Testing direct script execution...")
    
    script_path = Path(__file__).parent / "abov3.py"
    
    if not script_path.exists():
        print("[FAIL] abov3.py script not found")
        return False
    
    try:
        # Test with --version flag to avoid starting interactive session
        result = subprocess.run(
            [sys.executable, str(script_path), "--version"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and "ABOV3" in result.stdout:
            print(f"[OK] Direct script execution successful: {result.stdout.strip()}")
            return True
        else:
            print(f"[FAIL] Direct script execution failed. Return code: {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[FAIL] Direct script execution timed out")
        return False
    except Exception as e:
        print(f"[FAIL] Direct script execution error: {e}")
        return False

def test_batch_script():
    """Test the Windows batch script (if on Windows)."""
    if sys.platform != "win32":
        print("Skipping batch script test (not on Windows)")
        return True
        
    print("Testing Windows batch script...")
    
    batch_path = Path(__file__).parent / "abov3.bat"
    
    if not batch_path.exists():
        print("[FAIL] abov3.bat script not found")
        return False
    
    try:
        # Test with --version flag to avoid starting interactive session
        result = subprocess.run(
            [str(batch_path), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            shell=True
        )
        
        if result.returncode == 0 and "ABOV3" in result.stdout:
            print(f"[OK] Batch script execution successful: {result.stdout.strip()}")
            return True
        else:
            print(f"[FAIL] Batch script execution failed. Return code: {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[FAIL] Batch script execution timed out")
        return False
    except Exception as e:
        print(f"[FAIL] Batch script execution error: {e}")
        return False

def test_shell_script():
    """Test the Unix shell script (if on Unix-like system)."""
    if sys.platform == "win32":
        print("Skipping shell script test (on Windows)")
        return True
        
    print("Testing Unix shell script...")
    
    shell_path = Path(__file__).parent / "abov3.sh"
    
    if not shell_path.exists():
        print("[FAIL] abov3.sh script not found")
        return False
    
    try:
        # Test with --version flag to avoid starting interactive session
        result = subprocess.run(
            [str(shell_path), "--version"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and "ABOV3" in result.stdout:
            print(f"[OK] Shell script execution successful: {result.stdout.strip()}")
            return True
        else:
            print(f"[FAIL] Shell script execution failed. Return code: {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[FAIL] Shell script execution timed out")
        return False
    except Exception as e:
        print(f"[FAIL] Shell script execution error: {e}")
        return False

def test_setup_configuration():
    """Test setup.py and pyproject.toml configuration."""
    print("Testing setup configuration...")
    
    # Check setup.py
    setup_path = Path(__file__).parent / "setup.py"
    if setup_path.exists():
        print("[OK] setup.py found")
    else:
        print("[FAIL] setup.py not found")
        return False
    
    # Check pyproject.toml
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    if pyproject_path.exists():
        print("[OK] pyproject.toml found")
    else:
        print("[FAIL] pyproject.toml not found")
        return False
    
    return True

def main():
    """Run all entry point tests."""
    print("=" * 50)
    print("ABOV3 Entry Points Test")
    print("=" * 50)
    print()
    
    tests = [
        test_module_import,
        test_cli_import,
        test_main_module,
        test_direct_script,
        test_batch_script,
        test_shell_script,
        test_setup_configuration
    ]
    
    results = []
    
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Summary
    print("=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("[OK] All entry points are working correctly!")
        return 0
    else:
        print("[FAIL] Some entry points have issues. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())