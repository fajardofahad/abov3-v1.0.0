#!/usr/bin/env python3
"""
Simple test script to verify ABOV3 CLI functionality.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"

def test_cli():
    """Test CLI functionality."""
    print("Testing ABOV3 CLI...")
    
    # Test version
    print("\n1. Testing version command...")
    rc, stdout, stderr = run_command("python -m abov3.cli --version")
    if rc == 0:
        print(f"OK Version: {stdout.strip()}")
    else:
        print(f"ERROR Version failed: {stderr}")
    
    # Test help
    print("\n2. Testing help command...")
    rc, stdout, stderr = run_command("python -m abov3.cli --help")
    if rc == 0 and "ABOV3 4 Ollama" in stdout:
        print("OK Help command works")
    else:
        print(f"ERROR Help failed: {stderr}")
    
    # Test config show
    print("\n3. Testing config show...")
    rc, stdout, stderr = run_command("python -m abov3.cli config show")
    if rc == 0 and "ABOV3 Configuration" in stdout:
        print("OK Config show works")
    else:
        print(f"ERROR Config show failed: {stderr}")
    
    # Test config get
    print("\n4. Testing config get...")
    rc, stdout, stderr = run_command("python -m abov3.cli config get model.default_model")
    if rc == 0:
        print(f"OK Config get: {stdout.strip()}")
    else:
        print(f"ERROR Config get failed: {stderr}")
    
    # Test config set
    print("\n5. Testing config set...")
    rc, stdout, stderr = run_command("python -m abov3.cli config set model.temperature 0.9")
    if rc == 0:
        print("OK Config set works")
    else:
        print(f"ERROR Config set failed: {stderr}")
    
    # Test doctor
    print("\n6. Testing doctor command...")
    rc, stdout, stderr = run_command("python -m abov3.cli doctor")
    if rc == 0 and "Health Check" in stdout:
        print("OK Doctor command works")
    else:
        print(f"ERROR Doctor failed: {stderr}")
    
    print("\nOK CLI testing complete!")

if __name__ == "__main__":
    test_cli()