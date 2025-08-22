#!/usr/bin/env python3
"""
ABOV3 Installation Script

This script installs ABOV3 so you can run it from anywhere using just 'abov3'.
"""

import os
import sys
import shutil
from pathlib import Path

def main():
    print("ABOV3 Installation Script")
    print("=" * 50)
    
    # Get current directory (where ABOV3 is)
    abov3_dir = Path(__file__).parent.absolute()
    print(f"ABOV3 Directory: {abov3_dir}")
    
    # Option 1: Try pip install in development mode
    print("\n[1] Installing ABOV3 with pip (development mode)...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], cwd=abov3_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("SUCCESS: Successfully installed ABOV3 with pip!")
            print("\nYou can now run 'abov3' from anywhere!")
            print("\nTest it:")
            print("  abov3 --version")
            print("  abov3")
            return
        else:
            print(f"ERROR: Pip install failed: {result.stderr}")
    except Exception as e:
        print(f"ERROR: Pip install error: {e}")
    
    # Option 2: Create batch file in Windows directory
    print("\n[2] Creating global batch file...")
    try:
        windows_dir = Path(os.environ['WINDIR'])
        batch_content = f"""@echo off
cd /d "{abov3_dir}"
python abov3.py %*
"""
        
        batch_file = windows_dir / "abov3.bat"
        with open(batch_file, 'w') as f:
            f.write(batch_content)
        
        print(f"SUCCESS: Created global batch file: {batch_file}")
        print("\nYou can now run 'abov3' from anywhere in Command Prompt!")
        print("\nTest it:")
        print("  abov3 --version")
        print("  abov3")
        return
        
    except PermissionError:
        print("ERROR: Permission denied. Try running as administrator.")
    except Exception as e:
        print(f"ERROR: Error creating batch file: {e}")
    
    # Option 3: Create alias instructions
    print("\n[3] Manual Setup Instructions:")
    print(f"""
Add this directory to your PATH:
{abov3_dir}

Or create an alias in your shell:
- For PowerShell: Set-Alias abov3 "{abov3_dir / 'abov3.py'}"
- For Bash: alias abov3="python '{abov3_dir / 'abov3.py'}'"

Or run directly:
python "{abov3_dir / 'abov3.py'}"
""")

if __name__ == "__main__":
    main()