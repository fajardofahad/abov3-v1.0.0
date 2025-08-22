#!/bin/bash
# Setup ABOV3 alias for Git Bash/MINGW64

echo "Setting up ABOV3 for Git Bash..."

# Get current directory
ABOV3_DIR="$(pwd)"

# Add alias to .bashrc
echo "" >> ~/.bashrc
echo "# ABOV3 alias" >> ~/.bashrc
echo "alias abov3='python \"$ABOV3_DIR/abov3.py\"'" >> ~/.bashrc

echo "Added alias to ~/.bashrc"
echo ""
echo "To use immediately, run:"
echo "  source ~/.bashrc"
echo ""
echo "Or restart your Git Bash terminal."
echo ""
echo "Then you can run:"
echo "  abov3 --version"
echo "  abov3"