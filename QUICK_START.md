# 🚀 ABOV3 4 Ollama - Quick Start Guide

## Starting ABOV3 - Multiple Options

### Option 1: Direct Start (Simplest)
```bash
python start.py
```
This will check dependencies, guide you through setup, and start ABOV3.

### Option 2: Using abov3 Command
```bash
# Windows
abov3.bat

# Linux/Mac
./abov3.sh

# Or with Python
python abov3.py
```

### Option 3: Demo Mode (No Setup Required)
```bash
python demo.py
```
This runs a minimal demo that showcases ABOV3's capabilities.

### Option 4: Python Module
```bash
python -m abov3
```

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running (optional for demo)
   - Download from: https://ollama.ai/download
   - Start server: `ollama serve`
   - Pull a model: `ollama pull llama3.2:latest`

## 🎯 First Time Setup

If this is your first time running ABOV3:

1. Run `python start.py` - it will guide you through everything
2. Install dependencies when prompted
3. The setup wizard will help configure your preferences

## 💬 Interactive Commands

Once ABOV3 is running, you can use these commands:

- **Just type your question** - Start chatting with AI
- `/help` - Show available commands
- `/model` - Change AI model
- `/clear` - Clear conversation
- `/save` - Save conversation
- `/exit` - Exit ABOV3

## 🔧 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### "Ollama not running" error
1. Install Ollama: https://ollama.ai/download
2. Start it: `ollama serve`
3. Or run demo mode: `python demo.py`

### Windows Unicode errors
Set environment variable:
```bash
set PYTHONIOENCODING=utf-8
```

## 🎉 That's It!

You're ready to start coding with AI assistance. ABOV3 will help you:
- Generate code in any language
- Debug and optimize existing code
- Learn new programming concepts
- Build complete applications

Type `python start.py` to begin your AI coding journey!