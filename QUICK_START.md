# 🚀 ABOV3 4 Ollama - Quick Start Guide

**Get ABOV3 running in under 2 minutes! ⚡**

## 🎯 Fastest Start (Choose Any Method)

### Method 1: Direct Launcher (Windows - Recommended) ✅
```cmd
# Navigate to ABOV3 directory
cd "C:\path\to\above3-ollama-v1.0.0"

# Install dependencies (first time only)
pip install -r requirements.txt

# Launch immediately - works every time!
abov3_direct.bat
```

### Method 2: Automatic Setup ✅
```bash
# One command - handles everything automatically
python start.py
```

### Method 3: Direct Python ✅
```bash
# Simple direct execution
python abov3.py
```

### Method 4: Global Command (After Installation) ✅
```cmd
# If globally installed
abov3
```

## 📋 Prerequisites (5 minutes setup)

### 1. Python 3.8+ ✅ TESTED
- **Windows**: Download from python.org
- **macOS**: `brew install python`
- **Linux**: `sudo apt install python3`

### 2. Ollama (AI Model Platform) ✅ REQUIRED
1. **Download**: https://ollama.ai/download
2. **Install**: Run the installer
3. **Start service**: `ollama serve` (runs automatically after install)
4. **Pull a model**: `ollama pull llama3.2:latest`

### 3. Verify Setup
```bash
# Check Python version
python --version

# Check Ollama is running
ollama list
```

## ⚡ Launch ABOV3 (Choose One)

### Windows Users (Recommended)
```cmd
# Navigate to ABOV3 folder
cd "C:\Users\yourusername\Documents\ABOV3\abov3_ollama\above3-ollama-v1.0.0"

# Launch with the direct launcher
abov3_direct.bat --version
abov3_direct.bat
```

### Cross-Platform Users
```bash
# Automatic dependency check and launch
python start.py

# Or direct execution
python abov3.py
```

## 💬 Using ABOV3 - Interactive Commands

Once ABOV3 starts, you'll see the interactive chat interface:

```
╔═══════════════════════════════════════════════════════════════╗
║                        ABOV3 4 Ollama                        ║
║                   Production Ready v1.0.0                    ║
╚═══════════════════════════════════════════════════════════════╝

> Chat with your AI assistant (type your message):
```

### Built-in Commands ✅ ALL WORKING
```
/help          - Show all available commands
/model         - Change the current AI model (e.g., switch to codellama)
/clear         - Clear the current conversation
/save          - Save conversation to file
/exit          - Exit ABOV3 safely
```

### Example Session
```
> /help
Available commands:
- /help: Show this help
- /model: Change AI model
- /clear: Clear conversation
- /save: Save conversation
- /exit: Exit ABOV3

> /model
Available models:
1. llama3.2:latest
2. codellama:latest
Select model (1-2): 1

> Hello, can you help me write a Python function?
[AI responds with streaming text, properly formatted]

> /exit
Goodbye! Thanks for using ABOV3!
```

## 🎯 Working Features Confirmed ✅

- **✅ Error-Free Startup**: No more key binding errors
- **✅ Streaming Responses**: Real-time AI responses
- **✅ Interactive Commands**: All slash commands work
- **✅ Model Switching**: Switch between Ollama models
- **✅ Cross-Platform**: Windows, macOS, Linux tested
- **✅ Unicode Support**: Proper text encoding
- **✅ Exit Handling**: Clean shutdown with Ctrl+C or /exit

## 🔧 Troubleshooting (Common Issues Solved)

### Issue: "abov3 command not found" ❌
**Solution**: Use the direct launcher or Python execution:
```cmd
# Always works
abov3_direct.bat

# Or
python abov3.py
```

### Issue: "Invalid key" errors ❌
**Status**: ✅ FIXED - This issue has been completely resolved in the current version.

### Issue: "Module not found" errors ❌
**Solution**:
```bash
# Install all dependencies
pip install -r requirements.txt
```

### Issue: "Ollama connection failed" ❌
**Solution**:
1. Make sure Ollama is installed: https://ollama.ai/download
2. Start Ollama: `ollama serve`
3. Pull a model: `ollama pull llama3.2:latest`
4. Test connection: `ollama list`

### Issue: Windows Unicode errors ❌
**Solution**:
```cmd
set PYTHONIOENCODING=utf-8
```

### Issue: Git Bash compatibility ❌
**Solution**: Use Windows Command Prompt or PowerShell for the best experience.

## 🚀 Success! You're Ready

ABOV3 4 Ollama is now running successfully! You can:

- **Generate code** in any programming language
- **Debug and optimize** existing code
- **Learn programming concepts** with AI guidance
- **Build complete applications** with step-by-step assistance
- **Get real-time help** with streaming responses

## 📖 Next Steps

- Try asking for code examples: "Write a Python web scraper"
- Test debugging: "Find the bug in this code: [paste your code]"
- Explore model switching: `/model` to try different AI models
- Save important conversations: `/save`

## 💡 Pro Tips

1. **Use specific questions** for better AI responses
2. **Switch models** for different tasks (codellama for coding, llama3.2 for general questions)
3. **Save conversations** for future reference
4. **Use `/clear`** to start fresh conversations
5. **Close with `/exit`** for clean shutdown

---

**🎉 Welcome to the future of AI-assisted coding!**

**Need help?** Check the full documentation in `README.md` or `INSTALLATION_GUIDE.md`