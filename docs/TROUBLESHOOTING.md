# ABOV3 4 Ollama - Troubleshooting Guide

**Version 1.0.0**  
**Common Issues and Solutions**

---

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Connection Problems](#connection-problems)
4. [Model Issues](#model-issues)
5. [Performance Problems](#performance-problems)
6. [Configuration Errors](#configuration-errors)
7. [REPL and Interface Issues](#repl-and-interface-issues)
8. [Plugin Problems](#plugin-problems)
9. [Error Messages](#error-messages)
10. [Platform-Specific Issues](#platform-specific-issues)
11. [Advanced Troubleshooting](#advanced-troubleshooting)
12. [Getting Help](#getting-help)

---

## Quick Diagnostics

### Health Check Command

Start troubleshooting with the built-in diagnostic tool:

```bash
abov3 doctor
```

This command checks:
- ✅ Ollama connection status
- ✅ Default model availability
- ✅ Configuration directory structure
- ✅ Plugin system integrity
- ✅ Logging system status

### Common Quick Fixes

1. **Restart Ollama service:**
   ```bash
   # Linux/macOS
   systemctl restart ollama
   # or
   brew services restart ollama
   
   # Windows
   # Restart Ollama from system tray or services
   ```

2. **Check Ollama is running:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

3. **Verify Python version:**
   ```bash
   python --version  # Should be 3.8+
   ```

4. **Update ABOV3:**
   ```bash
   abov3 update
   ```

---

## Installation Issues

### Python Version Compatibility

**Problem:** Python version too old
```
❌ Error: Python 3.8+ required
```

**Solution:**
```bash
# Check Python version
python --version

# Install newer Python (Ubuntu/Debian)
sudo apt update
sudo apt install python3.10 python3.10-pip

# Install newer Python (macOS with Homebrew)
brew install python@3.10

# Windows: Download from python.org
```

### Dependency Installation Failures

**Problem:** Package installation fails
```
ERROR: Could not install packages due to an EnvironmentError
```

**Solutions:**

1. **Use virtual environment:**
   ```bash
   python -m venv abov3_env
   source abov3_env/bin/activate  # Linux/macOS
   # or
   abov3_env\Scripts\activate      # Windows
   
   pip install -r requirements.txt
   ```

2. **Update pip and setuptools:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

3. **Install with --user flag:**
   ```bash
   pip install --user -r requirements.txt
   ```

4. **Clear pip cache:**
   ```bash
   pip cache purge
   ```

### Permission Errors

**Problem:** Permission denied during installation
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**

1. **Use --user installation:**
   ```bash
   pip install --user -e .
   ```

2. **Linux/macOS with sudo (not recommended):**
   ```bash
   sudo pip install -e .
   ```

3. **Fix ownership (Linux/macOS):**
   ```bash
   sudo chown -R $USER ~/.local/lib/python*/site-packages/
   ```

### PATH Issues

**Problem:** Command not found
```bash
abov3: command not found
```

**Solutions:**

1. **Add to PATH (Linux/macOS):**
   ```bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Windows PATH:**
   - Add Python Scripts directory to system PATH
   - Usually: `C:\Users\{username}\AppData\Local\Programs\Python\Python3X\Scripts`

3. **Use full path:**
   ```bash
   python -m abov3.cli
   # or
   ~/.local/bin/abov3
   ```

---

## Connection Problems

### Ollama Server Issues

**Problem:** Cannot connect to Ollama server
```
❌ ERROR: Cannot connect to Ollama server at http://localhost:11434
```

**Diagnostic Steps:**

1. **Check if Ollama is running:**
   ```bash
   ps aux | grep ollama  # Linux/macOS
   tasklist | findstr ollama  # Windows
   ```

2. **Test connection manually:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

3. **Check port availability:**
   ```bash
   netstat -an | grep 11434  # Linux/macOS
   netstat -an | findstr 11434  # Windows
   ```

**Solutions:**

1. **Start Ollama service:**
   ```bash
   # Linux (systemd)
   sudo systemctl start ollama
   sudo systemctl enable ollama
   
   # macOS (Homebrew)
   brew services start ollama
   
   # Manual start
   ollama serve
   ```

2. **Check Ollama configuration:**
   ```bash
   # Default host and port
   export OLLAMA_HOST=0.0.0.0:11434
   ollama serve
   ```

3. **Update ABOV3 configuration:**
   ```bash
   # For remote Ollama server
   abov3 config set ollama.host http://192.168.1.100:11434
   
   # For different port
   abov3 config set ollama.host http://localhost:11435
   ```

### Network Connectivity Issues

**Problem:** Timeout errors
```
TimeoutError: Request timed out after 120 seconds
```

**Solutions:**

1. **Increase timeout:**
   ```bash
   abov3 config set ollama.timeout 300  # 5 minutes
   ```

2. **Check network latency:**
   ```bash
   ping 192.168.1.100  # Your Ollama server IP
   ```

3. **Test with curl:**
   ```bash
   time curl http://localhost:11434/api/tags
   ```

### Firewall and Security

**Problem:** Connection blocked by firewall
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solutions:**

1. **Linux (ufw):**
   ```bash
   sudo ufw allow 11434
   ```

2. **Linux (iptables):**
   ```bash
   sudo iptables -A INPUT -p tcp --dport 11434 -j ACCEPT
   ```

3. **Windows Firewall:**
   - Open Windows Defender Firewall
   - Allow app through firewall
   - Add Ollama or port 11434

4. **macOS:**
   ```bash
   # Check if firewall is blocking
   sudo pfctl -sr | grep 11434
   ```

---

## Model Issues

### Model Not Found

**Problem:** Default model not available
```
❌ ERROR: Default model 'llama3.2:latest' is not available
```

**Solutions:**

1. **List available models:**
   ```bash
   abov3 models list
   ollama list
   ```

2. **Install missing model:**
   ```bash
   abov3 models install llama3.2:latest
   # or directly with Ollama
   ollama pull llama3.2:latest
   ```

3. **Use different model:**
   ```bash
   abov3 config set model.default_model llama2:7b
   # or temporarily
   abov3 chat -m llama2:7b
   ```

### Model Installation Problems

**Problem:** Model installation fails
```
ERROR: Failed to install model llama3.2:latest
```

**Diagnostic Steps:**

1. **Check disk space:**
   ```bash
   df -h  # Linux/macOS
   dir C:\ # Windows
   ```

2. **Check Ollama logs:**
   ```bash
   journalctl -u ollama  # Linux (systemd)
   tail -f ~/.ollama/logs/server.log  # Manual installation
   ```

3. **Test direct installation:**
   ```bash
   ollama pull llama3.2:latest
   ```

**Solutions:**

1. **Free up disk space:**
   ```bash
   # Remove unused models
   ollama rm old_model:tag
   
   # Clean system cache
   sudo apt clean  # Ubuntu/Debian
   brew cleanup   # macOS
   ```

2. **Install smaller model:**
   ```bash
   abov3 models install llama2:7b  # Instead of 13b/70b
   ```

3. **Retry with progress:**
   ```bash
   abov3 models install llama3.2:latest --progress
   ```

### Model Performance Issues

**Problem:** Slow model responses
```
Responses are very slow or timing out
```

**Solutions:**

1. **Use smaller model:**
   ```bash
   abov3 config set model.default_model llama2:7b-chat
   ```

2. **Reduce token limits:**
   ```bash
   abov3 config set model.max_tokens 1024
   abov3 config set model.context_length 4096
   ```

3. **Optimize model parameters:**
   ```bash
   abov3 config set model.temperature 0.7
   abov3 config set model.top_p 0.9
   abov3 config set model.top_k 40
   ```

4. **Check system resources:**
   ```bash
   htop  # Linux/macOS
   # Task Manager on Windows
   ```

---

## Performance Problems

### High Memory Usage

**Problem:** ABOV3 using too much RAM
```
System running out of memory
```

**Solutions:**

1. **Monitor memory usage:**
   ```bash
   # Linux/macOS
   ps aux | grep abov3
   top -p $(pgrep abov3)
   
   # Windows
   tasklist | findstr abov3
   ```

2. **Reduce context length:**
   ```bash
   abov3 config set model.context_length 2048
   ```

3. **Limit conversation history:**
   ```bash
   abov3 config set history.max_conversations 20
   ```

4. **Clear cache:**
   ```bash
   # Linux/macOS
   rm -rf ~/.cache/abov3/*
   
   # Windows
   rmdir /s %LOCALAPPDATA%\abov3\cache
   ```

### Slow Response Times

**Problem:** AI responses are slow
```
Waiting too long for AI responses
```

**Solutions:**

1. **Use GPU acceleration (if available):**
   ```bash
   # Check if GPU is being used by Ollama
   nvidia-smi  # NVIDIA GPUs
   ollama ps   # Check running models
   ```

2. **Optimize Ollama:**
   ```bash
   # Set GPU layers (if supported)
   export OLLAMA_GPU_LAYERS=35
   ollama serve
   ```

3. **Use faster model:**
   ```bash
   abov3 models install mistral:7b-instruct
   abov3 config set model.default_model mistral:7b-instruct
   ```

### High CPU Usage

**Problem:** High CPU utilization
```
CPU usage constantly high
```

**Solutions:**

1. **Check background processes:**
   ```bash
   ps aux | grep -E "(ollama|abov3)"
   ```

2. **Limit concurrent operations:**
   ```bash
   abov3 config set ollama.max_retries 1
   ```

3. **Use lower precision models:**
   ```bash
   # Install quantized versions
   abov3 models install llama2:7b-chat-q4_0
   ```

---

## Configuration Errors

### Invalid Configuration File

**Problem:** Configuration validation fails
```
❌ ERROR: Configuration validation failed
```

**Solutions:**

1. **Reset configuration:**
   ```bash
   abov3 config reset --confirm
   ```

2. **Validate specific settings:**
   ```bash
   abov3 config validate
   ```

3. **Check configuration syntax:**
   ```bash
   # View current config
   abov3 config show
   
   # Edit manually
   nano ~/.config/abov3/config.toml  # Linux/macOS
   notepad %APPDATA%\abov3\config.toml  # Windows
   ```

### Permission Issues

**Problem:** Cannot write configuration
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**

1. **Fix directory permissions:**
   ```bash
   # Linux/macOS
   chmod 755 ~/.config/abov3/
   chmod 644 ~/.config/abov3/config.toml
   
   # Windows
   # Right-click folder → Properties → Security → Edit permissions
   ```

2. **Use alternative config location:**
   ```bash
   abov3 --config ./abov3-config.toml chat
   ```

### Environment Variable Conflicts

**Problem:** Environment variables override config
```
Settings not taking effect despite configuration changes
```

**Solutions:**

1. **Check environment variables:**
   ```bash
   env | grep ABOV3
   echo $ABOV3_DEFAULT_MODEL
   ```

2. **Unset conflicting variables:**
   ```bash
   unset ABOV3_DEFAULT_MODEL
   unset ABOV3_TEMPERATURE
   ```

3. **Use .env file:**
   ```bash
   # Create .env file in project directory
   echo "ABOV3_DEFAULT_MODEL=llama3.2:latest" > .env
   echo "ABOV3_DEBUG=false" >> .env
   ```

---

## REPL and Interface Issues

### Terminal Compatibility Problems

**Problem:** Broken or garbled display
```
Terminal display issues, weird characters
```

**Solutions:**

1. **Check terminal type:**
   ```bash
   echo $TERM
   ```

2. **Set compatibility mode:**
   ```bash
   export TERM=xterm-256color
   abov3 chat
   ```

3. **Disable advanced features:**
   ```bash
   abov3 config set ui.syntax_highlighting false
   abov3 config set ui.streaming_output false
   ```

4. **Use simple mode:**
   ```bash
   abov3 chat --no-banner
   ```

### Windows Terminal Issues

**Problem:** Colors not working on Windows
```
No syntax highlighting or colors in Command Prompt
```

**Solutions:**

1. **Use Windows Terminal:**
   - Install from Microsoft Store
   - Or use PowerShell Core

2. **Enable color support:**
   ```cmd
   reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1
   ```

3. **Use Git Bash or WSL:**
   ```bash
   # Install Git for Windows or Windows Subsystem for Linux
   ```

### Key Binding Issues

**Problem:** Keyboard shortcuts not working
```
Ctrl+C, Ctrl+D, arrow keys not functioning
```

**Solutions:**

1. **Switch key binding mode:**
   ```bash
   # In REPL
   /mode emacs
   /mode vi
   ```

2. **Check terminal capabilities:**
   ```bash
   infocmp $TERM
   ```

3. **Disable vim mode if problematic:**
   ```bash
   abov3 config set ui.vim_mode false
   ```

### History Problems

**Problem:** Command history not working
```
Arrow keys don't show previous commands
```

**Solutions:**

1. **Check history file permissions:**
   ```bash
   ls -la ~/.config/abov3/history.txt
   chmod 644 ~/.config/abov3/history.txt
   ```

2. **Reset history:**
   ```bash
   rm ~/.config/abov3/history.txt
   abov3 chat
   ```

3. **Use in-memory history:**
   ```bash
   abov3 config set history.auto_save false
   ```

---

## Plugin Problems

### Plugin Loading Failures

**Problem:** Plugins not loading
```
❌ WARNING: Failed to load plugin 'git'
```

**Solutions:**

1. **Check plugin status:**
   ```bash
   abov3 plugins list
   ```

2. **Verify plugin directories:**
   ```bash
   ls -la ~/.config/abov3/plugins/
   ```

3. **Re-enable plugin:**
   ```bash
   abov3 plugins disable git
   abov3 plugins enable git
   ```

4. **Check plugin dependencies:**
   ```bash
   # For git plugin, ensure git is installed
   git --version
   ```

### Plugin Conflicts

**Problem:** Plugins causing errors or conflicts
```
Plugin error: Multiple plugins trying to handle command
```

**Solutions:**

1. **Disable conflicting plugins:**
   ```bash
   abov3 plugins list --enabled-only
   abov3 plugins disable problematic_plugin
   ```

2. **Check plugin order:**
   ```bash
   abov3 config show | grep plugins
   ```

3. **Reset plugin configuration:**
   ```bash
   abov3 config set plugins.enabled []
   ```

### Custom Plugin Issues

**Problem:** Custom plugin not working
```
ImportError: No module named 'my_plugin'
```

**Solutions:**

1. **Verify plugin structure:**
   ```
   my_plugin/
   ├── __init__.py
   ├── plugin.py
   └── requirements.txt
   ```

2. **Check plugin installation:**
   ```bash
   abov3 plugins list | grep my_plugin
   ```

3. **Install plugin dependencies:**
   ```bash
   cd ~/.config/abov3/plugins/my_plugin/
   pip install -r requirements.txt
   ```

---

## Error Messages

### Common Error Messages and Solutions

#### "ModuleNotFoundError: No module named 'abov3'"

**Cause:** ABOV3 not properly installed or not in Python path

**Solutions:**
```bash
# Reinstall ABOV3
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"

# Use full module path
python -m abov3.cli chat
```

#### "JSONDecodeError: Expecting value: line 1 column 1"

**Cause:** Corrupted configuration or response data

**Solutions:**
```bash
# Reset configuration
abov3 config reset --confirm

# Clear cache
rm -rf ~/.cache/abov3/

# Check Ollama response
curl http://localhost:11434/api/tags
```

#### "OSError: [Errno 98] Address already in use"

**Cause:** Port already in use by another process

**Solutions:**
```bash
# Find process using port
lsof -i :11434  # Linux/macOS
netstat -ano | findstr :11434  # Windows

# Kill process or use different port
abov3 config set ollama.host http://localhost:11435
```

#### "KeyError: 'response'"

**Cause:** Unexpected API response format

**Solutions:**
```bash
# Update Ollama
ollama --version
# Download latest from ollama.ai

# Check API compatibility
curl http://localhost:11434/api/version
```

#### "SSL: CERTIFICATE_VERIFY_FAILED"

**Cause:** SSL certificate validation issues

**Solutions:**
```bash
# Disable SSL verification (not recommended for production)
abov3 config set ollama.verify_ssl false

# Update certificates
pip install --upgrade certifi
```

### Debug Mode for Detailed Errors

Enable debug mode for detailed error information:

```bash
# Temporary debug mode
abov3 --debug chat

# Persistent debug mode
abov3 config set debug true
export ABOV3_DEBUG=true
```

Debug mode shows:
- Complete stack traces
- API request/response details
- Configuration loading process
- Plugin initialization steps

---

## Platform-Specific Issues

### Windows Issues

**1. Path separators in configuration:**
```bash
# Use forward slashes or escape backslashes
abov3 config set logging.log_dir "C:/Users/user/logs"
# or
abov3 config set logging.log_dir "C:\\Users\\user\\logs"
```

**2. PowerShell execution policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**3. Long path support:**
```bash
# Enable long path support in Windows
# Computer Configuration → Administrative Templates → System → Filesystem
# Enable "Enable Win32 long paths"
```

### macOS Issues

**1. Homebrew installation conflicts:**
```bash
# Use specific Python version
/usr/local/bin/python3.10 -m pip install -e .

# Or create alias
alias abov3="/usr/local/bin/python3.10 -m abov3.cli"
```

**2. Permission issues with Homebrew:**
```bash
# Fix Homebrew permissions
sudo chown -R $(whoami) /usr/local/lib/python*/site-packages/
```

**3. Terminal.app compatibility:**
```bash
# Use iTerm2 for better compatibility
brew install --cask iterm2
```

### Linux Issues

**1. Distribution-specific package managers:**
```bash
# Ubuntu/Debian
sudo apt install python3-pip python3-venv

# CentOS/RHEL
sudo yum install python3-pip python3-venv

# Arch Linux
sudo pacman -S python-pip
```

**2. SELinux issues:**
```bash
# Check SELinux status
sestatus

# Temporarily disable
sudo setenforce 0

# Create policy for permanent fix
sudo setsebool -P httpd_can_network_connect 1
```

**3. SystemD service issues:**
```bash
# Check service status
systemctl status ollama

# Check logs
journalctl -u ollama -f

# Restart service
sudo systemctl restart ollama
```

---

## Advanced Troubleshooting

### Network Debugging

**1. Packet capture:**
```bash
# Linux
sudo tcpdump -i lo port 11434

# macOS
sudo tcpdump -i lo0 port 11434

# Windows (use Wireshark)
```

**2. HTTP debugging:**
```bash
# Verbose curl
curl -v http://localhost:11434/api/tags

# HTTP client debugging
export PYTHONHTTPSVERIFY=0
export CURL_CA_BUNDLE=""
```

### Performance Profiling

**1. Python profiling:**
```bash
# Profile ABOV3 execution
python -m cProfile -o abov3.prof -m abov3.cli chat

# Analyze profile
python -m pstats abov3.prof
```

**2. Memory profiling:**
```bash
# Install memory profiler
pip install memory_profiler

# Profile memory usage
mprof run python -m abov3.cli chat
mprof plot
```

### Log Analysis

**1. Enable comprehensive logging:**
```bash
abov3 config set logging.level DEBUG
abov3 config set logging.enable_file_logging true
abov3 config set logging.enable_json_logging true
```

**2. Log file locations:**
```bash
# Linux/macOS
~/.local/share/abov3/logs/
~/.cache/abov3/logs/

# Windows
%LOCALAPPDATA%\abov3\logs\
```

**3. Analyze logs:**
```bash
# View recent errors
tail -f ~/.local/share/abov3/logs/abov3.log | grep ERROR

# Search for specific issues
grep -n "timeout" ~/.local/share/abov3/logs/abov3.log

# JSON log analysis
jq '.level == "ERROR"' ~/.local/share/abov3/logs/abov3.jsonl
```

### System Resource Monitoring

**1. Monitor resource usage:**
```bash
# Linux
htop -p $(pgrep -f abov3)
iotop -p $(pgrep -f abov3)

# macOS
top -pid $(pgrep -f abov3)
fs_usage -f exec,pathname,diskio -p $(pgrep -f abov3)

# Windows
# Use Task Manager or Process Monitor
```

**2. Check disk I/O:**
```bash
# Linux
iotop -a -o -d 1

# macOS
sudo fs_usage -w -f diskio

# Windows
# Use Resource Monitor
```

### Database and Cache Issues

**1. Clear all caches:**
```bash
# Remove cache directories
rm -rf ~/.cache/abov3/
rm -rf ~/.local/share/abov3/cache/

# Windows
rmdir /s %LOCALAPPDATA%\abov3\cache
rmdir /s %APPDATA%\abov3\cache
```

**2. Reset databases:**
```bash
# Remove history database
rm ~/.local/share/abov3/history.db

# Remove session data
rm -rf ~/.local/share/abov3/sessions/
```

---

## Getting Help

### Before Asking for Help

1. **Run diagnostics:**
   ```bash
   abov3 doctor
   ```

2. **Collect system information:**
   ```bash
   abov3 --version
   python --version
   ollama --version
   uname -a  # Linux/macOS
   systeminfo  # Windows
   ```

3. **Check logs:**
   ```bash
   tail -100 ~/.local/share/abov3/logs/abov3.log
   ```

4. **Try minimal reproduction:**
   ```bash
   abov3 --debug config reset --confirm
   abov3 --debug chat
   ```

### Where to Get Help

1. **GitHub Issues:**
   - [Report bugs](https://github.com/abov3/abov3-ollama/issues)
   - Search existing issues first

2. **Documentation:**
   - [User Manual](USER_MANUAL.md)
   - [FAQ](FAQ.md)
   - [Examples directory](../examples/)

3. **Community:**
   - [Discussions](https://github.com/abov3/abov3-ollama/discussions)
   - Discord/Slack channels (if available)

### Information to Include in Bug Reports

**System Information:**
```bash
# Run this command and include output
abov3 doctor > system_info.txt
```

**Configuration:**
```bash
# Include sanitized configuration
abov3 config show --format json > config.json
# Remove any sensitive information
```

**Logs:**
```bash
# Include relevant log excerpts
tail -50 ~/.local/share/abov3/logs/abov3.log > recent_logs.txt
```

**Steps to Reproduce:**
1. Exact commands used
2. Expected behavior
3. Actual behavior
4. Error messages (full text)

### Emergency Recovery

If ABOV3 is completely broken:

1. **Complete reset:**
   ```bash
   # Backup first (optional)
   cp -r ~/.config/abov3/ ~/abov3_backup/
   
   # Remove all ABOV3 data
   rm -rf ~/.config/abov3/
   rm -rf ~/.local/share/abov3/
   rm -rf ~/.cache/abov3/
   
   # Reinstall
   pip uninstall abov3
   pip install -e .
   
   # First run
   abov3
   ```

2. **Use alternative interface:**
   ```bash
   # Direct Python module execution
   python -m abov3.cli --help
   
   # Direct Ollama usage
   curl -X POST http://localhost:11434/api/generate \
     -H "Content-Type: application/json" \
     -d '{"model": "llama2:7b", "prompt": "Hello"}'
   ```

---

## Conclusion

This troubleshooting guide covers the most common issues encountered with ABOV3 4 Ollama. Most problems can be resolved by:

1. Running `abov3 doctor` for diagnostics
2. Checking that Ollama is running and accessible
3. Verifying model availability
4. Reviewing configuration settings
5. Checking logs for detailed error information

For issues not covered in this guide, please:
- Search the GitHub issues
- Run diagnostics and collect system information
- Create a detailed bug report with reproduction steps

Remember to always backup your configuration and important conversations before making major changes to your system.

---

*This troubleshooting guide is for ABOV3 4 Ollama version 1.0.0. Some solutions may vary with different versions.*