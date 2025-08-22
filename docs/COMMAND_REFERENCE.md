# ABOV3 4 Ollama - Command Reference

**Version 1.0.0**  
**Complete CLI Command Documentation**

---

## Table of Contents

1. [Main Commands](#main-commands)
2. [Chat Commands](#chat-commands)
3. [Configuration Commands](#configuration-commands)
4. [Model Management](#model-management)
5. [History Commands](#history-commands)
6. [Plugin Commands](#plugin-commands)
7. [System Commands](#system-commands)
8. [REPL Commands](#repl-commands)
9. [CLI Arguments](#cli-arguments)
10. [Environment Variables](#environment-variables)
11. [Keyboard Shortcuts](#keyboard-shortcuts)
12. [Exit Codes](#exit-codes)

---

## Main Commands

### `abov3`
Main entry point for ABOV3 CLI.

**Syntax:**
```bash
abov3 [OPTIONS] [COMMAND]
```

**Options:**
- `--config, -c PATH` - Path to configuration file
- `--debug, -d` - Enable debug mode
- `--version, -v` - Show version and exit
- `--no-banner` - Don't display the banner
- `--help` - Show help message

**Examples:**
```bash
# Start with default settings
abov3

# Start with custom config
abov3 -c /path/to/config.toml

# Start in debug mode
abov3 --debug

# Show version
abov3 --version
```

---

## Chat Commands

### `abov3 chat`
Start an interactive chat session with ABOV3.

**Syntax:**
```bash
abov3 chat [OPTIONS]
```

**Options:**
- `--model, -m MODEL` - Model to use for the chat session
- `--system, -s PROMPT` - System prompt to use
- `--temperature, -t FLOAT` - Temperature for response generation (0.0-2.0)
- `--no-history` - Don't save conversation to history
- `--continue-last` - Continue the last conversation

**Examples:**
```bash
# Start basic chat
abov3 chat

# Use specific model
abov3 chat -m llama3.2:latest

# Set temperature and system prompt
abov3 chat -t 0.9 -s "You are a Python expert"

# Continue last conversation
abov3 chat --continue-last

# Chat without saving history
abov3 chat --no-history

# Combined options
abov3 chat -m codellama:latest -t 0.7 -s "You are a senior software developer"
```

**Chat Session Features:**
- Real-time streaming responses
- Syntax highlighting for code blocks
- Auto-completion support
- Command history navigation
- Multi-line input support
- Rich formatting and colors

---

## Configuration Commands

### `abov3 config`
Manage ABOV3 configuration settings.

**Subcommands:**
- `show` - Display current configuration
- `get` - Get specific configuration value
- `set` - Set configuration value
- `reset` - Reset configuration to defaults
- `validate` - Validate current configuration

### `abov3 config show`
Display current configuration.

**Syntax:**
```bash
abov3 config show [OPTIONS]
```

**Options:**
- `--format, -f FORMAT` - Output format: `table`, `json`, `yaml` (default: table)

**Examples:**
```bash
# Show configuration in table format
abov3 config show

# Show as JSON
abov3 config show -f json

# Show as YAML
abov3 config show -f yaml
```

### `abov3 config get`
Get a specific configuration value.

**Syntax:**
```bash
abov3 config get KEY
```

**Examples:**
```bash
# Get default model
abov3 config get model.default_model

# Get Ollama host
abov3 config get ollama.host

# Get UI theme
abov3 config get ui.theme

# Get temperature setting
abov3 config get model.temperature
```

### `abov3 config set`
Set a configuration value.

**Syntax:**
```bash
abov3 config set KEY VALUE
```

**Examples:**
```bash
# Set default model
abov3 config set model.default_model llama3.2:latest

# Set temperature
abov3 config set model.temperature 0.8

# Set max tokens
abov3 config set model.max_tokens 2048

# Set UI theme
abov3 config set ui.theme github

# Set Ollama host
abov3 config set ollama.host http://192.168.1.100:11434

# Enable/disable features
abov3 config set ui.syntax_highlighting true
abov3 config set ui.vim_mode false
```

### `abov3 config reset`
Reset configuration to defaults.

**Syntax:**
```bash
abov3 config reset [OPTIONS]
```

**Options:**
- `--confirm` - Skip confirmation prompt

**Examples:**
```bash
# Reset with confirmation
abov3 config reset

# Reset without confirmation
abov3 config reset --confirm
```

### `abov3 config validate`
Validate current configuration.

**Syntax:**
```bash
abov3 config validate
```

**Examples:**
```bash
# Validate configuration
abov3 config validate
```

---

## Model Management

### `abov3 models`
Manage AI models for ABOV3.

**Subcommands:**
- `list` - List available models
- `install` - Install a model
- `remove` - Remove a model
- `info` - Show model information
- `set-default` - Set default model

### `abov3 models list`
List available models.

**Syntax:**
```bash
abov3 models list [OPTIONS]
```

**Options:**
- `--format, -f FORMAT` - Output format: `table`, `json` (default: table)

**Examples:**
```bash
# List models in table format
abov3 models list

# List models in JSON format
abov3 models list -f json
```

### `abov3 models install`
Install a model.

**Syntax:**
```bash
abov3 models install MODEL_NAME [OPTIONS]
```

**Options:**
- `--progress` - Show download progress (default: true)

**Examples:**
```bash
# Install a model with progress
abov3 models install llama3.2:latest

# Install without progress display
abov3 models install codellama:7b --no-progress

# Install specific model versions
abov3 models install mistral:7b-instruct-v0.1-q4_0
abov3 models install deepseek-coder:6.7b
```

### `abov3 models remove`
Remove a model.

**Syntax:**
```bash
abov3 models remove MODEL_NAME [OPTIONS]
```

**Options:**
- `--confirm` - Skip confirmation prompt

**Examples:**
```bash
# Remove with confirmation
abov3 models remove llama2:7b

# Remove without confirmation
abov3 models remove llama2:7b --confirm
```

### `abov3 models info`
Show detailed information about a model.

**Syntax:**
```bash
abov3 models info MODEL_NAME
```

**Examples:**
```bash
# Get model information
abov3 models info llama3.2:latest
abov3 models info codellama:7b
```

### `abov3 models set-default`
Set the default model.

**Syntax:**
```bash
abov3 models set-default MODEL_NAME
```

**Examples:**
```bash
# Set default model
abov3 models set-default llama3.2:latest
abov3 models set-default codellama:latest
```

---

## History Commands

### `abov3 history`
Manage conversation history.

**Subcommands:**
- `list` - List all conversations
- `search` - Search conversations
- `show` - Show specific conversation
- `export` - Export conversation
- `clear` - Clear all history

### `abov3 history list`
List conversation history.

**Syntax:**
```bash
abov3 history list [OPTIONS]
```

**Options:**
- `--limit, -l INTEGER` - Maximum number of conversations to show (default: 10)
- `--format, -f FORMAT` - Output format: `table`, `json` (default: table)

**Examples:**
```bash
# List recent conversations
abov3 history list

# List more conversations
abov3 history list --limit 50

# List in JSON format
abov3 history list -f json
```

### `abov3 history search`
Search conversation history.

**Syntax:**
```bash
abov3 history search QUERY [OPTIONS]
```

**Options:**
- `--limit, -l INTEGER` - Maximum number of results (default: 10)

**Examples:**
```bash
# Search for Python-related conversations
abov3 history search "python"

# Search with more results
abov3 history search "REST API" --limit 20

# Search for specific functions
abov3 history search "lambda function"
```

### `abov3 history show`
Show specific conversation.

**Syntax:**
```bash
abov3 history show CONVERSATION_ID
```

**Examples:**
```bash
# Show conversation by ID
abov3 history show abc123def456
```

### `abov3 history export`
Export conversation.

**Syntax:**
```bash
abov3 history export CONVERSATION_ID OUTPUT_FILE
```

**Examples:**
```bash
# Export as JSON
abov3 history export abc123def456 conversation.json

# Export as Markdown
abov3 history export abc123def456 conversation.md
```

### `abov3 history clear`
Clear all history.

**Syntax:**
```bash
abov3 history clear [OPTIONS]
```

**Options:**
- `--confirm` - Skip confirmation prompt

**Examples:**
```bash
# Clear with confirmation
abov3 history clear

# Clear without confirmation
abov3 history clear --confirm
```

---

## Plugin Commands

### `abov3 plugins`
Manage ABOV3 plugins.

**Subcommands:**
- `list` - List available plugins
- `enable` - Enable a plugin
- `disable` - Disable a plugin
- `info` - Show plugin information
- `install` - Install plugin from path

### `abov3 plugins list`
List available plugins.

**Syntax:**
```bash
abov3 plugins list [OPTIONS]
```

**Options:**
- `--enabled-only` - Show only enabled plugins

**Examples:**
```bash
# List all plugins
abov3 plugins list

# List only enabled plugins
abov3 plugins list --enabled-only
```

### `abov3 plugins enable`
Enable a plugin.

**Syntax:**
```bash
abov3 plugins enable PLUGIN_NAME
```

**Examples:**
```bash
# Enable git plugin
abov3 plugins enable git

# Enable file operations plugin
abov3 plugins enable file_ops
```

### `abov3 plugins disable`
Disable a plugin.

**Syntax:**
```bash
abov3 plugins disable PLUGIN_NAME
```

**Examples:**
```bash
# Disable git plugin
abov3 plugins disable git

# Disable file operations plugin
abov3 plugins disable file_ops
```

### `abov3 plugins info`
Show plugin information.

**Syntax:**
```bash
abov3 plugins info PLUGIN_NAME
```

**Examples:**
```bash
# Get plugin information
abov3 plugins info git
abov3 plugins info file_ops
```

### `abov3 plugins install`
Install plugin from path.

**Syntax:**
```bash
abov3 plugins install PATH
```

**Examples:**
```bash
# Install from directory
abov3 plugins install /path/to/plugin

# Install from Git repository
abov3 plugins install git+https://github.com/user/plugin.git
```

---

## System Commands

### `abov3 update`
Check for and install updates.

**Syntax:**
```bash
abov3 update [OPTIONS]
```

**Options:**
- `--check-only` - Only check for updates, don't install

**Examples:**
```bash
# Check and install updates
abov3 update

# Only check for updates
abov3 update --check-only
```

### `abov3 doctor`
Run diagnostic checks on your ABOV3 installation.

**Syntax:**
```bash
abov3 doctor
```

**Examples:**
```bash
# Run health checks
abov3 doctor
```

**Checks Performed:**
- Ollama connection status
- Default model availability
- Configuration directory structure
- Plugin system integrity
- Logging system status

---

## REPL Commands

These commands are available within the interactive chat session:

### `/help`
Show help information.

**Syntax:**
```
/help
```

### `/clear`
Clear the screen.

**Syntax:**
```
/clear
```

### `/history`
Show command history.

**Syntax:**
```
/history
```

### `/save`
Save current session.

**Syntax:**
```
/save [filename]
```

**Examples:**
```
/save
/save my_session.json
/save important_conversation.json
```

### `/load`
Load a saved session.

**Syntax:**
```
/load filename
```

**Examples:**
```
/load my_session.json
/load important_conversation.json
```

### `/config`
Show or modify configuration.

**Syntax:**
```
/config [key=value]
```

**Examples:**
```
/config
/config theme=github
/config enable_streaming=true
```

### `/theme`
Change color theme.

**Syntax:**
```
/theme [theme_name]
```

**Available Themes:**
- `monokai` (default)
- `github`
- `solarized`
- `material`
- `dracula`

**Examples:**
```
/theme
/theme github
/theme monokai
```

### `/mode`
Switch key binding mode.

**Syntax:**
```
/mode [mode_name]
```

**Available Modes:**
- `emacs` (default)
- `vi`
- `custom`

**Examples:**
```
/mode
/mode vi
/mode emacs
```

### `/debug`
Toggle debug mode.

**Syntax:**
```
/debug
```

### `/context`
Show current context.

**Syntax:**
```
/context
```

### `/reset`
Reset the session.

**Syntax:**
```
/reset
```

### `/export`
Export session to file.

**Syntax:**
```
/export [filename]
```

**Examples:**
```
/export
/export session_report.md
/export conversation_export.md
```

### `/import`
Import session from file.

**Syntax:**
```
/import filename
```

**Examples:**
```
/import session_backup.json
/import previous_conversation.json
```

### `/status`
Show system status.

**Syntax:**
```
/status
```

### `/models`
List available AI models.

**Syntax:**
```
/models
```

### `/exit` or `/quit`
Exit the REPL.

**Syntax:**
```
/exit
/quit
```

---

## CLI Arguments

### Global Options

Available for all commands:

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Configuration file path | auto-detect |
| `--debug` | `-d` | Enable debug mode | false |
| `--version` | `-v` | Show version | - |
| `--no-banner` | | Suppress banner | false |
| `--help` | | Show help | - |

### Chat Options

Available for `abov3 chat`:

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--model` | `-m` | string | Model to use | from config |
| `--system` | `-s` | string | System prompt | none |
| `--temperature` | `-t` | float | Response temperature | from config |
| `--no-history` | | flag | Don't save to history | false |
| `--continue-last` | | flag | Continue last conversation | false |

### Configuration Options

Available for `abov3 config show`:

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--format` | `-f` | choice | Output format (table/json/yaml) | table |

### Model Options

Available for `abov3 models list`:

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--format` | `-f` | choice | Output format (table/json) | table |

Available for `abov3 models install`:

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--progress` | | flag | Show download progress | true |

### History Options

Available for `abov3 history list`:

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--limit` | `-l` | integer | Max results to show | 10 |
| `--format` | `-f` | choice | Output format (table/json) | table |

Available for `abov3 history search`:

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--limit` | `-l` | integer | Max results to show | 10 |

### Plugin Options

Available for `abov3 plugins list`:

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--enabled-only` | | flag | Show only enabled plugins | false |

### Update Options

Available for `abov3 update`:

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--check-only` | | flag | Only check, don't install | false |

---

## Environment Variables

### Core Configuration

| Variable | Description | Example |
|----------|-------------|---------|
| `ABOV3_CONFIG_PATH` | Path to configuration file | `/home/user/abov3.toml` |
| `ABOV3_DEBUG` | Enable debug mode | `true`, `false` |

### Ollama Configuration

| Variable | Description | Example |
|----------|-------------|---------|
| `ABOV3_OLLAMA_HOST` | Ollama server host | `http://localhost:11434` |
| `ABOV3_OLLAMA_TIMEOUT` | Request timeout (seconds) | `120` |

### Model Configuration

| Variable | Description | Example |
|----------|-------------|---------|
| `ABOV3_DEFAULT_MODEL` | Default model to use | `llama3.2:latest` |
| `ABOV3_TEMPERATURE` | Default temperature | `0.7` |
| `ABOV3_MAX_TOKENS` | Maximum tokens | `4096` |

### Logging Configuration

| Variable | Description | Example |
|----------|-------------|---------|
| `ABOV3_LOG_LEVEL` | Logging level | `DEBUG`, `INFO`, `WARNING` |
| `ABOV3_LOG_FILE` | Log file path | `/tmp/abov3.log` |

### Usage Examples

**Linux/macOS:**
```bash
export ABOV3_DEBUG=true
export ABOV3_OLLAMA_HOST=http://192.168.1.100:11434
export ABOV3_DEFAULT_MODEL=codellama:latest
abov3 chat
```

**Windows (Command Prompt):**
```cmd
set ABOV3_DEBUG=true
set ABOV3_OLLAMA_HOST=http://192.168.1.100:11434
set ABOV3_DEFAULT_MODEL=codellama:latest
abov3 chat
```

**Windows (PowerShell):**
```powershell
$env:ABOV3_DEBUG="true"
$env:ABOV3_OLLAMA_HOST="http://192.168.1.100:11434"
$env:ABOV3_DEFAULT_MODEL="codellama:latest"
abov3 chat
```

---

## Keyboard Shortcuts

### REPL Navigation

**Emacs Mode (Default):**
| Shortcut | Action |
|----------|--------|
| `Ctrl+A` | Move to beginning of line |
| `Ctrl+E` | Move to end of line |
| `Ctrl+F` | Move forward one character |
| `Ctrl+B` | Move backward one character |
| `Alt+F` | Move forward one word |
| `Alt+B` | Move backward one word |
| `Ctrl+D` | Delete character or exit |
| `Ctrl+K` | Kill to end of line |
| `Ctrl+U` | Kill to beginning of line |
| `Ctrl+W` | Kill previous word |
| `Ctrl+Y` | Yank (paste) |
| `Ctrl+P` | Previous history |
| `Ctrl+N` | Next history |
| `Ctrl+R` | Reverse search |
| `Ctrl+L` | Clear screen |
| `Ctrl+C` | Interrupt current operation |

**Vi Mode:**
| Shortcut | Action |
|----------|--------|
| `Esc` | Enter command mode |
| `i` | Enter insert mode |
| `a` | Append (insert after cursor) |
| `A` | Append at end of line |
| `0` | Move to beginning of line |
| `$` | Move to end of line |
| `w` | Move forward one word |
| `b` | Move backward one word |
| `x` | Delete character |
| `dd` | Delete line |
| `D` | Delete to end of line |
| `u` | Undo |
| `j` | Previous history |
| `k` | Next history |
| `/` | Search |

### Special Keys

| Key | Action |
|-----|--------|
| `Tab` | Auto-complete |
| `Shift+Tab` | Reverse auto-complete |
| `Enter` | Submit input |
| `Shift+Enter` | New line (multi-line mode) |
| `Ctrl+D` | Exit REPL |
| `Ctrl+C` | Interrupt/Cancel |
| `Page Up` | Scroll up |
| `Page Down` | Scroll down |

### Mouse Support

When enabled:
| Action | Function |
|--------|----------|
| `Click` | Position cursor |
| `Drag` | Select text |
| `Double-click` | Select word |
| `Right-click` | Context menu |
| `Scroll` | Navigate output |

---

## Exit Codes

ABOV3 uses standard exit codes to indicate the result of operations:

| Code | Name | Description |
|------|------|-------------|
| `0` | `EXIT_SUCCESS` | Operation completed successfully |
| `1` | `EXIT_FAILURE` | General failure |
| `2` | `EXIT_CONFIG_ERROR` | Configuration error |
| `3` | `EXIT_NETWORK_ERROR` | Network/connection error |
| `4` | `EXIT_MODEL_ERROR` | Model-related error |
| `5` | `EXIT_PLUGIN_ERROR` | Plugin system error |

### Usage in Scripts

```bash
# Check if ABOV3 command succeeded
abov3 models list
if [ $? -eq 0 ]; then
    echo "Models listed successfully"
else
    echo "Failed to list models"
    exit 1
fi

# Check specific error types
abov3 config validate
case $? in
    0)  echo "Configuration is valid" ;;
    2)  echo "Configuration error detected" ;;
    3)  echo "Cannot connect to Ollama" ;;
    *)  echo "Unexpected error" ;;
esac
```

---

## Command Examples

### Basic Workflow

```bash
# 1. Check system health
abov3 doctor

# 2. List available models
abov3 models list

# 3. Install a coding model
abov3 models install codellama:latest

# 4. Set as default
abov3 models set-default codellama:latest

# 5. Start coding session
abov3 chat -t 0.7 -s "You are an expert programmer"

# 6. Search previous conversations
abov3 history search "python function"

# 7. Enable useful plugins
abov3 plugins enable git
abov3 plugins enable file_ops
```

### Configuration Management

```bash
# View current configuration
abov3 config show

# Optimize for code generation
abov3 config set model.temperature 0.3
abov3 config set model.max_tokens 2048
abov3 config set ui.syntax_highlighting true

# Set up remote Ollama
abov3 config set ollama.host http://server.local:11434
abov3 config set ollama.timeout 180

# Validate changes
abov3 config validate
```

### Advanced Usage

```bash
# Debug mode with custom config
abov3 --debug --config ./dev-config.toml chat

# Batch operations
for model in codellama:7b mistral:latest llama3.2:latest; do
    abov3 models install "$model"
done

# Export all conversations
abov3 history list --format json | jq -r '.[].id' | while read id; do
    abov3 history export "$id" "export_${id}.md"
done
```

---

## Integration Examples

### Shell Scripts

```bash
#!/bin/bash
# Generate code documentation
generate_docs() {
    local file=$1
    echo "Generating documentation for $file"
    
    echo "Generate comprehensive documentation for this code file:" > prompt.txt
    cat "$file" >> prompt.txt
    
    abov3 chat --no-history < prompt.txt > "${file}.docs.md"
    
    if [ $? -eq 0 ]; then
        echo "Documentation generated: ${file}.docs.md"
    else
        echo "Failed to generate documentation for $file"
    fi
}

# Generate docs for all Python files
find . -name "*.py" -exec bash -c 'generate_docs "$0"' {} \;
```

### Git Hooks

```bash
#!/bin/bash
# pre-commit hook for code review
echo "Running ABOV3 code review..."

# Get modified files
modified_files=$(git diff --cached --name-only --diff-filter=AM | grep -E '\.(py|js|ts)$')

if [ -n "$modified_files" ]; then
    for file in $modified_files; do
        echo "Reviewing $file..."
        echo "Review this code for best practices and potential issues:" > review_prompt.txt
        cat "$file" >> review_prompt.txt
        
        abov3 chat --no-history < review_prompt.txt > "review_${file##*/}.md"
    done
    
    echo "Code review completed. Check review_*.md files for feedback."
fi
```

### API Integration

```python
#!/usr/bin/env python3
"""
ABOV3 API wrapper example
"""
import subprocess
import json

class ABOV3API:
    def __init__(self, model="llama3.2:latest"):
        self.model = model
    
    def chat(self, message, system_prompt=None, temperature=0.7):
        """Send a chat message and get response"""
        cmd = ["abov3", "chat", "--no-history", "-m", self.model, "-t", str(temperature)]
        
        if system_prompt:
            cmd.extend(["-s", system_prompt])
        
        # Send message via stdin
        result = subprocess.run(
            cmd,
            input=message,
            text=True,
            capture_output=True
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            raise Exception(f"ABOV3 error: {result.stderr}")
    
    def generate_code(self, specification):
        """Generate code from specification"""
        system_prompt = "You are an expert programmer. Generate clean, well-documented code."
        return self.chat(specification, system_prompt, temperature=0.3)
    
    def review_code(self, code):
        """Review code for issues"""
        system_prompt = "You are a senior code reviewer. Identify issues and suggest improvements."
        message = f"Review this code:\n\n{code}"
        return self.chat(message, system_prompt, temperature=0.2)

# Usage example
if __name__ == "__main__":
    api = ABOV3API()
    
    # Generate code
    spec = "Create a Python function to validate email addresses"
    code = api.generate_code(spec)
    print("Generated code:", code)
    
    # Review code
    review = api.review_code(code)
    print("Code review:", review)
```

---

This command reference provides comprehensive documentation for all ABOV3 4 Ollama CLI commands, options, and usage patterns. For additional help with specific commands, use the `--help` option:

```bash
abov3 --help
abov3 chat --help
abov3 config --help
# etc.
```

For practical examples and use cases, see the [User Manual](USER_MANUAL.md) and the examples directory in the project repository.