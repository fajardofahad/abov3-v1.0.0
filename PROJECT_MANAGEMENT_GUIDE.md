# ABOV3 Project Management Guide

## Overview

ABOV3 now includes comprehensive project management capabilities that transform it into a true coding assistant. This guide covers all the new features and how to use them effectively.

## New Features

### 🗂️ Project Directory Selection
- Select and switch between projects
- Automatic project type detection
- Project structure analysis
- Recent projects history

### 📁 File Management
- List, read, and edit project files
- Smart file filtering and search
- Directory tree visualization
- Safe file operations with backup

### 🧠 Context-Aware AI Assistance
- AI has access to your project structure and files
- Context-aware code suggestions
- Project-specific help and guidance
- Automatic context synchronization

### 🔍 Advanced Search
- Search across all project files
- Pattern-based file filtering
- Real-time file watching
- Change tracking

## Getting Started

### 1. Launch ABOV3 with Project Support

```bash
# Start ABOV3 in project mode
python -m abov3.core.project_app

# Or start with a specific project
python -m abov3.core.project_app /path/to/your/project
```

### 2. Basic Project Commands

```bash
# Select a project directory
/project /path/to/your/project

# View current project info
/project

# List project files
/files

# List specific file types
/files *.py
/files *.js 50  # limit to 50 results

# Show project directory tree
/tree
/tree 5  # max depth of 5 levels
```

### 3. File Operations

```bash
# Read a file
/read main.py
/read src/utils.py

# Edit a file (displays content for editing)
/edit main.py

# Save content to a file
/save_file new_script.py "print('Hello, World!')"

# Search across project files
/search "function_name"
/search "TODO" *.py 20  # search in Python files, max 20 results
```

### 4. Project Analysis

```bash
# Analyze current project
/analyze

# Get project statistics
/status

# View context information
/context
```

## Advanced Features

### Context-Aware Conversations

Once you select a project, ABOV3 automatically includes relevant project context in AI conversations:

```
You: "How can I improve the performance of my main.py file?"

AI: [Analyzes your actual main.py file and provides specific suggestions based on your code]
```

### Smart Code Assistance

The AI now understands your project structure and can provide:

- **Code Completion**: Context-aware suggestions based on your existing code
- **Refactoring Help**: Specific recommendations for your codebase
- **Bug Detection**: Analysis of your actual code for potential issues
- **Documentation**: Generate docs based on your code structure

### Real-Time Project Monitoring

ABOV3 automatically tracks changes to your project:
- Modified files are highlighted
- Context is updated when files change
- Recent changes are included in AI conversations

## Configuration

### Project Settings

You can customize project behavior in your configuration file:

```toml
[project]
# File size limit (10MB default)
max_file_size = 10485760

# Files to include in AI context
max_files_in_context = 50

# Auto-watch for file changes
auto_watch_changes = true

# Include project context in AI conversations
include_project_context = true

# File patterns to include
include_patterns = ["*.py", "*.js", "*.ts", "*.html", "*.css", "*.json", "*.md"]

# File patterns to exclude
exclude_patterns = ["node_modules/*", ".git/*", "__pycache__/*", "*.pyc"]
```

### Environment Variables

```bash
# Set maximum file size (in bytes)
export ABOV3_MAX_FILE_SIZE=20971520  # 20MB

# Enable/disable project context
export ABOV3_INCLUDE_PROJECT_CONTEXT=true

# Maximum files in context
export ABOV3_MAX_FILES_IN_CONTEXT=100
```

## Examples

### Example 1: Python Web App

```bash
# Select your Flask/Django project
/project ~/my-web-app

# Analyze the project structure
/analyze

# Look at the main application file
/read app.py

# Search for database models
/search "class.*Model" *.py

# Get AI help with optimization
You: "How can I optimize the database queries in my models?"
```

### Example 2: JavaScript/Node.js Project

```bash
# Select your Node.js project
/project ~/my-node-app

# Check project structure
/tree

# Look at package.json
/read package.json

# Find all async functions
/search "async function" *.js

# Get help with async/await patterns
You: "Can you help me convert these callback functions to async/await?"
```

### Example 3: Multi-Language Project

```bash
# Select a full-stack project
/project ~/my-fullstack-app

# See all file types
/files

# Look at frontend code
/read frontend/src/App.jsx

# Check backend code
/read backend/main.py

# Get architectural advice
You: "How should I structure the API communication between my React frontend and Python backend?"
```

## Best Practices

### 1. Project Organization
- Keep projects well-organized with clear directory structure
- Use meaningful file and directory names
- Include README files for better project understanding

### 2. File Management
- Regularly commit changes to version control
- Use `/tree` to understand project structure before making changes
- Leverage search functionality to find code patterns

### 3. AI Interactions
- Be specific about which files or components you're asking about
- Include context about what you're trying to achieve
- Ask for code examples that fit your project structure

### 4. Performance
- Exclude unnecessary files (node_modules, build artifacts) via configuration
- Use file patterns to limit context to relevant files
- Monitor project size - very large projects may need custom configuration

## Security

ABOV3 includes comprehensive security measures:

- **Path Validation**: Prevents directory traversal attacks
- **File Type Restrictions**: Configurable allowed/blocked file extensions
- **Size Limits**: Prevents processing of extremely large files
- **Permission Checks**: Validates file access permissions
- **Content Scanning**: Detects potentially malicious patterns

## Troubleshooting

### Common Issues

**"No project selected" error:**
```bash
# Make sure to select a project first
/project /path/to/your/project
```

**Permission denied errors:**
```bash
# Check file permissions
ls -la /path/to/your/project

# On Windows, run as administrator if needed
```

**Files not appearing in /files:**
```bash
# Check if files match include patterns
/files *  # show all files

# Check exclude patterns in configuration
```

**Search not finding results:**
```bash
# Try broader search patterns
/search "searchterm" *
/search "searchterm" *.* 100
```

### Performance Issues

**Large project taking too long:**
- Increase exclude patterns for build artifacts
- Reduce `max_files_in_context` setting
- Use more specific file patterns

**Memory usage high:**
- Check `max_file_size` setting
- Exclude large binary files
- Restart ABOV3 periodically for long sessions

### Getting Help

1. Check logs for detailed error information
2. Use `/debug` command for verbose output
3. Verify configuration with `/config`
4. Test with a smaller project first

## API Reference

### Project Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/project [path]` | Select project or show current | `/project ~/my-app` |
| `/files [pattern] [limit]` | List project files | `/files *.py 50` |
| `/read <file>` | Read file contents | `/read main.py` |
| `/edit <file>` | Edit file | `/edit config.py` |
| `/save_file <file> <content>` | Save file | `/save_file test.py "print('hi')"` |
| `/search <query> [pattern] [limit]` | Search files | `/search "TODO" *.py 20` |
| `/tree [depth]` | Show directory tree | `/tree 3` |
| `/analyze [file]` | Analyze project/file | `/analyze` |

### Configuration Options

See the configuration section above for all available options.

## What's Next?

This project management system opens up many possibilities:

- **IDE Integration**: Connect with VS Code, IntelliJ, etc.
- **Git Integration**: Enhanced version control awareness
- **Testing Support**: Automated test generation and running
- **Deployment**: Project-aware deployment assistance
- **Code Review**: AI-powered code review capabilities

The foundation is now in place for ABOV3 to become your comprehensive coding assistant!