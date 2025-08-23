"""
ABOV3 Interactive REPL Console

A sophisticated REPL interface with syntax highlighting, auto-completion,
and rich formatting capabilities.
"""

import asyncio
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union

from prompt_toolkit import Application, PromptSession, print_formatted_text
from prompt_toolkit.application import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.clipboard import ClipboardData
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    HSplit, VSplit, Window, ConditionalContainer, Float, FloatContainer
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import PromptMargin
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit.styles import Style, merge_styles
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.widgets import (
    Button, Dialog, Label, MenuContainer, MenuItem, TextArea, SearchToolbar
)

from pygments.lexers import PythonLexer, get_lexer_by_name
from pygments.styles import get_style_by_name

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.traceback import Traceback

from .completers import ContextAwareCompleter, MultiCompleter
from .formatters import OutputFormatter, StreamingFormatter
from .keybindings import create_keybindings, KeyBindingMode
from ...utils.project_errors import (
    ABOV3ProjectError, ProjectNotSelectedException, FileOperationError,
    handle_project_error, format_error_for_user
)

# Windows compatibility imports
if sys.platform == "win32":
    try:
        from prompt_toolkit.output.win32 import Win32Output
        from prompt_toolkit.input.win32 import Win32Input
    except ImportError:
        Win32Output = None
        Win32Input = None


@dataclass
class REPLConfig:
    """Configuration for the ABOV3 REPL."""
    
    # UI Configuration
    prompt_text: str = "ABOV3> "
    multiline_prompt: str = "...    "
    theme: str = "monokai"
    color_depth: ColorDepth = ColorDepth.TRUE_COLOR
    enable_syntax_highlighting: bool = True
    enable_auto_suggestions: bool = True
    enable_completion: bool = True
    
    # Key Bindings
    key_binding_mode: KeyBindingMode = KeyBindingMode.EMACS
    enable_vim_mode: bool = False
    enable_system_prompt: bool = True
    
    # History Configuration
    history_file: Optional[Path] = None
    max_history_size: int = 10000
    enable_search: bool = True
    
    # Session Management
    session_file: Optional[Path] = None
    auto_save_session: bool = True
    session_save_interval: int = 300  # seconds
    
    # Display Configuration
    max_output_lines: int = 1000
    enable_pager: bool = True
    wrap_lines: bool = True
    show_line_numbers: bool = False
    
    # Performance
    async_mode: bool = True
    streaming_output: bool = True
    buffer_size: int = 4096
    
    # Advanced Features
    enable_multiline: bool = False  # Disable multiline to prevent ... prompt issues
    enable_mouse_support: bool = True
    enable_bracketed_paste: bool = True
    enable_suspend: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            k: str(v) if isinstance(v, (Path, ColorDepth, KeyBindingMode)) else v
            for k, v in self.__dict__.items()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'REPLConfig':
        """Create configuration from dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                if key == 'history_file' or key == 'session_file':
                    value = Path(value) if value else None
                elif key == 'color_depth':
                    value = ColorDepth[value] if isinstance(value, str) else value
                elif key == 'key_binding_mode':
                    value = KeyBindingMode[value] if isinstance(value, str) else value
                setattr(config, key, value)
        return config


class CommandProcessor:
    """Process REPL commands and special directives."""
    
    COMMANDS = {
        '/help': 'Show help information',
        '/clear': 'Clear the screen',
        '/history': 'Show command history',
        '/save': 'Save current session',
        '/load': 'Load a saved session',
        '/config': 'Show/modify configuration',
        '/theme': 'Change color theme',
        '/exit': 'Exit the REPL',
        '/quit': 'Exit the REPL',
        '/mode': 'Switch key binding mode',
        '/debug': 'Toggle debug mode',
        '/context': 'Show current context',
        '/reset': 'Reset the session',
        '/export': 'Export session to file',
        '/import': 'Import session from file',
        '/status': 'Show system status',
        '/models': 'List available AI models',
        # Project commands
        '/project': 'Select/change working directory',
        '/files': 'List files in current project directory',
        '/read': 'Read and display file contents',
        '/edit': 'Open file for editing',
        '/save_file': 'Save content to file',
        '/search': 'Search for text across project files',
        '/tree': 'Show project directory tree structure',
        '/analyze': 'Analyze file or project structure',
    }
    
    def __init__(self, repl: 'ABOV3REPL'):
        self.repl = repl
        self.handlers = self._create_handlers()
    
    def _create_handlers(self) -> Dict[str, Callable]:
        """Create command handlers."""
        return {
            '/help': self._cmd_help,
            '/clear': self._cmd_clear,
            '/history': self._cmd_history,
            '/save': self._cmd_save,
            '/load': self._cmd_load,
            '/config': self._cmd_config,
            '/theme': self._cmd_theme,
            '/exit': self._cmd_exit,
            '/quit': self._cmd_exit,
            '/mode': self._cmd_mode,
            '/debug': self._cmd_debug,
            '/context': self._cmd_context,
            '/reset': self._cmd_reset,
            '/export': self._cmd_export,
            '/import': self._cmd_import,
            '/status': self._cmd_status,
            '/models': self._cmd_models,
            # Project command handlers
            '/project': self._cmd_project,
            '/files': self._cmd_files,
            '/read': self._cmd_read,
            '/edit': self._cmd_edit,
            '/save_file': self._cmd_save_file,
            '/search': self._cmd_search,
            '/tree': self._cmd_tree,
            '/analyze': self._cmd_analyze,
        }
    
    def is_command(self, text: str) -> bool:
        """Check if text is a command."""
        return text.strip().startswith('/')
    
    async def process(self, text: str) -> Optional[str]:
        """Process a command with security validation."""
        # Basic input validation
        if not text or not text.strip():
            return "Empty command"
        
        # Prevent command injection by validating input
        text = text.strip()
        if len(text) > 1000:  # Reasonable command length limit
            return "Command too long (max 1000 characters)"
        
        # Check for suspicious characters that might indicate injection attempts
        # Skip the check for the command part itself (before the first space)
        command_part = text.split(maxsplit=1)[0]
        args_part = text[len(command_part):].strip() if len(text) > len(command_part) else ""
        
        # Only check arguments for suspicious characters, not the command itself
        if args_part:
            suspicious_chars = [';', '&&', '||', '`', '$']
            for char in suspicious_chars:
                if char in args_part:
                    return f"Invalid character '{char}' in command arguments. Arguments should not contain shell operators."
        
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Validate command exists
        if command not in self.handlers:
            return f"Unknown command: {command}. Type /help for available commands."
        
        # Execute command with error handling
        try:
            handler = self.handlers[command]
            if asyncio.iscoroutinefunction(handler):
                return await handler(args)
            else:
                return handler(args)
        except Exception as e:
            return f"Command execution error: {str(e)}"
    
    def _cmd_help(self, args: str) -> str:
        """Show help information."""
        table = Table(title="ABOV3 REPL Commands", show_header=True)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        
        for cmd, desc in self.COMMANDS.items():
            table.add_row(cmd, desc)
        
        self.repl.console.print(table)
        return ""
    
    def _cmd_clear(self, args: str) -> str:
        """Clear the screen."""
        self.repl.console.clear()
        return ""
    
    def _cmd_history(self, args: str) -> str:
        """Show command history."""
        history = self.repl.get_history()
        if not history:
            return "No history available."
        
        table = Table(title="Command History", show_header=True)
        table.add_column("#", style="dim", width=6)
        table.add_column("Command", style="cyan")
        table.add_column("Timestamp", style="dim")
        
        for i, (cmd, timestamp) in enumerate(history[-20:], 1):
            table.add_row(str(i), cmd[:80] + "..." if len(cmd) > 80 else cmd, timestamp)
        
        self.repl.console.print(table)
        return ""
    
    async def _cmd_save(self, args: str) -> str:
        """Save current session."""
        filename = args.strip() or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        success = await self.repl.save_session(filename)
        return f"Session saved to {filename}" if success else "Failed to save session"
    
    async def _cmd_load(self, args: str) -> str:
        """Load a saved session."""
        filename = args.strip()
        if not filename:
            return "Please provide a session filename"
        success = await self.repl.load_session(filename)
        return f"Session loaded from {filename}" if success else "Failed to load session"
    
    def _cmd_config(self, args: str) -> str:
        """Show or modify configuration."""
        if not args:
            # Show current configuration
            config_dict = self.repl.config.to_dict()
            table = Table(title="Current Configuration", show_header=True)
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")
            
            for key, value in config_dict.items():
                table.add_row(key, str(value))
            
            self.repl.console.print(table)
            return ""
        else:
            # Modify configuration
            try:
                key, value = args.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if hasattr(self.repl.config, key):
                    # Convert value to appropriate type
                    current_value = getattr(self.repl.config, key)
                    if isinstance(current_value, bool):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(current_value, int):
                        value = int(value)
                    elif isinstance(current_value, Path):
                        value = Path(value)
                    
                    setattr(self.repl.config, key, value)
                    return f"Configuration updated: {key} = {value}"
                else:
                    return f"Unknown configuration key: {key}"
            except ValueError:
                return "Invalid format. Use: /config key=value"
    
    def _cmd_theme(self, args: str) -> str:
        """Change color theme."""
        theme = args.strip()
        if not theme:
            return "Available themes: monokai, github, solarized, material, dracula"
        
        try:
            self.repl.set_theme(theme)
            return f"Theme changed to: {theme}"
        except Exception as e:
            return f"Failed to set theme: {e}"
    
    def _cmd_exit(self, args: str) -> str:
        """Exit the REPL."""
        self.repl.running = False
        return "Goodbye!"
    
    def _cmd_mode(self, args: str) -> str:
        """Switch key binding mode."""
        mode = args.strip().upper()
        if mode in ['EMACS', 'VI', 'CUSTOM']:
            self.repl.config.key_binding_mode = KeyBindingMode[mode]
            self.repl.update_keybindings()
            return f"Key binding mode changed to: {mode}"
        return "Available modes: emacs, vi, custom"
    
    def _cmd_debug(self, args: str) -> str:
        """Toggle debug mode."""
        self.repl.debug_mode = not self.repl.debug_mode
        return f"Debug mode: {'ON' if self.repl.debug_mode else 'OFF'}"
    
    def _cmd_context(self, args: str) -> str:
        """Show current context."""
        context = self.repl.get_context()
        if not context:
            return "No context available"
        
        panel = Panel(
            json.dumps(context, indent=2),
            title="Current Context",
            border_style="cyan"
        )
        self.repl.console.print(panel)
        return ""
    
    def _cmd_reset(self, args: str) -> str:
        """Reset the session."""
        self.repl.reset_session()
        return "Session reset successfully"
    
    async def _cmd_export(self, args: str) -> str:
        """Export session to file."""
        filename = args.strip() or f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        success = await self.repl.export_session(filename)
        return f"Session exported to {filename}" if success else "Failed to export session"
    
    async def _cmd_import(self, args: str) -> str:
        """Import session from file."""
        filename = args.strip()
        if not filename:
            return "Please provide a filename to import"
        success = await self.repl.import_session(filename)
        return f"Session imported from {filename}" if success else "Failed to import session"
    
    def _cmd_status(self, args: str) -> str:
        """Show system status."""
        # Try to get status from app instance if available
        app_instance = getattr(self.repl, 'app_instance', None)
        if app_instance:
            try:
                # Get basic status info
                status_info = {
                    "State": app_instance.state.value,
                    "Uptime": f"{app_instance.metrics.get_uptime():.1f}s",
                    "Total Requests": app_instance.metrics.total_requests,
                    "Total Responses": app_instance.metrics.total_responses,
                    "Total Errors": app_instance.metrics.total_errors,
                    "Memory Usage": f"{app_instance.metrics.memory_usage_mb:.1f}MB",
                }
                
                # Add component health
                for component, health in app_instance._component_health.items():
                    status_info[f"{component.replace('_', ' ').title()}"] = "OK" if health else "FAILED"
                
                table = Table(title="System Status", show_header=True)
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="green")
                
                for key, value in status_info.items():
                    table.add_row(key, str(value))
                
                self.repl.console.print(table)
                return ""
            except Exception as e:
                return f"Error getting system status: {e}"
        else:
            return "System status not available - app instance not found"
    
    def _cmd_models(self, args: str) -> str:
        """List available AI models."""
        app_instance = getattr(self.repl, 'app_instance', None)
        if app_instance and app_instance.model_manager:
            try:
                # Try to get models synchronously if possible
                models = []
                if hasattr(app_instance.model_manager, 'list_models_sync'):
                    models = app_instance.model_manager.list_models_sync()
                else:
                    return "Model listing requires async operation - use CLI 'abov3 models list' instead"
                
                if not models:
                    return "No models available"
                
                table = Table(title="Available AI Models", show_header=True)
                table.add_column("Model Name", style="cyan")
                table.add_column("Size", style="green")
                table.add_column("Modified", style="yellow")
                
                for model in models:
                    table.add_row(
                        model.get("name", "Unknown"),
                        model.get("size", "Unknown"),
                        model.get("modified_at", "Unknown")
                    )
                
                self.repl.console.print(table)
                return ""
            except Exception as e:
                return f"Error listing models: {e}"
        else:
            return "Model manager not available"
    
    # Project command handlers
    async def _cmd_project(self, args: str) -> str:
        """Select/change working directory."""
        if not args.strip():
            # Show current project info
            project_manager = getattr(self.repl, 'project_manager', None)
            if project_manager:
                try:
                    project_info = await project_manager.get_project_info()
                    if project_info:
                        table = Table(title="Current Project", show_header=True)
                        table.add_column("Property", style="cyan")
                        table.add_column("Value", style="white")
                        
                        for key, value in project_info.items():
                            if isinstance(value, list):
                                value = ", ".join(map(str, value))
                            table.add_row(key.replace("_", " ").title(), str(value))
                        
                        self.repl.console.print(table)
                        return ""
                    else:
                        return "No project currently selected. Use '/project <path>' to select a project."
                except Exception as e:
                    return f"Error getting project info: {e}"
            else:
                return "Project manager not available"
        
        # Select new project
        project_path = args.strip()
        project_manager = getattr(self.repl, 'project_manager', None)
        
        if not project_manager:
            return "Project manager not available"
        
        try:
            success = await project_manager.select_project(project_path)
            if success:
                # Update REPL prompt to show project
                project_info = await project_manager.get_project_info()
                if project_info:
                    self.repl.config.prompt_text = f"ABOV3({project_info['name']})> "
                return f"Project selected: {project_path}"
            else:
                return f"Failed to select project: {project_path}"
        except ABOV3ProjectError as e:
            # Handle project-specific errors with user-friendly messages
            error_info = handle_project_error(e, {"command": "/project", "path": project_path})
            return format_error_for_user(error_info)
        except Exception as e:
            # Handle unexpected errors
            error_info = handle_project_error(e, {"command": "/project", "path": project_path})
            return format_error_for_user(error_info)
    
    async def _cmd_files(self, args: str) -> str:
        """List files in current project directory."""
        project_manager = getattr(self.repl, 'project_manager', None)
        if not project_manager:
            return "No project selected. Use '/project <path>' to select a project."
        
        # Parse arguments
        parts = args.split() if args.strip() else []
        pattern = parts[0] if parts else "*"
        max_results = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 50
        
        try:
            files = await project_manager.list_files(
                pattern=pattern,
                max_results=max_results
            )
            
            if not files:
                return f"No files found matching pattern: {pattern}"
            
            table = Table(title=f"Project Files ({len(files)} found)", show_header=True)
            table.add_column("Path", style="cyan", no_wrap=False)
            table.add_column("Size", style="green", justify="right")
            table.add_column("Modified", style="yellow")
            table.add_column("Type", style="magenta")
            table.add_column("Status", style="red")
            
            for file_info in files:
                size_str = self._format_file_size(file_info['size'])
                modified_str = file_info['modified'].split('T')[0]  # Just date
                status = "Modified" if file_info.get('is_modified') else ""
                
                table.add_row(
                    file_info['path'],
                    size_str,
                    modified_str,
                    file_info['type'],
                    status
                )
            
            self.repl.console.print(table)
            return ""
            
        except Exception as e:
            return f"Error listing files: {e}"
    
    async def _cmd_read(self, args: str) -> str:
        """Read and display file contents."""
        if not args.strip():
            return "Usage: /read <file_path>"
        
        project_manager = getattr(self.repl, 'project_manager', None)
        if not project_manager:
            return "No project selected. Use '/project <path>' to select a project."
        
        file_path = args.strip()
        
        try:
            content = await project_manager.read_file(file_path)
            
            # Detect file type for syntax highlighting
            file_ext = os.path.splitext(file_path)[1].lower()
            language_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.html': 'html',
                '.css': 'css',
                '.json': 'json',
                '.yaml': 'yaml',
                '.yml': 'yaml',
                '.md': 'markdown',
                '.sh': 'bash',
                '.sql': 'sql'
            }
            
            language = language_map.get(file_ext, 'text')
            
            # Create panel with syntax highlighting
            syntax = Syntax(content, language, theme=self.repl.config.theme, line_numbers=True)
            panel = Panel(
                syntax,
                title=f"File: {file_path}",
                border_style="cyan",
                expand=False
            )
            
            self.repl.console.print(panel)
            return ""
            
        except Exception as e:
            return f"Error reading file: {e}"
    
    async def _cmd_edit(self, args: str) -> str:
        """Open file for editing (placeholder - would integrate with external editor)."""
        if not args.strip():
            return "Usage: /edit <file_path>"
        
        file_path = args.strip()
        
        # For now, just read the file and suggest using /save_file to write changes
        try:
            result = await self._cmd_read(file_path)
            if result:
                return result
            
            return f"\nFile displayed above. To save changes, use: /save_file {file_path} <content>\nOr use your preferred editor to modify the file."
            
        except Exception as e:
            return f"Error accessing file: {e}"
    
    async def _cmd_save_file(self, args: str) -> str:
        """Save content to file."""
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /save_file <file_path> <content>"
        
        file_path = parts[0]
        content = parts[1]
        
        project_manager = getattr(self.repl, 'project_manager', None)
        if not project_manager:
            return "No project selected. Use '/project <path>' to select a project."
        
        try:
            await project_manager.write_file(file_path, content)
            return f"File saved: {file_path}"
            
        except Exception as e:
            return f"Error saving file: {e}"
    
    async def _cmd_search(self, args: str) -> str:
        """Search for text across project files."""
        if not args.strip():
            return "Usage: /search <query> [file_pattern] [max_results]"
        
        project_manager = getattr(self.repl, 'project_manager', None)
        if not project_manager:
            return "No project selected. Use '/project <path>' to select a project."
        
        # Parse arguments
        parts = args.split()
        query = parts[0]
        file_pattern = parts[1] if len(parts) > 1 else "*"
        max_results = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 20
        
        try:
            results = await project_manager.search_files(
                query=query,
                file_pattern=file_pattern,
                max_results=max_results
            )
            
            if not results:
                return f"No matches found for: {query}"
            
            # Display search results
            for result in results:
                file_panel = Panel(
                    f"File: {result['file_path']}\nMatches: {result['total_matches']}\n",
                    title="Search Result",
                    border_style="green"
                )
                self.repl.console.print(file_panel)
                
                # Show first few matches
                for match in result['matches'][:3]:  # Show max 3 matches per file
                    match_text = Text()
                    match_text.append(f"Line {match['line_number']}: ", style="bold yellow")
                    match_text.append(match['line_content'])
                    self.repl.console.print(match_text)
                
                if result['total_matches'] > 3:
                    self.repl.console.print(f"... and {result['total_matches'] - 3} more matches\n")
            
            return ""
            
        except Exception as e:
            return f"Error searching files: {e}"
    
    async def _cmd_tree(self, args: str) -> str:
        """Show project directory tree structure."""
        project_manager = getattr(self.repl, 'project_manager', None)
        if not project_manager:
            return "No project selected. Use '/project <path>' to select a project."
        
        # Parse max depth
        max_depth = 3
        if args.strip() and args.strip().isdigit():
            max_depth = int(args.strip())
        
        try:
            tree_data = await project_manager.get_project_tree(max_depth=max_depth)
            
            # Display tree structure
            self._display_tree(tree_data, "", is_last=True)
            return ""
            
        except Exception as e:
            return f"Error getting project tree: {e}"
    
    async def _cmd_analyze(self, args: str) -> str:
        """Analyze file or project structure."""
        project_manager = getattr(self.repl, 'project_manager', None)
        if not project_manager:
            return "No project selected. Use '/project <path>' to select a project."
        
        if not args.strip():
            # Analyze project
            try:
                project_info = await project_manager.get_project_info()
                if not project_info:
                    return "No project information available"
                
                # Create analysis panel
                analysis_text = []
                analysis_text.append(f"Project: {project_info['name']}")
                analysis_text.append(f"Type: {project_info['type']}")
                analysis_text.append(f"Total Files: {project_info['total_files']}")
                analysis_text.append(f"Total Size: {self._format_file_size(project_info['total_size'])}")
                
                if project_info['languages']:
                    analysis_text.append(f"Languages: {', '.join(project_info['languages'])}")
                
                if project_info['frameworks']:
                    analysis_text.append(f"Frameworks: {', '.join(project_info['frameworks'])}")
                
                if project_info['modified_files'] > 0:
                    analysis_text.append(f"Modified Files: {project_info['modified_files']}")
                
                panel = Panel(
                    "\n".join(analysis_text),
                    title="Project Analysis",
                    border_style="cyan"
                )
                self.repl.console.print(panel)
                return ""
                
            except Exception as e:
                return f"Error analyzing project: {e}"
        else:
            # Analyze specific file
            file_path = args.strip()
            return f"File analysis not yet implemented for: {file_path}"
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"
    
    def _display_tree(self, node: Dict[str, Any], prefix: str, is_last: bool) -> None:
        """Display tree structure recursively."""
        if node is None:
            return
        
        # Display current node
        connector = "└── " if is_last else "├── "
        name = node.get('name', 'Unknown')
        
        if node.get('type') == 'directory':
            name = f"[bold blue]{name}/[/bold blue]"
        elif node.get('type') == 'file':
            name = f"[green]{name}[/green]"
            if node.get('is_modified'):
                name += " [red]*[/red]"
        
        self.repl.console.print(f"{prefix}{connector}{name}")
        
        # Display children
        children = node.get('children', {})
        if children:
            child_prefix = prefix + ("    " if is_last else "│   ")
            child_items = list(children.items())
            
            for i, (child_name, child_node) in enumerate(child_items):
                is_last_child = i == len(child_items) - 1
                self._display_tree(child_node, child_prefix, is_last_child)


class ABOV3REPL:
    """Main REPL class for ABOV3 interactive console."""
    
    def __init__(
        self,
        config: Optional[REPLConfig] = None,
        process_callback: Optional[Callable[[str], Union[str, Any]]] = None,
        project_manager=None
    ):
        """
        Initialize the ABOV3 REPL.
        
        Args:
            config: REPL configuration
            process_callback: Callback function to process user input
            project_manager: Project manager instance for file operations
        """
        self.config = config or REPLConfig()
        self.process_callback = process_callback
        self.project_manager = project_manager
        self.running = False
        self.debug_mode = False
        
        # Initialize Rich console
        self.console = Console(
            color_system="truecolor" if self.config.color_depth == ColorDepth.TRUE_COLOR else "256",
            force_terminal=True,
            force_jupyter=False,
            stderr=False  # Prevent backend logs from appearing in REPL output
        )
        
        # Initialize prompt session
        self.session = None
        self.history = []
        self.context = {}
        self.command_processor = CommandProcessor(self)
        
        # Formatters
        self.output_formatter = OutputFormatter(self.console)
        self.streaming_formatter = StreamingFormatter(self.console)
        
        # Setup components (order is important!)
        self._setup_history()
        self._setup_style()  # Must come before _setup_session()
        self._setup_session()
    
    def _setup_history(self):
        """Setup command history."""
        if self.config.history_file:
            self.config.history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history_obj = FileHistory(str(self.config.history_file))
        else:
            self.history_obj = InMemoryHistory()
    
    def _setup_session(self):
        """Setup the prompt session."""
        # Create completer
        completer = ContextAwareCompleter(self)
        
        # Create key bindings
        key_bindings = create_keybindings(
            self.config.key_binding_mode,
            enable_vim=self.config.enable_vim_mode
        )
        
        # Ensure style is available (safety check)
        if not hasattr(self, 'style') or self.style is None:
            self._setup_style()
        
        # Prepare session arguments with Windows compatibility
        session_args = {
            # Don't set message here - we'll pass it in prompt_async calls
            'multiline': False,  # Always start in single-line mode
            'history': self.history_obj,
            'auto_suggest': AutoSuggestFromHistory() if self.config.enable_auto_suggestions else None,
            'completer': completer if self.config.enable_completion else None,
            'complete_while_typing': True,
            'key_bindings': key_bindings,
            'enable_history_search': self.config.enable_search,
            'mouse_support': self.config.enable_mouse_support,
            'wrap_lines': self.config.wrap_lines,
            'enable_suspend': self.config.enable_suspend,
            'color_depth': self.config.color_depth,
            'style': self.style,
            'include_default_pygments_style': False,
            'lexer': PygmentsLexer(PythonLexer) if self.config.enable_syntax_highlighting else None,
        }
        
        # Handle Windows and terminal compatibility issues
        if sys.platform == "win32":
            # Force disable mouse support and reduce color depth for compatibility
            session_args['mouse_support'] = False
            session_args['color_depth'] = ColorDepth.DEPTH_8_BIT
            
            if Win32Output and Win32Input:
                try:
                    session_args['output'] = Win32Output(sys.stdout)
                    session_args['input'] = Win32Input(sys.stdin)
                except Exception:
                    # If Windows I/O fails, fall back to defaults
                    pass
        
        # Additional compatibility for Git Bash/MinGW environments
        if 'TERM' in os.environ and 'xterm' in os.environ.get('TERM', ''):
            # Force compatibility mode for xterm-like terminals
            session_args['mouse_support'] = False
            session_args['color_depth'] = ColorDepth.DEPTH_8_BIT
        
        # Create the prompt session with fallback
        try:
            self.session = PromptSession(**session_args)
        except Exception as e:
            # If PromptSession fails, create a fallback session with minimal features
            fallback_args = {
                'multiline': False,
                'history': self.history_obj,
                'mouse_support': False,
                'color_depth': ColorDepth.DEPTH_8_BIT,
                'enable_suspend': False,
            }
            try:
                self.session = PromptSession(**fallback_args)
            except Exception as e2:
                # If even the fallback fails, use the simple fallback mode
                self.console.print(f"[yellow]Warning: Advanced REPL features disabled due to terminal compatibility issues[/yellow]")
                self.console.print(f"[dim]Reason: {e}[/dim]")
                self.session = None  # Will use simple input() fallback
    
    def _setup_style(self):
        """Setup the color style."""
        # Always initialize with a fallback style first
        default_style = Style.from_dict({
            'completion-menu.completion': 'bg:#008888 #ffffff',
            'completion-menu.completion.current': 'bg:#00aaaa #000000',
            'scrollbar.background': 'bg:#88aaaa',
            'scrollbar.button': 'bg:#222222',
            'prompt': 'cyan bold',
            'continuation': 'cyan',
        })
        
        try:
            # Try to load the configured theme
            pygments_style = get_style_by_name(self.config.theme)
            self.style = Style.from_pygments_cls(pygments_style)
        except Exception as e:
            # Log the error if debug mode is enabled
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"Warning: Failed to load theme '{self.config.theme}': {e}")
            # Use fallback style
            self.style = default_style
            # Reset theme to a known working one
            self.config.theme = "default"
    
    def set_theme(self, theme_name: str):
        """Change the color theme."""
        try:
            # Validate the theme exists first
            pygments_style = get_style_by_name(theme_name)
            new_style = Style.from_pygments_cls(pygments_style)
            
            # Only update if successful
            self.style = new_style
            self.config.theme = theme_name
            
            # Update the session if it exists
            if self.session:
                self.session.style = self.style
                
        except Exception as e:
            # Don't change anything if theme is invalid
            raise ValueError(f"Invalid theme: {theme_name}. Available themes include: monokai, github, solarized, material, dracula") from e
    
    def update_keybindings(self):
        """Update key bindings based on current configuration."""
        if self.session:
            key_bindings = create_keybindings(
                self.config.key_binding_mode,
                enable_vim=self.config.enable_vim_mode
            )
            self.session.key_bindings = key_bindings
    
    async def process_input(self, text: str) -> Any:
        """
        Process user input.
        
        Args:
            text: Input text from user
            
        Returns:
            Processed result
        """
        # Check if it's a command
        if self.command_processor.is_command(text):
            return await self.command_processor.process(text)
        
        # Process through callback if available
        if self.process_callback:
            if asyncio.iscoroutinefunction(self.process_callback):
                return await self.process_callback(text)
            else:
                return self.process_callback(text)
        
        # Default: echo back
        return f"Received: {text}"
    
    def display_output(self, output: Any):
        """
        Display output with rich formatting.
        
        Args:
            output: Output to display
        """
        if output is None or output == "":
            return
        
        if isinstance(output, str):
            # Check if it's code
            if output.startswith("```"):
                # Extract language and code
                lines = output.split('\n')
                lang = lines[0][3:].strip() or "python"
                code = '\n'.join(lines[1:-1] if lines[-1] == "```" else lines[1:])
                syntax = Syntax(code, lang, theme=self.config.theme, line_numbers=self.config.show_line_numbers)
                self.console.print(syntax)
            else:
                # Regular text or markdown
                if any(marker in output for marker in ['#', '**', '`', '-', '*']):
                    self.console.print(Markdown(output))
                else:
                    self.console.print(output)
        elif isinstance(output, dict):
            # Format as JSON
            syntax = Syntax(
                json.dumps(output, indent=2),
                "json",
                theme=self.config.theme,
                line_numbers=self.config.show_line_numbers
            )
            self.console.print(syntax)
        elif isinstance(output, (list, tuple)):
            # Format as table if appropriate
            if output and all(isinstance(item, dict) for item in output):
                self._display_table(output)
            else:
                for item in output:
                    self.display_output(item)
        else:
            # Default representation
            self.console.print(repr(output))
    
    def _display_table(self, data: List[Dict]):
        """Display list of dicts as a table."""
        if not data:
            return
        
        table = Table(show_header=True, header_style="bold magenta")
        
        # Add columns
        keys = list(data[0].keys())
        for key in keys:
            table.add_column(key, style="cyan")
        
        # Add rows
        for item in data:
            row = [str(item.get(key, "")) for key in keys]
            table.add_row(*row)
        
        self.console.print(table)
    
    def display_error(self, error: Exception):
        """
        Display error with rich formatting.
        
        Args:
            error: Exception to display
        """
        if self.debug_mode:
            # Show full traceback
            tb = Traceback.from_exception(
                type(error),
                error,
                error.__traceback__,
                show_locals=True,
                suppress=[asyncio],
                max_frames=20
            )
            self.console.print(tb)
        else:
            # Show simple error message
            self.console.print(f"[red]Error:[/red] {error}", style="bold red")
    
    async def stream_output(self, generator):
        """
        Stream output from a generator.
        
        Args:
            generator: Async generator yielding output chunks
        """
        with Live(console=self.console, refresh_per_second=10) as live:
            full_output = ""
            async for chunk in generator:
                full_output += chunk
                live.update(Text(full_output))
        
        return full_output
    
    def get_history(self) -> List[tuple]:
        """Get command history."""
        return self.history
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context."""
        return self.context
    
    def reset_session(self):
        """Reset the current session."""
        self.history.clear()
        self.context.clear()
        if self.history_obj:
            self.history_obj.reset()
    
    async def save_session(self, filename: str) -> bool:
        """
        Save the current session to a file.
        
        Args:
            filename: File to save session to
            
        Returns:
            True if successful
        """
        try:
            session_data = {
                'config': self.config.to_dict(),
                'history': self.history,
                'context': self.context,
                'timestamp': datetime.now().isoformat()
            }
            
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            self.display_error(e)
            return False
    
    async def load_session(self, filename: str) -> bool:
        """
        Load a session from a file.
        
        Args:
            filename: File to load session from
            
        Returns:
            True if successful
        """
        try:
            with open(filename, 'r') as f:
                session_data = json.load(f)
            
            if 'config' in session_data:
                self.config = REPLConfig.from_dict(session_data['config'])
                self._setup_session()
            
            if 'history' in session_data:
                self.history = session_data['history']
            
            if 'context' in session_data:
                self.context = session_data['context']
            
            return True
        except Exception as e:
            self.display_error(e)
            return False
    
    async def export_session(self, filename: str) -> bool:
        """
        Export session to a markdown file.
        
        Args:
            filename: File to export to
            
        Returns:
            True if successful
        """
        try:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                f.write(f"# ABOV3 Session Export\n\n")
                f.write(f"**Date:** {datetime.now().isoformat()}\n\n")
                f.write("## Command History\n\n")
                
                for i, (cmd, timestamp) in enumerate(self.history, 1):
                    f.write(f"{i}. `{cmd}` - {timestamp}\n")
                
                if self.context:
                    f.write("\n## Context\n\n")
                    f.write("```json\n")
                    f.write(json.dumps(self.context, indent=2, default=str))
                    f.write("\n```\n")
            
            return True
        except Exception as e:
            self.display_error(e)
            return False
    
    async def import_session(self, filename: str) -> bool:
        """
        Import session from a file.
        
        Args:
            filename: File to import from
            
        Returns:
            True if successful
        """
        # For now, just load JSON sessions
        return await self.load_session(filename)
    
    async def run(self):
        """Run the REPL main loop."""
        self.running = True
        
        # Display welcome message
        welcome = Panel(
            Text.from_markup(
                "[bold cyan]Welcome to ABOV3 Interactive Console[/bold cyan]\n\n"
                "Type [yellow]/help[/yellow] for available commands\n"
                "Press [green]Ctrl+D[/green] or type [yellow]/exit[/yellow] to quit"
            ),
            title="ABOV3 v1.0.0",
            border_style="cyan"
        )
        self.console.print(welcome)
        
        while self.running:
            try:
                # Get user input - use fallback if session is None
                if self.session is not None:
                    # Use advanced prompt session
                    text = await self.session.prompt_async(
                        message=self.config.prompt_text,
                        multiline=False  # Force single-line to prevent ... prompts
                    )
                else:
                    # Fallback to simple input for compatibility
                    self.console.print(self.config.prompt_text, end="", style="cyan bold")
                    text = await asyncio.get_event_loop().run_in_executor(None, input)
                
                if not text.strip():
                    continue
                
                # Add to history
                self.history.append((text, datetime.now().isoformat()))
                
                # Process input
                try:
                    result = await self.process_input(text)
                    
                    # Display output - handle both string and AsyncIterator responses
                    if result is not None:
                        # Check if result is an async iterator (streaming response)
                        if hasattr(result, '__aiter__'):
                            # Stream the response
                            streamed_result = await self.stream_output(result)
                            if streamed_result and streamed_result.strip():
                                # Display the final result if needed (already displayed during streaming)
                                pass
                        elif result != "":
                            # Regular string response
                            self.display_output(result)
                except Exception as e:
                    self.display_error(e)
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Use /exit to quit.[/yellow]")
                continue
            except EOFError:
                # Handle Ctrl+D (EOF) - same as /exit command
                self.console.print("\n[cyan]Ctrl+D detected - exiting...[/cyan]")
                self.running = False
                break
            except Exception as e:
                self.display_error(e)
                # Continue running instead of crashing
                continue
        
        # Cleanup
        if self.config.auto_save_session and self.config.session_file:
            try:
                await self.save_session(str(self.config.session_file))
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to save session: {e}[/yellow]")
        
        # Display goodbye message
        self.console.print("\n[cyan]Goodbye![/cyan]")
    
    def run_sync(self):
        """Run the REPL synchronously."""
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            self.console.print("\n[cyan]Interrupted. Goodbye![/cyan]")


def create_repl(
    config: Optional[REPLConfig] = None,
    process_callback: Optional[Callable] = None
) -> ABOV3REPL:
    """
    Create an ABOV3 REPL instance.
    
    Args:
        config: REPL configuration
        process_callback: Callback to process user input
        
    Returns:
        ABOV3REPL instance
    """
    return ABOV3REPL(config=config, process_callback=process_callback)


if __name__ == "__main__":
    # Example usage
    def process_input(text: str) -> str:
        """Example input processor."""
        if text.startswith("eval "):
            try:
                result = eval(text[5:])
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"
        return f"Echo: {text}"
    
    # Create and run REPL
    config = REPLConfig(
        prompt_text="ABOV3> ",
        theme="monokai",
        enable_vim_mode=False,
        history_file=Path.home() / ".abov3" / "history.txt",
        session_file=Path.home() / ".abov3" / "session.json"
    )
    
    repl = create_repl(config=config, process_callback=process_input)
    repl.run_sync()