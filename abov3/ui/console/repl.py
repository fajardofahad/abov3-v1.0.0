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
    enable_multiline: bool = True
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
        }
    
    def is_command(self, text: str) -> bool:
        """Check if text is a command."""
        return text.strip().startswith('/')
    
    async def process(self, text: str) -> Optional[str]:
        """Process a command."""
        parts = text.strip().split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command in self.handlers:
            handler = self.handlers[command]
            if asyncio.iscoroutinefunction(handler):
                return await handler(args)
            else:
                return handler(args)
        else:
            return f"Unknown command: {command}. Type /help for available commands."
    
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


class ABOV3REPL:
    """Main REPL class for ABOV3 interactive console."""
    
    def __init__(
        self,
        config: Optional[REPLConfig] = None,
        process_callback: Optional[Callable[[str], Union[str, Any]]] = None
    ):
        """
        Initialize the ABOV3 REPL.
        
        Args:
            config: REPL configuration
            process_callback: Callback function to process user input
        """
        self.config = config or REPLConfig()
        self.process_callback = process_callback
        self.running = False
        self.debug_mode = False
        
        # Initialize Rich console
        self.console = Console(
            color_system="truecolor" if self.config.color_depth == ColorDepth.TRUE_COLOR else "256",
            force_terminal=True,
            force_jupyter=False
        )
        
        # Initialize prompt session
        self.session = None
        self.history = []
        self.context = {}
        self.command_processor = CommandProcessor(self)
        
        # Formatters
        self.output_formatter = OutputFormatter(self.console)
        self.streaming_formatter = StreamingFormatter(self.console)
        
        # Setup components
        self._setup_history()
        self._setup_session()
        self._setup_style()
    
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
        
        # Create the prompt session
        self.session = PromptSession(
            message=self.config.prompt_text,
            multiline=self.config.enable_multiline,
            history=self.history_obj,
            auto_suggest=AutoSuggestFromHistory() if self.config.enable_auto_suggestions else None,
            completer=completer if self.config.enable_completion else None,
            complete_while_typing=True,
            key_bindings=key_bindings,
            enable_history_search=self.config.enable_search,
            mouse_support=self.config.enable_mouse_support,
            wrap_lines=self.config.wrap_lines,
            enable_suspend=self.config.enable_suspend,
            color_depth=self.config.color_depth,
            style=self.style,
            include_default_pygments_style=False,
            lexer=PygmentsLexer(PythonLexer) if self.config.enable_syntax_highlighting else None,
        )
    
    def _setup_style(self):
        """Setup the color style."""
        try:
            pygments_style = get_style_by_name(self.config.theme)
            self.style = Style.from_pygments_cls(pygments_style)
        except:
            # Fallback to default style
            self.style = Style.from_dict({
                'completion-menu.completion': 'bg:#008888 #ffffff',
                'completion-menu.completion.current': 'bg:#00aaaa #000000',
                'scrollbar.background': 'bg:#88aaaa',
                'scrollbar.button': 'bg:#222222',
            })
    
    def set_theme(self, theme_name: str):
        """Change the color theme."""
        try:
            pygments_style = get_style_by_name(theme_name)
            self.style = Style.from_pygments_cls(pygments_style)
            self.config.theme = theme_name
            if self.session:
                self.session.style = self.style
        except Exception as e:
            raise ValueError(f"Invalid theme: {theme_name}") from e
    
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
                # Get user input
                text = await self.session.prompt_async(
                    message=self.config.prompt_text,
                    multiline=self.config.enable_multiline,
                    prompt_continuation=self.config.multiline_prompt
                )
                
                if not text.strip():
                    continue
                
                # Add to history
                self.history.append((text, datetime.now().isoformat()))
                
                # Process input
                if self.config.async_mode:
                    result = await self.process_input(text)
                else:
                    result = self.process_input(text)
                
                # Display output
                if result is not None:
                    self.display_output(result)
                    
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                self.display_error(e)
        
        # Cleanup
        if self.config.auto_save_session and self.config.session_file:
            await self.save_session(str(self.config.session_file))
        
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