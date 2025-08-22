"""
ABOV3 Console Output Formatters

Provides rich formatting capabilities for console output.
"""

import json
import re
import textwrap
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from rich.console import Console, ConsoleOptions, RenderResult
from rich.highlighter import Highlighter, RegexHighlighter
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeRemainingColumn, TimeElapsedColumn
)
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.traceback import Traceback

from pygments import highlight
from pygments.formatters import TerminalFormatter, Terminal256Formatter
from pygments.lexers import (
    get_lexer_by_name, guess_lexer, PythonLexer,
    JsonLexer, YamlLexer, XmlLexer, SqlLexer
)
from pygments.styles import get_style_by_name


class OutputType(Enum):
    """Types of output that can be formatted."""
    TEXT = "text"
    CODE = "code"
    JSON = "json"
    TABLE = "table"
    TREE = "tree"
    ERROR = "error"
    MARKDOWN = "markdown"
    HTML = "html"
    XML = "xml"
    YAML = "yaml"
    SQL = "sql"
    DIFF = "diff"
    LOG = "log"


class CodeHighlighter(RegexHighlighter):
    """Custom syntax highlighter for code patterns."""
    
    base_style = "code."
    highlights = [
        # Python-like
        r"(?P<keyword>\b(def|class|import|from|as|if|else|elif|for|while|try|except|finally|with|return|yield|break|continue|pass|raise|assert|lambda|async|await)\b)",
        r"(?P<builtin>\b(True|False|None|self|cls)\b)",
        r"(?P<number>\b\d+\.?\d*\b)",
        r"(?P<string>\".*?\"|'.*?')",
        r"(?P<comment>#.*$)",
        r"(?P<decorator>@\w+)",
        
        # Function/method calls
        r"(?P<function>\w+)(?=\()",
        
        # URLs
        r"(?P<url>(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]*[-A-Za-z0-9+&@#/%=~_|])",
        
        # File paths
        r"(?P<path>(/[\w.-]+)+|[A-Za-z]:\\[\w\\.-]+)",
        
        # IP addresses
        r"(?P<ip>\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b)",
        
        # Hex values
        r"(?P<hex>0x[0-9a-fA-F]+)",
        
        # Timestamps
        r"(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})",
    ]


@dataclass
class FormatConfig:
    """Configuration for output formatting."""
    theme: str = "monokai"
    line_numbers: bool = False
    word_wrap: bool = True
    max_width: Optional[int] = None
    indent_size: int = 2
    highlight_code: bool = True
    use_colors: bool = True
    table_style: str = "rounded"
    panel_style: str = "cyan"
    error_style: str = "bold red"


class OutputFormatter:
    """Main formatter for all output types."""
    
    def __init__(self, console: Optional[Console] = None, config: Optional[FormatConfig] = None):
        """
        Initialize output formatter.
        
        Args:
            console: Rich console instance
            config: Formatting configuration
        """
        self.console = console or Console()
        self.config = config or FormatConfig()
        self.highlighter = CodeHighlighter()
        
        # Setup pygments style
        try:
            self.pygments_style = get_style_by_name(self.config.theme)
        except:
            self.pygments_style = get_style_by_name("default")
    
    def format(self, content: Any, output_type: Optional[OutputType] = None) -> None:
        """
        Format and display content based on its type.
        
        Args:
            content: Content to format
            output_type: Explicit output type
        """
        if output_type is None:
            output_type = self._detect_type(content)
        
        if output_type == OutputType.TEXT:
            self._format_text(content)
        elif output_type == OutputType.CODE:
            self._format_code(content)
        elif output_type == OutputType.JSON:
            self._format_json(content)
        elif output_type == OutputType.TABLE:
            self._format_table(content)
        elif output_type == OutputType.TREE:
            self._format_tree(content)
        elif output_type == OutputType.ERROR:
            self._format_error(content)
        elif output_type == OutputType.MARKDOWN:
            self._format_markdown(content)
        elif output_type == OutputType.HTML:
            self._format_html(content)
        elif output_type == OutputType.XML:
            self._format_xml(content)
        elif output_type == OutputType.YAML:
            self._format_yaml(content)
        elif output_type == OutputType.SQL:
            self._format_sql(content)
        elif output_type == OutputType.DIFF:
            self._format_diff(content)
        elif output_type == OutputType.LOG:
            self._format_log(content)
        else:
            self._format_text(str(content))
    
    def _detect_type(self, content: Any) -> OutputType:
        """Detect the type of content."""
        if isinstance(content, Exception):
            return OutputType.ERROR
        elif isinstance(content, dict):
            return OutputType.JSON
        elif isinstance(content, (list, tuple)) and content:
            if all(isinstance(item, dict) for item in content):
                return OutputType.TABLE
            else:
                return OutputType.JSON
        elif isinstance(content, str):
            # Try to detect content type from string
            if content.strip().startswith('{') or content.strip().startswith('['):
                try:
                    json.loads(content)
                    return OutputType.JSON
                except:
                    pass
            elif content.strip().startswith('```'):
                return OutputType.CODE
            elif content.strip().startswith('#') or '**' in content:
                return OutputType.MARKDOWN
            elif content.strip().startswith('<'):
                return OutputType.XML
            elif 'SELECT' in content.upper() or 'INSERT' in content.upper():
                return OutputType.SQL
            elif content.startswith('---') or content.startswith('diff'):
                return OutputType.DIFF
        
        return OutputType.TEXT
    
    def _format_text(self, content: Any):
        """Format plain text."""
        text = str(content)
        
        if self.config.highlight_code:
            text = self.highlighter.highlight(text)
        
        if self.config.word_wrap and self.config.max_width:
            text = textwrap.fill(text, width=self.config.max_width)
        
        self.console.print(text)
    
    def _format_code(self, content: str, language: str = "python"):
        """Format code with syntax highlighting."""
        # Extract language if specified in markdown-style code block
        if content.startswith('```'):
            lines = content.split('\n')
            language = lines[0][3:].strip() or language
            content = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
        
        syntax = Syntax(
            content,
            language,
            theme=self.config.theme,
            line_numbers=self.config.line_numbers,
            word_wrap=self.config.word_wrap
        )
        
        self.console.print(syntax)
    
    def _format_json(self, content: Union[str, dict, list]):
        """Format JSON data."""
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except:
                self._format_text(content)
                return
        
        json_str = json.dumps(content, indent=self.config.indent_size, default=str)
        
        syntax = Syntax(
            json_str,
            "json",
            theme=self.config.theme,
            line_numbers=self.config.line_numbers,
            word_wrap=self.config.word_wrap
        )
        
        self.console.print(syntax)
    
    def _format_table(self, content: List[Dict]):
        """Format data as a table."""
        if not content:
            return
        
        # Create table
        table = Table(
            show_header=True,
            header_style="bold magenta",
            show_lines=True if self.config.table_style == "grid" else False,
            box=getattr(Table, self.config.table_style.upper(), Table.ROUNDED)
        )
        
        # Add columns
        keys = list(content[0].keys())
        for key in keys:
            table.add_column(key, style="cyan")
        
        # Add rows
        for item in content:
            row = [str(item.get(key, "")) for key in keys]
            table.add_row(*row)
        
        self.console.print(table)
    
    def _format_tree(self, content: Dict, title: str = "Tree"):
        """Format hierarchical data as a tree."""
        tree = Tree(title)
        self._build_tree(tree, content)
        self.console.print(tree)
    
    def _build_tree(self, tree: Tree, data: Any, key: str = ""):
        """Recursively build a tree structure."""
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    branch = tree.add(f"[bold cyan]{k}[/bold cyan]")
                    self._build_tree(branch, v, k)
                else:
                    tree.add(f"[cyan]{k}[/cyan]: {v}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    branch = tree.add(f"[dim]Item {i}[/dim]")
                    self._build_tree(branch, item)
                else:
                    tree.add(str(item))
        else:
            tree.add(str(data))
    
    def _format_error(self, content: Exception):
        """Format error/exception."""
        if self.config.use_colors:
            # Use rich traceback
            tb = Traceback.from_exception(
                type(content),
                content,
                content.__traceback__,
                show_locals=True,
                suppress=[],
                max_frames=20
            )
            self.console.print(tb)
        else:
            # Plain text error
            import traceback
            tb_str = ''.join(traceback.format_exception(
                type(content),
                content,
                content.__traceback__
            ))
            self.console.print(tb_str, style=self.config.error_style)
    
    def _format_markdown(self, content: str):
        """Format markdown content."""
        md = Markdown(content)
        self.console.print(md)
    
    def _format_html(self, content: str):
        """Format HTML content."""
        # Strip HTML tags for console display
        import html
        text = re.sub('<[^<]+?>', '', html.unescape(content))
        self._format_text(text)
    
    def _format_xml(self, content: str):
        """Format XML content."""
        syntax = Syntax(
            content,
            "xml",
            theme=self.config.theme,
            line_numbers=self.config.line_numbers,
            word_wrap=self.config.word_wrap
        )
        self.console.print(syntax)
    
    def _format_yaml(self, content: str):
        """Format YAML content."""
        syntax = Syntax(
            content,
            "yaml",
            theme=self.config.theme,
            line_numbers=self.config.line_numbers,
            word_wrap=self.config.word_wrap
        )
        self.console.print(syntax)
    
    def _format_sql(self, content: str):
        """Format SQL content."""
        syntax = Syntax(
            content,
            "sql",
            theme=self.config.theme,
            line_numbers=self.config.line_numbers,
            word_wrap=self.config.word_wrap
        )
        self.console.print(syntax)
    
    def _format_diff(self, content: str):
        """Format diff output."""
        # Color diff lines
        lines = content.split('\n')
        for line in lines:
            if line.startswith('+'):
                self.console.print(line, style="green")
            elif line.startswith('-'):
                self.console.print(line, style="red")
            elif line.startswith('@'):
                self.console.print(line, style="cyan")
            else:
                self.console.print(line)
    
    def _format_log(self, content: str):
        """Format log output."""
        lines = content.split('\n')
        for line in lines:
            if 'ERROR' in line or 'CRITICAL' in line:
                self.console.print(line, style="bold red")
            elif 'WARNING' in line:
                self.console.print(line, style="yellow")
            elif 'INFO' in line:
                self.console.print(line, style="blue")
            elif 'DEBUG' in line:
                self.console.print(line, style="dim")
            else:
                self.console.print(line)


class StreamingFormatter:
    """Formatter for streaming output."""
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize streaming formatter.
        
        Args:
            console: Rich console instance
        """
        self.console = console or Console()
        self.live = None
        self.buffer = ""
    
    def start_stream(self):
        """Start a streaming session."""
        self.live = Live(
            "",
            console=self.console,
            refresh_per_second=10,
            transient=False
        )
        self.live.start()
        self.buffer = ""
    
    def update_stream(self, chunk: str):
        """Update the streaming output."""
        if self.live:
            self.buffer += chunk
            self.live.update(Text(self.buffer))
    
    def end_stream(self) -> str:
        """End the streaming session."""
        if self.live:
            self.live.stop()
            self.live = None
        
        result = self.buffer
        self.buffer = ""
        return result
    
    async def stream_async(self, generator):
        """Stream output from an async generator."""
        self.start_stream()
        try:
            async for chunk in generator:
                self.update_stream(chunk)
        finally:
            return self.end_stream()


class ErrorFormatter:
    """Specialized formatter for errors and exceptions."""
    
    def __init__(self, console: Optional[Console] = None, show_locals: bool = True):
        """
        Initialize error formatter.
        
        Args:
            console: Rich console instance
            show_locals: Show local variables in traceback
        """
        self.console = console or Console()
        self.show_locals = show_locals
    
    def format_error(self, error: Exception, show_full: bool = True):
        """
        Format an error with traceback.
        
        Args:
            error: Exception to format
            show_full: Show full traceback
        """
        if show_full:
            tb = Traceback.from_exception(
                type(error),
                error,
                error.__traceback__,
                show_locals=self.show_locals,
                suppress=[],
                max_frames=20
            )
            self.console.print(tb)
        else:
            # Simple error message
            panel = Panel(
                Text(str(error), style="red"),
                title=f"[bold red]{type(error).__name__}[/bold red]",
                border_style="red"
            )
            self.console.print(panel)
    
    def format_validation_error(self, field: str, message: str):
        """Format a validation error."""
        text = Text()
        text.append("Validation Error: ", style="bold red")
        text.append(f"{field} - ", style="yellow")
        text.append(message, style="white")
        self.console.print(text)
    
    def format_warning(self, message: str):
        """Format a warning message."""
        panel = Panel(
            Text(message, style="yellow"),
            title="[bold yellow]Warning[/bold yellow]",
            border_style="yellow"
        )
        self.console.print(panel)


class CodeFormatter:
    """Specialized formatter for code with advanced features."""
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize code formatter.
        
        Args:
            console: Rich console instance
        """
        self.console = console or Console()
    
    def format_code(
        self,
        code: str,
        language: str = "python",
        title: Optional[str] = None,
        line_numbers: bool = True,
        highlight_lines: Optional[List[int]] = None,
        theme: str = "monokai"
    ):
        """
        Format code with advanced options.
        
        Args:
            code: Code to format
            language: Programming language
            title: Optional title
            line_numbers: Show line numbers
            highlight_lines: Lines to highlight
            theme: Color theme
        """
        syntax = Syntax(
            code,
            language,
            theme=theme,
            line_numbers=line_numbers,
            line_range=None,
            highlight_lines=set(highlight_lines) if highlight_lines else None,
            word_wrap=False,
            indent_guides=True
        )
        
        if title:
            panel = Panel(syntax, title=title, border_style="cyan")
            self.console.print(panel)
        else:
            self.console.print(syntax)
    
    def format_diff(self, old: str, new: str, language: str = "python"):
        """
        Format a code diff.
        
        Args:
            old: Old version of code
            new: New version of code
            language: Programming language
        """
        import difflib
        
        diff = difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="old",
            tofile="new"
        )
        
        diff_text = ''.join(diff)
        
        # Color the diff
        lines = diff_text.split('\n')
        for line in lines:
            if line.startswith('+'):
                self.console.print(line, style="green")
            elif line.startswith('-'):
                self.console.print(line, style="red")
            elif line.startswith('@'):
                self.console.print(line, style="cyan")
            else:
                self.console.print(line)


class ProgressFormatter:
    """Formatter for progress indicators."""
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize progress formatter.
        
        Args:
            console: Rich console instance
        """
        self.console = console or Console()
        self.progress = None
    
    def create_progress(
        self,
        description: str = "Processing",
        total: Optional[int] = None,
        auto_refresh: bool = True
    ) -> Progress:
        """
        Create a progress bar.
        
        Args:
            description: Task description
            total: Total steps (None for indeterminate)
            auto_refresh: Auto-refresh display
            
        Returns:
            Progress instance
        """
        if total is None:
            # Indeterminate progress (spinner)
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                auto_refresh=auto_refresh
            )
        else:
            # Determinate progress (bar)
            self.progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                console=self.console,
                auto_refresh=auto_refresh
            )
        
        self.task_id = self.progress.add_task(description, total=total)
        return self.progress
    
    def update_progress(self, advance: int = 1, description: Optional[str] = None):
        """Update progress."""
        if self.progress and self.task_id is not None:
            self.progress.update(
                self.task_id,
                advance=advance,
                description=description
            )
    
    def finish_progress(self):
        """Finish and close progress."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.task_id = None