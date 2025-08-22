"""
ABOV3 Console Auto-completion System

Provides intelligent, context-aware auto-completion for the REPL interface.
"""

import ast
import builtins
import inspect
import keyword
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from prompt_toolkit.completion import (
    Completer, Completion, CompleteEvent, PathCompleter,
    WordCompleter, FuzzyCompleter, FuzzyWordCompleter,
    ThreadedCompleter, DynamicCompleter, merge_completers
)
from prompt_toolkit.document import Document

from pygments.lexers import get_all_lexers


class CommandCompleter(Completer):
    """Completer for REPL commands."""
    
    def __init__(self, commands: Optional[Dict[str, str]] = None):
        """
        Initialize command completer.
        
        Args:
            commands: Dictionary of command -> description
        """
        self.commands = commands or {
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
    
    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """Get command completions."""
        text = document.text_before_cursor.lstrip()
        
        # Only complete commands that start with /
        if not text.startswith('/'):
            return
        
        # Complete command names
        for cmd, description in self.commands.items():
            if cmd.startswith(text):
                yield Completion(
                    text=cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=description
                )


class CodeCompleter(Completer):
    """Python code completer with introspection."""
    
    def __init__(self, namespace: Optional[Dict[str, Any]] = None):
        """
        Initialize code completer.
        
        Args:
            namespace: Namespace for completion context
        """
        self.namespace = namespace or {}
        self._python_keywords = set(keyword.kwlist)
        self._builtins = set(dir(builtins))
    
    def update_namespace(self, namespace: Dict[str, Any]):
        """Update the completion namespace."""
        self.namespace.update(namespace)
    
    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """Get code completions."""
        text = document.text_before_cursor
        
        # Get the current word being typed
        word_match = re.search(r'[\w\.]+$', text)
        if not word_match:
            # Complete from scratch
            word = ''
            start_position = 0
        else:
            word = word_match.group()
            start_position = -len(word)
        
        completions = []
        
        # Handle attribute access (e.g., "obj.attr")
        if '.' in word:
            parts = word.rsplit('.', 1)
            obj_path = parts[0]
            attr_prefix = parts[1] if len(parts) > 1 else ''
            
            try:
                # Evaluate the object path
                obj = self._evaluate_path(obj_path)
                if obj is not None:
                    # Get attributes of the object
                    attrs = self._get_attributes(obj)
                    for attr in attrs:
                        if attr.startswith(attr_prefix):
                            completions.append(
                                Completion(
                                    text=f"{obj_path}.{attr}",
                                    start_position=start_position,
                                    display=attr,
                                    display_meta=self._get_attr_type(obj, attr)
                                )
                            )
            except:
                pass
        else:
            # Complete from namespace, keywords, and builtins
            prefix = word
            
            # Python keywords
            for kw in self._python_keywords:
                if kw.startswith(prefix):
                    completions.append(
                        Completion(
                            text=kw,
                            start_position=start_position,
                            display=kw,
                            display_meta='keyword'
                        )
                    )
            
            # Builtins
            for builtin_name in self._builtins:
                if builtin_name.startswith(prefix):
                    completions.append(
                        Completion(
                            text=builtin_name,
                            start_position=start_position,
                            display=builtin_name,
                            display_meta='builtin'
                        )
                    )
            
            # Namespace items
            for name, value in self.namespace.items():
                if name.startswith(prefix):
                    completions.append(
                        Completion(
                            text=name,
                            start_position=start_position,
                            display=name,
                            display_meta=self._get_type_name(value)
                        )
                    )
        
        # Sort and yield completions
        completions.sort(key=lambda c: c.text)
        for completion in completions:
            yield completion
    
    def _evaluate_path(self, path: str) -> Any:
        """Evaluate an object path in the namespace."""
        try:
            # Try to evaluate directly
            return eval(path, {'__builtins__': builtins.__dict__}, self.namespace)
        except:
            # Try to walk the path
            parts = path.split('.')
            obj = self.namespace.get(parts[0])
            for part in parts[1:]:
                if obj is None:
                    return None
                obj = getattr(obj, part, None)
            return obj
    
    def _get_attributes(self, obj: Any) -> List[str]:
        """Get attributes of an object."""
        attrs = []
        try:
            # Get all attributes
            for attr in dir(obj):
                # Filter private attributes unless explicitly requested
                if not attr.startswith('__'):
                    attrs.append(attr)
        except:
            pass
        return attrs
    
    def _get_attr_type(self, obj: Any, attr: str) -> str:
        """Get the type of an attribute."""
        try:
            value = getattr(obj, attr)
            if callable(value):
                if inspect.isclass(value):
                    return 'class'
                elif inspect.ismethod(value):
                    return 'method'
                elif inspect.isfunction(value):
                    return 'function'
                else:
                    return 'callable'
            else:
                return self._get_type_name(value)
        except:
            return 'attribute'
    
    def _get_type_name(self, value: Any) -> str:
        """Get a readable type name for a value."""
        if value is None:
            return 'None'
        elif callable(value):
            if inspect.isclass(value):
                return 'class'
            elif inspect.isfunction(value):
                return 'function'
            elif inspect.ismethod(value):
                return 'method'
            else:
                return 'callable'
        else:
            type_name = type(value).__name__
            if type_name in ('int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set'):
                return type_name
            else:
                return f'instance of {type_name}'


class FilePathCompleter(PathCompleter):
    """Enhanced file path completer."""
    
    def __init__(
        self,
        only_directories: bool = False,
        get_paths: Optional[callable] = None,
        file_filter: Optional[callable] = None,
        min_input_len: int = 0,
        expanduser: bool = True,
        show_hidden: bool = True
    ):
        """
        Initialize file path completer.
        
        Args:
            only_directories: Only complete directories
            get_paths: Callable to get paths
            file_filter: Filter for files
            min_input_len: Minimum input length
            expanduser: Expand ~ to home directory
            show_hidden: Show hidden files
        """
        super().__init__(
            only_directories=only_directories,
            get_paths=get_paths,
            file_filter=file_filter,
            min_input_len=min_input_len,
            expanduser=expanduser
        )
        self.show_hidden = show_hidden
    
    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """Get file path completions."""
        text = document.text_before_cursor
        
        # Check if we're in a string that looks like a path
        if '"' in text or "'" in text:
            # Extract the path from the string
            match = re.search(r'["\']([^"\']*?)$', text)
            if match:
                path_text = match.group(1)
                
                # Get completions from parent
                for completion in super().get_completions(
                    Document(path_text),
                    complete_event
                ):
                    # Filter hidden files if needed
                    if not self.show_hidden and completion.text.startswith('.'):
                        continue
                    
                    # Adjust position for the string context
                    yield Completion(
                        text=completion.text,
                        start_position=completion.start_position - len(path_text),
                        display=completion.display,
                        display_meta=completion.display_meta
                    )


class LanguageCompleter(Completer):
    """Completer for programming language names."""
    
    def __init__(self):
        """Initialize language completer."""
        # Get all available lexers from Pygments
        self.languages = set()
        for name, aliases, filenames, mimetypes in get_all_lexers():
            self.languages.add(name.lower())
            self.languages.update(alias.lower() for alias in aliases)
    
    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """Get language completions."""
        text = document.text_before_cursor.lower()
        
        # Check if we're in a code block context
        if '```' in document.text:
            # Get the text after the last ```
            match = re.search(r'```(\w*)$', text)
            if match:
                prefix = match.group(1)
                start_position = -len(prefix)
                
                for lang in sorted(self.languages):
                    if lang.startswith(prefix):
                        yield Completion(
                            text=lang,
                            start_position=start_position,
                            display=lang,
                            display_meta='language'
                        )


class ContextAwareCompleter(Completer):
    """
    Context-aware completer that combines multiple completers
    and chooses the appropriate one based on context.
    """
    
    def __init__(self, repl=None):
        """
        Initialize context-aware completer.
        
        Args:
            repl: Reference to the REPL instance
        """
        self.repl = repl
        self.command_completer = CommandCompleter()
        self.code_completer = CodeCompleter()
        self.path_completer = FilePathCompleter()
        self.language_completer = LanguageCompleter()
        
        # Additional specialized completers
        self.theme_completer = WordCompleter(
            ['monokai', 'github', 'solarized', 'material', 'dracula', 'vim', 'emacs'],
            ignore_case=True
        )
        
        self.config_key_completer = WordCompleter([
            'prompt_text', 'multiline_prompt', 'theme', 'color_depth',
            'enable_syntax_highlighting', 'enable_auto_suggestions',
            'enable_completion', 'key_binding_mode', 'enable_vim_mode',
            'history_file', 'max_history_size', 'enable_search',
            'session_file', 'auto_save_session', 'max_output_lines',
            'enable_pager', 'wrap_lines', 'show_line_numbers',
            'async_mode', 'streaming_output', 'enable_multiline'
        ])
        
        self.mode_completer = WordCompleter(['emacs', 'vi', 'custom'], ignore_case=True)
    
    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """Get context-aware completions."""
        text = document.text_before_cursor
        
        # Determine context and use appropriate completer
        
        # 1. Command context
        if text.startswith('/'):
            # Command completion
            yield from self.command_completer.get_completions(document, complete_event)
            
            # Additional context-specific completions
            if text.startswith('/theme'):
                # Theme names after /theme command
                theme_text = text[6:].strip()
                if theme_text:
                    theme_doc = Document(theme_text)
                    yield from self.theme_completer.get_completions(theme_doc, complete_event)
            
            elif text.startswith('/config'):
                # Config keys after /config command
                config_text = text[7:].strip()
                if config_text and '=' not in config_text:
                    config_doc = Document(config_text)
                    yield from self.config_key_completer.get_completions(config_doc, complete_event)
            
            elif text.startswith('/mode'):
                # Mode names after /mode command
                mode_text = text[5:].strip()
                if mode_text:
                    mode_doc = Document(mode_text)
                    yield from self.mode_completer.get_completions(mode_doc, complete_event)
            
            elif text.startswith('/load') or text.startswith('/save'):
                # File paths for load/save commands
                cmd_len = 5 if text.startswith('/load') else 5
                path_text = text[cmd_len:].strip()
                if path_text:
                    path_doc = Document(path_text)
                    yield from self.path_completer.get_completions(path_doc, complete_event)
        
        # 2. Code block context
        elif '```' in text:
            # Language completion for code blocks
            yield from self.language_completer.get_completions(document, complete_event)
        
        # 3. File path context
        elif any(marker in text for marker in ['"/', "'", '"./', '"../', 'open(', 'Path(']):
            # File path completion
            yield from self.path_completer.get_completions(document, complete_event)
        
        # 4. Python code context (default)
        else:
            # Update namespace if REPL is available
            if self.repl and hasattr(self.repl, 'context'):
                self.code_completer.update_namespace(self.repl.context)
            
            # Python code completion
            yield from self.code_completer.get_completions(document, complete_event)


class MultiCompleter(Completer):
    """
    Combine multiple completers with priority ordering.
    """
    
    def __init__(self, completers: List[Tuple[Completer, float]]):
        """
        Initialize multi-completer.
        
        Args:
            completers: List of (completer, priority) tuples
        """
        self.completers = sorted(completers, key=lambda x: x[1], reverse=True)
    
    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """Get completions from all completers."""
        seen = set()
        
        for completer, priority in self.completers:
            for completion in completer.get_completions(document, complete_event):
                # Avoid duplicates
                if completion.text not in seen:
                    seen.add(completion.text)
                    yield completion


class SmartCompleter(FuzzyCompleter):
    """
    Smart completer with fuzzy matching and learning capabilities.
    """
    
    def __init__(self, completer: Completer, enable_fuzzy: bool = True):
        """
        Initialize smart completer.
        
        Args:
            completer: Base completer to wrap
            enable_fuzzy: Enable fuzzy matching
        """
        if enable_fuzzy:
            super().__init__(completer)
        else:
            self.completer = completer
            self.enable_fuzzy = False
        
        self.usage_stats = {}
    
    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """Get smart completions."""
        if hasattr(self, 'enable_fuzzy') and not self.enable_fuzzy:
            completions = list(self.completer.get_completions(document, complete_event))
        else:
            completions = list(super().get_completions(document, complete_event))
        
        # Sort by usage frequency
        def sort_key(completion):
            return self.usage_stats.get(completion.text, 0)
        
        completions.sort(key=sort_key, reverse=True)
        
        for completion in completions:
            yield completion
    
    def record_usage(self, text: str):
        """Record usage of a completion."""
        self.usage_stats[text] = self.usage_stats.get(text, 0) + 1


def create_completer(
    repl=None,
    enable_fuzzy: bool = True,
    enable_threading: bool = True
) -> Completer:
    """
    Create a configured completer for the REPL.
    
    Args:
        repl: REPL instance for context
        enable_fuzzy: Enable fuzzy matching
        enable_threading: Enable threaded completion
        
    Returns:
        Configured completer
    """
    # Create base completer
    completer = ContextAwareCompleter(repl)
    
    # Wrap with fuzzy matching
    if enable_fuzzy:
        completer = SmartCompleter(completer, enable_fuzzy=True)
    
    # Wrap with threading for performance
    if enable_threading:
        completer = ThreadedCompleter(completer)
    
    return completer