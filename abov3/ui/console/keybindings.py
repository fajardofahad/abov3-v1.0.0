"""
ABOV3 Console Key Bindings

Custom key binding configurations for the REPL interface.
"""

from enum import Enum
from typing import Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.filters import (
    Condition, has_focus, has_selection, vi_mode, emacs_mode,
    has_completions, is_searching, is_multiline, vi_insert_mode
)
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.clipboard import ClipboardData


class KeyBindingMode(Enum):
    """Key binding modes."""
    EMACS = "emacs"
    VI = "vi"
    CUSTOM = "custom"


def create_keybindings(
    mode: KeyBindingMode = KeyBindingMode.EMACS,
    enable_vim: bool = False,
    enable_system_prompt: bool = True
) -> KeyBindings:
    """
    Create key bindings for the REPL.
    
    Args:
        mode: Key binding mode
        enable_vim: Enable Vim-style bindings
        enable_system_prompt: Enable system prompt key
        
    Returns:
        Configured KeyBindings instance
    """
    kb = KeyBindings()
    
    # Common bindings for all modes
    _add_common_bindings(kb, enable_system_prompt)
    
    if mode == KeyBindingMode.EMACS or not enable_vim:
        _add_emacs_bindings(kb)
    elif mode == KeyBindingMode.VI or enable_vim:
        _add_vi_bindings(kb)
    elif mode == KeyBindingMode.CUSTOM:
        _add_custom_bindings(kb)
    
    return kb


def _add_common_bindings(kb: KeyBindings, enable_system_prompt: bool = True):
    """Add common key bindings for all modes."""
    
    # Exit bindings
    @kb.add('c-d')
    def exit_on_ctrl_d(event: KeyPressEvent):
        """Exit on Ctrl+D."""
        event.app.exit()
    
    @kb.add('c-c')
    def keyboard_interrupt(event: KeyPressEvent):
        """Handle Ctrl+C."""
        event.app.exit(exception=KeyboardInterrupt())
    
    # Clear screen
    @kb.add('c-l')
    def clear_screen(event: KeyPressEvent):
        """Clear the screen."""
        event.app.renderer.clear()
    
    # History navigation
    @kb.add('c-p', filter=~has_completions)
    def previous_history(event: KeyPressEvent):
        """Go to previous history entry."""
        event.current_buffer.history_backward()
    
    @kb.add('c-n', filter=~has_completions)
    def next_history(event: KeyPressEvent):
        """Go to next history entry."""
        event.current_buffer.history_forward()
    
    # Search history
    @kb.add('c-r')
    def search_history_backward(event: KeyPressEvent):
        """Search history backward."""
        event.current_buffer.start_history_search()
    
    @kb.add('c-s')
    def search_history_forward(event: KeyPressEvent):
        """Search history forward."""
        event.current_buffer.start_history_search(direction='forward')
    
    # Completion
    @kb.add('tab', filter=~has_completions)
    def trigger_completion(event: KeyPressEvent):
        """Trigger auto-completion."""
        buff = event.current_buffer
        if buff.complete_state:
            buff.complete_next()
        else:
            buff.start_completion()
    
    @kb.add('s-tab', filter=has_completions)
    def complete_previous(event: KeyPressEvent):
        """Select previous completion."""
        event.current_buffer.complete_previous()
    
    @kb.add('c-space')
    def force_completion(event: KeyPressEvent):
        """Force trigger completion."""
        event.current_buffer.start_completion()
    
    # Accept completion
    @kb.add('enter', filter=has_completions)
    def accept_completion(event: KeyPressEvent):
        """Accept selected completion."""
        event.current_buffer.complete_state = None
    
    # Multiline editing
    @kb.add('escape', 'enter', filter=is_multiline)
    def accept_multiline(event: KeyPressEvent):
        """Accept multiline input."""
        event.current_buffer.validate_and_handle()
    
    @kb.add('c-j')
    def newline(event: KeyPressEvent):
        """Insert newline."""
        event.current_buffer.insert_text('\n')
    
    # Copy/Paste
    @kb.add('c-y')
    def paste_from_clipboard(event: KeyPressEvent):
        """Paste from clipboard."""
        data = event.app.clipboard.get_data()
        if data:
            event.current_buffer.insert_text(data.text)
    
    @kb.add('c-w', filter=has_selection)
    def cut_selection(event: KeyPressEvent):
        """Cut selected text."""
        data = event.current_buffer.cut_selection()
        if data:
            event.app.clipboard.set_data(data)
    
    @kb.add('escape', 'w', filter=has_selection)
    def copy_selection(event: KeyPressEvent):
        """Copy selected text."""
        data = event.current_buffer.copy_selection()
        if data:
            event.app.clipboard.set_data(data)
    
    # Undo/Redo
    @kb.add('c-z')
    def undo(event: KeyPressEvent):
        """Undo last change."""
        event.current_buffer.undo()
    
    @kb.add('c-y')
    def redo(event: KeyPressEvent):
        """Redo last undone change."""
        event.current_buffer.redo()
    
    # System prompt (if enabled)
    if enable_system_prompt:
        @kb.add('c-o')
        def system_prompt(event: KeyPressEvent):
            """Open system prompt."""
            # This can be customized to open a system command prompt
            event.app.exit(result='SYSTEM_PROMPT')
    
    # Help
    @kb.add('f1')
    def show_help(event: KeyPressEvent):
        """Show help."""
        event.current_buffer.insert_text('/help')
        event.current_buffer.validate_and_handle()
    
    # Toggle features
    @kb.add('f2')
    def toggle_multiline(event: KeyPressEvent):
        """Toggle multiline mode."""
        event.app.current_buffer.multiline = not event.app.current_buffer.multiline
    
    @kb.add('f3')
    def toggle_mouse_support(event: KeyPressEvent):
        """Toggle mouse support."""
        event.app.mouse_support = not event.app.mouse_support
    
    @kb.add('f4')
    def toggle_wrap_lines(event: KeyPressEvent):
        """Toggle line wrapping."""
        event.app.wrap_lines = not event.app.wrap_lines


def _add_emacs_bindings(kb: KeyBindings):
    """Add Emacs-style key bindings."""
    
    # Cursor movement
    @kb.add('c-a')
    def beginning_of_line(event: KeyPressEvent):
        """Move to beginning of line."""
        event.current_buffer.cursor_position = event.current_buffer.document.get_start_of_line_position()
    
    @kb.add('c-e')
    def end_of_line(event: KeyPressEvent):
        """Move to end of line."""
        event.current_buffer.cursor_position = event.current_buffer.document.get_end_of_line_position()
    
    @kb.add('c-f')
    def forward_char(event: KeyPressEvent):
        """Move forward one character."""
        event.current_buffer.cursor_right()
    
    @kb.add('c-b')
    def backward_char(event: KeyPressEvent):
        """Move backward one character."""
        event.current_buffer.cursor_left()
    
    @kb.add('escape', 'f')
    def forward_word(event: KeyPressEvent):
        """Move forward one word."""
        event.current_buffer.cursor_right(count=event.current_buffer.document.find_next_word_ending())
    
    @kb.add('escape', 'b')
    def backward_word(event: KeyPressEvent):
        """Move backward one word."""
        event.current_buffer.cursor_left(count=-event.current_buffer.document.find_previous_word_beginning())
    
    # Text manipulation
    @kb.add('c-k')
    def kill_line(event: KeyPressEvent):
        """Kill from cursor to end of line."""
        deleted = event.current_buffer.delete(count=event.current_buffer.document.get_end_of_line_position())
        event.app.clipboard.set_data(ClipboardData(deleted))
    
    @kb.add('c-u')
    def kill_line_backwards(event: KeyPressEvent):
        """Kill from cursor to beginning of line."""
        deleted = event.current_buffer.delete(count=event.current_buffer.document.get_start_of_line_position())
        event.app.clipboard.set_data(ClipboardData(deleted))
    
    @kb.add('escape', 'd')
    def kill_word(event: KeyPressEvent):
        """Kill word forward."""
        deleted = event.current_buffer.delete_word()
        event.app.clipboard.set_data(ClipboardData(deleted))
    
    @kb.add('escape', 'backspace')
    def kill_word_backwards(event: KeyPressEvent):
        """Kill word backward."""
        deleted = event.current_buffer.delete_word(before=True)
        event.app.clipboard.set_data(ClipboardData(deleted))
    
    @kb.add('c-t')
    def transpose_chars(event: KeyPressEvent):
        """Transpose characters."""
        event.current_buffer.transpose_chars()
    
    @kb.add('escape', 't')
    def transpose_words(event: KeyPressEvent):
        """Transpose words."""
        # Custom implementation needed
        pass
    
    @kb.add('escape', 'u')
    def uppercase_word(event: KeyPressEvent):
        """Uppercase word."""
        for _ in range(event.current_buffer.document.find_next_word_ending() or 1):
            event.current_buffer.transform_current_char(str.upper)
            event.current_buffer.cursor_right()
    
    @kb.add('escape', 'l')
    def lowercase_word(event: KeyPressEvent):
        """Lowercase word."""
        for _ in range(event.current_buffer.document.find_next_word_ending() or 1):
            event.current_buffer.transform_current_char(str.lower)
            event.current_buffer.cursor_right()
    
    @kb.add('escape', 'c')
    def capitalize_word(event: KeyPressEvent):
        """Capitalize word."""
        pos = event.current_buffer.document.find_next_word_ending()
        if pos:
            event.current_buffer.transform_current_char(str.upper)
            event.current_buffer.cursor_right()
            for _ in range(pos - 1):
                event.current_buffer.transform_current_char(str.lower)
                event.current_buffer.cursor_right()


def _add_vi_bindings(kb: KeyBindings):
    """Add Vi/Vim-style key bindings."""
    
    # Mode switching
    @kb.add('escape')
    def enter_vi_normal_mode(event: KeyPressEvent):
        """Enter Vi normal mode."""
        event.app.vi_state.input_mode = 'navigation'
    
    @kb.add('i', filter=vi_mode)
    def enter_vi_insert_mode(event: KeyPressEvent):
        """Enter Vi insert mode."""
        event.app.vi_state.input_mode = 'insert'
    
    @kb.add('a', filter=vi_mode)
    def enter_vi_insert_mode_after(event: KeyPressEvent):
        """Enter Vi insert mode after cursor."""
        event.current_buffer.cursor_right()
        event.app.vi_state.input_mode = 'insert'
    
    @kb.add('v', filter=vi_mode)
    def enter_vi_visual_mode(event: KeyPressEvent):
        """Enter Vi visual mode."""
        event.current_buffer.start_selection()
    
    # Vi navigation (normal mode)
    @kb.add('h', filter=vi_mode & ~vi_insert_mode)
    def vi_move_left(event: KeyPressEvent):
        """Move left in Vi mode."""
        event.current_buffer.cursor_left()
    
    @kb.add('l', filter=vi_mode & ~vi_insert_mode)
    def vi_move_right(event: KeyPressEvent):
        """Move right in Vi mode."""
        event.current_buffer.cursor_right()
    
    @kb.add('j', filter=vi_mode & ~vi_insert_mode)
    def vi_move_down(event: KeyPressEvent):
        """Move down in Vi mode."""
        event.current_buffer.cursor_down()
    
    @kb.add('k', filter=vi_mode & ~vi_insert_mode)
    def vi_move_up(event: KeyPressEvent):
        """Move up in Vi mode."""
        event.current_buffer.cursor_up()
    
    @kb.add('w', filter=vi_mode & ~vi_insert_mode)
    def vi_forward_word(event: KeyPressEvent):
        """Move forward one word in Vi mode."""
        event.current_buffer.cursor_right(count=event.current_buffer.document.find_next_word_ending())
    
    @kb.add('b', filter=vi_mode & ~vi_insert_mode)
    def vi_backward_word(event: KeyPressEvent):
        """Move backward one word in Vi mode."""
        event.current_buffer.cursor_left(count=-event.current_buffer.document.find_previous_word_beginning())
    
    @kb.add('0', filter=vi_mode & ~vi_insert_mode)
    def vi_beginning_of_line(event: KeyPressEvent):
        """Move to beginning of line in Vi mode."""
        event.current_buffer.cursor_position = event.current_buffer.document.get_start_of_line_position()
    
    @kb.add('$', filter=vi_mode & ~vi_insert_mode)
    def vi_end_of_line(event: KeyPressEvent):
        """Move to end of line in Vi mode."""
        event.current_buffer.cursor_position = event.current_buffer.document.get_end_of_line_position()
    
    # Vi editing commands
    @kb.add('x', filter=vi_mode & ~vi_insert_mode)
    def vi_delete_char(event: KeyPressEvent):
        """Delete character in Vi mode."""
        event.current_buffer.delete()
    
    @kb.add('d', 'd', filter=vi_mode & ~vi_insert_mode)
    def vi_delete_line(event: KeyPressEvent):
        """Delete line in Vi mode."""
        event.current_buffer.cursor_position = event.current_buffer.document.get_start_of_line_position()
        deleted = event.current_buffer.delete(count=len(event.current_buffer.document.current_line))
        event.app.clipboard.set_data(ClipboardData(deleted))
    
    @kb.add('y', 'y', filter=vi_mode & ~vi_insert_mode)
    def vi_yank_line(event: KeyPressEvent):
        """Yank line in Vi mode."""
        line = event.current_buffer.document.current_line
        event.app.clipboard.set_data(ClipboardData(line))
    
    @kb.add('p', filter=vi_mode & ~vi_insert_mode)
    def vi_paste(event: KeyPressEvent):
        """Paste in Vi mode."""
        data = event.app.clipboard.get_data()
        if data:
            event.current_buffer.insert_text(data.text)
    
    @kb.add('u', filter=vi_mode & ~vi_insert_mode)
    def vi_undo(event: KeyPressEvent):
        """Undo in Vi mode."""
        event.current_buffer.undo()
    
    @kb.add('c-r', filter=vi_mode & ~vi_insert_mode)
    def vi_redo(event: KeyPressEvent):
        """Redo in Vi mode."""
        event.current_buffer.redo()
    
    # Vi search
    @kb.add('/', filter=vi_mode & ~vi_insert_mode)
    def vi_search_forward(event: KeyPressEvent):
        """Search forward in Vi mode."""
        event.current_buffer.start_search()
    
    @kb.add('?', filter=vi_mode & ~vi_insert_mode)
    def vi_search_backward(event: KeyPressEvent):
        """Search backward in Vi mode."""
        event.current_buffer.start_search(direction='backward')
    
    @kb.add('n', filter=vi_mode & ~vi_insert_mode)
    def vi_search_next(event: KeyPressEvent):
        """Go to next search result in Vi mode."""
        event.current_buffer.apply_search()
    
    @kb.add('N', filter=vi_mode & ~vi_insert_mode)
    def vi_search_previous(event: KeyPressEvent):
        """Go to previous search result in Vi mode."""
        event.current_buffer.apply_search(direction='backward')


def _add_custom_bindings(kb: KeyBindings):
    """Add custom key bindings for special features."""
    
    # Quick commands
    @kb.add('c-q', 'h')
    def quick_help(event: KeyPressEvent):
        """Quick help command."""
        event.current_buffer.insert_text('/help')
        event.current_buffer.validate_and_handle()
    
    @kb.add('c-q', 'c')
    def quick_clear(event: KeyPressEvent):
        """Quick clear command."""
        event.current_buffer.insert_text('/clear')
        event.current_buffer.validate_and_handle()
    
    @kb.add('c-q', 's')
    def quick_save(event: KeyPressEvent):
        """Quick save command."""
        event.current_buffer.insert_text('/save')
        event.current_buffer.validate_and_handle()
    
    @kb.add('c-q', 'l')
    def quick_load(event: KeyPressEvent):
        """Quick load command."""
        event.current_buffer.insert_text('/load')
        event.current_buffer.validate_and_handle()
    
    @kb.add('c-q', 'q')
    def quick_quit(event: KeyPressEvent):
        """Quick quit command."""
        event.app.exit()
    
    # Smart features
    @kb.add('c-x', 'c-e')
    def edit_in_external_editor(event: KeyPressEvent):
        """Edit current buffer in external editor."""
        # This would open the current buffer in $EDITOR
        import os
        import tempfile
        import subprocess
        
        editor = os.environ.get('EDITOR', 'nano')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(event.current_buffer.text)
            f.flush()
            
            try:
                subprocess.call([editor, f.name])
                with open(f.name, 'r') as edited:
                    new_text = edited.read()
                    event.current_buffer.text = new_text
            finally:
                os.unlink(f.name)
    
    @kb.add('c-x', 'c-s')
    def save_buffer_to_file(event: KeyPressEvent):
        """Save current buffer to file."""
        # This would save the buffer content to a file
        text = event.current_buffer.text
        if text:
            import datetime
            filename = f"buffer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write(text)
            event.current_buffer.insert_text(f"\n# Saved to {filename}")
    
    # Code execution shortcuts
    @kb.add('c-x', 'c-r')
    def run_current_line(event: KeyPressEvent):
        """Run current line as code."""
        line = event.current_buffer.document.current_line
        if line.strip():
            # This would execute the current line
            event.current_buffer.insert_text(f"\n# Execute: {line}")
    
    @kb.add('c-x', 'c-a')
    def run_all(event: KeyPressEvent):
        """Run entire buffer as code."""
        text = event.current_buffer.text
        if text.strip():
            # This would execute the entire buffer
            event.current_buffer.validate_and_handle()
    
    # Advanced navigation
    @kb.add('c-x', '[')
    def jump_to_previous_prompt(event: KeyPressEvent):
        """Jump to previous prompt."""
        # Navigate to previous command in history
        event.current_buffer.history_backward()
    
    @kb.add('c-x', ']')
    def jump_to_next_prompt(event: KeyPressEvent):
        """Jump to next prompt."""
        # Navigate to next command in history
        event.current_buffer.history_forward()
    
    # Bracket matching
    @kb.add('c-]')
    def jump_to_matching_bracket(event: KeyPressEvent):
        """Jump to matching bracket."""
        doc = event.current_buffer.document
        pos = doc.cursor_position
        text = doc.text
        
        # Simple bracket matching
        brackets = {'(': ')', '[': ']', '{': '}', ')': '(', ']': '[', '}': '{'}
        if pos < len(text) and text[pos] in brackets:
            target = brackets[text[pos]]
            forward = text[pos] in '([{'
            
            count = 1
            i = pos + 1 if forward else pos - 1
            
            while 0 <= i < len(text) and count > 0:
                if text[i] == text[pos]:
                    count += 1
                elif text[i] == target:
                    count -= 1
                
                if count == 0:
                    event.current_buffer.cursor_position = i
                    break
                
                i += 1 if forward else -1


class KeyBindingManager:
    """Manager for dynamic key binding configuration."""
    
    def __init__(self):
        """Initialize key binding manager."""
        self.custom_bindings = {}
        self.modes = {}
        self.current_mode = KeyBindingMode.EMACS
    
    def register_binding(self, keys: str, handler: callable, mode: Optional[KeyBindingMode] = None):
        """
        Register a custom key binding.
        
        Args:
            keys: Key combination (e.g., "c-x c-s")
            handler: Handler function
            mode: Specific mode for this binding
        """
        mode = mode or self.current_mode
        if mode not in self.custom_bindings:
            self.custom_bindings[mode] = {}
        self.custom_bindings[mode][keys] = handler
    
    def unregister_binding(self, keys: str, mode: Optional[KeyBindingMode] = None):
        """
        Unregister a custom key binding.
        
        Args:
            keys: Key combination
            mode: Specific mode
        """
        mode = mode or self.current_mode
        if mode in self.custom_bindings and keys in self.custom_bindings[mode]:
            del self.custom_bindings[mode][keys]
    
    def switch_mode(self, mode: KeyBindingMode):
        """
        Switch key binding mode.
        
        Args:
            mode: New mode
        """
        self.current_mode = mode
    
    def get_bindings(self) -> KeyBindings:
        """
        Get current key bindings.
        
        Returns:
            KeyBindings instance
        """
        kb = create_keybindings(self.current_mode)
        
        # Add custom bindings for current mode
        if self.current_mode in self.custom_bindings:
            for keys, handler in self.custom_bindings[self.current_mode].items():
                kb.add(keys)(handler)
        
        return kb
    
    def list_bindings(self, mode: Optional[KeyBindingMode] = None) -> dict:
        """
        List all key bindings for a mode.
        
        Args:
            mode: Mode to list (defaults to current)
            
        Returns:
            Dictionary of key bindings
        """
        mode = mode or self.current_mode
        bindings = {}
        
        # Get default bindings based on mode
        if mode == KeyBindingMode.EMACS:
            bindings.update({
                'C-a': 'Beginning of line',
                'C-e': 'End of line',
                'C-k': 'Kill line',
                'C-y': 'Paste',
                'M-f': 'Forward word',
                'M-b': 'Backward word',
            })
        elif mode == KeyBindingMode.VI:
            bindings.update({
                'h/j/k/l': 'Vi navigation',
                'i': 'Insert mode',
                'dd': 'Delete line',
                'yy': 'Yank line',
                'p': 'Paste',
                '/': 'Search forward',
            })
        
        # Add common bindings
        bindings.update({
            'Tab': 'Auto-complete',
            'C-d': 'Exit',
            'C-l': 'Clear screen',
            'C-p/C-n': 'History navigation',
            'C-r': 'Search history',
            'F1': 'Help',
        })
        
        # Add custom bindings
        if mode in self.custom_bindings:
            for keys in self.custom_bindings[mode]:
                bindings[keys] = 'Custom binding'
        
        return bindings