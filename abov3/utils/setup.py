"""
Setup wizard for ABOV3 4 Ollama first-time configuration.

This module provides an interactive setup wizard that guides users through
the initial configuration of ABOV3, including Ollama connection setup,
model selection, and preference configuration.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import ollama
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.table import Table
from rich.text import Text

from ..core.config import Config
from ..models.manager import ModelManager


class SetupWizard:
    """Interactive setup wizard for first-time configuration."""
    
    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
    
    def run(self) -> None:
        """Run the complete setup wizard."""
        self.console.print("\n[bold cyan]ABOV3 4 Ollama Setup Wizard[/bold cyan]")
        self.console.print("Let's configure ABOV3 for your system.\n")
        
        try:
            # Step 1: Ollama connection
            self._setup_ollama_connection()
            
            # Step 2: Model selection
            self._setup_model_selection()
            
            # Step 3: UI preferences
            self._setup_ui_preferences()
            
            # Step 4: Advanced settings (optional)
            if Confirm.ask("Would you like to configure advanced settings?", default=False):
                self._setup_advanced_settings()
            
            # Step 5: Summary and confirmation
            self._show_summary()
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Setup cancelled by user.[/yellow]")
            sys.exit(0)
        except Exception as e:
            self.console.print(f"\n[red]Setup failed: {e}[/red]")
            sys.exit(1)
    
    def _setup_ollama_connection(self) -> None:
        """Configure Ollama server connection."""
        self.console.print(Panel(
            "[bold]Step 1: Ollama Connection[/bold]\n"
            "ABOV3 needs to connect to an Ollama server to use AI models.\n"
            "If you haven't installed Ollama yet, please visit: https://ollama.ai",
            title="Ollama Setup",
            border_style="blue"
        ))
        
        # Test default connection first
        if self._test_ollama_connection(self.config.ollama.host):
            self.console.print(f"[green]OK[/green] Connected to Ollama at {self.config.ollama.host}")
            
            if not Confirm.ask("Would you like to use a different Ollama server?", default=False):
                return
        
        # Get custom host
        while True:
            host = Prompt.ask(
                "Ollama server URL",
                default=self.config.ollama.host
            )
            
            if self._test_ollama_connection(host):
                self.config.ollama.host = host
                self.console.print(f"[green]OK[/green] Connected to Ollama at {host}")
                break
            else:
                self.console.print(f"[red]ERROR[/red] Cannot connect to Ollama at {host}")
                if not Confirm.ask("Try a different URL?", default=True):
                    self.console.print("[yellow]Warning: Continuing with unverified connection.[/yellow]")
                    self.config.ollama.host = host
                    break
        
        # Configure timeout if needed
        if Confirm.ask("Would you like to configure connection timeout?", default=False):
            timeout = IntPrompt.ask(
                "Connection timeout (seconds)",
                default=self.config.ollama.timeout,
                show_default=True
            )
            self.config.ollama.timeout = timeout
    
    def _setup_model_selection(self) -> None:
        """Configure model selection and download models if needed."""
        self.console.print(Panel(
            "[bold]Step 2: Model Selection[/bold]\n"
            "Choose which AI model you'd like to use as your default.\n"
            "We'll help you download it if it's not already available.",
            title="Model Setup",
            border_style="green"
        ))
        
        try:
            # Get available models
            client = ollama.Client(host=self.config.ollama.host)
            available_models = []
            
            try:
                model_list = client.list()
                available_models = [model["name"] for model in model_list.get("models", [])]
            except Exception:
                pass
            
            if available_models:
                self.console.print("\n[bold]Available models:[/bold]")
                table = Table(show_header=False)
                table.add_column("Index", style="cyan")
                table.add_column("Model", style="green")
                
                for i, model in enumerate(available_models, 1):
                    table.add_row(str(i), model)
                
                self.console.print(table)
                
                if Confirm.ask("Would you like to use one of these models?", default=True):
                    while True:
                        try:
                            choice = IntPrompt.ask(
                                "Select model number",
                                show_default=False
                            )
                            if 1 <= choice <= len(available_models):
                                self.config.model.default_model = available_models[choice - 1]
                                break
                            else:
                                self.console.print("[red]Invalid choice. Please try again.[/red]")
                        except Exception:
                            self.console.print("[red]Invalid input. Please enter a number.[/red]")
                else:
                    self._setup_custom_model()
            else:
                self.console.print("[yellow]No models found. Let's download one.[/yellow]")
                self._setup_custom_model()
                
        except Exception as e:
            self.console.print(f"[yellow]Could not check available models: {e}[/yellow]")
            self._setup_custom_model()
    
    def _setup_custom_model(self) -> None:
        """Setup custom model download."""
        self.console.print("\n[bold]Recommended models:[/bold]")
        
        recommended_models = [
            ("llama3.2:latest", "Latest Llama 3.2 model (recommended)"),
            ("llama3.1:latest", "Llama 3.1 model"),
            ("codellama:latest", "Code-focused Llama model"),
            ("mixtral:latest", "Mixtral model"),
            ("custom", "Enter custom model name")
        ]
        
        table = Table(show_header=False)
        table.add_column("Index", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Description", style="dim")
        
        for i, (model, desc) in enumerate(recommended_models, 1):
            table.add_row(str(i), model, desc)
        
        self.console.print(table)
        
        while True:
            try:
                choice = IntPrompt.ask("Select model", show_default=False)
                if 1 <= choice <= len(recommended_models):
                    if choice == len(recommended_models):  # Custom option
                        model_name = Prompt.ask("Enter model name")
                    else:
                        model_name = recommended_models[choice - 1][0]
                    
                    self.config.model.default_model = model_name
                    
                    # Offer to download the model
                    if Confirm.ask(f"Would you like to download '{model_name}' now?", default=True):
                        self._download_model(model_name)
                    
                    break
                else:
                    self.console.print("[red]Invalid choice. Please try again.[/red]")
            except Exception:
                self.console.print("[red]Invalid input. Please enter a number.[/red]")
    
    def _setup_ui_preferences(self) -> None:
        """Configure UI preferences."""
        self.console.print(Panel(
            "[bold]Step 3: Interface Preferences[/bold]\n"
            "Customize the ABOV3 interface to your liking.",
            title="UI Setup",
            border_style="yellow"
        ))
        
        # Theme selection
        theme_choices = ["dark", "light", "auto"]
        self.console.print("\nTheme options:")
        for i, theme in enumerate(theme_choices, 1):
            self.console.print(f"  {i}. {theme}")
        
        while True:
            try:
                choice = IntPrompt.ask(
                    "Select theme",
                    default=1,
                    show_default=True
                )
                if 1 <= choice <= len(theme_choices):
                    self.config.ui.theme = theme_choices[choice - 1]
                    break
                else:
                    self.console.print("[red]Invalid choice.[/red]")
            except Exception:
                self.console.print("[red]Invalid input.[/red]")
        
        # Other UI preferences
        self.config.ui.syntax_highlighting = Confirm.ask(
            "Enable syntax highlighting?",
            default=self.config.ui.syntax_highlighting
        )
        
        self.config.ui.line_numbers = Confirm.ask(
            "Show line numbers in code?",
            default=self.config.ui.line_numbers
        )
        
        self.config.ui.vim_mode = Confirm.ask(
            "Enable vim key bindings?",
            default=self.config.ui.vim_mode
        )
    
    def _setup_advanced_settings(self) -> None:
        """Configure advanced settings."""
        self.console.print(Panel(
            "[bold]Advanced Settings[/bold]\n"
            "Configure advanced model and behavior settings.",
            title="Advanced Setup",
            border_style="red"
        ))
        
        # Model parameters
        if Confirm.ask("Configure model parameters?", default=False):
            self.config.model.temperature = FloatPrompt.ask(
                "Temperature (0.0-2.0)",
                default=self.config.model.temperature,
                show_default=True
            )
            
            self.config.model.max_tokens = IntPrompt.ask(
                "Maximum tokens",
                default=self.config.model.max_tokens,
                show_default=True
            )
            
            self.config.model.context_length = IntPrompt.ask(
                "Context length",
                default=self.config.model.context_length,
                show_default=True
            )
        
        # History settings
        if Confirm.ask("Configure history settings?", default=False):
            self.config.history.max_conversations = IntPrompt.ask(
                "Maximum conversations to keep",
                default=self.config.history.max_conversations,
                show_default=True
            )
            
            self.config.history.auto_save = Confirm.ask(
                "Auto-save conversations?",
                default=self.config.history.auto_save
            )
        
        # Update checking
        self.config.check_updates = Confirm.ask(
            "Check for updates on startup?",
            default=self.config.check_updates
        )
    
    def _show_summary(self) -> None:
        """Show configuration summary and confirm."""
        self.console.print(Panel(
            "[bold]Configuration Summary[/bold]\n"
            "Please review your configuration below.",
            title="Summary",
            border_style="cyan"
        ))
        
        table = Table(title="Your ABOV3 Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Ollama Host", self.config.ollama.host)
        table.add_row("Default Model", self.config.model.default_model)
        table.add_row("Theme", self.config.ui.theme)
        table.add_row("Syntax Highlighting", "Yes" if self.config.ui.syntax_highlighting else "No")
        table.add_row("Line Numbers", "Yes" if self.config.ui.line_numbers else "No")
        table.add_row("Vim Mode", "Yes" if self.config.ui.vim_mode else "No")
        table.add_row("Temperature", str(self.config.model.temperature))
        table.add_row("Max Tokens", str(self.config.model.max_tokens))
        table.add_row("Auto-save History", "Yes" if self.config.history.auto_save else "No")
        table.add_row("Check Updates", "Yes" if self.config.check_updates else "No")
        
        self.console.print(table)
        
        if not Confirm.ask("\nSave this configuration?", default=True):
            self.console.print("[yellow]Configuration not saved. You can run the setup wizard again later.[/yellow]")
            sys.exit(0)
    
    def _test_ollama_connection(self, host: str) -> bool:
        """Test connection to Ollama server."""
        try:
            client = ollama.Client(host=host)
            client.list()
            return True
        except Exception:
            return False
    
    def _download_model(self, model_name: str) -> None:
        """Download a model with progress indication."""
        try:
            self.console.print(f"\n[bold]Downloading {model_name}...[/bold]")
            self.console.print("[dim]This may take a while depending on the model size and your internet connection.[/dim]")
            
            client = ollama.Client(host=self.config.ollama.host)
            
            # Start download (this is a simplified version - you might want to add progress tracking)
            with self.console.status(f"Downloading {model_name}..."):
                client.pull(model_name)
            
            self.console.print(f"[green]OK[/green] Successfully downloaded {model_name}")
            
        except Exception as e:
            self.console.print(f"[red]ERROR[/red] Failed to download {model_name}: {e}")
            self.console.print("[yellow]You can download it later using: abov3 models install {model_name}[/yellow]")