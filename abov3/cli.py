#!/usr/bin/env python3
"""
ABOV3 4 Ollama - Production-Ready CLI Interface

A comprehensive command-line interface for the ABOV3 AI coding assistant,
providing powerful AI-powered code generation, debugging, and refactoring
capabilities using local Ollama models.

Author: ABOV3 Team
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import __version__, __description__
from .core.config import Config, get_config, reload_config, save_config
from .core.app import ABOV3App
from .models.manager import ModelManager
from .utils.updater import UpdateChecker
# Import SetupWizard directly to avoid circular import
from .utils.setup import SetupWizard


def suppress_third_party_logging():
    """Early configuration to suppress third-party library logging noise."""
    # Configure root logger first to ensure proper inheritance
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # List of noisy third-party loggers to suppress immediately
    noisy_loggers = [
        'httpx', 'httpcore', 'urllib3', 'requests', 'aiohttp', 'asyncio',
        'ollama', 'websockets', 'prompt_toolkit', 'pygments', 'rich',
        'markdown', 'watchdog', 'gitpython', 'paramiko', 'cryptography',
        'ssl', 'chardet'
    ]
    
    # Set ERROR level for the noisiest loggers
    for logger_name in ['httpx', 'httpcore']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = True  # Allow propagation to file handlers
    
    # Set WARNING level for other third-party loggers
    for logger_name in noisy_loggers:
        if logger_name not in ['httpx', 'httpcore']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            logger.propagate = True  # Allow propagation to file handlers

# Suppress third-party logging as early as possible
suppress_third_party_logging()


# Global console instance for rich output
console = Console()

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_CONFIG_ERROR = 2
EXIT_NETWORK_ERROR = 3
EXIT_MODEL_ERROR = 4
EXIT_PLUGIN_ERROR = 5


class CLIError(Exception):
    """Base exception for CLI errors."""
    
    def __init__(self, message: str, exit_code: int = EXIT_FAILURE):
        self.message = message
        self.exit_code = exit_code
        super().__init__(message)


def handle_error(func):
    """Decorator to handle CLI errors gracefully."""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CLIError as e:
            console.print(f"[bold red]Error:[/bold red] {e.message}")
            sys.exit(e.exit_code)
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user.[/yellow]")
            sys.exit(EXIT_SUCCESS)
        except Exception as e:
            # Try to get config, but don't fail if it's not available
            try:
                debug_mode = get_config().debug
            except:
                debug_mode = False
                
            if debug_mode:
                console.print(f"[bold red]Unexpected error:[/bold red]\n{traceback.format_exc()}")
            else:
                console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}")
                console.print("[dim]Run with --debug flag for more details.[/dim]")
            sys.exit(EXIT_FAILURE)
    return wrapper


def check_ollama_connection(config: Config) -> bool:
    """Check if Ollama server is accessible."""
    try:
        client = ollama.Client(host=config.ollama.host)
        client.list()
        return True
    except Exception:
        return False


def display_banner():
    """Display the ABOV3 banner."""
    banner = Text()
    banner.append("  █████  ██████   ██████  ██    ██ ██████  \n", style="bold cyan")
    banner.append(" ██   ██ ██   ██ ██    ██ ██    ██      ██ \n", style="bold cyan")
    banner.append(" ███████ ██████  ██    ██ ██    ██  █████  \n", style="bold cyan")
    banner.append(" ██   ██ ██   ██ ██    ██  ██  ██       ██ \n", style="bold cyan")
    banner.append(" ██   ██ ██████   ██████    ████   ██████  \n", style="bold cyan")
    banner.append(f"\n{__description__}", style="bold white")
    banner.append(f"\nVersion {__version__}", style="dim")
    
    console.print(Panel(banner, expand=False, border_style="blue"))


# Global options shared across commands
@click.group(invoke_without_command=True)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file"
)
@click.option(
    "--debug", "-d",
    is_flag=True,
    envvar="ABOV3_DEBUG",
    help="Enable debug mode"
)
@click.option(
    "--version", "-v",
    is_flag=True,
    help="Show version and exit"
)
@click.option(
    "--no-banner",
    is_flag=True,
    help="Don't display the banner"
)
@click.pass_context
@handle_error
def cli(ctx: click.Context, config: Optional[Path], debug: bool, version: bool, no_banner: bool):
    """
    ABOV3 4 Ollama - Advanced Interactive AI Coding Assistant
    
    A powerful Python-based console application that provides an interactive CLI interface
    for AI-powered code generation, debugging, and refactoring using local Ollama models.
    
    Examples:
        abov3 chat                    # Start interactive chat session
        abov3 config set model.default_model llama3.2:latest
        abov3 models list             # List available models
        abov3 history search "python" # Search conversation history
        abov3 plugins enable git      # Enable git plugin
    
    Environment Variables:
        ABOV3_CONFIG_PATH             # Path to configuration file
        ABOV3_DEBUG                   # Enable debug mode
        ABOV3_OLLAMA_HOST             # Ollama server host
        ABOV3_DEFAULT_MODEL           # Default model to use
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Handle version flag
    if version:
        console.print(f"ABOV3 4 Ollama v{__version__}")
        sys.exit(EXIT_SUCCESS)
    
    # Load configuration
    try:
        if config:
            app_config = Config.load_from_file(config)
        else:
            app_config = get_config()
    except Exception as e:
        raise CLIError(f"Failed to load configuration: {e}", EXIT_CONFIG_ERROR)
    
    # Override debug setting if provided
    if debug:
        app_config.debug = True
    
    # Store in context
    ctx.obj["config"] = app_config
    ctx.obj["debug"] = app_config.debug
    
    # Display banner unless explicitly disabled or in non-interactive mode
    if not no_banner and sys.stdout.isatty() and ctx.invoked_subcommand != "chat":
        display_banner()
    
    # Check for first run
    if app_config.first_run:
        if ctx.invoked_subcommand is None:
            # No subcommand provided on first run, start setup wizard
            console.print("\n[bold yellow]Welcome to ABOV3 4 Ollama![/bold yellow]")
            console.print("It looks like this is your first time running ABOV3.")
            
            if Confirm.ask("Would you like to run the setup wizard?", default=True):
                wizard = SetupWizard(app_config)
                wizard.run()
                app_config.first_run = False
                save_config(app_config)
                console.print("\n[bold green]Setup completed![/bold green]")
            else:
                app_config.first_run = False
                save_config(app_config)
    
    # Check for updates (if enabled)
    if app_config.check_updates and ctx.invoked_subcommand != "config":
        try:
            updater = UpdateChecker()
            if updater.check_for_updates():
                latest_version = updater.get_latest_version()
                console.print(f"\n[bold yellow]Update available![/bold yellow] v{latest_version}")
                console.print("Run 'abov3 update' to install the latest version.")
        except Exception:
            # Silently ignore update check failures
            pass
    
    # If no subcommand provided, start chat session
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


@cli.command()
@click.option(
    "--model", "-m",
    help="Model to use for the chat session"
)
@click.option(
    "--system", "-s",
    help="System prompt to use"
)
@click.option(
    "--temperature", "-t",
    type=float,
    help="Temperature for response generation"
)
@click.option(
    "--no-history",
    is_flag=True,
    help="Don't save conversation to history"
)
@click.option(
    "--continue-last",
    is_flag=True,
    help="Continue the last conversation"
)
@click.pass_context
@handle_error
def chat(ctx: click.Context, model: Optional[str], system: Optional[str], 
         temperature: Optional[float], no_history: bool, continue_last: bool):
    """
    Start an interactive chat session with ABOV3.
    
    This command launches the main interactive interface where you can have
    conversations with AI models for coding assistance, debugging, and more.
    
    Examples:
        abov3 chat                              # Start with default settings
        abov3 chat -m llama3.2:latest          # Use specific model
        abov3 chat -t 0.9                      # Set temperature
        abov3 chat --system "You are a Python expert"  # Custom system prompt
        abov3 chat --continue-last              # Continue last conversation
    """
    config = ctx.obj["config"]
    
    # Validate Ollama connection
    if not check_ollama_connection(config):
        raise CLIError(
            f"Cannot connect to Ollama server at {config.ollama.host}. "
            f"Please ensure Ollama is running and accessible.",
            EXIT_NETWORK_ERROR
        )
    
    # Override model if specified
    if model:
        config.model.default_model = model
    
    # Override temperature if specified
    if temperature is not None:
        config.model.temperature = temperature
    
    # Create and run the app
    try:
        app = ABOV3App(config)
        
        # Configure app options
        app.save_history = not no_history
        if system:
            app.set_system_prompt(system)
        if continue_last:
            app.load_last_conversation()
        
        # Run the interactive session
        asyncio.run(app.run())
        
    except Exception as e:
        raise CLIError(f"Failed to start chat session: {e}")


@cli.group()
@click.pass_context
def config(ctx: click.Context):
    """
    Manage ABOV3 configuration settings.
    
    The config command allows you to view, modify, and validate your ABOV3
    configuration. Settings can be organized hierarchically using dot notation.
    
    Examples:
        abov3 config show                       # Show all configuration
        abov3 config get model.default_model    # Get specific setting
        abov3 config set model.temperature 0.8  # Set specific setting
        abov3 config reset                      # Reset to defaults
        abov3 config validate                   # Validate configuration
    """
    pass


@config.command()
@click.option(
    "--format", "-f",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format"
)
@click.pass_context
@handle_error
def show(ctx: click.Context, format: str):
    """Show current configuration."""
    config = ctx.obj["config"]
    
    if format == "json":
        import json
        console.print_json(json.dumps(config.model_dump(), indent=2))
    elif format == "yaml":
        import yaml
        console.print(yaml.dump(config.model_dump(), default_flow_style=False))
    else:  # table format
        table = Table(title="ABOV3 Configuration", show_header=True)
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Description", style="dim")
        
        def add_config_rows(obj, prefix=""):
            # Check if obj is a pydantic model or a dict
            if hasattr(obj, 'model_dump'):
                data = obj.model_dump()
                model_fields = obj.__class__.model_fields
            else:
                data = obj
                model_fields = {}
                
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    # This is a nested configuration
                    # Add a section header
                    table.add_row(f"[bold]{full_key}[/bold]", "", "")
                    for nested_key, nested_value in value.items():
                        nested_full_key = f"{full_key}.{nested_key}"
                        table.add_row(f"  {nested_key}", str(nested_value), "")
                else:
                    # Get field info for description
                    field_info = model_fields.get(key)
                    description = field_info.description if field_info else ""
                    table.add_row(full_key, str(value), description)
        
        add_config_rows(config)
        console.print(table)


@config.command()
@click.argument("key")
@click.pass_context
@handle_error
def get(ctx: click.Context, key: str):
    """Get a specific configuration value."""
    config = ctx.obj["config"]
    
    try:
        # Navigate through nested keys
        keys = key.split(".")
        value = config
        for k in keys:
            value = getattr(value, k)
        
        console.print(f"[cyan]{key}[/cyan] = [green]{value}[/green]")
    except AttributeError:
        raise CLIError(f"Configuration key '{key}' not found")


@config.command()
@click.argument("key")
@click.argument("value")
@click.pass_context
@handle_error
def set(ctx: click.Context, key: str, value: str):
    """Set a configuration value."""
    config = ctx.obj["config"]
    
    try:
        # Navigate to the parent object
        keys = key.split(".")
        parent = config
        for k in keys[:-1]:
            parent = getattr(parent, k)
        
        final_key = keys[-1]
        
        # Get the field type and convert value accordingly
        field = parent.__class__.model_fields.get(final_key)
        if field:
            # Get annotation from field
            field_annotation = field.annotation
            if field_annotation == bool:
                value = value.lower() in ("true", "1", "yes", "on")
            elif field_annotation == int:
                value = int(value)
            elif field_annotation == float:
                value = float(value)
            # For other types, keep as string
        
        # Set the value
        setattr(parent, final_key, value)
        
        # Save configuration
        save_config(config)
        
        console.print(f"[green]OK[/green] Set [cyan]{key}[/cyan] = [green]{value}[/green]")
        
    except AttributeError:
        raise CLIError(f"Configuration key '{key}' not found")
    except ValueError as e:
        raise CLIError(f"Invalid value for '{key}': {e}")


@config.command()
@click.option(
    "--confirm",
    is_flag=True,
    help="Skip confirmation prompt"
)
@click.pass_context
@handle_error
def reset(ctx: click.Context, confirm: bool):
    """Reset configuration to defaults."""
    if not confirm:
        if not Confirm.ask("This will reset all configuration to defaults. Continue?"):
            console.print("[yellow]Reset cancelled.[/yellow]")
            return
    
    # Create new default config
    new_config = Config()
    new_config.save_to_file()
    
    # Reload global config
    reload_config()
    
    console.print("[green]OK[/green] Configuration reset to defaults")


@config.command()
@click.pass_context
@handle_error
def validate(ctx: click.Context):
    """Validate current configuration."""
    config = ctx.obj["config"]
    
    try:
        # Test Ollama connection
        if not check_ollama_connection(config):
            console.print("[red]ERROR[/red] Cannot connect to Ollama server")
            return
        
        # Test model availability
        model_manager = ModelManager(config)
        if not model_manager.is_model_available(config.model.default_model):
            console.print(f"[yellow]WARNING[/yellow] Default model '{config.model.default_model}' not available")
        
        console.print("[green]OK[/green] Configuration is valid")
        
    except Exception as e:
        console.print(f"[red]ERROR[/red] Configuration validation failed: {e}")


@cli.group()
@click.pass_context
def models(ctx: click.Context):
    """
    Manage AI models for ABOV3.
    
    The models command provides functionality to list, install, remove, and
    manage AI models available through Ollama.
    
    Examples:
        abov3 models list                       # List all models
        abov3 models install llama3.2:latest   # Install a model
        abov3 models remove llama2:7b           # Remove a model
        abov3 models info llama3.2:latest      # Show model information
        abov3 models set-default llama3.2:latest # Set default model
    """
    pass


@models.command()
@click.option(
    "--format", "-f",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format"
)
@click.pass_context
@handle_error
def list(ctx: click.Context, format: str):
    """List available models."""
    config = ctx.obj["config"]
    
    if not check_ollama_connection(config):
        raise CLIError("Cannot connect to Ollama server", EXIT_NETWORK_ERROR)
    
    try:
        model_manager = ModelManager(config)
        models = model_manager.list_models_sync()
        
        if format == "json":
            import json
            console.print_json(json.dumps(models, indent=2))
        else:
            table = Table(title="Available Models")
            table.add_column("Name", style="cyan")
            table.add_column("Size", style="green")
            table.add_column("Modified", style="yellow")
            table.add_column("Current", style="bold red")
            
            for model in models:
                is_current = "OK" if model["name"] == config.model.default_model else ""
                table.add_row(
                    model["name"],
                    model.get("size", "Unknown"),
                    model.get("modified_at", "Unknown"),
                    is_current
                )
            
            console.print(table)
            
    except Exception as e:
        raise CLIError(f"Failed to list models: {e}", EXIT_MODEL_ERROR)


@models.command()
@click.argument("model_name")
@click.option(
    "--progress",
    is_flag=True,
    default=True,
    help="Show download progress"
)
@click.pass_context
@handle_error
def install(ctx: click.Context, model_name: str, progress: bool):
    """Install a model."""
    config = ctx.obj["config"]
    
    if not check_ollama_connection(config):
        raise CLIError("Cannot connect to Ollama server", EXIT_NETWORK_ERROR)
    
    try:
        model_manager = ModelManager(config)
        
        console.print(f"Installing model: [cyan]{model_name}[/cyan]")
        
        if progress:
            # Install with progress callback
            def progress_callback(status):
                console.print(f"Status: {status}")
            
            success = model_manager.install_model_sync(model_name, progress_callback)
            if not success:
                raise CLIError(f"Failed to install model {model_name}")
        else:
            success = model_manager.install_model_sync(model_name)
            if not success:
                raise CLIError(f"Failed to install model {model_name}")
        
        console.print(f"[green]OK[/green] Model [cyan]{model_name}[/cyan] installed successfully")
        
    except Exception as e:
        raise CLIError(f"Failed to install model: {e}", EXIT_MODEL_ERROR)


@models.command()
@click.argument("model_name")
@click.option(
    "--confirm",
    is_flag=True,
    help="Skip confirmation prompt"
)
@click.pass_context
@handle_error
def remove(ctx: click.Context, model_name: str, confirm: bool):
    """Remove a model."""
    config = ctx.obj["config"]
    
    if not confirm:
        if not Confirm.ask(f"Remove model '{model_name}'?"):
            console.print("[yellow]Remove cancelled.[/yellow]")
            return
    
    if not check_ollama_connection(config):
        raise CLIError("Cannot connect to Ollama server", EXIT_NETWORK_ERROR)
    
    try:
        model_manager = ModelManager(config)
        success = model_manager.remove_model_sync(model_name)
        if not success:
            raise CLIError(f"Failed to remove model {model_name}")
        
        console.print(f"[green]OK[/green] Model [cyan]{model_name}[/cyan] removed successfully")
        
    except Exception as e:
        raise CLIError(f"Failed to remove model: {e}", EXIT_MODEL_ERROR)


@models.command()
@click.argument("model_name")
@click.pass_context
@handle_error
def info(ctx: click.Context, model_name: str):
    """Show detailed information about a model."""
    config = ctx.obj["config"]
    
    if not check_ollama_connection(config):
        raise CLIError("Cannot connect to Ollama server", EXIT_NETWORK_ERROR)
    
    try:
        model_manager = ModelManager(config)
        info = model_manager.get_model_info_sync(model_name)
        
        table = Table(title=f"Model Information: {model_name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in info.items():
            table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(table)
        
    except Exception as e:
        raise CLIError(f"Failed to get model info: {e}", EXIT_MODEL_ERROR)


@models.command(name="set-default")
@click.argument("model_name")
@click.pass_context
@handle_error
def set_default(ctx: click.Context, model_name: str):
    """Set the default model."""
    config = ctx.obj["config"]
    
    # Verify model exists
    if not check_ollama_connection(config):
        raise CLIError("Cannot connect to Ollama server", EXIT_NETWORK_ERROR)
    
    try:
        model_manager = ModelManager(config)
        if not model_manager.is_model_available(model_name):
            raise CLIError(f"Model '{model_name}' is not available")
        
        config.model.default_model = model_name
        save_config(config)
        
        console.print(f"[green]OK[/green] Default model set to [cyan]{model_name}[/cyan]")
        
    except Exception as e:
        raise CLIError(f"Failed to set default model: {e}", EXIT_MODEL_ERROR)


@cli.group()
@click.pass_context
def history(ctx: click.Context):
    """
    Manage conversation history.
    
    The history command provides functionality to view, search, and manage
    your conversation history with ABOV3.
    
    Examples:
        abov3 history list                      # List all conversations
        abov3 history search "python"           # Search conversations
        abov3 history show <id>                 # Show specific conversation
        abov3 history export <id> output.json   # Export conversation
        abov3 history clear                     # Clear all history
    """
    pass


@history.command()
@click.option(
    "--limit", "-l",
    type=int,
    default=10,
    help="Maximum number of conversations to show"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format"
)
@click.pass_context
@handle_error
def list(ctx: click.Context, limit: int, format: str):
    """List conversation history."""
    # Implementation would depend on history storage system
    console.print("[yellow]History list functionality not yet implemented.[/yellow]")


@history.command()
@click.argument("query")
@click.option(
    "--limit", "-l",
    type=int,
    default=10,
    help="Maximum number of results"
)
@click.pass_context
@handle_error
def search(ctx: click.Context, query: str, limit: int):
    """Search conversation history."""
    # Implementation would depend on history storage system
    console.print(f"[yellow]Searching for: {query} (not yet implemented)[/yellow]")


@cli.group()
@click.pass_context
def plugins(ctx: click.Context):
    """
    Manage ABOV3 plugins.
    
    The plugins command provides functionality to list, enable, disable,
    and manage plugins that extend ABOV3's capabilities.
    
    Examples:
        abov3 plugins list                      # List all plugins
        abov3 plugins enable git                # Enable git plugin
        abov3 plugins disable git               # Disable git plugin
        abov3 plugins info git                  # Show plugin information
        abov3 plugins install <path>            # Install plugin from path
    """
    pass


@plugins.command()
@click.option(
    "--enabled-only",
    is_flag=True,
    help="Show only enabled plugins"
)
@click.pass_context
@handle_error
def list(ctx: click.Context, enabled_only: bool):
    """List available plugins."""
    # Implementation would depend on plugin system
    console.print("[yellow]Plugin list functionality not yet implemented.[/yellow]")


@plugins.command()
@click.argument("plugin_name")
@click.pass_context
@handle_error
def enable(ctx: click.Context, plugin_name: str):
    """Enable a plugin."""
    config = ctx.obj["config"]
    
    if plugin_name not in config.plugins.enabled:
        config.plugins.enabled.append(plugin_name)
        save_config(config)
        console.print(f"[green]OK[/green] Plugin [cyan]{plugin_name}[/cyan] enabled")
    else:
        console.print(f"[yellow]Plugin [cyan]{plugin_name}[/cyan] is already enabled[/yellow]")


@plugins.command()
@click.argument("plugin_name")
@click.pass_context
@handle_error
def disable(ctx: click.Context, plugin_name: str):
    """Disable a plugin."""
    config = ctx.obj["config"]
    
    if plugin_name in config.plugins.enabled:
        config.plugins.enabled.remove(plugin_name)
        save_config(config)
        console.print(f"[green]OK[/green] Plugin [cyan]{plugin_name}[/cyan] disabled")
    else:
        console.print(f"[yellow]Plugin [cyan]{plugin_name}[/cyan] is not enabled[/yellow]")


@cli.command()
@click.option(
    "--check-only",
    is_flag=True,
    help="Only check for updates, don't install"
)
@click.pass_context
@handle_error
def update(ctx: click.Context, check_only: bool):
    """
    Check for and install updates.
    
    This command checks for available updates to ABOV3 and can install them
    automatically.
    
    Examples:
        abov3 update                            # Check and install updates
        abov3 update --check-only               # Only check for updates
    """
    try:
        updater = UpdateChecker()
        
        console.print("Checking for updates...")
        
        if updater.check_for_updates():
            latest_version = updater.get_latest_version()
            console.print(f"[green]Update available![/green] v{latest_version}")
            
            if check_only:
                console.print("Use 'abov3 update' to install the latest version.")
                return
            
            if Confirm.ask("Would you like to install the update?"):
                console.print("Installing update...")
                updater.install_update()
                console.print("[green]OK[/green] Update installed successfully!")
                console.print("Please restart ABOV3 to use the new version.")
            else:
                console.print("[yellow]Update cancelled.[/yellow]")
        else:
            console.print("[green]You are running the latest version.[/green]")
            
    except Exception as e:
        raise CLIError(f"Update check failed: {e}")


@cli.command()
@click.pass_context
@handle_error
def doctor(ctx: click.Context):
    """
    Run diagnostic checks on your ABOV3 installation.
    
    This command performs comprehensive health checks on your ABOV3 installation,
    configuration, and dependencies.
    """
    config = ctx.obj["config"]
    
    console.print("[bold]ABOV3 Health Check[/bold]\n")
    
    checks = []
    
    # Check Ollama connection
    console.print("* Checking Ollama connection...")
    if check_ollama_connection(config):
        console.print("[green]OK[/green] Ollama server is accessible")
        checks.append(True)
    else:
        console.print(f"[red]ERROR[/red] Cannot connect to Ollama server at {config.ollama.host}")
        checks.append(False)
    
    # Check default model
    console.print("* Checking default model...")
    try:
        model_manager = ModelManager(config)
        if model_manager.is_model_available(config.model.default_model):
            console.print(f"[green]OK[/green] Default model '{config.model.default_model}' is available")
            checks.append(True)
        else:
            console.print(f"[red]ERROR[/red] Default model '{config.model.default_model}' is not available")
            checks.append(False)
    except Exception as e:
        console.print(f"[red]ERROR[/red] Failed to check model: {e}")
        checks.append(False)
    
    # Check configuration directories
    console.print("* Checking configuration directories...")
    try:
        config_dir = Config.get_config_dir()
        data_dir = Config.get_data_dir()
        cache_dir = Config.get_cache_dir()
        
        if config_dir.exists() and data_dir.exists() and cache_dir.exists():
            console.print("[green]OK[/green] Configuration directories exist")
            checks.append(True)
        else:
            console.print("[red]ERROR[/red] Some configuration directories are missing")
            checks.append(False)
    except Exception as e:
        console.print(f"[red]ERROR[/red] Failed to check directories: {e}")
        checks.append(False)
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold] {sum(checks)}/{len(checks)} checks passed")
    
    if all(checks):
        console.print("[green]SUCCESS: ABOV3 is healthy and ready to use![/green]")
    else:
        console.print("[yellow]WARNING Some issues were found. Please address them for optimal performance.[/yellow]")


def main():
    """Main entry point for the CLI."""
    try:
        # Ensure proper terminal encoding
        if sys.platform == "win32":
            import locale
            locale.setlocale(locale.LC_ALL, "")
        
        cli()
    except Exception as e:
        console.print(f"[bold red]Fatal error:[/bold red] {e}")
        sys.exit(EXIT_FAILURE)


if __name__ == "__main__":
    main()