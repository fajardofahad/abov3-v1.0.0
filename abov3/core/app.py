"""
Core application class for ABOV3 4 Ollama.

This module contains the main ABOV3 application class that orchestrates all subsystems
including API communication, context management, UI components, security, and plugins.
Provides a production-ready foundation for AI-powered coding assistance.
"""

import asyncio
import logging
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.traceback import Traceback

from .config import Config, get_config, save_config
from .api.ollama_client import OllamaClient, ChatMessage, get_ollama_client, RetryConfig
from .context.manager import ContextManager
from ..models.manager import ModelManager
from ..ui.console.repl import ABOV3REPL, REPLConfig, create_repl
from ..utils.security import SecurityManager
from .api.exceptions import APIError, ConnectionError, ModelNotFoundError


class AppState(Enum):
    """Application state enumeration."""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AppMetrics:
    """Application metrics and statistics."""
    start_time: datetime = field(default_factory=datetime.now)
    total_requests: int = 0
    total_responses: int = 0
    total_errors: int = 0
    active_sessions: int = 0
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    
    def get_uptime(self) -> float:
        """Get current uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()


class ABOV3App:
    """
    Main application class for ABOV3 - AI-powered coding assistant.
    
    This class orchestrates all subsystems including:
    - API client for Ollama communication
    - Context management for conversation state
    - Model management for AI models
    - Security management for safe operations
    - Plugin system for extensibility
    - REPL interface for interactive mode
    - Health monitoring and diagnostics
    """
    
    def __init__(
        self, 
        config: Optional[Config] = None,
        interactive: bool = True,
        debug: bool = False
    ):
        """
        Initialize the ABOV3 application.
        
        Args:
            config: Configuration object, uses default if None
            interactive: Whether to run in interactive mode
            debug: Enable debug mode
        """
        # Core configuration
        self.config = config or get_config()
        self.interactive = interactive
        self.debug = debug or self.config.debug
        
        # Application state
        self.state = AppState.INITIALIZING
        self.metrics = AppMetrics()
        self._shutdown_event = asyncio.Event()
        self._background_tasks: List[asyncio.Task] = []
        self._cleanup_callbacks: List[Callable] = []
        
        # Logging setup
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Rich console for output
        self.console = Console(
            color_system="truecolor",
            force_terminal=True,
            force_jupyter=False
        )
        
        # Core components (initialized in startup)
        self.ollama_client: Optional[OllamaClient] = None
        self.context_manager: Optional[ContextManager] = None
        self.model_manager: Optional[ModelManager] = None
        self.security_manager: Optional[SecurityManager] = None
        self.repl: Optional[ABOV3REPL] = None
        
        # Plugin system (placeholder for future implementation)
        self.plugins: Dict[str, Any] = {}
        
        # Session state
        self.current_session_id: Optional[str] = None
        self.system_prompt: Optional[str] = None
        
        # Performance monitoring
        self._performance_lock = threading.Lock()
        self._last_health_check = time.time()
        
        # Signal handlers
        self._setup_signal_handlers()
        
        self.logger.info("ABOV3 application initialized")
    
    def _setup_logging(self) -> None:
        """Setup application logging."""
        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=self.config.logging.format,
            handlers=[]
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(self.config.logging.format)
        console_handler.setFormatter(formatter)
        
        # File handler if specified
        handlers = [console_handler]
        if self.config.logging.file_path:
            file_handler = logging.FileHandler(self.config.logging.file_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Setup logger
        logger = logging.getLogger()
        logger.handlers = handlers
        logger.setLevel(log_level)
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        asyncio.create_task(self.shutdown())
    
    async def startup(self) -> None:
        """
        Initialize all application subsystems.
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            self.state = AppState.STARTING
            self.logger.info("Starting ABOV3 application...")
            
            # Initialize core components
            await self._initialize_ollama_client()
            await self._initialize_context_manager()
            await self._initialize_model_manager()
            await self._initialize_security_manager()
            
            # Initialize REPL if in interactive mode
            if self.interactive:
                await self._initialize_repl()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Perform health checks
            await self._initial_health_checks()
            
            self.state = AppState.RUNNING
            self.metrics.start_time = datetime.now()
            
            self.logger.info("ABOV3 application started successfully")
            
            if self.interactive:
                self._display_startup_banner()
        
        except Exception as e:
            self.state = AppState.ERROR
            self.logger.error(f"Failed to start ABOV3 application: {e}")
            raise RuntimeError(f"Application startup failed: {e}") from e
    
    async def _initialize_ollama_client(self) -> None:
        """Initialize the Ollama API client."""
        self.logger.debug("Initializing Ollama client...")
        
        retry_config = RetryConfig(
            max_retries=self.config.ollama.max_retries,
            base_delay=1.0,
            max_delay=30.0
        )
        
        self.ollama_client = OllamaClient(
            config=self.config,
            retry_config=retry_config
        )
        
        # Test connection
        if not await self.ollama_client.health_check():
            raise ConnectionError("Cannot connect to Ollama server")
        
        self.logger.info("Ollama client initialized successfully")
    
    async def _initialize_context_manager(self) -> None:
        """Initialize the context manager."""
        self.logger.debug("Initializing context manager...")
        
        context_config = {
            "max_tokens": self.config.model.context_length,
            "compression_threshold": 0.8,
            "sliding_window_size": 20
        }
        
        self.context_manager = ContextManager(context_config)
        self.logger.info("Context manager initialized successfully")
    
    async def _initialize_model_manager(self) -> None:
        """Initialize the model manager."""
        self.logger.debug("Initializing model manager...")
        
        self.model_manager = ModelManager(self.config)
        
        # Verify default model exists
        if not self.model_manager.is_model_available(self.config.model.default_model):
            self.logger.warning(f"Default model {self.config.model.default_model} not available")
        
        self.logger.info("Model manager initialized successfully")
    
    async def _initialize_security_manager(self) -> None:
        """Initialize the security manager."""
        self.logger.debug("Initializing security manager...")
        
        security_config = {
            "rate_limit_requests": 100,
            "rate_limit_window": 3600,
            "session_timeout": 1800
        }
        
        self.security_manager = SecurityManager(security_config)
        self.logger.info("Security manager initialized successfully")
    
    async def _initialize_repl(self) -> None:
        """Initialize the REPL interface."""
        self.logger.debug("Initializing REPL interface...")
        
        repl_config = REPLConfig(
            prompt_text="ABOV3> ",
            theme=self.config.ui.theme,
            enable_syntax_highlighting=self.config.ui.syntax_highlighting,
            enable_vim_mode=self.config.ui.vim_mode,
            history_file=self.config.get_data_dir() / "history.txt",
            session_file=self.config.get_data_dir() / "session.json",
            max_history_size=self.config.ui.max_history_display
        )
        
        self.repl = create_repl(
            config=repl_config,
            process_callback=self._process_user_input
        )
        
        self.logger.info("REPL interface initialized successfully")
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""
        self.logger.debug("Starting background tasks...")
        
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitor_loop())
        self._background_tasks.append(health_task)
        
        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self._background_tasks.append(metrics_task)
        
        # Context cleanup task
        if self.context_manager:
            cleanup_task = asyncio.create_task(self._context_cleanup_loop())
            self._background_tasks.append(cleanup_task)
        
        self.logger.info(f"Started {len(self._background_tasks)} background tasks")
    
    async def _initial_health_checks(self) -> None:
        """Perform initial health checks on all subsystems."""
        self.logger.debug("Performing initial health checks...")
        
        health_status = await self.get_health_status()
        
        critical_issues = [
            component for component, status in health_status.items()
            if not status.get("healthy", False) and status.get("critical", False)
        ]
        
        if critical_issues:
            raise RuntimeError(f"Critical health check failures: {critical_issues}")
        
        self.logger.info("Initial health checks completed successfully")
    
    def _display_startup_banner(self) -> None:
        """Display the application startup banner."""
        banner = Panel(
            Text.from_markup(
                "[bold cyan]ABOV3 - AI-Powered Coding Assistant[/bold cyan]\n\n"
                f"[green]✓[/green] Connected to Ollama at {self.config.ollama.host}\n"
                f"[green]✓[/green] Default Model: {self.config.model.default_model}\n"
                f"[green]✓[/green] Context Window: {self.config.model.context_length:,} tokens\n"
                f"[green]✓[/green] Interactive Mode: {'Enabled' if self.interactive else 'Disabled'}\n\n"
                "[dim]Type '/help' for available commands or start coding![/dim]"
            ),
            title="[bold blue]Welcome to ABOV3[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(banner)
    
    async def _process_user_input(self, user_input: str) -> Union[str, AsyncIterator[str]]:
        """
        Process user input and generate AI response.
        
        Args:
            user_input: Input from the user
            
        Returns:
            AI response or async iterator for streaming
        """
        try:
            self.metrics.total_requests += 1
            
            # Security check
            if self.security_manager:
                is_safe, issues = self.security_manager.is_content_safe(user_input)
                if not is_safe:
                    return f"[red]Security Warning:[/red] {'; '.join(issues)}"
            
            # Add to context
            if self.context_manager:
                self.context_manager.add_message("user", user_input)
            
            # Prepare messages for API
            messages = []
            
            # Add system prompt if set
            if self.system_prompt:
                messages.append(ChatMessage(role="system", content=self.system_prompt))
            
            # Add context
            if self.context_manager:
                context_messages = self.context_manager.get_context_for_model()
                for msg in context_messages:
                    messages.append(ChatMessage(
                        role=msg["role"],
                        content=msg["content"]
                    ))
            else:
                # Fallback: use current input only
                messages.append(ChatMessage(role="user", content=user_input))
            
            # Generate response
            if self.ollama_client:
                if self.config.ui.syntax_highlighting:  # Use as streaming indicator
                    # Streaming response
                    return self._stream_response(messages)
                else:
                    # Non-streaming response
                    response = await self.ollama_client.chat(
                        model=self.config.model.default_model,
                        messages=messages,
                        stream=False
                    )
                    
                    response_content = response.message.content
                    
                    # Add to context
                    if self.context_manager:
                        self.context_manager.add_message("assistant", response_content)
                    
                    self.metrics.total_responses += 1
                    return response_content
            else:
                return "[red]Error:[/red] Ollama client not available"
        
        except Exception as e:
            self.metrics.total_errors += 1
            self.logger.error(f"Error processing user input: {e}")
            return f"[red]Error:[/red] {str(e)}"
    
    async def _stream_response(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        """Stream AI response in real-time."""
        try:
            full_response = ""
            async for chunk in await self.ollama_client.chat(
                model=self.config.model.default_model,
                messages=messages,
                stream=True
            ):
                if chunk.message.content:
                    full_response += chunk.message.content
                    yield chunk.message.content
            
            # Add complete response to context
            if self.context_manager and full_response:
                self.context_manager.add_message("assistant", full_response)
            
            self.metrics.total_responses += 1
        
        except Exception as e:
            self.metrics.total_errors += 1
            yield f"[red]Error:[/red] {str(e)}"
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if self.ollama_client:
                    healthy = await self.ollama_client.health_check()
                    if not healthy:
                        self.logger.warning("Ollama health check failed")
                
                self._last_health_check = time.time()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds
                
                # Update metrics
                self.metrics.uptime_seconds = self.metrics.get_uptime()
                
                # Memory usage (simple approximation)
                import psutil
                process = psutil.Process()
                self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
    
    async def _context_cleanup_loop(self) -> None:
        """Background context cleanup loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
                if self.context_manager:
                    await self.context_manager.async_compress()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in context cleanup: {e}")
    
    async def run(self) -> None:
        """
        Main application run loop.
        
        This method starts the application and runs the appropriate interface
        (REPL for interactive mode, or keeps running for non-interactive).
        """
        try:
            await self.startup()
            
            if self.interactive and self.repl:
                # Run interactive REPL
                await self.repl.run()
            else:
                # Non-interactive mode - wait for shutdown
                await self._shutdown_event.wait()
        
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Application run error: {e}")
            if self.debug:
                self.console.print(Traceback())
        finally:
            await self.shutdown()
    
    def run_sync(self) -> None:
        """Run the application synchronously."""
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            self.console.print("\n[cyan]Goodbye![/cyan]")
    
    async def shutdown(self) -> None:
        """
        Graceful application shutdown.
        
        Stops all background tasks, saves state, and cleans up resources.
        """
        if self.state in [AppState.STOPPING, AppState.STOPPED]:
            return
        
        self.state = AppState.STOPPING
        self.logger.info("Shutting down ABOV3 application...")
        
        try:
            # Signal shutdown to background tasks
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Save configuration
            try:
                save_config(self.config)
            except Exception as e:
                self.logger.warning(f"Failed to save configuration: {e}")
            
            # Save context state
            if self.context_manager:
                try:
                    context_file = self.config.get_data_dir() / "context_state.json"
                    context_data = self.context_manager.export_context("dict")
                    
                    import json
                    with open(context_file, 'w') as f:
                        json.dump(context_data, f, default=str, indent=2)
                    
                    self.logger.debug("Context state saved")
                except Exception as e:
                    self.logger.warning(f"Failed to save context state: {e}")
            
            # Close API client
            if self.ollama_client:
                await self.ollama_client.close()
            
            # Run cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.warning(f"Cleanup callback error: {e}")
            
            self.state = AppState.STOPPED
            self.logger.info("ABOV3 application stopped successfully")
        
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.state = AppState.ERROR
    
    async def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive health status of all subsystems.
        
        Returns:
            Dictionary with health status of each component
        """
        status = {}
        
        # Ollama client health
        if self.ollama_client:
            try:
                healthy = await self.ollama_client.health_check()
                status["ollama"] = {
                    "healthy": healthy,
                    "critical": True,
                    "details": "Ollama API connection"
                }
            except Exception as e:
                status["ollama"] = {
                    "healthy": False,
                    "critical": True,
                    "error": str(e)
                }
        
        # Context manager health
        if self.context_manager:
            try:
                stats = self.context_manager.get_context_stats()
                status["context"] = {
                    "healthy": True,
                    "critical": False,
                    "details": stats
                }
            except Exception as e:
                status["context"] = {
                    "healthy": False,
                    "critical": False,
                    "error": str(e)
                }
        
        # Model manager health
        if self.model_manager:
            try:
                models = self.model_manager.list_models()
                status["models"] = {
                    "healthy": len(models) > 0,
                    "critical": False,
                    "details": f"{len(models)} models available"
                }
            except Exception as e:
                status["models"] = {
                    "healthy": False,
                    "critical": False,
                    "error": str(e)
                }
        
        # Application health
        status["application"] = {
            "healthy": self.state == AppState.RUNNING,
            "critical": True,
            "details": {
                "state": self.state.value,
                "uptime": self.metrics.get_uptime(),
                "requests": self.metrics.total_requests,
                "errors": self.metrics.total_errors
            }
        }
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get application metrics and statistics."""
        with self._performance_lock:
            return {
                "uptime_seconds": self.metrics.get_uptime(),
                "total_requests": self.metrics.total_requests,
                "total_responses": self.metrics.total_responses,
                "total_errors": self.metrics.total_errors,
                "error_rate": (
                    self.metrics.total_errors / max(self.metrics.total_requests, 1)
                ),
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "state": self.state.value,
                "last_health_check": self._last_health_check,
                "background_tasks": len(self._background_tasks)
            }
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for conversations."""
        self.system_prompt = prompt
        self.logger.info("System prompt updated")
    
    def add_cleanup_callback(self, callback: Callable) -> None:
        """Add a callback to be called during shutdown."""
        self._cleanup_callbacks.append(callback)
    
    # Context manager integration
    def get_context_stats(self) -> Optional[Dict[str, Any]]:
        """Get context manager statistics."""
        if self.context_manager:
            return self.context_manager.get_context_stats()
        return None
    
    def clear_context(self, keep_system: bool = True) -> None:
        """Clear the conversation context."""
        if self.context_manager:
            self.context_manager.clear_context(keep_system)
            self.logger.info("Context cleared")
    
    def search_context(self, query: str, max_results: int = 10) -> List[Any]:
        """Search conversation context."""
        if self.context_manager:
            return self.context_manager.search_context(query, max_results)
        return []
    
    # Model manager integration
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        if self.model_manager:
            return self.model_manager.list_models()
        return []
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available."""
        if self.model_manager:
            return self.model_manager.is_model_available(model_name)
        return False
    
    async def install_model(self, model_name: str, progress_callback: Optional[Callable] = None) -> bool:
        """Install a model."""
        try:
            if self.model_manager:
                self.model_manager.install_model(model_name, progress_callback)
                self.logger.info(f"Model {model_name} installed successfully")
                return True
        except Exception as e:
            self.logger.error(f"Failed to install model {model_name}: {e}")
        return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        if self.model_manager:
            try:
                return self.model_manager.get_model_info(model_name)
            except Exception as e:
                self.logger.error(f"Failed to get model info for {model_name}: {e}")
        return None


@asynccontextmanager
async def create_app(
    config: Optional[Config] = None,
    interactive: bool = True,
    debug: bool = False
) -> ABOV3App:
    """
    Async context manager for creating and managing ABOV3 app lifecycle.
    
    Args:
        config: Configuration object
        interactive: Whether to run in interactive mode
        debug: Enable debug mode
        
    Yields:
        Configured and started ABOV3App instance
    """
    app = ABOV3App(config=config, interactive=interactive, debug=debug)
    try:
        await app.startup()
        yield app
    finally:
        await app.shutdown()


def create_app_sync(
    config: Optional[Config] = None,
    interactive: bool = True,
    debug: bool = False
) -> ABOV3App:
    """
    Create ABOV3 app instance synchronously (without auto-startup).
    
    Args:
        config: Configuration object
        interactive: Whether to run in interactive mode
        debug: Enable debug mode
        
    Returns:
        ABOV3App instance (not started)
    """
    return ABOV3App(config=config, interactive=interactive, debug=debug)