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
    DEGRADED = "degraded"  # Some services failed but core functionality works
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
    failed_components: List[str] = field(default_factory=list)
    degraded_features: List[str] = field(default_factory=list)
    
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
        self._component_health: Dict[str, bool] = {}
        self._startup_timeout = 30.0  # 30 seconds startup timeout
        self._health_check_timeout = 5.0  # 5 seconds per health check
        
        # Logging setup - do this FIRST
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
        
        # Now we can safely log since logger is set up
        self.logger.info("ABOV3 application initialized")
    
    def _setup_logging(self) -> None:
        """Setup application logging with proper third-party library suppression."""
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
        
        # Suppress noisy third-party library logging to avoid user-facing noise
        self._configure_third_party_logging()
    
    def _configure_third_party_logging(self) -> None:
        """Configure third-party library logging to suppress noise in user interface."""
        # List of third-party loggers to suppress based on common noisy libraries
        third_party_loggers = [
            'httpx',           # HTTP client library logs
            'httpcore',        # HTTP core library logs
            'urllib3',         # HTTP library logs
            'requests',        # HTTP library logs
            'aiohttp',         # Async HTTP library logs
            'asyncio',         # Asyncio debug logs
            'ollama',          # Ollama library logs (keep at WARNING+ only)
            'websockets',      # WebSocket library logs
            'prompt_toolkit',  # Terminal library logs
            'pygments',        # Syntax highlighting logs
            'rich',            # Rich text library logs
            'markdown',        # Markdown parsing logs
            'watchdog',        # File watching logs
            'gitpython',       # Git library logs
            'paramiko',        # SSH library logs
            'cryptography',    # Crypto library logs
            'ssl',             # SSL library logs
            'chardet',         # Character detection logs
        ]
        
        # Set WARNING level for third-party libraries to suppress INFO/DEBUG noise
        for logger_name in third_party_loggers:
            third_party_logger = logging.getLogger(logger_name)
            third_party_logger.setLevel(logging.WARNING)
        
        # Special handling for specific loggers that are particularly noisy
        # httpx is very verbose with HTTP request logs - set to ERROR level
        httpx_logger = logging.getLogger('httpx')
        httpx_logger.setLevel(logging.ERROR)
        
        # httpcore is also verbose with connection logs
        httpcore_logger = logging.getLogger('httpcore')
        httpcore_logger.setLevel(logging.ERROR)
        
        # Ollama library should only show warnings and errors to users
        ollama_logger = logging.getLogger('ollama')
        ollama_logger.setLevel(logging.WARNING)
        
        # Only show ABOV3 application logs and higher severity third-party logs
        abov3_logger = logging.getLogger('abov3')
        abov3_logger.setLevel(self.config.logging.level.upper() if hasattr(self.config.logging.level, 'upper') else logging.INFO)
        
        # Note: Can't use self.logger here since it's not created yet, but that's OK
        # The debug message will be logged later when logger is available
    
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
        Initialize all application subsystems with resilient startup logic.
        
        Supports degraded mode where the app can start even if some
        non-critical components fail.
        
        Raises:
            RuntimeError: If critical initialization fails
        """
        startup_start = time.time()
        
        try:
            self.state = AppState.STARTING
            self.logger.info("Starting ABOV3 application...")
            
            # Initialize components with timeout and error handling
            self._report_startup_progress("Initializing components...")
            components_status = await self._initialize_components_resilient()
            
            # Report component status
            self._report_component_status(components_status)
            
            # Check if we have minimum required components for basic operation
            # For now, we allow the app to start even without Ollama in degraded mode
            # Only completely fail if ALL components are broken
            all_failed = all(not status for status in components_status.values())
            
            if all_failed:
                self.state = AppState.ERROR
                error_msg = "All components failed to initialize"
                self.logger.error(error_msg)
                self._report_startup_progress(f"FAILED - {error_msg}", is_error=True)
                raise RuntimeError(error_msg)
            
            # Start background tasks (non-blocking)
            self._report_startup_progress("Starting background tasks...")
            self._start_background_tasks_non_blocking()
            
            # Determine final state
            failed_components = [name for name, status in components_status.items() if not status]
            if failed_components:
                self.state = AppState.DEGRADED
                self.metrics.failed_components = failed_components
                self.logger.warning(f"Application started in degraded mode. Failed components: {failed_components}")
                self._report_startup_progress(f"Started in DEGRADED mode - some components failed: {', '.join(failed_components)}", is_error=False)
            else:
                self.state = AppState.RUNNING
                self.logger.info("ABOV3 application started successfully - all components healthy")
                self._report_startup_progress("Successfully started - all components healthy")
            
            self.metrics.start_time = datetime.now()
            
            # Display banner with status
            if self.interactive:
                self._display_startup_banner()
        
        except Exception as e:
            self.state = AppState.ERROR
            elapsed = time.time() - startup_start
            self.logger.error(f"Failed to start ABOV3 application after {elapsed:.2f}s: {e}")
            raise RuntimeError(f"Application startup failed: {e}") from e
    
    async def _initialize_components_resilient(self) -> Dict[str, bool]:
        """
        Initialize all components with error handling and timeouts.
        
        Returns:
            Dictionary mapping component names to initialization success status
        """
        components_status = {}
        
        # Define component initialization tasks
        component_tasks = [
            ('ollama_client', self._initialize_ollama_client_safe()),
            ('context_manager', self._initialize_context_manager_safe()),
            ('model_manager', self._initialize_model_manager_safe()),
            ('security_manager', self._initialize_security_manager_safe()),
        ]
        
        # Add REPL if in interactive mode
        if self.interactive:
            component_tasks.append(('repl', self._initialize_repl_safe()))
        
        # Initialize components concurrently with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[task for _, task in component_tasks], return_exceptions=True),
                timeout=self._startup_timeout
            )
            
            # Process results
            for i, (component_name, _) in enumerate(component_tasks):
                result = results[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to initialize {component_name}: {result}")
                    components_status[component_name] = False
                    self._component_health[component_name] = False
                else:
                    components_status[component_name] = result
                    self._component_health[component_name] = result
                    if result:
                        self.logger.info(f"{component_name} initialized successfully")
                    else:
                        self.logger.warning(f"{component_name} initialization failed")
        
        except asyncio.TimeoutError:
            self.logger.error(f"Component initialization timed out after {self._startup_timeout}s")
            # Mark all unfinished components as failed
            for component_name, _ in component_tasks:
                if component_name not in components_status:
                    components_status[component_name] = False
                    self._component_health[component_name] = False
        
        return components_status
    
    async def _initialize_ollama_client_safe(self) -> bool:
        """Safe initialization of Ollama client with timeout."""
        try:
            await asyncio.wait_for(self._initialize_ollama_client(), timeout=10.0)
            return True
        except Exception as e:
            self.logger.error(f"Ollama client initialization failed: {e}")
            return False
    
    async def _initialize_context_manager_safe(self) -> bool:
        """Safe initialization of context manager."""
        try:
            await self._initialize_context_manager()
            return True
        except Exception as e:
            self.logger.error(f"Context manager initialization failed: {e}")
            return False
    
    async def _initialize_model_manager_safe(self) -> bool:
        """Safe initialization of model manager."""
        try:
            await self._initialize_model_manager()
            return True
        except Exception as e:
            self.logger.error(f"Model manager initialization failed: {e}")
            return False
    
    async def _initialize_security_manager_safe(self) -> bool:
        """Safe initialization of security manager."""
        try:
            await self._initialize_security_manager()
            return True
        except Exception as e:
            self.logger.error(f"Security manager initialization failed: {e}")
            return False
    
    async def _initialize_repl_safe(self) -> bool:
        """Safe initialization of REPL interface."""
        try:
            await self._initialize_repl()
            return True
        except Exception as e:
            self.logger.error(f"REPL initialization failed: {e}")
            return False

    async def _initialize_ollama_client(self) -> None:
        """Initialize the Ollama API client with timeout and retry."""
        self.logger.debug("Initializing Ollama client...")
        
        retry_config = RetryConfig(
            max_retries=self.config.ollama.max_retries,
            base_delay=1.0,
            max_delay=10.0  # Reduced for faster startup
        )
        
        self.ollama_client = OllamaClient(
            config=self.config,
            retry_config=retry_config
        )
        
        # Test connection with timeout and retries
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.logger.debug(f"Testing Ollama connection (attempt {attempt + 1}/{max_attempts})...")
                healthy = await asyncio.wait_for(
                    self.ollama_client.health_check(), 
                    timeout=self._health_check_timeout
                )
                if healthy:
                    self.logger.info("Ollama client initialized and connected successfully")
                    return
                else:
                    self.logger.warning(f"Ollama health check failed (attempt {attempt + 1})")
            except asyncio.TimeoutError:
                self.logger.warning(f"Ollama health check timed out (attempt {attempt + 1})")
            except Exception as e:
                self.logger.warning(f"Ollama connection test failed (attempt {attempt + 1}): {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < max_attempts - 1:
                await asyncio.sleep(1.0)
        
        # If we get here, all attempts failed
        raise ConnectionError(f"Cannot connect to Ollama server after {max_attempts} attempts")
    
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
            enable_syntax_highlighting=False,  # Disable to prevent streaming mode initially
            enable_vim_mode=self.config.ui.vim_mode,
            enable_multiline=False,  # Explicitly disable multiline to prevent ... prompts
            history_file=self.config.get_data_dir() / "history.txt",
            session_file=self.config.get_data_dir() / "session.json",
            max_history_size=self.config.ui.max_history_display
        )
        
        self.repl = create_repl(
            config=repl_config,
            process_callback=self._process_user_input
        )
        
        # Add reference to app instance for status commands
        self.repl.app_instance = self
        
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
    
    def _start_background_tasks_non_blocking(self) -> None:
        """Start background monitoring and maintenance tasks in a non-blocking way."""
        self.logger.debug("Starting background tasks (non-blocking)...")
        
        # Start tasks only if components are available
        if self._component_health.get('ollama_client', False):
            health_task = asyncio.create_task(self._health_monitor_loop())
            self._background_tasks.append(health_task)
        
        # Metrics collection doesn't depend on external services
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self._background_tasks.append(metrics_task)
        
        # Context cleanup task (only if context manager is available)
        if self._component_health.get('context_manager', False) and self.context_manager:
            cleanup_task = asyncio.create_task(self._context_cleanup_loop())
            self._background_tasks.append(cleanup_task)
        
        self.logger.info(f"Started {len(self._background_tasks)} background tasks")
    
    async def _initial_health_checks(self) -> None:
        """Perform initial health checks on all subsystems."""
        self.logger.debug("Performing initial health checks...")
        
        health_status = await self.get_health_status(startup_mode=True)
        
        critical_issues = []
        warnings = []
        
        for component, status in health_status.items():
            if not status.get("healthy", False):
                if status.get("critical", False):
                    critical_issues.append(component)
                    self.logger.error(f"Critical health check failure for {component}: {status.get('error', 'Unknown error')}")
                else:
                    warnings.append(component)
                    self.logger.warning(f"Health check warning for {component}: {status.get('error', 'Minor issue')}")
            else:
                self.logger.debug(f"Health check passed for {component}")
        
        if warnings:
            self.logger.info(f"Health check warnings (non-critical): {warnings}")
        
        # Only fail on critical issues, and be more lenient during startup
        if critical_issues:
            # Filter out known startup-related issues
            startup_tolerant_critical_issues = []
            for issue in critical_issues:
                if issue == "ollama" and not self.config.ollama.required_for_startup:
                    self.logger.warning("Ollama service not available but marked as non-required for startup")
                    continue
                startup_tolerant_critical_issues.append(issue)
            
            if startup_tolerant_critical_issues:
                raise RuntimeError(f"Critical health check failures: {startup_tolerant_critical_issues}")
        
        self.logger.info("Initial health checks completed successfully")
    
    def _display_startup_banner(self) -> None:
        """Display the application startup banner with actual component status."""
        # Check component statuses - use ASCII characters for Windows compatibility
        ollama_status = "OK" if self._component_health.get('ollama_client', False) else "FAIL"
        context_status = "OK" if self._component_health.get('context_manager', False) else "FAIL"
        model_status = "OK" if self._component_health.get('model_manager', False) else "FAIL"
        security_status = "OK" if self._component_health.get('security_manager', False) else "FAIL"
        
        # Set colors based on status
        ollama_color = "green" if ollama_status == "OK" else "red"
        context_color = "green" if context_status == "OK" else "yellow"
        model_color = "green" if model_status == "OK" else "yellow"
        security_color = "green" if security_status == "OK" else "yellow"
        
        # App state indicator
        state_text = f"State: {self.state.value.title()}"
        state_color = "green" if self.state == AppState.RUNNING else "yellow" if self.state == AppState.DEGRADED else "red"
        
        banner_content = f"""[bold cyan]ABOV3 - AI-Powered Coding Assistant[/bold cyan]

[{state_color}]{state_text}[/{state_color}]

[{ollama_color}]{ollama_status}[/{ollama_color}] Ollama Connection: {"Connected" if ollama_status == "OK" else "Failed"} ({self.config.ollama.host})
[{context_color}]{context_status}[/{context_color}] Context Manager: {"Active" if context_status == "OK" else "Failed"}
[{model_color}]{model_status}[/{model_color}] Model Manager: {"Active" if model_status == "OK" else "Failed"}
[{security_color}]{security_status}[/{security_color}] Security Manager: {"Active" if security_status == "OK" else "Failed"}
[green]OK[/green] Interactive Mode: {'Enabled' if self.interactive else 'Disabled'}"""
        
        if self.metrics.failed_components:
            banner_content += f"\n\n[yellow]WARN[/yellow] Failed Components: {', '.join(self.metrics.failed_components)}"
        
        banner_content += "\n\n[dim]Type '/help' for available commands or start coding![/dim]"
        
        border_style = "green" if self.state == AppState.RUNNING else "yellow" if self.state == AppState.DEGRADED else "red"
        
        banner = Panel(
            Text.from_markup(banner_content),
            title="[bold blue]Welcome to ABOV3[/bold blue]",
            border_style=border_style,
            padding=(1, 2)
        )
        self.console.print(banner)
    
    async def _process_user_input(self, user_input: str) -> Union[str, AsyncIterator[str]]:
        """
        Process user input and generate AI response.
        
        This method is called by the REPL after it has already handled commands.
        It processes the input as a chat message and sends it to the AI.
        
        Args:
            user_input: Input from the user
            
        Returns:
            AI response or async iterator for streaming
        """
        try:
            # Process as AI chat input (commands are already handled by REPL)
            self.metrics.total_requests += 1
            
            # Security check for chat input
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
            if self.ollama_client and self._component_health.get('ollama_client', False):
                # Check if default model is available
                model_to_use = self.config.model.default_model
                try:
                    if not await self.ollama_client.model_exists(model_to_use):
                        # Try to find an alternative model
                        available_models = await self.ollama_client.list_models()
                        if available_models:
                            model_to_use = available_models[0].name
                            self.logger.warning(f"Default model {self.config.model.default_model} not found, using {model_to_use}")
                        else:
                            return "[red]Error:[/red] No models available. Please install a model with: ollama pull llama3.2:latest"
                except Exception as e:
                    self.logger.error(f"Error checking model availability: {e}")
                    # Continue with configured model and let the API handle the error
                
                if self.config.ui.syntax_highlighting:  # Use as streaming indicator
                    # Streaming response
                    return self._stream_response(messages, model_to_use)
                else:
                    # Non-streaming response
                    response = await self.ollama_client.chat(
                        model=model_to_use,
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
                # Provide helpful message based on current state
                if self.state == AppState.DEGRADED:
                    return "[yellow]Warning:[/yellow] ABOV3 is running in degraded mode. AI chat functionality is not available due to Ollama connection issues. Please check your Ollama server and restart ABOV3."
                else:
                    return "[red]Error:[/red] Ollama client not available"
        
        except Exception as e:
            self.metrics.total_errors += 1
            self.logger.error(f"Error processing user input: {e}")
            return f"[red]Error:[/red] {str(e)}"
    
    async def _stream_response(self, messages: List[ChatMessage], model: Optional[str] = None) -> AsyncIterator[str]:
        """Stream AI response in real-time."""
        model_to_use = model or self.config.model.default_model
        try:
            full_response = ""
            async for chunk in await self.ollama_client.chat(
                model=model_to_use,
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
        """Background health monitoring loop with resilient error handling."""
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Only monitor if we have components to monitor
                if self._component_health.get('ollama_client', False) and self.ollama_client:
                    try:
                        healthy = await asyncio.wait_for(
                            self.ollama_client.health_check(), 
                            timeout=10.0
                        )
                        if healthy:
                            consecutive_failures = 0
                            # If we were in degraded mode and Ollama is back, try to recover
                            if self.state == AppState.DEGRADED and 'ollama_client' in self.metrics.failed_components:
                                self.logger.info("Ollama service recovered - removing from failed components")
                                self.metrics.failed_components.remove('ollama_client')
                                self._component_health['ollama_client'] = True
                                
                                # Check if we can transition back to RUNNING state
                                if not self.metrics.failed_components:
                                    self.state = AppState.RUNNING
                                    self.logger.info("All components recovered - transitioning to RUNNING state")
                        else:
                            consecutive_failures += 1
                            self.logger.warning(f"Ollama health check failed (consecutive failures: {consecutive_failures})")
                            
                            # If too many consecutive failures, mark as degraded
                            if consecutive_failures >= max_consecutive_failures and self.state == AppState.RUNNING:
                                self.logger.error("Too many consecutive Ollama failures - marking as degraded")
                                self.state = AppState.DEGRADED
                                if 'ollama_client' not in self.metrics.failed_components:
                                    self.metrics.failed_components.append('ollama_client')
                                self._component_health['ollama_client'] = False
                    
                    except asyncio.TimeoutError:
                        consecutive_failures += 1
                        self.logger.warning(f"Ollama health check timeout (consecutive: {consecutive_failures})")
                    except Exception as e:
                        consecutive_failures += 1
                        self.logger.warning(f"Ollama health check error (consecutive: {consecutive_failures}): {e}")
                
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
                # If REPL exits, signal shutdown
                if not self.repl.running:
                    self._shutdown_event.set()
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
    
    async def get_health_status(self, startup_mode: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive health status of all subsystems.
        
        Args:
            startup_mode: If True, use more lenient health checks suitable for startup
        
        Returns:
            Dictionary with health status of each component
        """
        status = {}
        
        # Ollama client health
        if self.ollama_client:
            try:
                # Add timeout for health check to prevent hanging
                healthy = await asyncio.wait_for(
                    self.ollama_client.health_check(), 
                    timeout=5.0 if startup_mode else 10.0
                )
                status["ollama"] = {
                    "healthy": healthy,
                    "critical": not startup_mode,  # Less critical during startup
                    "details": "Ollama API connection"
                }
                if not healthy:
                    status["ollama"]["error"] = "Ollama server not responding"
            except asyncio.TimeoutError:
                status["ollama"] = {
                    "healthy": False,
                    "critical": not startup_mode,  # Less critical during startup
                    "error": "Ollama health check timeout"
                }
            except Exception as e:
                status["ollama"] = {
                    "healthy": False,
                    "critical": not startup_mode,  # Less critical during startup
                    "error": str(e)
                }
        else:
            # If client not initialized yet
            status["ollama"] = {
                "healthy": False,
                "critical": False,
                "error": "Ollama client not initialized"
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
                # Use timeout for model listing to prevent hanging
                models = await asyncio.wait_for(
                    self.model_manager.list_models(), 
                    timeout=10.0 if startup_mode else 15.0
                )
                status["models"] = {
                    "healthy": len(models) > 0,
                    "critical": False,
                    "details": f"{len(models)} models available"
                }
            except asyncio.TimeoutError:
                status["models"] = {
                    "healthy": False,
                    "critical": False,
                    "error": "Model listing timed out"
                }
            except Exception as e:
                status["models"] = {
                    "healthy": False,
                    "critical": False,
                    "error": str(e)
                }
        else:
            status["models"] = {
                "healthy": False,
                "critical": False,
                "error": "Model manager not initialized"
            }
        
        # Application health
        # During startup, STARTING and INITIALIZING states are acceptable
        valid_states = [AppState.RUNNING]
        if startup_mode:
            valid_states.extend([AppState.STARTING, AppState.INITIALIZING])
        
        app_healthy = self.state in valid_states
        status["application"] = {
            "healthy": app_healthy,
            "critical": not startup_mode and self.state == AppState.ERROR,  # Only critical if in error state outside startup
            "details": {
                "state": self.state.value,
                "uptime": self.metrics.get_uptime(),
                "requests": self.metrics.total_requests,
                "errors": self.metrics.total_errors,
                "startup_mode": startup_mode,
                "valid_states": [s.value for s in valid_states]
            }
        }
        
        if not app_healthy:
            status["application"]["error"] = f"Application in {self.state.value} state (expected one of {[s.value for s in valid_states]})"
        
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
    
    def is_healthy(self) -> bool:
        """Check if the application is in a healthy state."""
        return self.state in [AppState.RUNNING, AppState.DEGRADED]
    
    def is_functional(self) -> bool:
        """Check if core functionality is available."""
        return self._component_health.get('ollama_client', False)
    
    def get_component_status(self, component: str) -> bool:
        """Get the health status of a specific component."""
        return self._component_health.get(component, False)
    
    def force_component_recovery(self, component: str) -> None:
        """Force a component recovery attempt (restart background task if needed)."""
        self.logger.info(f"Forcing recovery attempt for component: {component}")
        
        if component == 'ollama_client' and component in self.metrics.failed_components:
            # Try to re-establish Ollama connection
            asyncio.create_task(self._attempt_ollama_recovery())
    
    async def _attempt_ollama_recovery(self) -> None:
        """Attempt to recover Ollama connection."""
        try:
            if self.ollama_client:
                self.logger.info("Attempting Ollama connection recovery...")
                healthy = await asyncio.wait_for(
                    self.ollama_client.health_check(), 
                    timeout=self._health_check_timeout
                )
                
                if healthy:
                    self.logger.info("Ollama connection recovered successfully")
                    if 'ollama_client' in self.metrics.failed_components:
                        self.metrics.failed_components.remove('ollama_client')
                    self._component_health['ollama_client'] = True
                    
                    # Update state if all components are healthy
                    if not self.metrics.failed_components and self.state == AppState.DEGRADED:
                        self.state = AppState.RUNNING
                        self.logger.info("Application state upgraded to RUNNING")
                else:
                    self.logger.warning("Ollama connection recovery failed - health check failed")
        except Exception as e:
            self.logger.error(f"Ollama recovery attempt failed: {e}")
    
    def _report_startup_progress(self, message: str, is_error: bool = False) -> None:
        """Report startup progress with consistent formatting."""
        if is_error:
            self.logger.error(f"STARTUP: {message}")
            if self.interactive:
                self.console.print(f"[red]ERROR: {message}[/red]")
        else:
            self.logger.info(f"STARTUP: {message}")
            if self.interactive:
                self.console.print(f"[blue]INFO: {message}[/blue]")
    
    def _report_component_status(self, components_status: Dict[str, bool]) -> None:
        """Report the status of all components during startup."""
        self.logger.info("Component initialization results:")
        
        if self.interactive:
            self.console.print("\n[bold]Component Status:[/bold]")
        
        for component, success in components_status.items():
            status_text = "OK" if success else "FAILED"
            color = "green" if success else "red"
            
            self.logger.info(f"  {component}: {'SUCCESS' if success else 'FAILED'}")
            if self.interactive:
                self.console.print(f"  [{color}]{status_text}[/{color}] {component.replace('_', ' ').title()}")
        
        if self.interactive:
            self.console.print()  # Add blank line
    
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
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        if self.model_manager:
            return await self.model_manager.list_models()
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
                success = await self.model_manager.install_model(model_name, progress_callback)
                if success:
                    self.logger.info(f"Model {model_name} installed successfully")
                    return True
                else:
                    self.logger.error(f"Failed to install model {model_name}")
                    return False
        except Exception as e:
            self.logger.error(f"Failed to install model {model_name}: {e}")
        return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        if self.model_manager:
            try:
                model_info = self.model_manager.get_model_info_sync(model_name)
                if model_info:
                    # Convert ModelInfo object to dictionary
                    return {
                        'name': model_info.name,
                        'tag': model_info.tag,
                        'full_name': model_info.full_name,
                        'size': model_info.size,
                        'size_gb': model_info.size_gb,
                        'digest': model_info.digest,
                        'modified_at': model_info.modified_at.isoformat() if model_info.modified_at else None,
                        'parameter_count': model_info.parameter_count,
                        'quantization': model_info.quantization,
                        'architecture': model_info.architecture,
                        'size_category': model_info.size_category.value if model_info.size_category else None
                    }
                return None
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