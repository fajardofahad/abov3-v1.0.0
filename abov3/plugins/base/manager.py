"""
ABOV3 Plugin Manager

Comprehensive plugin management system with discovery, loading, dependency resolution,
hot reloading, and secure execution capabilities.

Features:
    - Plugin discovery from multiple directories
    - Dependency resolution and loading order
    - Hot reloading and lifecycle management
    - Hook system for extensibility
    - Security validation and sandboxing
    - Configuration management
    - Error handling and recovery

Author: ABOV3 Enterprise Plugin System
Version: 1.0.0
"""

import os
import sys
import json
import importlib
import importlib.util
import threading
from typing import Dict, List, Optional, Set, Any, Type, Callable
from pathlib import Path
from collections import defaultdict, deque
import logging

from .plugin import BasePlugin, PluginState, PluginConfig, PluginMetadata
from .registry import PluginRegistry
from ...utils.security import SecurityManager
from ...core.config import get_config


class PluginLoadError(Exception):
    """Exception raised when plugin loading fails."""
    pass


class PluginDependencyError(Exception):
    """Exception raised when plugin dependencies cannot be resolved."""
    pass


class PluginManager:
    """
    Central plugin management system for ABOV3.
    
    Handles plugin discovery, loading, lifecycle management, dependency resolution,
    and provides a hook system for extending functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger('abov3.plugins.manager')
        self.registry = PluginRegistry()
        self.security_manager = SecurityManager()
        
        # Plugin storage
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugin_classes: Dict[str, Type[BasePlugin]] = {}
        self._plugin_modules: Dict[str, Any] = {}
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # Hook system
        self._hooks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Plugin directories
        self._plugin_directories: List[Path] = []
        
        # State tracking
        self._initialized = False
        self._loading_order: List[str] = []
        
        # Setup logging
        self._setup_logging()
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the plugin manager with configuration.
        
        Args:
            config: Configuration dictionary
        """
        if self._initialized:
            self.logger.warning("Plugin manager already initialized")
            return
        
        self.config.update(config)
        
        # Setup plugin directories
        self._setup_plugin_directories()
        
        # Discover and load plugins
        self.discover_plugins()
        
        # Load enabled plugins if auto-load is enabled
        if self.config.get('auto_load', True):
            self._load_enabled_plugins()
        
        self._initialized = True
        self.logger.info("Plugin manager initialized successfully")
    
    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in configured directories.
        
        Returns:
            List of discovered plugin names
        """
        discovered = []
        
        for directory in self._plugin_directories:
            if not directory.exists():
                self.logger.warning(f"Plugin directory does not exist: {directory}")
                continue
            
            discovered.extend(self._discover_in_directory(directory))
        
        self.logger.info(f"Discovered {len(discovered)} plugins: {discovered}")
        return discovered
    
    def load_plugin(self, plugin_name: str, force: bool = False) -> Optional[BasePlugin]:
        """
        Load a specific plugin by name.
        
        Args:
            plugin_name: Name of the plugin to load
            force: Force loading even if already loaded
            
        Returns:
            Loaded plugin instance or None if failed
        """
        with self._lock:
            if plugin_name in self._plugins and not force:
                self.logger.debug(f"Plugin {plugin_name} already loaded")
                return self._plugins[plugin_name]
            
            # Check if plugin is discovered
            if not self.registry.is_plugin_registered(plugin_name):
                self.logger.error(f"Plugin {plugin_name} not found in registry")
                raise PluginLoadError(f"Plugin {plugin_name} not found")
            
            try:
                # Load dependencies first
                self._load_dependencies(plugin_name)
                
                # Load the plugin
                plugin = self._load_single_plugin(plugin_name)
                if plugin:
                    self._plugins[plugin_name] = plugin
                    self.logger.info(f"Plugin {plugin_name} loaded successfully")
                    
                    # Execute post-load hooks
                    self.execute_hook('plugin_loaded', plugin)
                    
                return plugin
                
            except Exception as e:
                self.logger.error(f"Failed to load plugin {plugin_name}: {e}")
                raise PluginLoadError(f"Failed to load plugin {plugin_name}: {e}")
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a specific plugin.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if unloaded successfully, False otherwise
        """
        with self._lock:
            if plugin_name not in self._plugins:
                self.logger.debug(f"Plugin {plugin_name} not loaded")
                return True
            
            plugin = self._plugins[plugin_name]
            
            # Check for dependent plugins
            dependents = self._reverse_dependencies.get(plugin_name, set())
            if dependents:
                loaded_dependents = [dep for dep in dependents if dep in self._plugins]
                if loaded_dependents:
                    self.logger.error(f"Cannot unload {plugin_name}: dependent plugins loaded: {loaded_dependents}")
                    return False
            
            try:
                # Execute pre-unload hooks
                self.execute_hook('plugin_unloading', plugin)
                
                # Unload the plugin
                success = plugin.unload()
                
                if success:
                    del self._plugins[plugin_name]
                    
                    # Clear from loading order
                    if plugin_name in self._loading_order:
                        self._loading_order.remove(plugin_name)
                    
                    # Execute post-unload hooks
                    self.execute_hook('plugin_unloaded', plugin_name)
                    
                    self.logger.info(f"Plugin {plugin_name} unloaded successfully")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Failed to unload plugin {plugin_name}: {e}")
                return False
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a specific plugin.
        
        Args:
            plugin_name: Name of the plugin to reload
            
        Returns:
            True if reloaded successfully, False otherwise
        """
        with self._lock:
            if plugin_name not in self._plugins:
                return self.load_plugin(plugin_name) is not None
            
            plugin = self._plugins[plugin_name]
            was_active = plugin.get_state() == PluginState.ACTIVE
            
            # Unload the plugin
            if not self.unload_plugin(plugin_name):
                return False
            
            # Clear cached module
            if plugin_name in self._plugin_modules:
                module = self._plugin_modules[plugin_name]
                if hasattr(module, '__file__') and module.__file__:
                    # Remove from sys.modules to force reload
                    modules_to_remove = [name for name in sys.modules.keys() 
                                       if name.startswith(f"{module.__name__}.")]
                    for mod_name in modules_to_remove:
                        del sys.modules[mod_name]
                    
                    if module.__name__ in sys.modules:
                        del sys.modules[module.__name__]
                
                del self._plugin_modules[plugin_name]
            
            # Reload the plugin
            new_plugin = self.load_plugin(plugin_name)
            
            if new_plugin and was_active:
                new_plugin.activate()
            
            return new_plugin is not None
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a loaded plugin by name."""
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all available plugins."""
        return self.registry.list_plugins()
    
    def list_loaded_plugins(self) -> List[str]:
        """List all currently loaded plugins."""
        return list(self._plugins.keys())
    
    def list_active_plugins(self) -> List[str]:
        """List all currently active plugins."""
        return [name for name, plugin in self._plugins.items() 
                if plugin.get_state() == PluginState.ACTIVE]
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get metadata information for a plugin."""
        return self.registry.get_plugin_metadata(plugin_name)
    
    def activate_plugin(self, plugin_name: str) -> bool:
        """Activate a loaded plugin."""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            return plugin.activate()
        return False
    
    def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate an active plugin."""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            return plugin.deactivate()
        return False
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            plugin.enable()
            return True
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            plugin.disable()
            return True
        return False
    
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a callback for a specific hook."""
        with self._lock:
            self._hooks[hook_name].append(callback)
        self.logger.debug(f"Registered hook '{hook_name}'")
    
    def unregister_hook(self, hook_name: str, callback: Callable) -> None:
        """Unregister a callback for a specific hook."""
        with self._lock:
            if callback in self._hooks[hook_name]:
                self._hooks[hook_name].remove(callback)
        self.logger.debug(f"Unregistered hook '{hook_name}'")
    
    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute all hooks registered for the given hook name."""
        results = []
        
        with self._lock:
            callbacks = self._hooks[hook_name].copy()
        
        for callback in callbacks:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Hook '{hook_name}' callback failed: {e}")
        
        # Also execute plugin hooks
        for plugin in self._plugins.values():
            if plugin.get_state() == PluginState.ACTIVE:
                try:
                    plugin_results = plugin.execute_hooks(hook_name, *args, **kwargs)
                    results.extend(plugin_results)
                except Exception as e:
                    self.logger.error(f"Plugin hook '{hook_name}' failed in {plugin.get_name()}: {e}")
        
        return results
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """Get runtime information about the plugin manager."""
        with self._lock:
            plugin_info = {}
            for name, plugin in self._plugins.items():
                plugin_info[name] = plugin.get_runtime_info()
        
        return {
            "initialized": self._initialized,
            "total_plugins": len(self.registry.list_plugins()),
            "loaded_plugins": len(self._plugins),
            "active_plugins": len(self.list_active_plugins()),
            "plugin_directories": [str(d) for d in self._plugin_directories],
            "loading_order": self._loading_order.copy(),
            "hooks": {hook: len(callbacks) for hook, callbacks in self._hooks.items()},
            "plugins": plugin_info
        }
    
    def validate_plugin_security(self, plugin_path: Path) -> bool:
        """Validate plugin security before loading."""
        try:
            # Check if path is safe
            if not plugin_path.exists() or not plugin_path.is_dir():
                return False
            
            # Check for required files
            if not (plugin_path / "__init__.py").exists():
                return False
            
            # Validate Python files for malicious content
            for py_file in plugin_path.glob("**/*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    is_safe, issues = self.security_manager.is_content_safe(content)
                    if not is_safe:
                        self.logger.error(f"Security issues in {py_file}: {issues}")
                        return False
                        
                except Exception as e:
                    self.logger.warning(f"Failed to validate {py_file}: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Security validation failed for {plugin_path}: {e}")
            return False
    
    # Private methods
    def _setup_plugin_directories(self) -> None:
        """Setup plugin search directories."""
        # Default directories
        default_dirs = [
            Path(__file__).parent.parent / "builtin",  # Built-in plugins
            get_config().get_data_dir() / "plugins",   # User plugins
        ]
        
        # Add configured directories
        config_dirs = self.config.get('directories', [])
        for dir_path in config_dirs:
            default_dirs.append(Path(dir_path))
        
        # Ensure directories exist and are valid
        for directory in default_dirs:
            if directory.exists() or directory.parent.exists():
                directory.mkdir(parents=True, exist_ok=True)
                self._plugin_directories.append(directory)
                self.logger.debug(f"Added plugin directory: {directory}")
    
    def _discover_in_directory(self, directory: Path) -> List[str]:
        """Discover plugins in a specific directory."""
        discovered = []
        
        for item in directory.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if self._is_valid_plugin_directory(item):
                    plugin_name = item.name
                    
                    # Validate security
                    if not self.validate_plugin_security(item):
                        self.logger.warning(f"Plugin {plugin_name} failed security validation")
                        continue
                    
                    # Register in registry
                    metadata = self._extract_plugin_metadata(item)
                    self.registry.register_plugin(plugin_name, item, metadata)
                    
                    discovered.append(plugin_name)
                    self.logger.debug(f"Discovered plugin: {plugin_name}")
        
        return discovered
    
    def _is_valid_plugin_directory(self, path: Path) -> bool:
        """Check if directory contains a valid plugin."""
        return (path / "__init__.py").exists()
    
    def _extract_plugin_metadata(self, plugin_path: Path) -> Optional[PluginMetadata]:
        """Extract metadata from plugin directory."""
        metadata_file = plugin_path / "plugin.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                return PluginMetadata(
                    name=data.get('name', plugin_path.name),
                    version=data.get('version', '1.0.0'),
                    description=data.get('description', ''),
                    author=data.get('author', 'Unknown'),
                    license=data.get('license', 'MIT'),
                    website=data.get('website'),
                    email=data.get('email'),
                    tags=data.get('tags', []),
                    dependencies=data.get('dependencies', []),
                    min_abov3_version=data.get('min_abov3_version', '1.0.0'),
                    max_abov3_version=data.get('max_abov3_version'),
                    python_min_version=data.get('python_min_version', '3.8'),
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to parse metadata for {plugin_path.name}: {e}")
        
        return None
    
    def _load_enabled_plugins(self) -> None:
        """Load all enabled plugins."""
        enabled_plugins = self.config.get('enabled', [])
        
        if not enabled_plugins:
            # Load all discovered plugins if none specifically enabled
            enabled_plugins = self.registry.list_plugins()
        
        # Sort by loading order based on dependencies
        loading_order = self._resolve_loading_order(enabled_plugins)
        
        for plugin_name in loading_order:
            try:
                plugin = self.load_plugin(plugin_name)
                if plugin and plugin.config.enabled:
                    plugin.activate()
            except Exception as e:
                self.logger.error(f"Failed to load enabled plugin {plugin_name}: {e}")
    
    def _load_dependencies(self, plugin_name: str) -> None:
        """Load plugin dependencies."""
        metadata = self.registry.get_plugin_metadata(plugin_name)
        if not metadata or not metadata.dependencies:
            return
        
        for dep_name in metadata.dependencies:
            if dep_name not in self._plugins:
                self.logger.info(f"Loading dependency {dep_name} for {plugin_name}")
                self.load_plugin(dep_name)
            
            # Track dependency relationship
            self._dependency_graph[plugin_name].add(dep_name)
            self._reverse_dependencies[dep_name].add(plugin_name)
    
    def _load_single_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Load a single plugin instance."""
        plugin_info = self.registry.get_plugin_info(plugin_name)
        if not plugin_info:
            raise PluginLoadError(f"Plugin {plugin_name} not found in registry")
        
        plugin_path = plugin_info['path']
        
        try:
            # Import the plugin module
            spec = importlib.util.spec_from_file_location(
                plugin_name,
                plugin_path / "__init__.py"
            )
            
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Cannot create spec for plugin {plugin_name}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Cache the module
            self._plugin_modules[plugin_name] = module
            
            # Find plugin class
            plugin_class = self._find_plugin_class(module, plugin_name)
            if not plugin_class:
                raise PluginLoadError(f"No plugin class found in {plugin_name}")
            
            # Cache the class
            self._plugin_classes[plugin_name] = plugin_class
            
            # Create plugin configuration
            config = self._create_plugin_config(plugin_name)
            
            # Create plugin instance
            plugin = plugin_class(plugin_path, config)
            
            # Load the plugin
            if not plugin.load():
                raise PluginLoadError(f"Plugin {plugin_name} load() method failed")
            
            # Track loading order
            if plugin_name not in self._loading_order:
                self._loading_order.append(plugin_name)
            
            return plugin
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_name}: {e}")
            raise PluginLoadError(f"Failed to load plugin {plugin_name}: {e}")
    
    def _find_plugin_class(self, module: Any, plugin_name: str) -> Optional[Type[BasePlugin]]:
        """Find the plugin class in the module."""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, BasePlugin) and 
                attr is not BasePlugin):
                return attr
        
        # Look for specifically named class
        expected_names = [
            f"{plugin_name.title()}Plugin",
            f"{plugin_name.replace('_', '')}Plugin",
            "Plugin"
        ]
        
        for name in expected_names:
            if hasattr(module, name):
                cls = getattr(module, name)
                if isinstance(cls, type) and issubclass(cls, BasePlugin):
                    return cls
        
        return None
    
    def _create_plugin_config(self, plugin_name: str) -> PluginConfig:
        """Create configuration for a plugin."""
        # Get global plugin config
        global_config = get_config().plugins
        
        # Plugin-specific config
        plugin_config = self.config.get('plugin_configs', {}).get(plugin_name, {})
        
        # Merge configurations
        config_data = {
            'enabled': plugin_name in global_config.enabled if global_config.enabled else True,
            'auto_load': global_config.auto_load,
            **plugin_config
        }
        
        return PluginConfig(**config_data)
    
    def _resolve_loading_order(self, plugin_names: List[str]) -> List[str]:
        """Resolve plugin loading order based on dependencies."""
        # Build dependency graph
        graph = {}
        in_degree = {}
        
        for plugin_name in plugin_names:
            graph[plugin_name] = []
            in_degree[plugin_name] = 0
        
        for plugin_name in plugin_names:
            metadata = self.registry.get_plugin_metadata(plugin_name)
            if metadata and metadata.dependencies:
                for dep in metadata.dependencies:
                    if dep in graph:
                        graph[dep].append(plugin_name)
                        in_degree[plugin_name] += 1
        
        # Topological sort
        queue = deque([name for name in plugin_names if in_degree[name] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for circular dependencies
        if len(result) != len(plugin_names):
            remaining = [name for name in plugin_names if name not in result]
            raise PluginDependencyError(f"Circular dependency detected: {remaining}")
        
        return result
    
    def _setup_logging(self) -> None:
        """Setup plugin manager logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)