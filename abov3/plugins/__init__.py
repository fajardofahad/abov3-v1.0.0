"""
ABOV3 4 Ollama Plugin System

A comprehensive, secure plugin architecture for extending ABOV3 functionality.
Provides plugin discovery, loading, dependency management, and secure execution.

Features:
    - Abstract base plugin class with lifecycle management
    - Plugin discovery and automatic loading
    - Dependency resolution and management
    - Secure plugin execution with sandboxing
    - Plugin registry with metadata and versioning
    - Hook system for extending core functionality
    - Configuration management for plugins
    - Hot reloading capability
    - Built-in plugins for common tasks
    - Integration with security framework

Author: ABOV3 Enterprise Plugin System
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Type
import logging

from .base.plugin import BasePlugin, PluginError, PluginMetadata
from .base.manager import PluginManager
from .base.registry import PluginRegistry

# Configure plugin logging
logger = logging.getLogger('abov3.plugins')
logger.setLevel(logging.INFO)

# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def initialize_plugins(config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize the plugin system with optional configuration."""
    manager = get_plugin_manager()
    manager.initialize(config or {})


def load_plugin(plugin_name: str) -> Optional[BasePlugin]:
    """Load a specific plugin by name."""
    manager = get_plugin_manager()
    return manager.load_plugin(plugin_name)


def get_plugin(plugin_name: str) -> Optional[BasePlugin]:
    """Get a loaded plugin by name."""
    manager = get_plugin_manager()
    return manager.get_plugin(plugin_name)


def list_plugins() -> List[str]:
    """List all available plugins."""
    manager = get_plugin_manager()
    return manager.list_plugins()


def list_loaded_plugins() -> List[str]:
    """List all currently loaded plugins."""
    manager = get_plugin_manager()
    return manager.list_loaded_plugins()


def reload_plugin(plugin_name: str) -> bool:
    """Reload a specific plugin."""
    manager = get_plugin_manager()
    return manager.reload_plugin(plugin_name)


def unload_plugin(plugin_name: str) -> bool:
    """Unload a specific plugin."""
    manager = get_plugin_manager()
    return manager.unload_plugin(plugin_name)


def discover_plugins() -> None:
    """Discover available plugins in configured directories."""
    manager = get_plugin_manager()
    manager.discover_plugins()


def get_plugin_info(plugin_name: str) -> Optional[PluginMetadata]:
    """Get metadata information for a plugin."""
    manager = get_plugin_manager()
    return manager.get_plugin_info(plugin_name)


def execute_hook(hook_name: str, *args, **kwargs) -> List[Any]:
    """Execute all hooks registered for the given hook name."""
    manager = get_plugin_manager()
    return manager.execute_hook(hook_name, *args, **kwargs)


def register_hook(hook_name: str, callback) -> None:
    """Register a callback for a specific hook."""
    manager = get_plugin_manager()
    manager.register_hook(hook_name, callback)


def unregister_hook(hook_name: str, callback) -> None:
    """Unregister a callback for a specific hook."""
    manager = get_plugin_manager()
    manager.unregister_hook(hook_name, callback)


# Export main classes and functions
__all__ = [
    'BasePlugin',
    'PluginError', 
    'PluginMetadata',
    'PluginManager',
    'PluginRegistry',
    'get_plugin_manager',
    'initialize_plugins',
    'load_plugin',
    'get_plugin',
    'list_plugins',
    'list_loaded_plugins',
    'reload_plugin',
    'unload_plugin',
    'discover_plugins',
    'get_plugin_info',
    'execute_hook',
    'register_hook',
    'unregister_hook',
]