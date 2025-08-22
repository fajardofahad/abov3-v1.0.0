"""
ABOV3 Base Plugin Class

Abstract base class for all ABOV3 plugins with comprehensive lifecycle management,
security integration, and configuration support.

This module defines the core plugin interface that all plugins must implement,
providing standardized methods for initialization, execution, and cleanup.

Author: ABOV3 Enterprise Plugin System
Version: 1.0.0
"""

import os
import sys
import json
import logging
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, validator
from ...utils.security import SecurityManager, detect_malicious_patterns


class PluginState(Enum):
    """Plugin lifecycle states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UNLOADING = "unloading"


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


@dataclass
class PluginMetadata:
    """Plugin metadata information."""
    name: str
    version: str
    description: str
    author: str
    license: str = "MIT"
    website: Optional[str] = None
    email: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    min_abov3_version: str = "1.0.0"
    max_abov3_version: Optional[str] = None
    python_min_version: str = "3.8"
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created is None:
            self.created = datetime.utcnow()
        if self.updated is None:
            self.updated = datetime.utcnow()


class PluginConfig(BaseModel):
    """Plugin configuration model with validation."""
    
    enabled: bool = Field(default=True, description="Whether the plugin is enabled")
    auto_load: bool = Field(default=True, description="Auto-load plugin on startup")
    priority: int = Field(default=50, ge=0, le=100, description="Plugin loading priority")
    timeout: int = Field(default=30, ge=1, le=300, description="Plugin operation timeout")
    max_memory: int = Field(default=100, ge=1, description="Max memory usage in MB")
    permissions: List[str] = Field(default_factory=list, description="Required permissions")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Plugin-specific settings")
    
    class Config:
        validate_assignment = True
        extra = "allow"  # Allow plugin-specific configuration


class BasePlugin(ABC):
    """
    Abstract base class for all ABOV3 plugins.
    
    Provides standardized lifecycle management, configuration handling,
    security integration, and hook system support.
    """
    
    def __init__(self, plugin_path: Path, config: Optional[PluginConfig] = None):
        """
        Initialize the plugin.
        
        Args:
            plugin_path: Path to the plugin directory
            config: Plugin configuration
        """
        self.plugin_path = plugin_path
        self.config = config or PluginConfig()
        self.state = PluginState.UNLOADED
        self.logger = logging.getLogger(f"abov3.plugins.{self.get_name()}")
        self.security_manager = SecurityManager()
        self._hooks: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._start_time: Optional[datetime] = None
        self._error_count = 0
        self._last_error: Optional[str] = None
        
        # Load metadata
        self._metadata = self._load_metadata()
        
        # Initialize plugin-specific logger
        self._setup_logging()
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the plugin name."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Return the plugin version."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return the plugin description."""
        pass
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        if self._metadata is None:
            self._metadata = PluginMetadata(
                name=self.get_name(),
                version=self.get_version(),
                description=self.get_description(),
                author="Unknown"
            )
        return self._metadata
    
    def get_state(self) -> PluginState:
        """Return current plugin state."""
        return self.state
    
    def get_config(self) -> PluginConfig:
        """Return plugin configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update plugin configuration."""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.settings[key] = value
    
    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self.config.enabled
    
    def enable(self) -> None:
        """Enable the plugin."""
        self.config.enabled = True
        self.logger.info(f"Plugin {self.get_name()} enabled")
    
    def disable(self) -> None:
        """Disable the plugin."""
        self.config.enabled = False
        self.logger.info(f"Plugin {self.get_name()} disabled")
    
    def load(self) -> bool:
        """
        Load the plugin.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if self.state != PluginState.UNLOADED:
            self.logger.warning(f"Plugin {self.get_name()} is already loaded")
            return False
        
        try:
            self._set_state(PluginState.LOADING)
            
            # Validate plugin security
            if not self._validate_security():
                self._set_state(PluginState.ERROR)
                self._last_error = "Security validation failed"
                return False
            
            # Check dependencies
            if not self._check_dependencies():
                self._set_state(PluginState.ERROR)
                self._last_error = "Dependency check failed"
                return False
            
            # Initialize plugin
            if not self.on_load():
                self._set_state(PluginState.ERROR)
                self._last_error = "Plugin initialization failed"
                return False
            
            self._set_state(PluginState.LOADED)
            self._start_time = datetime.utcnow()
            self.logger.info(f"Plugin {self.get_name()} loaded successfully")
            return True
            
        except Exception as e:
            self._set_state(PluginState.ERROR)
            self._last_error = str(e)
            self._error_count += 1
            self.logger.error(f"Failed to load plugin {self.get_name()}: {e}")
            return False
    
    def unload(self) -> bool:
        """
        Unload the plugin.
        
        Returns:
            True if unloaded successfully, False otherwise
        """
        if self.state == PluginState.UNLOADED:
            return True
        
        try:
            self._set_state(PluginState.UNLOADING)
            
            # Deactivate if active
            if self.state == PluginState.ACTIVE:
                self.deactivate()
            
            # Cleanup plugin
            if not self.on_unload():
                self.logger.warning(f"Plugin {self.get_name()} cleanup failed")
            
            # Clear hooks
            self._hooks.clear()
            
            self._set_state(PluginState.UNLOADED)
            self._start_time = None
            self.logger.info(f"Plugin {self.get_name()} unloaded")
            return True
            
        except Exception as e:
            self._set_state(PluginState.ERROR)
            self._last_error = str(e)
            self._error_count += 1
            self.logger.error(f"Failed to unload plugin {self.get_name()}: {e}")
            return False
    
    def activate(self) -> bool:
        """
        Activate the plugin.
        
        Returns:
            True if activated successfully, False otherwise
        """
        if self.state != PluginState.LOADED:
            self.logger.warning(f"Plugin {self.get_name()} must be loaded before activation")
            return False
        
        if not self.config.enabled:
            self.logger.warning(f"Plugin {self.get_name()} is disabled")
            return False
        
        try:
            if not self.on_activate():
                self._last_error = "Plugin activation failed"
                return False
            
            self._set_state(PluginState.ACTIVE)
            self.logger.info(f"Plugin {self.get_name()} activated")
            return True
            
        except Exception as e:
            self._set_state(PluginState.ERROR)
            self._last_error = str(e)
            self._error_count += 1
            self.logger.error(f"Failed to activate plugin {self.get_name()}: {e}")
            return False
    
    def deactivate(self) -> bool:
        """
        Deactivate the plugin.
        
        Returns:
            True if deactivated successfully, False otherwise
        """
        if self.state != PluginState.ACTIVE:
            return True
        
        try:
            if not self.on_deactivate():
                self.logger.warning(f"Plugin {self.get_name()} deactivation failed")
            
            self._set_state(PluginState.LOADED)
            self.logger.info(f"Plugin {self.get_name()} deactivated")
            return True
            
        except Exception as e:
            self._set_state(PluginState.ERROR)
            self._last_error = str(e)
            self._error_count += 1
            self.logger.error(f"Failed to deactivate plugin {self.get_name()}: {e}")
            return False
    
    def reload(self) -> bool:
        """
        Reload the plugin.
        
        Returns:
            True if reloaded successfully, False otherwise
        """
        was_active = self.state == PluginState.ACTIVE
        
        if not self.unload():
            return False
        
        if not self.load():
            return False
        
        if was_active:
            return self.activate()
        
        return True
    
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a callback for a specific hook."""
        with self._lock:
            if hook_name not in self._hooks:
                self._hooks[hook_name] = []
            self._hooks[hook_name].append(callback)
        
        self.logger.debug(f"Registered hook '{hook_name}' for plugin {self.get_name()}")
    
    def unregister_hook(self, hook_name: str, callback: Callable) -> None:
        """Unregister a callback for a specific hook."""
        with self._lock:
            if hook_name in self._hooks and callback in self._hooks[hook_name]:
                self._hooks[hook_name].remove(callback)
                if not self._hooks[hook_name]:
                    del self._hooks[hook_name]
        
        self.logger.debug(f"Unregistered hook '{hook_name}' for plugin {self.get_name()}")
    
    def execute_hooks(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute all hooks registered for the given hook name."""
        results = []
        
        with self._lock:
            callbacks = self._hooks.get(hook_name, []).copy()
        
        for callback in callbacks:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Hook '{hook_name}' failed in plugin {self.get_name()}: {e}")
        
        return results
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """Get runtime information about the plugin."""
        uptime = None
        if self._start_time:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()
        
        return {
            "name": self.get_name(),
            "version": self.get_version(),
            "state": self.state.value,
            "enabled": self.config.enabled,
            "uptime": uptime,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "hooks": list(self._hooks.keys()),
            "memory_limit": self.config.max_memory,
            "timeout": self.config.timeout,
        }
    
    def validate_input(self, data: Any) -> bool:
        """
        Validate input data for security.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if safe, False otherwise
        """
        if isinstance(data, str):
            patterns = detect_malicious_patterns(data)
            if patterns:
                self.logger.warning(f"Malicious patterns detected in input: {patterns}")
                return False
        
        return True
    
    # Abstract lifecycle methods
    @abstractmethod
    def on_load(self) -> bool:
        """
        Called when the plugin is loaded.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def on_unload(self) -> bool:
        """
        Called when the plugin is unloaded.
        
        Returns:
            True if successful, False otherwise
        """
        return True
    
    def on_activate(self) -> bool:
        """
        Called when the plugin is activated.
        
        Returns:
            True if successful, False otherwise
        """
        return True
    
    def on_deactivate(self) -> bool:
        """
        Called when the plugin is deactivated.
        
        Returns:
            True if successful, False otherwise
        """
        return True
    
    def on_config_changed(self, old_config: PluginConfig, new_config: PluginConfig) -> None:
        """
        Called when plugin configuration changes.
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
        """
        pass
    
    # Private methods
    def _load_metadata(self) -> Optional[PluginMetadata]:
        """Load plugin metadata from plugin.json file."""
        metadata_file = self.plugin_path / "plugin.json"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return PluginMetadata(
                name=data.get('name', self.get_name()),
                version=data.get('version', self.get_version()),
                description=data.get('description', self.get_description()),
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
            self.logger.warning(f"Failed to load metadata for plugin {self.get_name()}: {e}")
            return None
    
    def _validate_security(self) -> bool:
        """Validate plugin security."""
        try:
            # Check plugin files for malicious content
            for py_file in self.plugin_path.glob("**/*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    is_safe, issues = self.security_manager.is_content_safe(content)
                    if not is_safe:
                        self.logger.error(f"Security issues in {py_file}: {issues}")
                        return False
                        
                except Exception as e:
                    self.logger.warning(f"Failed to check security of {py_file}: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Security validation failed: {e}")
            return False
    
    def _check_dependencies(self) -> bool:
        """Check if plugin dependencies are satisfied."""
        metadata = self.get_metadata()
        
        # Check Python version
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if python_version < metadata.python_min_version:
            self.logger.error(f"Python {metadata.python_min_version}+ required, found {python_version}")
            return False
        
        # Check plugin dependencies (would need plugin manager reference)
        # For now, just log dependencies
        if metadata.dependencies:
            self.logger.info(f"Plugin {self.get_name()} requires: {metadata.dependencies}")
        
        return True
    
    def _set_state(self, new_state: PluginState) -> None:
        """Set plugin state and log the change."""
        old_state = self.state
        self.state = new_state
        
        if old_state != new_state:
            self.logger.debug(f"Plugin {self.get_name()} state changed: {old_state.value} -> {new_state.value}")
    
    def _setup_logging(self) -> None:
        """Setup plugin-specific logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)