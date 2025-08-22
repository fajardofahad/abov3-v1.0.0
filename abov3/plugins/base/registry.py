"""
ABOV3 Plugin Registry

Comprehensive plugin registry system for managing plugin metadata, versions,
dependencies, and discovery information.

Features:
    - Plugin metadata storage and retrieval
    - Version management and compatibility checking
    - Dependency tracking and resolution
    - Plugin discovery and indexing
    - Persistent storage of plugin information
    - Search and filtering capabilities
    - Plugin validation and verification

Author: ABOV3 Enterprise Plugin System
Version: 1.0.0
"""

import json
import sqlite3
import threading
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import logging

from .plugin import PluginMetadata
from ...core.config import get_config


class RegistryError(Exception):
    """Exception raised for registry-related errors."""
    pass


class PluginRegistry:
    """
    Plugin registry for managing plugin metadata and discovery information.
    
    Provides persistent storage of plugin information, dependency tracking,
    version management, and search capabilities.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the plugin registry.
        
        Args:
            db_path: Optional path to the database file
        """
        self.logger = logging.getLogger('abov3.plugins.registry')
        self._lock = threading.RLock()
        
        # Database setup
        if db_path is None:
            db_path = get_config().get_data_dir() / "plugins.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._plugins: Dict[str, Dict[str, Any]] = {}
        self._metadata_cache: Dict[str, PluginMetadata] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._reverse_dependencies: Dict[str, Set[str]] = {}
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_from_database()
        
        self.logger.info(f"Plugin registry initialized with {len(self._plugins)} plugins")
    
    def register_plugin(self, name: str, path: Path, metadata: Optional[PluginMetadata] = None) -> bool:
        """
        Register a plugin in the registry.
        
        Args:
            name: Plugin name
            path: Path to plugin directory
            metadata: Optional plugin metadata
            
        Returns:
            True if registered successfully, False otherwise
        """
        with self._lock:
            try:
                # Validate plugin
                if not self._validate_plugin_path(path):
                    self.logger.error(f"Invalid plugin path: {path}")
                    return False
                
                # Extract or use provided metadata
                if metadata is None:
                    metadata = self._extract_metadata_from_path(path, name)
                
                # Create plugin info
                plugin_info = {
                    'name': name,
                    'path': str(path),
                    'registered_at': datetime.utcnow().isoformat(),
                    'last_updated': datetime.utcnow().isoformat(),
                    'verified': self._verify_plugin(path),
                    'enabled': True,
                    'metadata': asdict(metadata) if metadata else {}
                }
                
                # Update in-memory cache
                self._plugins[name] = plugin_info
                if metadata:
                    self._metadata_cache[name] = metadata
                    self._update_dependency_graph(name, metadata.dependencies)
                
                # Save to database
                self._save_plugin_to_database(name, plugin_info)
                
                self.logger.info(f"Registered plugin: {name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to register plugin {name}: {e}")
                return False
    
    def unregister_plugin(self, name: str) -> bool:
        """
        Unregister a plugin from the registry.
        
        Args:
            name: Plugin name
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        with self._lock:
            try:
                if name not in self._plugins:
                    self.logger.warning(f"Plugin {name} not registered")
                    return True
                
                # Remove from dependency graph
                self._remove_from_dependency_graph(name)
                
                # Remove from caches
                del self._plugins[name]
                if name in self._metadata_cache:
                    del self._metadata_cache[name]
                
                # Remove from database
                self._remove_plugin_from_database(name)
                
                self.logger.info(f"Unregistered plugin: {name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to unregister plugin {name}: {e}")
                return False
    
    def is_plugin_registered(self, name: str) -> bool:
        """Check if a plugin is registered."""
        return name in self._plugins
    
    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get plugin information."""
        return self._plugins.get(name)
    
    def get_plugin_metadata(self, name: str) -> Optional[PluginMetadata]:
        """Get plugin metadata."""
        return self._metadata_cache.get(name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())
    
    def list_enabled_plugins(self) -> List[str]:
        """List enabled plugin names."""
        return [name for name, info in self._plugins.items() if info.get('enabled', True)]
    
    def list_verified_plugins(self) -> List[str]:
        """List verified plugin names."""
        return [name for name, info in self._plugins.items() if info.get('verified', False)]
    
    def search_plugins(self, query: str, tags: Optional[List[str]] = None) -> List[str]:
        """
        Search plugins by name, description, or tags.
        
        Args:
            query: Search query string
            tags: Optional list of tags to filter by
            
        Returns:
            List of matching plugin names
        """
        results = []
        query_lower = query.lower()
        
        for name, plugin_info in self._plugins.items():
            metadata = self._metadata_cache.get(name)
            
            # Search in name and description
            if (query_lower in name.lower() or 
                (metadata and query_lower in metadata.description.lower())):
                
                # Filter by tags if specified
                if tags and metadata:
                    if not any(tag in metadata.tags for tag in tags):
                        continue
                
                results.append(name)
        
        return results
    
    def get_plugins_by_author(self, author: str) -> List[str]:
        """Get plugins by a specific author."""
        results = []
        for name, metadata in self._metadata_cache.items():
            if metadata.author.lower() == author.lower():
                results.append(name)
        return results
    
    def get_plugins_with_tag(self, tag: str) -> List[str]:
        """Get plugins with a specific tag."""
        results = []
        for name, metadata in self._metadata_cache.items():
            if tag in metadata.tags:
                results.append(name)
        return results
    
    def get_plugin_dependencies(self, name: str) -> List[str]:
        """Get direct dependencies of a plugin."""
        metadata = self._metadata_cache.get(name)
        return metadata.dependencies if metadata else []
    
    def get_plugin_dependents(self, name: str) -> List[str]:
        """Get plugins that depend on this plugin."""
        return list(self._reverse_dependencies.get(name, set()))
    
    def check_dependency_chain(self, name: str) -> List[str]:
        """
        Get the complete dependency chain for a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            List of dependencies in loading order
        """
        visited = set()
        result = []
        
        def visit(plugin_name: str):
            if plugin_name in visited:
                return
            
            visited.add(plugin_name)
            dependencies = self.get_plugin_dependencies(plugin_name)
            
            for dep in dependencies:
                if dep in self._plugins:
                    visit(dep)
            
            result.append(plugin_name)
        
        visit(name)
        return result[:-1]  # Exclude the plugin itself
    
    def validate_dependencies(self, name: str) -> Tuple[bool, List[str]]:
        """
        Validate that all dependencies for a plugin are available.
        
        Args:
            name: Plugin name
            
        Returns:
            Tuple of (is_valid, missing_dependencies)
        """
        dependencies = self.get_plugin_dependencies(name)
        missing = [dep for dep in dependencies if dep not in self._plugins]
        return len(missing) == 0, missing
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detect circular dependencies in the plugin graph.
        
        Returns:
            List of circular dependency chains
        """
        visited = set()
        path = []
        cycles = []
        
        def dfs(node: str):
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            dependencies = self.get_plugin_dependencies(node)
            for dep in dependencies:
                if dep in self._plugins:
                    dfs(dep)
            
            path.pop()
        
        for plugin_name in self._plugins:
            if plugin_name not in visited:
                dfs(plugin_name)
        
        return cycles
    
    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin in the registry."""
        with self._lock:
            if name in self._plugins:
                self._plugins[name]['enabled'] = True
                self._update_plugin_in_database(name, {'enabled': True})
                return True
        return False
    
    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin in the registry."""
        with self._lock:
            if name in self._plugins:
                self._plugins[name]['enabled'] = False
                self._update_plugin_in_database(name, {'enabled': False})
                return True
        return False
    
    def update_plugin_metadata(self, name: str, metadata: PluginMetadata) -> bool:
        """Update plugin metadata."""
        with self._lock:
            if name not in self._plugins:
                return False
            
            try:
                # Update cache
                self._metadata_cache[name] = metadata
                self._plugins[name]['metadata'] = asdict(metadata)
                self._plugins[name]['last_updated'] = datetime.utcnow().isoformat()
                
                # Update dependency graph
                self._update_dependency_graph(name, metadata.dependencies)
                
                # Save to database
                self._update_plugin_in_database(name, {
                    'metadata': asdict(metadata),
                    'last_updated': self._plugins[name]['last_updated']
                })
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to update metadata for {name}: {e}")
                return False
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            total_plugins = len(self._plugins)
            enabled_plugins = len(self.list_enabled_plugins())
            verified_plugins = len(self.list_verified_plugins())
            
            # Count plugins by author
            authors = {}
            for metadata in self._metadata_cache.values():
                authors[metadata.author] = authors.get(metadata.author, 0) + 1
            
            # Count plugins by tag
            tags = {}
            for metadata in self._metadata_cache.values():
                for tag in metadata.tags:
                    tags[tag] = tags.get(tag, 0) + 1
            
            return {
                'total_plugins': total_plugins,
                'enabled_plugins': enabled_plugins,
                'verified_plugins': verified_plugins,
                'disabled_plugins': total_plugins - enabled_plugins,
                'authors': authors,
                'tags': tags,
                'dependency_count': sum(len(deps) for deps in self._dependency_graph.values()),
                'circular_dependencies': len(self.detect_circular_dependencies())
            }
    
    def export_registry(self, export_path: Path) -> bool:
        """Export registry data to a JSON file."""
        try:
            with self._lock:
                data = {
                    'plugins': self._plugins,
                    'exported_at': datetime.utcnow().isoformat(),
                    'version': '1.0.0'
                }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Registry exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export registry: {e}")
            return False
    
    def import_registry(self, import_path: Path) -> bool:
        """Import registry data from a JSON file."""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            plugins_data = data.get('plugins', {})
            
            with self._lock:
                for name, plugin_info in plugins_data.items():
                    # Validate plugin path still exists
                    plugin_path = Path(plugin_info['path'])
                    if not plugin_path.exists():
                        self.logger.warning(f"Plugin path no longer exists: {plugin_path}")
                        continue
                    
                    # Register the plugin
                    metadata_dict = plugin_info.get('metadata', {})
                    if metadata_dict:
                        metadata = PluginMetadata(**metadata_dict)
                    else:
                        metadata = None
                    
                    self.register_plugin(name, plugin_path, metadata)
            
            self.logger.info(f"Registry imported from {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import registry: {e}")
            return False
    
    # Private methods
    def _init_database(self) -> None:
        """Initialize the SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS plugins (
                        name TEXT PRIMARY KEY,
                        path TEXT NOT NULL,
                        registered_at TEXT NOT NULL,
                        last_updated TEXT NOT NULL,
                        verified BOOLEAN NOT NULL DEFAULT 0,
                        enabled BOOLEAN NOT NULL DEFAULT 1,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_plugins_enabled 
                    ON plugins(enabled)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_plugins_verified 
                    ON plugins(verified)
                ''')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise RegistryError(f"Database initialization failed: {e}")
    
    def _load_from_database(self) -> None:
        """Load existing plugins from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('SELECT * FROM plugins')
                
                for row in cursor:
                    name = row['name']
                    plugin_info = {
                        'name': name,
                        'path': row['path'],
                        'registered_at': row['registered_at'],
                        'last_updated': row['last_updated'],
                        'verified': bool(row['verified']),
                        'enabled': bool(row['enabled']),
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                    }
                    
                    self._plugins[name] = plugin_info
                    
                    # Load metadata if available
                    if plugin_info['metadata']:
                        try:
                            metadata = PluginMetadata(**plugin_info['metadata'])
                            self._metadata_cache[name] = metadata
                            self._update_dependency_graph(name, metadata.dependencies)
                        except Exception as e:
                            self.logger.warning(f"Failed to load metadata for {name}: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to load from database: {e}")
    
    def _save_plugin_to_database(self, name: str, plugin_info: Dict[str, Any]) -> None:
        """Save plugin information to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO plugins 
                    (name, path, registered_at, last_updated, verified, enabled, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    name,
                    plugin_info['path'],
                    plugin_info['registered_at'],
                    plugin_info['last_updated'],
                    plugin_info['verified'],
                    plugin_info['enabled'],
                    json.dumps(plugin_info['metadata']) if plugin_info['metadata'] else None
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save plugin {name} to database: {e}")
    
    def _update_plugin_in_database(self, name: str, updates: Dict[str, Any]) -> None:
        """Update plugin information in database."""
        try:
            # Build SET clause
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                if key == 'metadata':
                    set_clauses.append('metadata = ?')
                    values.append(json.dumps(value) if value else None)
                else:
                    set_clauses.append(f'{key} = ?')
                    values.append(value)
            
            values.append(name)  # For WHERE clause
            
            with sqlite3.connect(self.db_path) as conn:
                query = f"UPDATE plugins SET {', '.join(set_clauses)} WHERE name = ?"
                conn.execute(query, values)
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update plugin {name} in database: {e}")
    
    def _remove_plugin_from_database(self, name: str) -> None:
        """Remove plugin from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM plugins WHERE name = ?', (name,))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to remove plugin {name} from database: {e}")
    
    def _validate_plugin_path(self, path: Path) -> bool:
        """Validate that plugin path is valid."""
        return path.exists() and path.is_dir() and (path / "__init__.py").exists()
    
    def _extract_metadata_from_path(self, path: Path, name: str) -> PluginMetadata:
        """Extract metadata from plugin directory."""
        metadata_file = path / "plugin.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                return PluginMetadata(
                    name=data.get('name', name),
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
                self.logger.warning(f"Failed to parse metadata for {name}: {e}")
        
        # Return default metadata
        return PluginMetadata(
            name=name,
            version='1.0.0',
            description=f'Plugin {name}',
            author='Unknown'
        )
    
    def _verify_plugin(self, path: Path) -> bool:
        """Verify plugin integrity and safety."""
        try:
            # Check required files
            if not (path / "__init__.py").exists():
                return False
            
            # Check for malicious content would go here
            # For now, just return True if basic structure is valid
            return True
            
        except Exception:
            return False
    
    def _update_dependency_graph(self, name: str, dependencies: List[str]) -> None:
        """Update the dependency graph for a plugin."""
        # Clear existing dependencies
        if name in self._dependency_graph:
            for old_dep in self._dependency_graph[name]:
                if old_dep in self._reverse_dependencies:
                    self._reverse_dependencies[old_dep].discard(name)
        
        # Set new dependencies
        self._dependency_graph[name] = set(dependencies)
        
        # Update reverse dependencies
        for dep in dependencies:
            if dep not in self._reverse_dependencies:
                self._reverse_dependencies[dep] = set()
            self._reverse_dependencies[dep].add(name)
    
    def _remove_from_dependency_graph(self, name: str) -> None:
        """Remove a plugin from the dependency graph."""
        # Remove dependencies
        if name in self._dependency_graph:
            for dep in self._dependency_graph[name]:
                if dep in self._reverse_dependencies:
                    self._reverse_dependencies[dep].discard(name)
            del self._dependency_graph[name]
        
        # Remove reverse dependencies
        if name in self._reverse_dependencies:
            for dependent in self._reverse_dependencies[name]:
                if dependent in self._dependency_graph:
                    self._dependency_graph[dependent].discard(name)
            del self._reverse_dependencies[name]