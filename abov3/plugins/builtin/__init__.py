"""
ABOV3 Built-in Plugins

Collection of built-in plugins for common ABOV3 functionality.
These plugins provide essential features and serve as examples for
custom plugin development.

Built-in plugins include:
    - System Information Plugin: Display system and environment info
    - Model Manager Plugin: Enhanced model management capabilities
    - History Plugin: Advanced conversation history management
    - Security Plugin: Security monitoring and validation
    - Performance Plugin: Performance monitoring and optimization
    - Export Plugin: Data export and backup functionality
    - Notification Plugin: System notifications and alerts
    - Theme Plugin: UI theme management
    - Shortcut Plugin: Custom keyboard shortcuts
    - Voice Plugin: Voice input/output capabilities

Author: ABOV3 Enterprise Plugin System
Version: 1.0.0
"""

import logging
from typing import List, Dict, Any

# Configure built-in plugins logging
logger = logging.getLogger('abov3.plugins.builtin')

# Registry of built-in plugins
BUILTIN_PLUGINS = {
    'system_info': {
        'name': 'System Information',
        'description': 'Display system and environment information',
        'version': '1.0.0',
        'author': 'ABOV3 Team',
        'enabled': True,
        'category': 'system'
    },
    'model_manager': {
        'name': 'Enhanced Model Manager',
        'description': 'Advanced model management and configuration',
        'version': '1.0.0', 
        'author': 'ABOV3 Team',
        'enabled': True,
        'category': 'core'
    },
    'history_manager': {
        'name': 'History Manager',
        'description': 'Advanced conversation history management',
        'version': '1.0.0',
        'author': 'ABOV3 Team', 
        'enabled': True,
        'category': 'core'
    },
    'security_monitor': {
        'name': 'Security Monitor',
        'description': 'Security monitoring and validation',
        'version': '1.0.0',
        'author': 'ABOV3 Team',
        'enabled': True,
        'category': 'security'
    },
    'performance_monitor': {
        'name': 'Performance Monitor',
        'description': 'Performance monitoring and optimization',
        'version': '1.0.0',
        'author': 'ABOV3 Team',
        'enabled': True,
        'category': 'performance'
    },
    'export_manager': {
        'name': 'Export Manager',
        'description': 'Data export and backup functionality',
        'version': '1.0.0',
        'author': 'ABOV3 Team',
        'enabled': True,
        'category': 'utility'
    },
    'notification_system': {
        'name': 'Notification System',
        'description': 'System notifications and alerts',
        'version': '1.0.0',
        'author': 'ABOV3 Team',
        'enabled': True,
        'category': 'ui'
    },
    'theme_manager': {
        'name': 'Theme Manager',
        'description': 'UI theme management and customization',
        'version': '1.0.0',
        'author': 'ABOV3 Team',
        'enabled': True,
        'category': 'ui'
    },
    'shortcut_manager': {
        'name': 'Shortcut Manager',
        'description': 'Custom keyboard shortcuts and commands',
        'version': '1.0.0',
        'author': 'ABOV3 Team',
        'enabled': True,
        'category': 'ui'
    },
    'voice_interface': {
        'name': 'Voice Interface',
        'description': 'Voice input and output capabilities',
        'version': '1.0.0',
        'author': 'ABOV3 Team',
        'enabled': False,  # Disabled by default
        'category': 'interface'
    }
}


def get_builtin_plugins() -> List[str]:
    """Get list of all built-in plugin names."""
    return list(BUILTIN_PLUGINS.keys())


def get_enabled_builtin_plugins() -> List[str]:
    """Get list of enabled built-in plugin names."""
    return [name for name, info in BUILTIN_PLUGINS.items() if info.get('enabled', False)]


def get_builtin_plugin_info(plugin_name: str) -> Dict[str, Any]:
    """Get information about a specific built-in plugin."""
    return BUILTIN_PLUGINS.get(plugin_name, {})


def get_plugins_by_category(category: str) -> List[str]:
    """Get built-in plugins by category."""
    return [name for name, info in BUILTIN_PLUGINS.items() 
            if info.get('category') == category]


def is_builtin_plugin(plugin_name: str) -> bool:
    """Check if a plugin is a built-in plugin."""
    return plugin_name in BUILTIN_PLUGINS


def get_builtin_categories() -> List[str]:
    """Get list of built-in plugin categories."""
    categories = set()
    for info in BUILTIN_PLUGINS.values():
        if 'category' in info:
            categories.add(info['category'])
    return sorted(list(categories))


def enable_builtin_plugin(plugin_name: str) -> bool:
    """Enable a built-in plugin."""
    if plugin_name in BUILTIN_PLUGINS:
        BUILTIN_PLUGINS[plugin_name]['enabled'] = True
        logger.info(f"Enabled built-in plugin: {plugin_name}")
        return True
    return False


def disable_builtin_plugin(plugin_name: str) -> bool:
    """Disable a built-in plugin."""
    if plugin_name in BUILTIN_PLUGINS:
        BUILTIN_PLUGINS[plugin_name]['enabled'] = False
        logger.info(f"Disabled built-in plugin: {plugin_name}")
        return True
    return False


def get_builtin_plugin_stats() -> Dict[str, Any]:
    """Get statistics about built-in plugins."""
    total = len(BUILTIN_PLUGINS)
    enabled = len(get_enabled_builtin_plugins())
    categories = {}
    
    for info in BUILTIN_PLUGINS.values():
        category = info.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
    
    return {
        'total': total,
        'enabled': enabled,
        'disabled': total - enabled,
        'categories': categories
    }


# Export main functions
__all__ = [
    'BUILTIN_PLUGINS',
    'get_builtin_plugins',
    'get_enabled_builtin_plugins', 
    'get_builtin_plugin_info',
    'get_plugins_by_category',
    'is_builtin_plugin',
    'get_builtin_categories',
    'enable_builtin_plugin',
    'disable_builtin_plugin',
    'get_builtin_plugin_stats'
]