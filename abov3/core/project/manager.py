"""
Project Manager for ABOV3.

This module provides comprehensive project directory selection, file management,
and context-aware project operations with full integration into the ABOV3 ecosystem.
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ...utils.file_ops import (
    SafeFileOperations, DirectoryAnalyzer, ProjectAnalyzer, FileWatcher,
    ProjectStructure, FileInfo, DirectoryInfo, ChangeEvent
)
from ...utils.security import SecurityManager
from ...utils.project_errors import (
    ProjectNotFoundError, ProjectPermissionError, FileOperationError,
    ProjectAnalysisError, ContextSyncError, FileWatchingError,
    handle_project_error
)
from ..config import get_config


logger = logging.getLogger(__name__)


class ProjectState(Enum):
    """Project state enumeration."""
    NONE = "none"
    LOADING = "loading"
    ACTIVE = "active"
    ERROR = "error"


@dataclass
class ProjectContext:
    """Project context information."""
    root_path: str
    name: str
    project_type: str
    structure: Optional[ProjectStructure] = None
    current_files: Dict[str, FileInfo] = field(default_factory=dict)
    modified_files: Set[str] = field(default_factory=set)
    watched_files: Set[str] = field(default_factory=set)
    last_analyzed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectConfig:
    """Project-specific configuration."""
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_files_in_context: int = 50
    auto_watch_changes: bool = True
    include_in_context: List[str] = field(default_factory=lambda: [
        "*.py", "*.js", "*.ts", "*.html", "*.css", "*.json", "*.md", "*.txt", "*.yml", "*.yaml"
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "node_modules/*", ".git/*", "__pycache__/*", "*.pyc", "*.pyo", "*.pyd",
        ".venv/*", "venv/*", "env/*", "build/*", "dist/*", "target/*"
    ])
    search_patterns: List[str] = field(default_factory=lambda: [
        "*.py", "*.js", "*.ts", "*.java", "*.cpp", "*.c", "*.h", "*.go", "*.rs",
        "*.html", "*.css", "*.md", "*.txt", "*.json", "*.yml", "*.yaml"
    ])


class ProjectManager:
    """
    Comprehensive project management for ABOV3.
    
    Provides:
    - Project directory selection and validation
    - File operations with context awareness
    - Project structure analysis
    - Real-time file watching
    - Integration with ABOV3 context system
    """
    
    def __init__(self, 
                 security_manager: Optional[SecurityManager] = None,
                 config: Optional[ProjectConfig] = None):
        """Initialize the project manager."""
        self.config = config or ProjectConfig()
        self.app_config = get_config()
        
        # Initialize components
        self.security_manager = security_manager or SecurityManager()
        self.file_ops = SafeFileOperations(
            security_manager=self.security_manager,
            max_file_size=self.config.max_file_size
        )
        self.dir_analyzer = DirectoryAnalyzer(self.file_ops)
        self.project_analyzer = ProjectAnalyzer(self.file_ops)
        self.file_watcher = FileWatcher(self.file_ops)
        
        # State management
        self.state = ProjectState.NONE
        self.current_project: Optional[ProjectContext] = None
        self.project_history: List[str] = []
        self.change_callbacks: List[callable] = []
        
        logger.info("ProjectManager initialized")
    
    async def select_project(self, project_path: str) -> bool:
        """
        Select and initialize a project directory.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            bool: True if successful
        """
        try:
            self.state = ProjectState.LOADING
            
            # Normalize and validate path
            project_path = os.path.abspath(os.path.expanduser(project_path))
            
            # Security validation
            if not self.security_manager.validate_path(project_path):
                error_context = {"project_path": project_path, "operation": "security_validation"}
                error_info = handle_project_error(
                    ProjectPermissionError("path validation", project_path),
                    error_context
                )
                logger.error("Path validation failed", extra=error_info)
                raise ProjectPermissionError("path validation", project_path)
            
            if not os.path.exists(project_path):
                error_context = {"project_path": project_path, "operation": "path_existence_check"}
                error_info = handle_project_error(
                    ProjectNotFoundError(project_path),
                    error_context
                )
                logger.error("Project directory not found", extra=error_info)
                raise ProjectNotFoundError(project_path)
            
            if not os.path.isdir(project_path):
                error_context = {"project_path": project_path, "operation": "directory_check"}
                error_info = handle_project_error(
                    FileOperationError("directory validation", project_path, "Path is not a directory"),
                    error_context
                )
                logger.error("Path is not a directory", extra=error_info)
                raise FileOperationError("directory validation", project_path, "Path is not a directory")
            
            # Check permissions
            if not os.access(project_path, os.R_OK):
                error_context = {"project_path": project_path, "operation": "permission_check"}
                error_info = handle_project_error(
                    ProjectPermissionError("read access", project_path),
                    error_context
                )
                logger.error("No read permission for project directory", extra=error_info)
                raise ProjectPermissionError("read access", project_path)
            
            # Clean up previous project
            if self.current_project:
                await self._cleanup_project()
            
            # Analyze project structure
            logger.info(f"Analyzing project structure: {project_path}")
            structure = await self.project_analyzer.analyze_project(project_path)
            
            # Create project context
            project_name = os.path.basename(project_path)
            self.current_project = ProjectContext(
                root_path=project_path,
                name=project_name,
                project_type=structure.project_type,
                structure=structure,
                last_analyzed=datetime.now()
            )
            
            # Start watching if enabled
            if self.config.auto_watch_changes:
                await self._start_watching()
            
            # Load initial file context
            await self._load_initial_context()
            
            # Update history
            if project_path not in self.project_history:
                self.project_history.append(project_path)
                if len(self.project_history) > 10:  # Keep last 10 projects
                    self.project_history.pop(0)
            
            self.state = ProjectState.ACTIVE
            logger.info(f"Project selected successfully: {project_name} ({structure.project_type})")
            
            # Notify callbacks
            await self._notify_change_callbacks("project_selected", {
                "path": project_path,
                "name": project_name,
                "type": structure.project_type
            })
            
            return True
            
        except Exception as e:
            self.state = ProjectState.ERROR
            logger.error(f"Failed to select project {project_path}: {e}")
            raise
    
    async def get_project_info(self) -> Optional[Dict[str, Any]]:
        """Get current project information."""
        if not self.current_project:
            return None
        
        return {
            "path": self.current_project.root_path,
            "name": self.current_project.name,
            "type": self.current_project.project_type,
            "state": self.state.value,
            "total_files": self.current_project.structure.total_files if self.current_project.structure else 0,
            "total_size": self.current_project.structure.total_size if self.current_project.structure else 0,
            "languages": list(self.current_project.structure.languages) if self.current_project.structure else [],
            "frameworks": list(self.current_project.structure.frameworks) if self.current_project.structure else [],
            "modified_files": len(self.current_project.modified_files),
            "watched_files": len(self.current_project.watched_files),
            "last_analyzed": self.current_project.last_analyzed.isoformat() if self.current_project.last_analyzed else None
        }
    
    async def list_files(self, 
                        pattern: str = "*",
                        directory: str = "",
                        recursive: bool = True,
                        max_results: int = 100) -> List[Dict[str, Any]]:
        """
        List files in the current project.
        
        Args:
            pattern: File pattern to match
            directory: Subdirectory to search in (relative to project root)
            recursive: Whether to search recursively
            max_results: Maximum number of results
            
        Returns:
            List of file information dictionaries
        """
        if not self.current_project:
            raise RuntimeError("No project selected")
        
        search_path = self.current_project.root_path
        if directory:
            search_path = os.path.join(search_path, directory)
            if not os.path.exists(search_path):
                raise FileNotFoundError(f"Directory does not exist: {directory}")
        
        # Find matching files
        matches = await self.dir_analyzer.find_files(search_path, pattern, recursive)
        
        # Apply exclusion patterns
        filtered_matches = []
        for file_path in matches:
            relative_path = os.path.relpath(file_path, self.current_project.root_path)
            if not self._is_excluded(relative_path):
                filtered_matches.append(file_path)
        
        # Limit results
        if len(filtered_matches) > max_results:
            filtered_matches = filtered_matches[:max_results]
        
        # Get file info
        file_list = []
        for file_path in filtered_matches:
            try:
                file_info = await self.file_ops.get_file_info(file_path)
                relative_path = os.path.relpath(file_path, self.current_project.root_path)
                
                file_list.append({
                    "path": relative_path,
                    "absolute_path": file_path,
                    "name": file_info.name,
                    "size": file_info.size,
                    "modified": file_info.modified.isoformat(),
                    "type": file_info.extension or "no_extension",
                    "is_modified": file_path in self.current_project.modified_files
                })
            except Exception as e:
                logger.warning(f"Error getting info for {file_path}: {e}")
                continue
        
        return file_list
    
    async def read_file(self, file_path: str, encoding: str = "utf-8") -> str:
        """
        Read a file from the project.
        
        Args:
            file_path: Relative or absolute file path
            encoding: File encoding
            
        Returns:
            File content
        """
        if not self.current_project:
            raise RuntimeError("No project selected")
        
        # Convert relative path to absolute
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.current_project.root_path, file_path)
        
        # Validate path is within project
        if not file_path.startswith(self.current_project.root_path):
            raise ValueError("File path is outside project directory")
        
        # Read file
        content = await self.file_ops.read_file(file_path, encoding)
        
        # Update context
        relative_path = os.path.relpath(file_path, self.current_project.root_path)
        if relative_path not in self.current_project.watched_files:
            self.current_project.watched_files.add(relative_path)
        
        # Store file info in context
        file_info = await self.file_ops.get_file_info(file_path)
        self.current_project.current_files[relative_path] = file_info
        
        return content
    
    async def write_file(self, 
                        file_path: str, 
                        content: str,
                        encoding: str = "utf-8",
                        create_dirs: bool = True) -> None:
        """
        Write content to a file in the project.
        
        Args:
            file_path: Relative or absolute file path
            content: Content to write
            encoding: File encoding
            create_dirs: Whether to create parent directories
        """
        if not self.current_project:
            raise RuntimeError("No project selected")
        
        # Convert relative path to absolute
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.current_project.root_path, file_path)
        
        # Validate path is within project
        if not file_path.startswith(self.current_project.root_path):
            raise ValueError("File path is outside project directory")
        
        # Create parent directories if needed
        if create_dirs:
            parent_dir = os.path.dirname(file_path)
            await self.file_ops.create_directory(parent_dir)
        
        # Write file
        await self.file_ops.write_file(file_path, content, encoding, backup=True)
        
        # Update tracking
        relative_path = os.path.relpath(file_path, self.current_project.root_path)
        self.current_project.modified_files.add(relative_path)
        
        # Update file info
        file_info = await self.file_ops.get_file_info(file_path)
        self.current_project.current_files[relative_path] = file_info
        
        # Notify callbacks
        await self._notify_change_callbacks("file_modified", {
            "path": relative_path,
            "absolute_path": file_path
        })
    
    async def search_files(self, 
                          query: str,
                          file_pattern: str = "*",
                          case_sensitive: bool = False,
                          max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Search for text within project files.
        
        Args:
            query: Search query
            file_pattern: File pattern to search in
            case_sensitive: Whether search is case sensitive
            max_results: Maximum number of results
            
        Returns:
            List of search results with file paths and matching lines
        """
        if not self.current_project:
            raise RuntimeError("No project selected")
        
        results = []
        search_patterns = self.config.search_patterns if file_pattern == "*" else [file_pattern]
        
        for pattern in search_patterns:
            files = await self.dir_analyzer.find_files(
                self.current_project.root_path, 
                pattern, 
                recursive=True
            )
            
            for file_path in files:
                relative_path = os.path.relpath(file_path, self.current_project.root_path)
                
                # Skip excluded files
                if self._is_excluded(relative_path):
                    continue
                
                try:
                    # Read file content
                    content = await self.file_ops.read_file(file_path)
                    
                    # Search for query
                    lines = content.split('\n')
                    matches = []
                    
                    for line_num, line in enumerate(lines, 1):
                        search_line = line if case_sensitive else line.lower()
                        search_query = query if case_sensitive else query.lower()
                        
                        if search_query in search_line:
                            matches.append({
                                "line_number": line_num,
                                "line_content": line.strip(),
                                "match_start": search_line.find(search_query),
                                "match_end": search_line.find(search_query) + len(search_query)
                            })
                    
                    if matches:
                        results.append({
                            "file_path": relative_path,
                            "absolute_path": file_path,
                            "matches": matches[:10],  # Limit matches per file
                            "total_matches": len(matches)
                        })
                    
                    # Limit total results
                    if len(results) >= max_results:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error searching file {file_path}: {e}")
                    continue
            
            if len(results) >= max_results:
                break
        
        return results
    
    async def get_project_tree(self, max_depth: int = 5) -> Dict[str, Any]:
        """
        Get project directory tree structure.
        
        Args:
            max_depth: Maximum depth to traverse
            
        Returns:
            Tree structure as nested dictionary
        """
        if not self.current_project:
            raise RuntimeError("No project selected")
        
        def build_tree(path: str, current_depth: int = 0) -> Dict[str, Any]:
            if current_depth >= max_depth:
                return {"type": "directory", "name": os.path.basename(path), "truncated": True}
            
            try:
                if os.path.isfile(path):
                    relative_path = os.path.relpath(path, self.current_project.root_path)
                    if self._is_excluded(relative_path):
                        return None
                    
                    file_info = os.stat(path)
                    return {
                        "type": "file",
                        "name": os.path.basename(path),
                        "size": file_info.st_size,
                        "modified": datetime.fromtimestamp(file_info.st_mtime).isoformat(),
                        "is_modified": path in self.current_project.modified_files
                    }
                
                elif os.path.isdir(path):
                    relative_path = os.path.relpath(path, self.current_project.root_path)
                    if relative_path != "." and self._is_excluded(relative_path):
                        return None
                    
                    children = {}
                    try:
                        entries = sorted(os.listdir(path))
                        for entry in entries:
                            entry_path = os.path.join(path, entry)
                            child_node = build_tree(entry_path, current_depth + 1)
                            if child_node:
                                children[entry] = child_node
                    except PermissionError:
                        return {"type": "directory", "name": os.path.basename(path), "error": "Permission denied"}
                    
                    return {
                        "type": "directory",
                        "name": os.path.basename(path) if current_depth > 0 else self.current_project.name,
                        "children": children,
                        "child_count": len(children)
                    }
            
            except Exception as e:
                logger.warning(f"Error building tree for {path}: {e}")
                return {"type": "error", "name": os.path.basename(path), "error": str(e)}
        
        return build_tree(self.current_project.root_path)
    
    async def get_project_context(self, max_files: Optional[int] = None) -> Dict[str, Any]:
        """
        Get project context for AI integration.
        
        Args:
            max_files: Maximum number of files to include in context
            
        Returns:
            Project context dictionary
        """
        if not self.current_project:
            return {}
        
        max_files = max_files or self.config.max_files_in_context
        
        # Get project info
        context = {
            "project": await self.get_project_info(),
            "structure": {
                "type": self.current_project.project_type,
                "languages": list(self.current_project.structure.languages) if self.current_project.structure else [],
                "frameworks": list(self.current_project.structure.frameworks) if self.current_project.structure else [],
            },
            "files": {},
            "recent_changes": []
        }
        
        # Include important files in context
        important_files = []
        
        # Add modified files (highest priority)
        for relative_path in list(self.current_project.modified_files)[:max_files//2]:
            if relative_path in self.current_project.current_files:
                important_files.append((relative_path, "modified"))
        
        # Add recently accessed files
        for relative_path in list(self.current_project.current_files.keys())[:max_files-len(important_files)]:
            if relative_path not in self.current_project.modified_files:
                important_files.append((relative_path, "accessed"))
        
        # Load file contents
        for relative_path, reason in important_files:
            try:
                full_path = os.path.join(self.current_project.root_path, relative_path)
                content = await self.file_ops.read_file(full_path)
                file_info = self.current_project.current_files[relative_path]
                
                context["files"][relative_path] = {
                    "content": content,
                    "size": file_info.size,
                    "modified": file_info.modified.isoformat(),
                    "extension": file_info.extension,
                    "reason": reason
                }
            except Exception as e:
                logger.warning(f"Error loading context file {relative_path}: {e}")
        
        return context
    
    def add_change_callback(self, callback: callable) -> None:
        """Add callback for project changes."""
        self.change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: callable) -> None:
        """Remove change callback."""
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    async def close_project(self) -> None:
        """Close the current project and clean up resources."""
        if self.current_project:
            await self._cleanup_project()
            self.current_project = None
            self.state = ProjectState.NONE
            
            await self._notify_change_callbacks("project_closed", {})
    
    def _is_excluded(self, relative_path: str) -> bool:
        """Check if a file path should be excluded."""
        import fnmatch
        
        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(relative_path, pattern):
                return True
        return False
    
    async def _cleanup_project(self) -> None:
        """Clean up current project resources."""
        try:
            # Stop file watching
            await self.file_watcher.stop_all_watching()
            
            # Clear context
            if self.current_project:
                self.current_project.current_files.clear()
                self.current_project.modified_files.clear()
                self.current_project.watched_files.clear()
            
            logger.info("Project cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during project cleanup: {e}")
    
    async def _start_watching(self) -> None:
        """Start watching project files for changes."""
        if not self.current_project:
            return
        
        async def file_change_callback(event: ChangeEvent) -> None:
            """Handle file change events."""
            try:
                relative_path = os.path.relpath(event.file_path, self.current_project.root_path)
                
                # Skip excluded files
                if self._is_excluded(relative_path):
                    return
                
                # Update tracking
                if event.event_type in ['created', 'modified']:
                    self.current_project.modified_files.add(relative_path)
                    
                    # Update file info if available
                    if event.file_info:
                        self.current_project.current_files[relative_path] = event.file_info
                
                elif event.event_type == 'deleted':
                    self.current_project.modified_files.discard(relative_path)
                    self.current_project.current_files.pop(relative_path, None)
                    self.current_project.watched_files.discard(relative_path)
                
                # Notify callbacks
                await self._notify_change_callbacks("file_changed", {
                    "event_type": event.event_type,
                    "path": relative_path,
                    "timestamp": event.timestamp.isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error handling file change event: {e}")
        
        try:
            # Watch project root
            await self.file_watcher.start_watching(
                self.current_project.root_path,
                file_change_callback,
                recursive=True,
                patterns=self.config.include_in_context
            )
            
            logger.info(f"Started watching project: {self.current_project.root_path}")
            
        except Exception as e:
            logger.error(f"Failed to start file watching: {e}")
    
    async def _load_initial_context(self) -> None:
        """Load initial file context for the project."""
        if not self.current_project:
            return
        
        # Load important project files into context
        important_files = []
        
        # Configuration files
        if self.current_project.structure:
            important_files.extend(self.current_project.structure.config_files[:5])
            important_files.extend(self.current_project.structure.dependency_files[:3])
        
        # README and documentation
        readme_patterns = ["README*", "readme*"]
        for pattern in readme_patterns:
            matches = await self.dir_analyzer.find_files(
                self.current_project.root_path, 
                pattern, 
                recursive=False
            )
            if matches:
                important_files.extend(matches[:1])  # Just the first README
                break
        
        # Load files into context
        for file_path in important_files[:10]:  # Limit initial load
            try:
                relative_path = os.path.relpath(file_path, self.current_project.root_path)
                file_info = await self.file_ops.get_file_info(file_path)
                self.current_project.current_files[relative_path] = file_info
                
            except Exception as e:
                logger.warning(f"Error loading initial context file {file_path}: {e}")
    
    async def _notify_change_callbacks(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify all change callbacks."""
        for callback in self.change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in change callback: {e}")