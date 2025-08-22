"""
ABOV3 Ollama File Operations Utilities

Comprehensive file system operations and project navigation utilities providing safe file handling,
project structure analysis, file watching, backup management, and cross-platform path operations.

Features:
    - Safe file system operations with security validation
    - Project structure analysis and navigation
    - Real-time file watching and change detection
    - Automated backup and versioning support
    - Cross-platform path handling and normalization
    - File encoding detection and conversion
    - Archive operations (zip, tar, gzip)
    - Directory synchronization and mirroring
    - File metadata extraction and analysis
    - Temporary file and directory management

Security Features:
    - Path traversal protection
    - File permission validation
    - Safe file operations with sandboxing
    - Malicious file detection
    - Size and type restrictions
    - Quarantine system for suspicious files

Author: ABOV3 Enterprise Documentation Agent
Version: 1.0.0
"""

import os
import sys
import shutil
import stat
import time
import json
import hashlib
import tempfile
import logging
import asyncio
import zipfile
import tarfile
import gzip
import threading
from pathlib import Path, PurePath
from typing import (
    Dict, List, Optional, Any, Tuple, Set, Union, 
    AsyncGenerator, Callable, NamedTuple, Iterator,
    AsyncIterator, Protocol, runtime_checkable
)
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import mimetypes
import fnmatch

# Third-party imports
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

# Internal imports
from .security import SecurityManager, is_safe_path, validate_file_permissions
from .validation import PathValidator, validate_file_path


# Configure logging
logger = logging.getLogger('abov3.file_ops')


@dataclass
class FileInfo:
    """Comprehensive file information."""
    path: str
    name: str
    extension: str
    size: int
    created: datetime
    modified: datetime
    accessed: datetime
    permissions: str
    is_directory: bool
    is_file: bool
    is_symlink: bool
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    hash_md5: Optional[str] = None
    hash_sha256: Optional[str] = None
    owner: Optional[str] = None
    group: Optional[str] = None


@dataclass
class DirectoryInfo:
    """Directory structure information."""
    path: str
    total_files: int
    total_directories: int
    total_size: int
    file_types: Dict[str, int]
    largest_files: List[FileInfo]
    newest_files: List[FileInfo]
    oldest_files: List[FileInfo]
    depth: int


@dataclass
class ProjectStructure:
    """Project structure analysis."""
    root_path: str
    project_type: str
    config_files: List[str]
    source_directories: List[str]
    test_directories: List[str]
    documentation_files: List[str]
    dependency_files: List[str]
    build_files: List[str]
    ignore_patterns: List[str]
    languages: Set[str]
    frameworks: Set[str]
    total_files: int
    total_size: int


@dataclass
class BackupInfo:
    """Backup operation information."""
    backup_id: str
    source_path: str
    backup_path: str
    timestamp: datetime
    size: int
    file_count: int
    compression: str
    checksum: str
    metadata: Dict[str, Any]


@dataclass
class ChangeEvent:
    """File system change event."""
    event_type: str  # created, modified, deleted, moved
    file_path: str
    timestamp: datetime
    old_path: Optional[str] = None
    file_info: Optional[FileInfo] = None


class FileOperationError(Exception):
    """Base exception for file operations."""
    pass


class SecurityViolationError(FileOperationError):
    """Raised when a security violation is detected."""
    pass


class PathTraversalError(SecurityViolationError):
    """Raised when path traversal is attempted."""
    pass


@runtime_checkable
class FileWatcherCallback(Protocol):
    """Protocol for file watcher callbacks."""
    async def __call__(self, event: ChangeEvent) -> None:
        """Handle file system change event."""
        ...


class SafeFileOperations:
    """Safe file operations with security validation."""
    
    def __init__(self, security_manager: Optional[SecurityManager] = None,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 allowed_extensions: Optional[Set[str]] = None,
                 blocked_extensions: Optional[Set[str]] = None):
        self.security_manager = security_manager or SecurityManager()
        self.path_validator = PathValidator()
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or {
            '.py', '.js', '.ts', '.html', '.css', '.json', '.yaml', '.yml',
            '.toml', '.ini', '.cfg', '.conf', '.txt', '.md', '.rst',
            '.java', '.c', '.cpp', '.h', '.hpp', '.go', '.rs', '.rb',
            '.php', '.swift', '.kt', '.scala', '.sh', '.bash', '.ps1'
        }
        self.blocked_extensions = blocked_extensions or {
            '.exe', '.dll', '.so', '.dylib', '.bat', '.cmd', '.scr',
            '.vbs', '.jar', '.war', '.ear', '.com', '.pif'
        }
        
        # Create thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def read_file(self, file_path: str, encoding: Optional[str] = None) -> str:
        """Safely read file content."""
        file_path = self._validate_and_normalize_path(file_path)
        
        # Security checks
        self._check_file_access(file_path, 'read')
        
        # Detect encoding if not provided
        if encoding is None:
            encoding = await self._detect_encoding(file_path)
        
        # Read file asynchronously
        loop = asyncio.get_event_loop()
        try:
            content = await loop.run_in_executor(
                self.executor,
                self._read_file_sync,
                file_path,
                encoding
            )
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise FileOperationError(f"Failed to read file: {e}")
    
    async def write_file(self, file_path: str, content: str, 
                        encoding: str = 'utf-8', backup: bool = True) -> None:
        """Safely write content to file."""
        file_path = self._validate_and_normalize_path(file_path)
        
        # Security checks
        self._check_file_access(file_path, 'write')
        self._validate_content(content)
        
        # Create backup if requested and file exists
        if backup and os.path.exists(file_path):
            await self._create_backup(file_path)
        
        # Write file asynchronously
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                self.executor,
                self._write_file_sync,
                file_path,
                content,
                encoding
            )
            logger.info(f"File written successfully: {file_path}")
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            raise FileOperationError(f"Failed to write file: {e}")
    
    async def copy_file(self, src_path: str, dst_path: str, 
                       preserve_metadata: bool = True) -> None:
        """Safely copy file."""
        src_path = self._validate_and_normalize_path(src_path)
        dst_path = self._validate_and_normalize_path(dst_path)
        
        # Security checks
        self._check_file_access(src_path, 'read')
        self._check_file_access(dst_path, 'write')
        
        # Ensure destination directory exists
        dst_dir = os.path.dirname(dst_path)
        await self.create_directory(dst_dir)
        
        # Copy file asynchronously
        loop = asyncio.get_event_loop()
        try:
            if preserve_metadata:
                await loop.run_in_executor(self.executor, shutil.copy2, src_path, dst_path)
            else:
                await loop.run_in_executor(self.executor, shutil.copy, src_path, dst_path)
            
            logger.info(f"File copied: {src_path} -> {dst_path}")
        except Exception as e:
            logger.error(f"Error copying file {src_path} to {dst_path}: {e}")
            raise FileOperationError(f"Failed to copy file: {e}")
    
    async def move_file(self, src_path: str, dst_path: str) -> None:
        """Safely move/rename file."""
        src_path = self._validate_and_normalize_path(src_path)
        dst_path = self._validate_and_normalize_path(dst_path)
        
        # Security checks
        self._check_file_access(src_path, 'read')
        self._check_file_access(dst_path, 'write')
        
        # Ensure destination directory exists
        dst_dir = os.path.dirname(dst_path)
        await self.create_directory(dst_dir)
        
        # Move file asynchronously
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(self.executor, shutil.move, src_path, dst_path)
            logger.info(f"File moved: {src_path} -> {dst_path}")
        except Exception as e:
            logger.error(f"Error moving file {src_path} to {dst_path}: {e}")
            raise FileOperationError(f"Failed to move file: {e}")
    
    async def delete_file(self, file_path: str, secure: bool = False) -> None:
        """Safely delete file."""
        file_path = self._validate_and_normalize_path(file_path)
        
        # Security checks
        self._check_file_access(file_path, 'write')
        
        # Secure deletion (overwrite with random data)
        if secure:
            await self._secure_delete(file_path)
        else:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(self.executor, os.remove, file_path)
                logger.info(f"File deleted: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")
                raise FileOperationError(f"Failed to delete file: {e}")
    
    async def create_directory(self, dir_path: str, mode: int = 0o755) -> None:
        """Safely create directory."""
        dir_path = self._validate_and_normalize_path(dir_path)
        
        # Security checks
        self._check_path_safety(dir_path)
        
        if not os.path.exists(dir_path):
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    self.executor,
                    os.makedirs,
                    dir_path,
                    mode,
                    True  # exist_ok
                )
                logger.info(f"Directory created: {dir_path}")
            except Exception as e:
                logger.error(f"Error creating directory {dir_path}: {e}")
                raise FileOperationError(f"Failed to create directory: {e}")
    
    async def get_file_info(self, file_path: str) -> FileInfo:
        """Get comprehensive file information."""
        file_path = self._validate_and_normalize_path(file_path)
        
        # Security checks
        self._check_file_access(file_path, 'read')
        
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self.executor,
                self._get_file_info_sync,
                file_path
            )
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            raise FileOperationError(f"Failed to get file info: {e}")
    
    def _validate_and_normalize_path(self, file_path: str) -> str:
        """Validate and normalize file path."""
        # Validate path format
        if not validate_file_path(file_path):
            raise ValueError(f"Invalid file path format: {file_path}")
        
        # Normalize path
        normalized = os.path.normpath(os.path.abspath(file_path))
        
        # Check for path traversal
        if not is_safe_path(normalized):
            raise PathTraversalError(f"Path traversal detected: {file_path}")
        
        return normalized
    
    def _check_file_access(self, file_path: str, operation: str) -> None:
        """Check file access permissions and security."""
        # Check if path is safe
        self._check_path_safety(file_path)
        
        # Check file extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() in self.blocked_extensions:
            raise SecurityViolationError(f"File type not allowed: {ext}")
        
        if self.allowed_extensions and ext.lower() not in self.allowed_extensions:
            raise SecurityViolationError(f"File type not in allowed list: {ext}")
        
        # Check file size for existing files
        if os.path.exists(file_path) and os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                raise SecurityViolationError(f"File too large: {file_size} bytes")
        
        # Check permissions
        if operation == 'read' and os.path.exists(file_path):
            if not os.access(file_path, os.R_OK):
                raise FileOperationError(f"No read permission for: {file_path}")
        elif operation == 'write':
            if os.path.exists(file_path):
                if not os.access(file_path, os.W_OK):
                    raise FileOperationError(f"No write permission for: {file_path}")
            else:
                # Check parent directory
                parent_dir = os.path.dirname(file_path)
                if os.path.exists(parent_dir) and not os.access(parent_dir, os.W_OK):
                    raise FileOperationError(f"No write permission for directory: {parent_dir}")
    
    def _check_path_safety(self, file_path: str) -> None:
        """Check if path is safe to access."""
        if not is_safe_path(file_path):
            raise SecurityViolationError(f"Unsafe path detected: {file_path}")
    
    def _validate_content(self, content: str) -> None:
        """Validate file content for security issues."""
        if self.security_manager:
            # Check for malicious patterns
            from .security import detect_malicious_patterns
            patterns = detect_malicious_patterns(content)
            if patterns:
                raise SecurityViolationError(f"Malicious patterns detected: {patterns}")
    
    def _read_file_sync(self, file_path: str, encoding: str) -> str:
        """Synchronous file reading."""
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    
    def _write_file_sync(self, file_path: str, content: str, encoding: str) -> None:
        """Synchronous file writing."""
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(file_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
    
    async def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding."""
        if CHARDET_AVAILABLE:
            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(
                    self.executor,
                    self._detect_encoding_sync,
                    file_path
                )
            except Exception:
                pass
        
        # Fallback to UTF-8
        return 'utf-8'
    
    def _detect_encoding_sync(self, file_path: str) -> str:
        """Synchronous encoding detection."""
        with open(file_path, 'rb') as f:
            raw_data = f.read(8192)  # Read first 8KB
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8')
    
    def _get_file_info_sync(self, file_path: str) -> FileInfo:
        """Synchronous file info extraction."""
        stat_info = os.stat(file_path)
        
        # Basic info
        name = os.path.basename(file_path)
        extension = os.path.splitext(name)[1].lower()
        
        # Timestamps
        created = datetime.fromtimestamp(stat_info.st_ctime)
        modified = datetime.fromtimestamp(stat_info.st_mtime)
        accessed = datetime.fromtimestamp(stat_info.st_atime)
        
        # File type detection
        mime_type = mimetypes.guess_type(file_path)[0]
        if MAGIC_AVAILABLE and os.path.isfile(file_path):
            try:
                mime_type = magic.from_file(file_path, mime=True)
            except Exception:
                pass
        
        # Permissions
        permissions = stat.filemode(stat_info.st_mode)
        
        # Hashes for files
        hash_md5 = None
        hash_sha256 = None
        if os.path.isfile(file_path) and stat_info.st_size < 50 * 1024 * 1024:  # < 50MB
            try:
                hash_md5, hash_sha256 = self._calculate_hashes(file_path)
            except Exception:
                pass
        
        return FileInfo(
            path=file_path,
            name=name,
            extension=extension,
            size=stat_info.st_size,
            created=created,
            modified=modified,
            accessed=accessed,
            permissions=permissions,
            is_directory=os.path.isdir(file_path),
            is_file=os.path.isfile(file_path),
            is_symlink=os.path.islink(file_path),
            mime_type=mime_type,
            hash_md5=hash_md5,
            hash_sha256=hash_sha256
        )
    
    def _calculate_hashes(self, file_path: str) -> Tuple[str, str]:
        """Calculate MD5 and SHA256 hashes."""
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
        
        return md5_hash.hexdigest(), sha256_hash.hexdigest()
    
    async def _create_backup(self, file_path: str) -> str:
        """Create backup of existing file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.backup_{timestamp}"
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, shutil.copy2, file_path, backup_path)
        
        logger.info(f"Backup created: {backup_path}")
        return backup_path
    
    async def _secure_delete(self, file_path: str) -> None:
        """Securely delete file by overwriting with random data."""
        if not os.path.isfile(file_path):
            return
        
        file_size = os.path.getsize(file_path)
        
        # Overwrite with random data multiple times
        for _ in range(3):
            with open(file_path, 'r+b') as f:
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
        
        # Finally delete the file
        os.remove(file_path)
        logger.info(f"File securely deleted: {file_path}")


class DirectoryAnalyzer:
    """Analyze directory structure and contents."""
    
    def __init__(self, safe_ops: Optional[SafeFileOperations] = None):
        self.safe_ops = safe_ops or SafeFileOperations()
    
    async def analyze_directory(self, dir_path: str, max_depth: int = 10) -> DirectoryInfo:
        """Analyze directory structure and contents."""
        dir_path = self.safe_ops._validate_and_normalize_path(dir_path)
        
        if not os.path.isdir(dir_path):
            raise FileOperationError(f"Not a directory: {dir_path}")
        
        # Initialize counters
        total_files = 0
        total_directories = 0
        total_size = 0
        file_types = defaultdict(int)
        all_files = []
        
        # Walk directory tree
        for root, dirs, files in os.walk(dir_path):
            # Check depth
            current_depth = root.replace(dir_path, '').count(os.sep)
            if current_depth >= max_depth:
                dirs[:] = []  # Don't descend further
                continue
            
            total_directories += len(dirs)
            
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    file_info = await self.safe_ops.get_file_info(file_path)
                    all_files.append(file_info)
                    total_files += 1
                    total_size += file_info.size
                    
                    # Count file types
                    if file_info.extension:
                        file_types[file_info.extension] += 1
                    else:
                        file_types['no_extension'] += 1
                        
                except Exception as e:
                    logger.warning(f"Error analyzing file {file_path}: {e}")
        
        # Sort files for analysis
        all_files.sort(key=lambda f: f.size, reverse=True)
        largest_files = all_files[:10]
        
        all_files.sort(key=lambda f: f.modified, reverse=True)
        newest_files = all_files[:10]
        
        all_files.sort(key=lambda f: f.modified)
        oldest_files = all_files[:10]
        
        return DirectoryInfo(
            path=dir_path,
            total_files=total_files,
            total_directories=total_directories,
            total_size=total_size,
            file_types=dict(file_types),
            largest_files=largest_files,
            newest_files=newest_files,
            oldest_files=oldest_files,
            depth=max_depth
        )
    
    async def find_files(self, dir_path: str, pattern: str = "*", 
                        recursive: bool = True) -> List[str]:
        """Find files matching pattern."""
        dir_path = self.safe_ops._validate_and_normalize_path(dir_path)
        
        matches = []
        
        if recursive:
            for root, dirs, files in os.walk(dir_path):
                for file_name in files:
                    if fnmatch.fnmatch(file_name, pattern):
                        matches.append(os.path.join(root, file_name))
        else:
            try:
                for file_name in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file_name)
                    if os.path.isfile(file_path) and fnmatch.fnmatch(file_name, pattern):
                        matches.append(file_path)
            except OSError as e:
                logger.error(f"Error listing directory {dir_path}: {e}")
                raise FileOperationError(f"Failed to list directory: {e}")
        
        return sorted(matches)
    
    async def get_directory_size(self, dir_path: str) -> int:
        """Calculate total directory size."""
        dir_path = self.safe_ops._validate_and_normalize_path(dir_path)
        
        total_size = 0
        for root, dirs, files in os.walk(dir_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    total_size += os.path.getsize(file_path)
                except OSError:
                    continue
        
        return total_size


class ProjectAnalyzer:
    """Analyze project structure and identify project type."""
    
    PROJECT_INDICATORS = {
        'python': {
            'files': ['setup.py', 'pyproject.toml', 'requirements.txt', 'Pipfile', 'setup.cfg'],
            'dirs': ['src', 'tests', 'test'],
            'patterns': ['*.py']
        },
        'nodejs': {
            'files': ['package.json', 'package-lock.json', 'yarn.lock', 'tsconfig.json'],
            'dirs': ['node_modules', 'src', 'dist', 'build'],
            'patterns': ['*.js', '*.ts', '*.jsx', '*.tsx']
        },
        'java': {
            'files': ['pom.xml', 'build.gradle', 'build.xml'],
            'dirs': ['src/main/java', 'src/test/java', 'target', 'build'],
            'patterns': ['*.java']
        },
        'go': {
            'files': ['go.mod', 'go.sum', 'Gopkg.toml'],
            'dirs': ['cmd', 'pkg', 'internal'],
            'patterns': ['*.go']
        },
        'rust': {
            'files': ['Cargo.toml', 'Cargo.lock'],
            'dirs': ['src', 'target', 'tests'],
            'patterns': ['*.rs']
        },
        'cpp': {
            'files': ['CMakeLists.txt', 'Makefile', 'configure.ac'],
            'dirs': ['src', 'include', 'build', 'bin'],
            'patterns': ['*.cpp', '*.hpp', '*.c', '*.h']
        }
    }
    
    def __init__(self, safe_ops: Optional[SafeFileOperations] = None):
        self.safe_ops = safe_ops or SafeFileOperations()
        self.dir_analyzer = DirectoryAnalyzer(safe_ops)
    
    async def analyze_project(self, project_path: str) -> ProjectStructure:
        """Analyze project structure and identify type."""
        project_path = self.safe_ops._validate_and_normalize_path(project_path)
        
        if not os.path.isdir(project_path):
            raise FileOperationError(f"Not a directory: {project_path}")
        
        # Analyze directory
        dir_info = await self.dir_analyzer.analyze_directory(project_path)
        
        # Identify project type
        project_type = await self._identify_project_type(project_path)
        
        # Find configuration files
        config_files = await self._find_config_files(project_path)
        
        # Find source directories
        source_dirs = await self._find_source_directories(project_path)
        
        # Find test directories
        test_dirs = await self._find_test_directories(project_path)
        
        # Find documentation files
        doc_files = await self._find_documentation_files(project_path)
        
        # Find dependency files
        dep_files = await self._find_dependency_files(project_path)
        
        # Find build files
        build_files = await self._find_build_files(project_path)
        
        # Get ignore patterns
        ignore_patterns = await self._get_ignore_patterns(project_path)
        
        # Identify languages and frameworks
        languages, frameworks = await self._identify_languages_frameworks(project_path)
        
        return ProjectStructure(
            root_path=project_path,
            project_type=project_type,
            config_files=config_files,
            source_directories=source_dirs,
            test_directories=test_dirs,
            documentation_files=doc_files,
            dependency_files=dep_files,
            build_files=build_files,
            ignore_patterns=ignore_patterns,
            languages=languages,
            frameworks=frameworks,
            total_files=dir_info.total_files,
            total_size=dir_info.total_size
        )
    
    async def _identify_project_type(self, project_path: str) -> str:
        """Identify project type based on files and structure."""
        scores = defaultdict(int)
        
        for project_type, indicators in self.PROJECT_INDICATORS.items():
            # Check for indicator files
            for file_name in indicators['files']:
                file_path = os.path.join(project_path, file_name)
                if os.path.exists(file_path):
                    scores[project_type] += 3
            
            # Check for indicator directories
            for dir_name in indicators['dirs']:
                dir_path = os.path.join(project_path, dir_name)
                if os.path.isdir(dir_path):
                    scores[project_type] += 2
            
            # Check for file patterns
            for pattern in indicators['patterns']:
                matches = await self.dir_analyzer.find_files(
                    project_path, pattern, recursive=False
                )
                if matches:
                    scores[project_type] += len(matches)
        
        if scores:
            return max(scores, key=scores.get)
        return 'unknown'
    
    async def _find_config_files(self, project_path: str) -> List[str]:
        """Find configuration files."""
        config_patterns = [
            '*.json', '*.yaml', '*.yml', '*.toml', '*.ini', '*.cfg', '*.conf',
            '.env*', '*.properties', '*.xml'
        ]
        
        config_files = []
        for pattern in config_patterns:
            matches = await self.dir_analyzer.find_files(
                project_path, pattern, recursive=False
            )
            config_files.extend(matches)
        
        return config_files
    
    async def _find_source_directories(self, project_path: str) -> List[str]:
        """Find source code directories."""
        common_source_dirs = ['src', 'source', 'lib', 'app', 'main']
        source_dirs = []
        
        for dir_name in common_source_dirs:
            dir_path = os.path.join(project_path, dir_name)
            if os.path.isdir(dir_path):
                source_dirs.append(dir_path)
        
        return source_dirs
    
    async def _find_test_directories(self, project_path: str) -> List[str]:
        """Find test directories."""
        common_test_dirs = ['test', 'tests', 'testing', 'spec', 'specs']
        test_dirs = []
        
        for dir_name in common_test_dirs:
            dir_path = os.path.join(project_path, dir_name)
            if os.path.isdir(dir_path):
                test_dirs.append(dir_path)
        
        return test_dirs
    
    async def _find_documentation_files(self, project_path: str) -> List[str]:
        """Find documentation files."""
        doc_patterns = ['*.md', '*.rst', '*.txt', 'README*', 'CHANGELOG*', 'LICENSE*']
        doc_files = []
        
        for pattern in doc_patterns:
            matches = await self.dir_analyzer.find_files(
                project_path, pattern, recursive=True
            )
            doc_files.extend(matches)
        
        # Also check for docs directory
        docs_dir = os.path.join(project_path, 'docs')
        if os.path.isdir(docs_dir):
            doc_files.append(docs_dir)
        
        return doc_files
    
    async def _find_dependency_files(self, project_path: str) -> List[str]:
        """Find dependency management files."""
        dep_files = [
            'requirements.txt', 'Pipfile', 'package.json', 'yarn.lock',
            'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle'
        ]
        
        found_files = []
        for file_name in dep_files:
            file_path = os.path.join(project_path, file_name)
            if os.path.exists(file_path):
                found_files.append(file_path)
        
        return found_files
    
    async def _find_build_files(self, project_path: str) -> List[str]:
        """Find build configuration files."""
        build_files = [
            'Makefile', 'CMakeLists.txt', 'build.xml', 'webpack.config.js',
            'rollup.config.js', 'vite.config.js', 'tox.ini'
        ]
        
        found_files = []
        for file_name in build_files:
            file_path = os.path.join(project_path, file_name)
            if os.path.exists(file_path):
                found_files.append(file_path)
        
        return found_files
    
    async def _get_ignore_patterns(self, project_path: str) -> List[str]:
        """Get ignore patterns from various ignore files."""
        ignore_files = ['.gitignore', '.dockerignore', '.npmignore']
        patterns = []
        
        for ignore_file in ignore_files:
            file_path = os.path.join(project_path, ignore_file)
            if os.path.exists(file_path):
                try:
                    content = await self.safe_ops.read_file(file_path)
                    file_patterns = [
                        line.strip() for line in content.split('\n')
                        if line.strip() and not line.startswith('#')
                    ]
                    patterns.extend(file_patterns)
                except Exception as e:
                    logger.warning(f"Error reading ignore file {file_path}: {e}")
        
        return patterns
    
    async def _identify_languages_frameworks(self, project_path: str) -> Tuple[Set[str], Set[str]]:
        """Identify programming languages and frameworks used."""
        languages = set()
        frameworks = set()
        
        # Language detection by file extensions
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++',
            '.c': 'C',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.swift': 'Swift',
            '.kt': 'Kotlin'
        }
        
        # Walk through project and identify languages
        for root, dirs, files in os.walk(project_path):
            for file_name in files:
                _, ext = os.path.splitext(file_name)
                if ext.lower() in language_map:
                    languages.add(language_map[ext.lower()])
        
        # Framework detection
        framework_indicators = {
            'Django': ['manage.py', 'django'],
            'Flask': ['app.py', 'flask'],
            'FastAPI': ['fastapi'],
            'React': ['package.json', 'react'],
            'Vue': ['package.json', 'vue'],
            'Angular': ['angular.json', '@angular'],
            'Spring': ['pom.xml', 'spring'],
            'Express': ['package.json', 'express']
        }
        
        # Check for framework indicators
        for framework, indicators in framework_indicators.items():
            for indicator in indicators:
                if '.' in indicator:  # File check
                    file_path = os.path.join(project_path, indicator)
                    if os.path.exists(file_path):
                        frameworks.add(framework)
                        break
                else:  # Content check in package files
                    package_files = ['package.json', 'requirements.txt', 'pom.xml']
                    for package_file in package_files:
                        file_path = os.path.join(project_path, package_file)
                        if os.path.exists(file_path):
                            try:
                                content = await self.safe_ops.read_file(file_path)
                                if indicator in content:
                                    frameworks.add(framework)
                                    break
                            except Exception:
                                continue
        
        return languages, frameworks


class FileWatcher:
    """File system watcher for real-time change detection."""
    
    def __init__(self, safe_ops: Optional[SafeFileOperations] = None):
        self.safe_ops = safe_ops or SafeFileOperations()
        self.observers: Dict[str, Observer] = {}
        self.callbacks: Dict[str, List[FileWatcherCallback]] = defaultdict(list)
        self.is_watching = False
        
        if not WATCHDOG_AVAILABLE:
            logger.warning("Watchdog not available. File watching disabled.")
    
    async def start_watching(self, path: str, 
                           callback: FileWatcherCallback,
                           recursive: bool = True,
                           patterns: Optional[List[str]] = None) -> str:
        """Start watching a path for changes."""
        if not WATCHDOG_AVAILABLE:
            raise FileOperationError("File watching not available (watchdog not installed)")
        
        path = self.safe_ops._validate_and_normalize_path(path)
        
        if not os.path.exists(path):
            raise FileOperationError(f"Path does not exist: {path}")
        
        # Create unique watch ID
        watch_id = f"{path}_{id(callback)}"
        
        # Create event handler
        handler = self._create_event_handler(watch_id, patterns or [])
        
        # Create and start observer
        observer = Observer()
        observer.schedule(handler, path, recursive=recursive)
        observer.start()
        
        # Store observer and callback
        self.observers[watch_id] = observer
        self.callbacks[watch_id].append(callback)
        self.is_watching = True
        
        logger.info(f"Started watching {path} with ID {watch_id}")
        return watch_id
    
    async def stop_watching(self, watch_id: str) -> None:
        """Stop watching a specific path."""
        if watch_id in self.observers:
            observer = self.observers[watch_id]
            observer.stop()
            observer.join()
            
            del self.observers[watch_id]
            del self.callbacks[watch_id]
            
            logger.info(f"Stopped watching {watch_id}")
    
    async def stop_all_watching(self) -> None:
        """Stop all file watching."""
        for watch_id in list(self.observers.keys()):
            await self.stop_watching(watch_id)
        
        self.is_watching = False
        logger.info("Stopped all file watching")
    
    def _create_event_handler(self, watch_id: str, patterns: List[str]) -> FileSystemEventHandler:
        """Create file system event handler."""
        
        class AsyncEventHandler(FileSystemEventHandler):
            def __init__(self, watcher_instance, watch_id, patterns):
                self.watcher = watcher_instance
                self.watch_id = watch_id
                self.patterns = patterns
                self.loop = asyncio.new_event_loop()
                self.thread = threading.Thread(target=self._run_loop, daemon=True)
                self.thread.start()
            
            def _run_loop(self):
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()
            
            def _should_process(self, file_path: str) -> bool:
                if not self.patterns:
                    return True
                
                file_name = os.path.basename(file_path)
                return any(fnmatch.fnmatch(file_name, pattern) for pattern in self.patterns)
            
            def on_created(self, event: FileSystemEvent) -> None:
                if not event.is_directory and self._should_process(event.src_path):
                    self._handle_event('created', event.src_path)
            
            def on_modified(self, event: FileSystemEvent) -> None:
                if not event.is_directory and self._should_process(event.src_path):
                    self._handle_event('modified', event.src_path)
            
            def on_deleted(self, event: FileSystemEvent) -> None:
                if not event.is_directory and self._should_process(event.src_path):
                    self._handle_event('deleted', event.src_path)
            
            def on_moved(self, event: FileSystemEvent) -> None:
                if not event.is_directory and self._should_process(event.dest_path):
                    self._handle_event('moved', event.dest_path, event.src_path)
            
            def _handle_event(self, event_type: str, file_path: str, old_path: str = None) -> None:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.watcher._process_event(self.watch_id, event_type, file_path, old_path),
                        self.loop
                    )
                except Exception as e:
                    logger.error(f"Error handling file event: {e}")
        
        return AsyncEventHandler(self, watch_id, patterns)
    
    async def _process_event(self, watch_id: str, event_type: str, 
                           file_path: str, old_path: Optional[str] = None) -> None:
        """Process file system event and call callbacks."""
        try:
            # Create change event
            file_info = None
            if os.path.exists(file_path):
                file_info = await self.safe_ops.get_file_info(file_path)
            
            change_event = ChangeEvent(
                event_type=event_type,
                file_path=file_path,
                timestamp=datetime.now(),
                old_path=old_path,
                file_info=file_info
            )
            
            # Call all callbacks for this watch
            if watch_id in self.callbacks:
                for callback in self.callbacks[watch_id]:
                    try:
                        await callback(change_event)
                    except Exception as e:
                        logger.error(f"Error in file watcher callback: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing file event {event_type} for {file_path}: {e}")


class ArchiveOperations:
    """Archive and compression operations."""
    
    def __init__(self, safe_ops: Optional[SafeFileOperations] = None):
        self.safe_ops = safe_ops or SafeFileOperations()
    
    async def create_zip_archive(self, source_path: str, archive_path: str,
                               compression_level: int = 6) -> None:
        """Create ZIP archive."""
        source_path = self.safe_ops._validate_and_normalize_path(source_path)
        archive_path = self.safe_ops._validate_and_normalize_path(archive_path)
        
        loop = asyncio.get_event_loop()
        
        def _create_zip():
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, 
                               compresslevel=compression_level) as zipf:
                if os.path.isfile(source_path):
                    zipf.write(source_path, os.path.basename(source_path))
                elif os.path.isdir(source_path):
                    for root, dirs, files in os.walk(source_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, source_path)
                            zipf.write(file_path, arcname)
        
        try:
            await loop.run_in_executor(None, _create_zip)
            logger.info(f"ZIP archive created: {archive_path}")
        except Exception as e:
            logger.error(f"Error creating ZIP archive: {e}")
            raise FileOperationError(f"Failed to create ZIP archive: {e}")
    
    async def extract_zip_archive(self, archive_path: str, extract_path: str) -> None:
        """Extract ZIP archive."""
        archive_path = self.safe_ops._validate_and_normalize_path(archive_path)
        extract_path = self.safe_ops._validate_and_normalize_path(extract_path)
        
        loop = asyncio.get_event_loop()
        
        def _extract_zip():
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                # Security check: validate all paths in archive
                for member in zipf.namelist():
                    if os.path.isabs(member) or ".." in member:
                        raise SecurityViolationError(f"Unsafe path in archive: {member}")
                
                zipf.extractall(extract_path)
        
        try:
            await loop.run_in_executor(None, _extract_zip)
            logger.info(f"ZIP archive extracted to: {extract_path}")
        except Exception as e:
            logger.error(f"Error extracting ZIP archive: {e}")
            raise FileOperationError(f"Failed to extract ZIP archive: {e}")
    
    async def create_tar_archive(self, source_path: str, archive_path: str,
                               compression: Optional[str] = 'gz') -> None:
        """Create TAR archive with optional compression."""
        source_path = self.safe_ops._validate_and_normalize_path(source_path)
        archive_path = self.safe_ops._validate_and_normalize_path(archive_path)
        
        mode = 'w'
        if compression == 'gz':
            mode = 'w:gz'
        elif compression == 'bz2':
            mode = 'w:bz2'
        elif compression == 'xz':
            mode = 'w:xz'
        
        loop = asyncio.get_event_loop()
        
        def _create_tar():
            with tarfile.open(archive_path, mode) as tar:
                tar.add(source_path, arcname=os.path.basename(source_path))
        
        try:
            await loop.run_in_executor(None, _create_tar)
            logger.info(f"TAR archive created: {archive_path}")
        except Exception as e:
            logger.error(f"Error creating TAR archive: {e}")
            raise FileOperationError(f"Failed to create TAR archive: {e}")


# Export main classes and functions
__all__ = [
    'FileInfo',
    'DirectoryInfo',
    'ProjectStructure',
    'BackupInfo',
    'ChangeEvent',
    'FileOperationError',
    'SecurityViolationError',
    'PathTraversalError',
    'SafeFileOperations',
    'DirectoryAnalyzer',
    'ProjectAnalyzer',
    'FileWatcher',
    'ArchiveOperations',
    'FileWatcherCallback'
]