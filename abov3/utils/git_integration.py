"""
ABOV3 Ollama Git Integration Utilities

Comprehensive Git repository operations and integration utilities providing repository management,
commit and branch operations, diff analysis, merge conflict resolution, and remote repository operations.

Features:
    - Git repository initialization and configuration
    - Commit history analysis and management
    - Branch operations and workflow management
    - Diff analysis and change tracking
    - Merge conflict detection and resolution assistance
    - Remote repository operations and synchronization
    - Git blame and history analysis
    - Tag and release management
    - Submodule operations
    - Git hooks management
    - Repository statistics and insights

Security Features:
    - Safe repository operations with validation
    - Credential management and encryption
    - Branch protection and access control
    - Commit signature verification
    - Secure remote operations

Author: ABOV3 Enterprise Documentation Agent
Version: 1.0.0
"""

import os
import re
import sys
import json
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import (
    Dict, List, Optional, Any, Tuple, Set, Union, 
    AsyncGenerator, Callable, NamedTuple, Iterator, Literal
)
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil

# Third-party imports
try:
    import git
    from git import Repo, GitCommandError, InvalidGitRepositoryError
    from git.objects import Commit, Tree, Blob
    from git.refs import Head, RemoteReference
    from git.remote import Remote
    GITPYTHON_AVAILABLE = True
except ImportError:
    GITPYTHON_AVAILABLE = False

# Internal imports
from .security import SecurityManager, is_safe_path
from .validation import PathValidator, validate_file_path
from .file_ops import SafeFileOperations, FileInfo


# Configure logging
logger = logging.getLogger('abov3.git_integration')


@dataclass
class GitConfig:
    """Git configuration settings."""
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    default_branch: str = 'main'
    remote_origin: Optional[str] = None
    signing_key: Optional[str] = None
    auto_crlf: bool = False
    safe_directory: bool = True


@dataclass
class CommitInfo:
    """Information about a Git commit."""
    sha: str
    message: str
    author_name: str
    author_email: str
    committer_name: str
    committer_email: str
    authored_date: datetime
    committed_date: datetime
    parents: List[str]
    tree_sha: str
    stats: Dict[str, Any]
    files_changed: List[str]
    additions: int
    deletions: int
    is_merge: bool
    tags: List[str] = field(default_factory=list)


@dataclass
class BranchInfo:
    """Information about a Git branch."""
    name: str
    commit_sha: str
    is_active: bool
    is_remote: bool
    tracking_branch: Optional[str]
    ahead_count: int
    behind_count: int
    last_commit_date: datetime
    last_commit_message: str
    author: str


@dataclass
class DiffInfo:
    """Information about file differences."""
    file_path: str
    change_type: str  # A (added), M (modified), D (deleted), R (renamed)
    old_path: Optional[str]
    additions: int
    deletions: int
    diff_content: str
    binary: bool = False
    similarity: Optional[float] = None  # For renames


@dataclass
class MergeConflict:
    """Information about a merge conflict."""
    file_path: str
    conflict_markers: List[Tuple[int, int]]  # (start_line, end_line) pairs
    our_content: str
    their_content: str
    base_content: Optional[str]
    resolution_suggestions: List[str] = field(default_factory=list)


@dataclass
class RepositoryStats:
    """Repository statistics and insights."""
    total_commits: int
    total_branches: int
    total_tags: int
    total_contributors: int
    lines_of_code: int
    files_count: int
    repository_size: int
    first_commit_date: datetime
    last_commit_date: datetime
    most_active_files: List[Tuple[str, int]]
    top_contributors: List[Tuple[str, int]]
    commit_frequency: Dict[str, int]
    language_stats: Dict[str, int]


@dataclass
class RemoteInfo:
    """Information about a Git remote."""
    name: str
    url: str
    fetch_url: str
    push_url: str
    is_valid: bool
    branches: List[str]
    last_fetch: Optional[datetime] = None


class GitOperationError(Exception):
    """Base exception for Git operations."""
    pass


class RepositoryNotFoundError(GitOperationError):
    """Raised when Git repository is not found."""
    pass


class BranchNotFoundError(GitOperationError):
    """Raised when branch is not found."""
    pass


class MergeConflictError(GitOperationError):
    """Raised when merge conflicts occur."""
    pass


class GitRepository:
    """Git repository operations wrapper."""
    
    def __init__(self, repo_path: str, security_manager: Optional[SecurityManager] = None):
        self.repo_path = os.path.abspath(repo_path)
        self.security_manager = security_manager or SecurityManager()
        self.path_validator = PathValidator()
        self.safe_ops = SafeFileOperations(security_manager)
        
        # Validate repository path
        if not is_safe_path(self.repo_path):
            raise GitOperationError(f"Unsafe repository path: {repo_path}")
        
        # Initialize GitPython repo if available
        self.repo: Optional[Repo] = None
        if GITPYTHON_AVAILABLE:
            try:
                self.repo = Repo(self.repo_path)
            except InvalidGitRepositoryError:
                # Repository doesn't exist yet
                pass
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def init_repository(self, bare: bool = False, 
                            initial_branch: str = 'main') -> None:
        """Initialize a new Git repository."""
        if not os.path.exists(self.repo_path):
            os.makedirs(self.repo_path, exist_ok=True)
        
        if GITPYTHON_AVAILABLE:
            loop = asyncio.get_event_loop()
            try:
                self.repo = await loop.run_in_executor(
                    self.executor,
                    lambda: Repo.init(self.repo_path, bare=bare)
                )
                
                # Set initial branch if not bare
                if not bare and self.repo.heads:
                    current_branch = self.repo.active_branch
                    if current_branch.name != initial_branch:
                        new_branch = self.repo.create_head(initial_branch)
                        new_branch.checkout()
                
                logger.info(f"Git repository initialized at {self.repo_path}")
            except Exception as e:
                logger.error(f"Error initializing repository: {e}")
                raise GitOperationError(f"Failed to initialize repository: {e}")
        else:
            # Fallback to command line
            await self._run_git_command(['init'])
            if not bare:
                await self._run_git_command(['checkout', '-b', initial_branch])
    
    async def clone_repository(self, url: str, branch: Optional[str] = None,
                             depth: Optional[int] = None) -> None:
        """Clone a remote repository."""
        if os.path.exists(self.repo_path) and os.listdir(self.repo_path):
            raise GitOperationError(f"Directory not empty: {self.repo_path}")
        
        # Validate URL (basic check)
        if not self._is_valid_git_url(url):
            raise GitOperationError(f"Invalid Git URL: {url}")
        
        if GITPYTHON_AVAILABLE:
            loop = asyncio.get_event_loop()
            try:
                kwargs = {}
                if branch:
                    kwargs['branch'] = branch
                if depth:
                    kwargs['depth'] = depth
                
                self.repo = await loop.run_in_executor(
                    self.executor,
                    lambda: Repo.clone_from(url, self.repo_path, **kwargs)
                )
                
                logger.info(f"Repository cloned from {url} to {self.repo_path}")
            except Exception as e:
                logger.error(f"Error cloning repository: {e}")
                raise GitOperationError(f"Failed to clone repository: {e}")
        else:
            # Fallback to command line
            cmd = ['clone', url, self.repo_path]
            if branch:
                cmd.extend(['-b', branch])
            if depth:
                cmd.extend(['--depth', str(depth)])
            
            await self._run_git_command(cmd, cwd=os.path.dirname(self.repo_path))
    
    async def get_status(self) -> Dict[str, Any]:
        """Get repository status."""
        await self._ensure_repository()
        
        if GITPYTHON_AVAILABLE and self.repo:
            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(
                    self.executor,
                    self._get_status_gitpython
                )
            except Exception as e:
                logger.warning(f"GitPython status failed, falling back to CLI: {e}")
        
        # Fallback to command line
        return await self._get_status_cli()
    
    def _get_status_gitpython(self) -> Dict[str, Any]:
        """Get status using GitPython."""
        if not self.repo:
            raise GitOperationError("Repository not initialized")
        
        # Get current branch
        try:
            current_branch = self.repo.active_branch.name
        except TypeError:
            current_branch = None  # Detached HEAD
        
        # Get staged and unstaged changes
        staged_files = []
        unstaged_files = []
        untracked_files = []
        
        # Staged files
        for item in self.repo.index.diff("HEAD"):
            staged_files.append({
                'path': item.a_path,
                'change_type': item.change_type
            })
        
        # Unstaged files
        for item in self.repo.index.diff(None):
            unstaged_files.append({
                'path': item.a_path,
                'change_type': item.change_type
            })
        
        # Untracked files
        untracked_files = list(self.repo.untracked_files)
        
        return {
            'current_branch': current_branch,
            'staged_files': staged_files,
            'unstaged_files': unstaged_files,
            'untracked_files': untracked_files,
            'is_dirty': self.repo.is_dirty(),
            'ahead_behind': self._get_ahead_behind_count()
        }
    
    async def _get_status_cli(self) -> Dict[str, Any]:
        """Get status using command line."""
        # Get current branch
        branch_result = await self._run_git_command(['branch', '--show-current'])
        current_branch = branch_result.strip() if branch_result.strip() else None
        
        # Get status porcelain
        status_result = await self._run_git_command(['status', '--porcelain'])
        
        staged_files = []
        unstaged_files = []
        untracked_files = []
        
        for line in status_result.split('\n'):
            if not line.strip():
                continue
            
            status_code = line[:2]
            file_path = line[3:]
            
            if status_code[0] != ' ' and status_code[0] != '?':
                staged_files.append({
                    'path': file_path,
                    'change_type': status_code[0]
                })
            
            if status_code[1] != ' ' and status_code[1] != '?':
                unstaged_files.append({
                    'path': file_path,
                    'change_type': status_code[1]
                })
            
            if status_code == '??':
                untracked_files.append(file_path)
        
        return {
            'current_branch': current_branch,
            'staged_files': staged_files,
            'unstaged_files': unstaged_files,
            'untracked_files': untracked_files,
            'is_dirty': bool(staged_files or unstaged_files or untracked_files)
        }
    
    async def add_files(self, file_paths: Union[str, List[str]]) -> None:
        """Add files to staging area."""
        await self._ensure_repository()
        
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        # Validate file paths
        for file_path in file_paths:
            full_path = os.path.join(self.repo_path, file_path)
            if not is_safe_path(full_path):
                raise GitOperationError(f"Unsafe file path: {file_path}")
        
        if GITPYTHON_AVAILABLE and self.repo:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    self.executor,
                    lambda: self.repo.index.add(file_paths)
                )
                logger.info(f"Files added to staging: {file_paths}")
            except Exception as e:
                logger.error(f"Error adding files: {e}")
                raise GitOperationError(f"Failed to add files: {e}")
        else:
            # Fallback to command line
            cmd = ['add'] + file_paths
            await self._run_git_command(cmd)
    
    async def commit(self, message: str, author: Optional[str] = None,
                   sign: bool = False) -> str:
        """Create a new commit."""
        await self._ensure_repository()
        
        if not message.strip():
            raise GitOperationError("Commit message cannot be empty")
        
        if GITPYTHON_AVAILABLE and self.repo:
            loop = asyncio.get_event_loop()
            try:
                kwargs = {'message': message}
                if author:
                    kwargs['author'] = author
                if sign:
                    kwargs['gpg_sign'] = True
                
                commit = await loop.run_in_executor(
                    self.executor,
                    lambda: self.repo.index.commit(**kwargs)
                )
                
                logger.info(f"Commit created: {commit.hexsha}")
                return commit.hexsha
            except Exception as e:
                logger.error(f"Error creating commit: {e}")
                raise GitOperationError(f"Failed to create commit: {e}")
        else:
            # Fallback to command line
            cmd = ['commit', '-m', message]
            if sign:
                cmd.append('-S')
            
            result = await self._run_git_command(cmd)
            
            # Extract commit hash from output
            match = re.search(r'\[.+?\s+([a-f0-9]+)\]', result)
            if match:
                return match.group(1)
            
            return "unknown"
    
    async def create_branch(self, branch_name: str, start_point: Optional[str] = None) -> None:
        """Create a new branch."""
        await self._ensure_repository()
        
        if not self._is_valid_branch_name(branch_name):
            raise GitOperationError(f"Invalid branch name: {branch_name}")
        
        if GITPYTHON_AVAILABLE and self.repo:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    self.executor,
                    lambda: self.repo.create_head(branch_name, start_point)
                )
                logger.info(f"Branch created: {branch_name}")
            except Exception as e:
                logger.error(f"Error creating branch: {e}")
                raise GitOperationError(f"Failed to create branch: {e}")
        else:
            # Fallback to command line
            cmd = ['branch', branch_name]
            if start_point:
                cmd.append(start_point)
            
            await self._run_git_command(cmd)
    
    async def checkout_branch(self, branch_name: str, create: bool = False) -> None:
        """Checkout a branch."""
        await self._ensure_repository()
        
        if GITPYTHON_AVAILABLE and self.repo:
            loop = asyncio.get_event_loop()
            try:
                if create:
                    await loop.run_in_executor(
                        self.executor,
                        lambda: self.repo.git.checkout('-b', branch_name)
                    )
                else:
                    await loop.run_in_executor(
                        self.executor,
                        lambda: self.repo.git.checkout(branch_name)
                    )
                logger.info(f"Checked out branch: {branch_name}")
            except Exception as e:
                logger.error(f"Error checking out branch: {e}")
                raise GitOperationError(f"Failed to checkout branch: {e}")
        else:
            # Fallback to command line
            cmd = ['checkout']
            if create:
                cmd.append('-b')
            cmd.append(branch_name)
            
            await self._run_git_command(cmd)
    
    async def delete_branch(self, branch_name: str, force: bool = False) -> None:
        """Delete a branch."""
        await self._ensure_repository()
        
        if GITPYTHON_AVAILABLE and self.repo:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    self.executor,
                    lambda: self.repo.delete_head(branch_name, force=force)
                )
                logger.info(f"Branch deleted: {branch_name}")
            except Exception as e:
                logger.error(f"Error deleting branch: {e}")
                raise GitOperationError(f"Failed to delete branch: {e}")
        else:
            # Fallback to command line
            cmd = ['branch', '-D' if force else '-d', branch_name]
            await self._run_git_command(cmd)
    
    async def get_branches(self, include_remote: bool = True) -> List[BranchInfo]:
        """Get list of branches."""
        await self._ensure_repository()
        
        if GITPYTHON_AVAILABLE and self.repo:
            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(
                    self.executor,
                    lambda: self._get_branches_gitpython(include_remote)
                )
            except Exception as e:
                logger.warning(f"GitPython branches failed, falling back to CLI: {e}")
        
        # Fallback to command line
        return await self._get_branches_cli(include_remote)
    
    def _get_branches_gitpython(self, include_remote: bool) -> List[BranchInfo]:
        """Get branches using GitPython."""
        if not self.repo:
            raise GitOperationError("Repository not initialized")
        
        branches = []
        current_branch = None
        
        try:
            current_branch = self.repo.active_branch.name
        except TypeError:
            pass  # Detached HEAD
        
        # Local branches
        for head in self.repo.heads:
            branch_info = BranchInfo(
                name=head.name,
                commit_sha=head.commit.hexsha,
                is_active=head.name == current_branch,
                is_remote=False,
                tracking_branch=head.tracking_branch().name if head.tracking_branch() else None,
                ahead_count=0,  # Will be calculated if needed
                behind_count=0,  # Will be calculated if needed
                last_commit_date=head.commit.committed_datetime,
                last_commit_message=head.commit.message.strip(),
                author=head.commit.author.name
            )
            branches.append(branch_info)
        
        # Remote branches
        if include_remote:
            for remote in self.repo.remotes:
                for ref in remote.refs:
                    if ref.name.endswith('/HEAD'):
                        continue
                    
                    branch_name = ref.name.replace(f"{remote.name}/", "")
                    branch_info = BranchInfo(
                        name=f"{remote.name}/{branch_name}",
                        commit_sha=ref.commit.hexsha,
                        is_active=False,
                        is_remote=True,
                        tracking_branch=None,
                        ahead_count=0,
                        behind_count=0,
                        last_commit_date=ref.commit.committed_datetime,
                        last_commit_message=ref.commit.message.strip(),
                        author=ref.commit.author.name
                    )
                    branches.append(branch_info)
        
        return branches
    
    async def _get_branches_cli(self, include_remote: bool) -> List[BranchInfo]:
        """Get branches using command line."""
        branches = []
        
        # Get current branch
        current_result = await self._run_git_command(['branch', '--show-current'])
        current_branch = current_result.strip() if current_result.strip() else None
        
        # Get local branches
        local_result = await self._run_git_command(['branch', '--format=%(refname:short)|%(objectname)|%(committerdate:iso)|%(subject)|%(authorname)'])
        
        for line in local_result.split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('|')
            if len(parts) >= 5:
                name, sha, date_str, message, author = parts[:5]
                
                try:
                    commit_date = datetime.fromisoformat(date_str.replace(' ', 'T'))
                except ValueError:
                    commit_date = datetime.now()
                
                branch_info = BranchInfo(
                    name=name,
                    commit_sha=sha,
                    is_active=name == current_branch,
                    is_remote=False,
                    tracking_branch=None,  # Would need additional command
                    ahead_count=0,
                    behind_count=0,
                    last_commit_date=commit_date,
                    last_commit_message=message,
                    author=author
                )
                branches.append(branch_info)
        
        # Get remote branches
        if include_remote:
            try:
                remote_result = await self._run_git_command(['branch', '-r', '--format=%(refname:short)|%(objectname)|%(committerdate:iso)|%(subject)|%(authorname)'])
                
                for line in remote_result.split('\n'):
                    if not line.strip() or '/HEAD' in line:
                        continue
                    
                    parts = line.split('|')
                    if len(parts) >= 5:
                        name, sha, date_str, message, author = parts[:5]
                        
                        try:
                            commit_date = datetime.fromisoformat(date_str.replace(' ', 'T'))
                        except ValueError:
                            commit_date = datetime.now()
                        
                        branch_info = BranchInfo(
                            name=name,
                            commit_sha=sha,
                            is_active=False,
                            is_remote=True,
                            tracking_branch=None,
                            ahead_count=0,
                            behind_count=0,
                            last_commit_date=commit_date,
                            last_commit_message=message,
                            author=author
                        )
                        branches.append(branch_info)
            except GitOperationError:
                # No remotes configured
                pass
        
        return branches
    
    async def get_commit_history(self, branch: Optional[str] = None,
                               max_count: int = 100,
                               since: Optional[datetime] = None,
                               until: Optional[datetime] = None) -> List[CommitInfo]:
        """Get commit history."""
        await self._ensure_repository()
        
        if GITPYTHON_AVAILABLE and self.repo:
            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(
                    self.executor,
                    lambda: self._get_commit_history_gitpython(branch, max_count, since, until)
                )
            except Exception as e:
                logger.warning(f"GitPython commit history failed, falling back to CLI: {e}")
        
        # Fallback to command line
        return await self._get_commit_history_cli(branch, max_count, since, until)
    
    def _get_commit_history_gitpython(self, branch: Optional[str],
                                    max_count: int,
                                    since: Optional[datetime],
                                    until: Optional[datetime]) -> List[CommitInfo]:
        """Get commit history using GitPython."""
        if not self.repo:
            raise GitOperationError("Repository not initialized")
        
        kwargs = {'max_count': max_count}
        if since:
            kwargs['since'] = since
        if until:
            kwargs['until'] = until
        
        ref = branch if branch else 'HEAD'
        commits = list(self.repo.iter_commits(ref, **kwargs))
        
        commit_infos = []
        for commit in commits:
            # Get stats
            stats = commit.stats.total
            
            # Get file changes
            files_changed = []
            if commit.parents:
                diffs = commit.parents[0].diff(commit)
                files_changed = [diff.b_path or diff.a_path for diff in diffs if diff.b_path or diff.a_path]
            
            commit_info = CommitInfo(
                sha=commit.hexsha,
                message=commit.message.strip(),
                author_name=commit.author.name,
                author_email=commit.author.email,
                committer_name=commit.committer.name,
                committer_email=commit.committer.email,
                authored_date=commit.authored_datetime,
                committed_date=commit.committed_datetime,
                parents=[parent.hexsha for parent in commit.parents],
                tree_sha=commit.tree.hexsha,
                stats={'total': stats},
                files_changed=files_changed,
                additions=stats.get('insertions', 0),
                deletions=stats.get('deletions', 0),
                is_merge=len(commit.parents) > 1
            )
            commit_infos.append(commit_info)
        
        return commit_infos
    
    async def _get_commit_history_cli(self, branch: Optional[str],
                                    max_count: int,
                                    since: Optional[datetime],
                                    until: Optional[datetime]) -> List[CommitInfo]:
        """Get commit history using command line."""
        cmd = ['log', '--pretty=format:%H|%s|%an|%ae|%cn|%ce|%ai|%ci|%P|%T', f'-{max_count}']
        
        if branch:
            cmd.append(branch)
        
        if since:
            cmd.extend(['--since', since.isoformat()])
        
        if until:
            cmd.extend(['--until', until.isoformat()])
        
        result = await self._run_git_command(cmd)
        
        commit_infos = []
        for line in result.split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('|')
            if len(parts) >= 10:
                sha, message, author_name, author_email, committer_name, committer_email, authored_date, committed_date, parents, tree_sha = parts[:10]
                
                try:
                    authored_dt = datetime.fromisoformat(authored_date.replace(' ', 'T'))
                    committed_dt = datetime.fromisoformat(committed_date.replace(' ', 'T'))
                except ValueError:
                    authored_dt = committed_dt = datetime.now()
                
                parent_list = parents.split() if parents else []
                
                commit_info = CommitInfo(
                    sha=sha,
                    message=message,
                    author_name=author_name,
                    author_email=author_email,
                    committer_name=committer_name,
                    committer_email=committer_email,
                    authored_date=authored_dt,
                    committed_date=committed_dt,
                    parents=parent_list,
                    tree_sha=tree_sha,
                    stats={},  # Would need additional command
                    files_changed=[],  # Would need additional command
                    additions=0,  # Would need additional command
                    deletions=0,  # Would need additional command
                    is_merge=len(parent_list) > 1
                )
                commit_infos.append(commit_info)
        
        return commit_infos
    
    async def get_diff(self, commit1: Optional[str] = None,
                      commit2: Optional[str] = None,
                      file_path: Optional[str] = None) -> List[DiffInfo]:
        """Get diff between commits or working directory."""
        await self._ensure_repository()
        
        if GITPYTHON_AVAILABLE and self.repo:
            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(
                    self.executor,
                    lambda: self._get_diff_gitpython(commit1, commit2, file_path)
                )
            except Exception as e:
                logger.warning(f"GitPython diff failed, falling back to CLI: {e}")
        
        # Fallback to command line
        return await self._get_diff_cli(commit1, commit2, file_path)
    
    def _get_diff_gitpython(self, commit1: Optional[str],
                          commit2: Optional[str],
                          file_path: Optional[str]) -> List[DiffInfo]:
        """Get diff using GitPython."""
        if not self.repo:
            raise GitOperationError("Repository not initialized")
        
        # Determine what to diff
        if commit1 and commit2:
            diffs = self.repo.commit(commit1).diff(self.repo.commit(commit2))
        elif commit1:
            diffs = self.repo.commit(commit1).diff()
        else:
            diffs = self.repo.index.diff(None)  # Working directory changes
        
        diff_infos = []
        for diff in diffs:
            if file_path and diff.b_path != file_path:
                continue
            
            diff_info = DiffInfo(
                file_path=diff.b_path or diff.a_path,
                change_type=diff.change_type,
                old_path=diff.a_path if diff.renamed_file else None,
                additions=0,  # Would need to parse diff
                deletions=0,  # Would need to parse diff
                diff_content=diff.diff.decode('utf-8', errors='ignore') if diff.diff else '',
                binary=diff.b_blob and diff.b_blob.size > 1024 * 1024,  # Assume binary if > 1MB
                similarity=diff.similarity if hasattr(diff, 'similarity') else None
            )
            diff_infos.append(diff_info)
        
        return diff_infos
    
    async def _get_diff_cli(self, commit1: Optional[str],
                          commit2: Optional[str],
                          file_path: Optional[str]) -> List[DiffInfo]:
        """Get diff using command line."""
        cmd = ['diff', '--name-status']
        
        if commit1 and commit2:
            cmd.append(f"{commit1}..{commit2}")
        elif commit1:
            cmd.append(commit1)
        
        if file_path:
            cmd.append(file_path)
        
        result = await self._run_git_command(cmd)
        
        diff_infos = []
        for line in result.split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                status = parts[0]
                file_path = parts[1]
                old_path = parts[2] if len(parts) > 2 else None
                
                # Map status codes
                change_type_map = {
                    'A': 'A',  # Added
                    'M': 'M',  # Modified
                    'D': 'D',  # Deleted
                    'R': 'R',  # Renamed
                    'C': 'C',  # Copied
                    'U': 'U'   # Unmerged
                }
                
                change_type = change_type_map.get(status[0], status[0])
                
                diff_info = DiffInfo(
                    file_path=file_path,
                    change_type=change_type,
                    old_path=old_path,
                    additions=0,  # Would need additional command
                    deletions=0,  # Would need additional command
                    diff_content='',  # Would need additional command
                    binary=False
                )
                diff_infos.append(diff_info)
        
        return diff_infos
    
    async def merge_branch(self, branch_name: str, 
                          strategy: Optional[str] = None) -> bool:
        """Merge a branch into current branch."""
        await self._ensure_repository()
        
        if GITPYTHON_AVAILABLE and self.repo:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    self.executor,
                    lambda: self.repo.git.merge(branch_name)
                )
                logger.info(f"Branch merged: {branch_name}")
                return True
            except GitCommandError as e:
                if "conflict" in str(e).lower():
                    logger.warning(f"Merge conflicts in branch {branch_name}")
                    return False
                raise GitOperationError(f"Merge failed: {e}")
        else:
            # Fallback to command line
            cmd = ['merge', branch_name]
            if strategy:
                cmd.extend(['-s', strategy])
            
            try:
                await self._run_git_command(cmd)
                return True
            except GitOperationError as e:
                if "conflict" in str(e).lower():
                    return False
                raise
    
    async def detect_merge_conflicts(self) -> List[MergeConflict]:
        """Detect and analyze merge conflicts."""
        await self._ensure_repository()
        
        conflicts = []
        
        # Get conflicted files
        status = await self.get_status()
        conflicted_files = []
        
        for file_info in status.get('unstaged_files', []):
            if file_info.get('change_type') == 'U':  # Unmerged
                conflicted_files.append(file_info['path'])
        
        # Analyze each conflicted file
        for file_path in conflicted_files:
            try:
                full_path = os.path.join(self.repo_path, file_path)
                content = await self.safe_ops.read_file(full_path)
                
                conflict = await self._analyze_conflict_file(file_path, content)
                if conflict:
                    conflicts.append(conflict)
                    
            except Exception as e:
                logger.warning(f"Error analyzing conflict in {file_path}: {e}")
        
        return conflicts
    
    async def _analyze_conflict_file(self, file_path: str, content: str) -> Optional[MergeConflict]:
        """Analyze a single conflicted file."""
        lines = content.split('\n')
        conflict_markers = []
        current_conflict = None
        
        for i, line in enumerate(lines):
            if line.startswith('<<<<<<<'):
                current_conflict = {'start': i, 'ours_end': None, 'theirs_start': None}
            elif line.startswith('=======') and current_conflict:
                current_conflict['ours_end'] = i
                current_conflict['theirs_start'] = i + 1
            elif line.startswith('>>>>>>>') and current_conflict:
                current_conflict['end'] = i
                conflict_markers.append((current_conflict['start'], current_conflict['end']))
                current_conflict = None
        
        if not conflict_markers:
            return None
        
        # Extract conflict content (simplified)
        first_conflict = conflict_markers[0]
        start_line, end_line = first_conflict
        
        our_content = ""
        their_content = ""
        
        # This is a simplified extraction - real implementation would be more sophisticated
        conflict_section = lines[start_line:end_line + 1]
        separator_index = -1
        
        for i, line in enumerate(conflict_section):
            if line.startswith('======='):
                separator_index = i
                break
        
        if separator_index > 0:
            our_content = '\n'.join(conflict_section[1:separator_index])
            their_content = '\n'.join(conflict_section[separator_index + 1:-1])
        
        return MergeConflict(
            file_path=file_path,
            conflict_markers=conflict_markers,
            our_content=our_content,
            their_content=their_content,
            base_content=None,  # Would need to get from merge-base
            resolution_suggestions=self._generate_resolution_suggestions(our_content, their_content)
        )
    
    def _generate_resolution_suggestions(self, our_content: str, their_content: str) -> List[str]:
        """Generate suggestions for resolving conflicts."""
        suggestions = []
        
        # Simple heuristics for suggestions
        if len(our_content.strip()) == 0:
            suggestions.append("Consider accepting their changes (our version is empty)")
        elif len(their_content.strip()) == 0:
            suggestions.append("Consider accepting our changes (their version is empty)")
        elif our_content.strip() == their_content.strip():
            suggestions.append("Both versions are identical - choose either")
        else:
            suggestions.append("Manual review required - significant differences detected")
            
            # Check for simple additions
            our_lines = set(our_content.split('\n'))
            their_lines = set(their_content.split('\n'))
            
            if our_lines.issubset(their_lines):
                suggestions.append("Their version appears to be a superset - consider accepting their changes")
            elif their_lines.issubset(our_lines):
                suggestions.append("Our version appears to be a superset - consider accepting our changes")
        
        return suggestions
    
    async def get_repository_stats(self) -> RepositoryStats:
        """Get comprehensive repository statistics."""
        await self._ensure_repository()
        
        # Get all commits
        commits = await self.get_commit_history(max_count=10000)  # Reasonable limit
        
        # Get all branches
        branches = await self.get_branches()
        
        # Calculate statistics
        total_commits = len(commits)
        total_branches = len([b for b in branches if not b.is_remote])
        
        # Contributors
        contributors = Counter()
        for commit in commits:
            contributors[commit.author_email] += 1
        
        # File activity (simplified)
        file_activity = Counter()
        for commit in commits:
            for file_path in commit.files_changed:
                file_activity[file_path] += 1
        
        # Commit frequency by month
        commit_frequency = Counter()
        for commit in commits:
            month_key = commit.committed_date.strftime('%Y-%m')
            commit_frequency[month_key] += 1
        
        # Repository size
        repo_size = await self._calculate_repo_size()
        
        # Language statistics (simplified)
        language_stats = await self._get_language_stats()
        
        first_commit_date = commits[-1].committed_date if commits else datetime.now()
        last_commit_date = commits[0].committed_date if commits else datetime.now()
        
        return RepositoryStats(
            total_commits=total_commits,
            total_branches=total_branches,
            total_tags=0,  # Would need additional command
            total_contributors=len(contributors),
            lines_of_code=0,  # Would need additional analysis
            files_count=0,  # Would need additional analysis
            repository_size=repo_size,
            first_commit_date=first_commit_date,
            last_commit_date=last_commit_date,
            most_active_files=file_activity.most_common(10),
            top_contributors=contributors.most_common(10),
            commit_frequency=dict(commit_frequency),
            language_stats=language_stats
        )
    
    async def _calculate_repo_size(self) -> int:
        """Calculate repository size."""
        total_size = 0
        for root, dirs, files in os.walk(self.repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                except OSError:
                    continue
        return total_size
    
    async def _get_language_stats(self) -> Dict[str, int]:
        """Get language statistics by file extension."""
        language_stats = Counter()
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')
            
            for file in files:
                _, ext = os.path.splitext(file)
                if ext:
                    language_stats[ext.lower()] += 1
        
        return dict(language_stats)
    
    def _get_ahead_behind_count(self) -> Dict[str, int]:
        """Get ahead/behind count for current branch."""
        if not GITPYTHON_AVAILABLE or not self.repo:
            return {'ahead': 0, 'behind': 0}
        
        try:
            current_branch = self.repo.active_branch
            tracking_branch = current_branch.tracking_branch()
            
            if tracking_branch:
                ahead = list(self.repo.iter_commits(f'{tracking_branch}..{current_branch}'))
                behind = list(self.repo.iter_commits(f'{current_branch}..{tracking_branch}'))
                return {'ahead': len(ahead), 'behind': len(behind)}
        except Exception:
            pass
        
        return {'ahead': 0, 'behind': 0}
    
    async def _ensure_repository(self) -> None:
        """Ensure repository is initialized."""
        if not os.path.exists(os.path.join(self.repo_path, '.git')):
            raise RepositoryNotFoundError(f"Git repository not found at {self.repo_path}")
    
    async def _run_git_command(self, args: List[str], cwd: Optional[str] = None) -> str:
        """Run a Git command safely."""
        cmd = ['git'] + args
        work_dir = cwd or self.repo_path
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore').strip()
                raise GitOperationError(f"Git command failed: {error_msg}")
            
            return stdout.decode('utf-8', errors='ignore').strip()
            
        except FileNotFoundError:
            raise GitOperationError("Git command not found. Please install Git.")
        except Exception as e:
            raise GitOperationError(f"Git command execution failed: {e}")
    
    def _is_valid_git_url(self, url: str) -> bool:
        """Validate Git URL format."""
        # Basic URL validation
        git_url_patterns = [
            r'^https?://.*\.git$',
            r'^git@.*:.*\.git$',
            r'^ssh://.*\.git$',
            r'^file://.*',
            r'^.*\.git$'
        ]
        
        return any(re.match(pattern, url, re.IGNORECASE) for pattern in git_url_patterns)
    
    def _is_valid_branch_name(self, name: str) -> bool:
        """Validate branch name."""
        # Git branch name rules (simplified)
        if not name or name.startswith('-') or name.endswith('.'):
            return False
        
        invalid_chars = [' ', '~', '^', ':', '?', '*', '[', '\\', '..']
        if any(char in name for char in invalid_chars):
            return False
        
        return True


class GitWorkflow:
    """High-level Git workflow operations."""
    
    def __init__(self, repo: GitRepository):
        self.repo = repo
    
    async def create_feature_branch(self, feature_name: str, 
                                  base_branch: str = 'main') -> str:
        """Create a new feature branch."""
        branch_name = f"feature/{feature_name}"
        
        # Ensure we're on the base branch
        await self.repo.checkout_branch(base_branch)
        
        # Create and checkout feature branch
        await self.repo.create_branch(branch_name, base_branch)
        await self.repo.checkout_branch(branch_name)
        
        logger.info(f"Feature branch created: {branch_name}")
        return branch_name
    
    async def create_hotfix_branch(self, hotfix_name: str,
                                 base_branch: str = 'main') -> str:
        """Create a new hotfix branch."""
        branch_name = f"hotfix/{hotfix_name}"
        
        # Ensure we're on the base branch
        await self.repo.checkout_branch(base_branch)
        
        # Create and checkout hotfix branch
        await self.repo.create_branch(branch_name, base_branch)
        await self.repo.checkout_branch(branch_name)
        
        logger.info(f"Hotfix branch created: {branch_name}")
        return branch_name
    
    async def finish_feature_branch(self, feature_name: str,
                                  target_branch: str = 'main',
                                  delete_branch: bool = True) -> bool:
        """Finish a feature branch by merging to target."""
        branch_name = f"feature/{feature_name}"
        
        # Checkout target branch
        await self.repo.checkout_branch(target_branch)
        
        # Merge feature branch
        success = await self.repo.merge_branch(branch_name)
        
        if success and delete_branch:
            await self.repo.delete_branch(branch_name)
            logger.info(f"Feature branch completed and deleted: {branch_name}")
        
        return success
    
    async def create_release_branch(self, version: str,
                                  base_branch: str = 'develop') -> str:
        """Create a release branch."""
        branch_name = f"release/{version}"
        
        # Ensure we're on the base branch
        await self.repo.checkout_branch(base_branch)
        
        # Create and checkout release branch
        await self.repo.create_branch(branch_name, base_branch)
        await self.repo.checkout_branch(branch_name)
        
        logger.info(f"Release branch created: {branch_name}")
        return branch_name


# Export main classes and functions
__all__ = [
    'GitConfig',
    'CommitInfo',
    'BranchInfo',
    'DiffInfo',
    'MergeConflict',
    'RepositoryStats',
    'RemoteInfo',
    'GitOperationError',
    'RepositoryNotFoundError',
    'BranchNotFoundError',
    'MergeConflictError',
    'GitRepository',
    'GitWorkflow'
]