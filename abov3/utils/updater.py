"""
Update checker and installer for ABOV3 4 Ollama.

This module provides functionality to check for updates and install new versions
of ABOV3, ensuring users always have access to the latest features and security fixes.
"""

import json
import subprocess
import sys
from typing import Optional, Dict, Any
from urllib.request import urlopen
from urllib.error import URLError
import packaging.version

from .. import __version__


class UpdateChecker:
    """Handles checking for and installing updates."""
    
    def __init__(self):
        self.current_version = __version__
        self.pypi_url = "https://pypi.org/pypi/abov3/json"
        self.github_api_url = "https://api.github.com/repos/abov3/abov3-ollama/releases/latest"
    
    def check_for_updates(self) -> bool:
        """
        Check if updates are available.
        
        Returns:
            True if an update is available, False otherwise
        """
        try:
            latest_version = self.get_latest_version()
            if latest_version:
                current = packaging.version.parse(self.current_version)
                latest = packaging.version.parse(latest_version)
                return latest > current
        except Exception:
            # Silently fail on network errors
            pass
        
        return False
    
    def get_latest_version(self) -> Optional[str]:
        """
        Get the latest available version.
        
        Returns:
            The latest version string, or None if unable to determine
        """
        try:
            # First try PyPI
            version = self._get_version_from_pypi()
            if version:
                return version
            
            # Fallback to GitHub releases
            return self._get_version_from_github()
            
        except Exception:
            return None
    
    def _get_version_from_pypi(self) -> Optional[str]:
        """Get version from PyPI API."""
        try:
            with urlopen(self.pypi_url, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data.get("info", {}).get("version")
        except (URLError, json.JSONDecodeError, KeyError):
            return None
    
    def _get_version_from_github(self) -> Optional[str]:
        """Get version from GitHub releases API."""
        try:
            with urlopen(self.github_api_url, timeout=10) as response:
                data = json.loads(response.read().decode())
                tag_name = data.get("tag_name", "")
                # Remove 'v' prefix if present
                return tag_name.lstrip("v") if tag_name else None
        except (URLError, json.JSONDecodeError, KeyError):
            return None
    
    def get_release_notes(self, version: Optional[str] = None) -> Optional[str]:
        """
        Get release notes for a specific version.
        
        Args:
            version: Version to get notes for (defaults to latest)
            
        Returns:
            Release notes as markdown string, or None if unavailable
        """
        try:
            if version is None:
                # Get latest release notes
                url = self.github_api_url
            else:
                url = f"https://api.github.com/repos/abov3/abov3-ollama/releases/tags/v{version}"
            
            with urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data.get("body", "")
                
        except Exception:
            return None
    
    def install_update(self) -> bool:
        """
        Install the latest update using pip.
        
        Returns:
            True if installation was successful, False otherwise
        """
        try:
            # Use pip to upgrade the package
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "abov3"],
                capture_output=True,
                text=True,
                check=True
            )
            
            return result.returncode == 0
            
        except subprocess.CalledProcessError:
            return False
        except Exception:
            return False
    
    def can_auto_update(self) -> bool:
        """
        Check if automatic updates are possible in the current environment.
        
        Returns:
            True if auto-update is possible, False otherwise
        """
        try:
            # Check if we can write to the installation directory
            # This is a simplified check - in practice, you might want more sophisticated logic
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "abov3"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # If we can query the package, we likely can update it
            return result.returncode == 0
            
        except subprocess.CalledProcessError:
            return False
        except Exception:
            return False