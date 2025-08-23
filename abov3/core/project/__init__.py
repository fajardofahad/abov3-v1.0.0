"""
Project management for ABOV3.

This module provides comprehensive project directory selection and file management
capabilities, transforming ABOV3 into a true coding assistant that can work with
user projects.
"""

from .manager import ProjectManager, ProjectContext, ProjectState

__all__ = ['ProjectManager', 'ProjectContext', 'ProjectState']