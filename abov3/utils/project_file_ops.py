"""
Project-aware file operations for ABOV3.

This module extends the base file operations with project-specific functionality,
providing context-aware file handling for coding assistance tasks.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from .file_ops import SafeFileOperations, DirectoryAnalyzer, ProjectAnalyzer, FileInfo
from .security import SecurityManager
from ..core.config import get_config


logger = logging.getLogger(__name__)


@dataclass
class CodeAnalysis:
    """Code analysis result."""
    file_path: str
    language: str
    lines_of_code: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity_score: float
    issues: List[str]
    suggestions: List[str]


class ProjectFileOperations:
    """
    Project-aware file operations with enhanced capabilities for coding assistance.
    
    Features:
    - Context-aware file operations
    - Code analysis and parsing
    - Smart file suggestions
    - Template generation
    - Refactoring support
    """
    
    def __init__(self, 
                 security_manager: Optional[SecurityManager] = None,
                 project_root: Optional[str] = None):
        """Initialize project file operations."""
        self.config = get_config()
        self.security_manager = security_manager or SecurityManager()
        self.project_root = project_root
        
        # Initialize base operations
        self.safe_ops = SafeFileOperations(security_manager=self.security_manager)
        self.dir_analyzer = DirectoryAnalyzer(self.safe_ops)
        self.project_analyzer = ProjectAnalyzer(self.safe_ops)
        
        # Code analysis patterns
        self.language_patterns = {
            'python': {
                'extensions': ['.py'],
                'function_patterns': [r'def\s+(\w+)', r'async\s+def\s+(\w+)'],
                'class_patterns': [r'class\s+(\w+)'],
                'import_patterns': [r'import\s+([\w.]+)', r'from\s+([\w.]+)\s+import']
            },
            'javascript': {
                'extensions': ['.js', '.jsx', '.ts', '.tsx'],
                'function_patterns': [r'function\s+(\w+)', r'(\w+)\s*:\s*function', r'(\w+)\s*=>\s*{'],
                'class_patterns': [r'class\s+(\w+)'],
                'import_patterns': [r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]', r'require\([\'"]([^\'"]+)[\'"]\)']
            },
            'java': {
                'extensions': ['.java'],
                'function_patterns': [r'public\s+\w+\s+(\w+)\s*\(', r'private\s+\w+\s+(\w+)\s*\('],
                'class_patterns': [r'class\s+(\w+)', r'interface\s+(\w+)'],
                'import_patterns': [r'import\s+([\w.]+)']
            }
        }
    
    def set_project_root(self, project_root: str) -> None:
        """Set the project root directory."""
        self.project_root = os.path.abspath(project_root)
        logger.info(f"Project root set to: {self.project_root}")
    
    async def smart_read_file(self, 
                             file_path: str, 
                             include_analysis: bool = True) -> Dict[str, Any]:
        """
        Smart file reading with optional code analysis.
        
        Args:
            file_path: File path (relative to project root or absolute)
            include_analysis: Whether to include code analysis
            
        Returns:
            Dictionary with file content and analysis
        """
        # Resolve path
        resolved_path = self._resolve_project_path(file_path)
        
        # Read file content
        content = await self.safe_ops.read_file(resolved_path)
        file_info = await self.safe_ops.get_file_info(resolved_path)
        
        result = {
            'path': file_path,
            'absolute_path': resolved_path,
            'content': content,
            'size': file_info.size,
            'modified': file_info.modified.isoformat(),
            'extension': file_info.extension,
            'mime_type': file_info.mime_type
        }
        
        # Add code analysis if requested
        if include_analysis:
            analysis = await self.analyze_code(resolved_path, content)
            result['analysis'] = analysis
        
        return result
    
    async def smart_write_file(self, 
                              file_path: str, 
                              content: str,
                              auto_format: bool = True,
                              create_backup: bool = True) -> Dict[str, Any]:
        """
        Smart file writing with optional formatting and validation.
        
        Args:
            file_path: File path (relative to project root or absolute)
            content: Content to write
            auto_format: Whether to auto-format the code
            create_backup: Whether to create a backup
            
        Returns:
            Dictionary with write operation results
        """
        # Resolve path
        resolved_path = self._resolve_project_path(file_path)
        
        # Auto-format if requested
        if auto_format:
            content = await self._format_code(resolved_path, content)
        
        # Validate content
        validation_result = await self._validate_code(resolved_path, content)
        
        # Write file
        await self.safe_ops.write_file(resolved_path, content, backup=create_backup)
        
        # Get updated file info
        file_info = await self.safe_ops.get_file_info(resolved_path)
        
        return {
            'path': file_path,
            'absolute_path': resolved_path,
            'size': file_info.size,
            'modified': file_info.modified.isoformat(),
            'validation': validation_result,
            'formatted': auto_format
        }
    
    async def analyze_code(self, file_path: str, content: str = None) -> CodeAnalysis:
        """
        Analyze code file for structure and complexity.
        
        Args:
            file_path: Path to the code file
            content: File content (will be read if not provided)
            
        Returns:
            CodeAnalysis object with analysis results
        """
        if content is None:
            content = await self.safe_ops.read_file(file_path)
        
        # Detect language
        language = self._detect_language(file_path)
        
        # Initialize analysis
        analysis = CodeAnalysis(
            file_path=file_path,
            language=language,
            lines_of_code=0,
            functions=[],
            classes=[],
            imports=[],
            complexity_score=0.0,
            issues=[],
            suggestions=[]
        )
        
        if language not in self.language_patterns:
            analysis.issues.append(f"Unsupported language: {language}")
            return analysis
        
        lines = content.split('\n')
        analysis.lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        # Extract code elements
        patterns = self.language_patterns[language]
        
        import re
        
        # Extract functions
        for pattern in patterns['function_patterns']:
            matches = re.findall(pattern, content, re.MULTILINE)
            analysis.functions.extend(matches)
        
        # Extract classes
        for pattern in patterns['class_patterns']:
            matches = re.findall(pattern, content, re.MULTILINE)
            analysis.classes.extend(matches)
        
        # Extract imports
        for pattern in patterns['import_patterns']:
            matches = re.findall(pattern, content, re.MULTILINE)
            analysis.imports.extend(matches)
        
        # Calculate complexity (simple heuristic)
        complexity_factors = {
            'if': 1, 'elif': 1, 'else': 1,
            'for': 2, 'while': 2,
            'try': 1, 'except': 1, 'finally': 1,
            'with': 1, 'and': 0.5, 'or': 0.5
        }
        
        complexity_score = 0
        for keyword, weight in complexity_factors.items():
            count = len(re.findall(r'\b' + keyword + r'\b', content))
            complexity_score += count * weight
        
        analysis.complexity_score = complexity_score / max(analysis.lines_of_code, 1) * 100
        
        # Add suggestions based on analysis
        if analysis.complexity_score > 20:
            analysis.suggestions.append("Consider breaking down complex functions")
        
        if len(analysis.functions) > 20:
            analysis.suggestions.append("File contains many functions - consider splitting")
        
        if analysis.lines_of_code > 500:
            analysis.suggestions.append("Large file - consider refactoring into smaller modules")
        
        return analysis
    
    async def suggest_files(self, 
                           query: str, 
                           file_type: str = "any",
                           limit: int = 10) -> List[Dict[str, Any]]:
        """
        Suggest relevant files based on query and context.
        
        Args:
            query: Search query or context
            file_type: Type of files to suggest ('source', 'config', 'doc', 'any')
            limit: Maximum number of suggestions
            
        Returns:
            List of file suggestions with relevance scores
        """
        if not self.project_root:
            raise RuntimeError("No project root set")
        
        suggestions = []
        
        # Define file type patterns
        type_patterns = {
            'source': ['*.py', '*.js', '*.ts', '*.java', '*.cpp', '*.c', '*.go', '*.rs'],
            'config': ['*.json', '*.yaml', '*.yml', '*.toml', '*.ini', '*.cfg'],
            'doc': ['*.md', '*.rst', '*.txt', 'README*', 'CHANGELOG*'],
            'any': ['*']
        }
        
        patterns = type_patterns.get(file_type, ['*'])
        
        # Find matching files
        for pattern in patterns:
            files = await self.dir_analyzer.find_files(self.project_root, pattern, recursive=True)
            
            for file_path in files:
                try:
                    # Calculate relevance score
                    relative_path = os.path.relpath(file_path, self.project_root)
                    relevance_score = self._calculate_file_relevance(relative_path, query)
                    
                    if relevance_score > 0:
                        file_info = await self.safe_ops.get_file_info(file_path)
                        
                        suggestions.append({
                            'path': relative_path,
                            'absolute_path': file_path,
                            'relevance_score': relevance_score,
                            'size': file_info.size,
                            'modified': file_info.modified.isoformat(),
                            'type': file_info.extension or 'no_extension'
                        })
                
                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {e}")
                    continue
        
        # Sort by relevance and return top results
        suggestions.sort(key=lambda x: x['relevance_score'], reverse=True)
        return suggestions[:limit]
    
    async def create_from_template(self, 
                                  template_name: str, 
                                  target_path: str,
                                  variables: Dict[str, str] = None) -> str:
        """
        Create a file from a template.
        
        Args:
            template_name: Name of the template to use
            target_path: Target file path
            variables: Variables to substitute in template
            
        Returns:
            Path to created file
        """
        variables = variables or {}
        
        # Load template
        template_content = self._get_template(template_name)
        
        # Substitute variables
        for var_name, var_value in variables.items():
            template_content = template_content.replace(f"{{{{ {var_name} }}}}", var_value)
        
        # Resolve target path
        resolved_path = self._resolve_project_path(target_path)
        
        # Create file
        await self.safe_ops.write_file(resolved_path, template_content, backup=False)
        
        logger.info(f"Created file from template '{template_name}': {target_path}")
        return resolved_path
    
    async def refactor_file(self, 
                           file_path: str,
                           operation: str,
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform refactoring operations on a file.
        
        Args:
            file_path: Path to file to refactor
            operation: Refactoring operation ('rename_function', 'extract_method', etc.)
            params: Operation parameters
            
        Returns:
            Refactoring results
        """
        resolved_path = self._resolve_project_path(file_path)
        content = await self.safe_ops.read_file(resolved_path)
        
        # Perform refactoring based on operation
        if operation == 'rename_function':
            new_content = self._rename_function(content, params['old_name'], params['new_name'])
        elif operation == 'extract_method':
            new_content = self._extract_method(content, params)
        elif operation == 'add_docstring':
            new_content = self._add_docstring(content, params)
        else:
            raise ValueError(f"Unsupported refactoring operation: {operation}")
        
        # Write refactored content
        await self.safe_ops.write_file(resolved_path, new_content, backup=True)
        
        return {
            'operation': operation,
            'file_path': file_path,
            'changes_made': True,
            'backup_created': True
        }
    
    def _resolve_project_path(self, file_path: str) -> str:
        """Resolve file path relative to project root."""
        if os.path.isabs(file_path):
            return file_path
        
        if self.project_root:
            return os.path.join(self.project_root, file_path)
        
        return os.path.abspath(file_path)
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        extension = os.path.splitext(file_path)[1].lower()
        
        for language, info in self.language_patterns.items():
            if extension in info['extensions']:
                return language
        
        return 'unknown'
    
    async def _format_code(self, file_path: str, content: str) -> str:
        """Auto-format code content."""
        language = self._detect_language(file_path)
        
        # Simple formatting rules (can be enhanced with proper formatters)
        if language == 'python':
            # Basic Python formatting
            lines = content.split('\n')
            formatted_lines = []
            
            for line in lines:
                stripped = line.strip()
                if stripped:
                    # Basic indentation fixing
                    if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'with ')):
                        formatted_lines.append(stripped)
                    elif stripped.startswith(('else:', 'elif ', 'except:', 'finally:')):
                        formatted_lines.append(stripped)
                    else:
                        # Preserve existing indentation for now
                        formatted_lines.append(line)
                else:
                    formatted_lines.append('')
            
            return '\n'.join(formatted_lines)
        
        # Return original content if no formatter available
        return content
    
    async def _validate_code(self, file_path: str, content: str) -> Dict[str, Any]:
        """Validate code syntax and structure."""
        language = self._detect_language(file_path)
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'language': language
        }
        
        if language == 'python':
            # Basic Python validation
            try:
                import ast
                ast.parse(content)
            except SyntaxError as e:
                validation['valid'] = False
                validation['errors'].append(f"Syntax error at line {e.lineno}: {e.msg}")
        
        # Add more validation rules as needed
        
        return validation
    
    def _calculate_file_relevance(self, file_path: str, query: str) -> float:
        """Calculate relevance score for a file based on query."""
        query_lower = query.lower()
        file_path_lower = file_path.lower()
        
        score = 0.0
        
        # Exact name match
        if query_lower in os.path.basename(file_path_lower):
            score += 10.0
        
        # Path component match
        path_parts = file_path_lower.split(os.sep)
        for part in path_parts:
            if query_lower in part:
                score += 2.0
        
        # Extension relevance
        extension = os.path.splitext(file_path)[1].lower()
        if extension in ['.py', '.js', '.ts', '.java', '.cpp']:
            score += 1.0
        
        return score
    
    def _get_template(self, template_name: str) -> str:
        """Get template content."""
        templates = {
            'python_class': '''class {{ class_name }}:
    """{{ description }}"""
    
    def __init__(self):
        """Initialize {{ class_name }}."""
        pass
    
    def {{ method_name }}(self):
        """{{ method_description }}"""
        pass
''',
            'python_function': '''def {{ function_name }}({{ parameters }}):
    """{{ description }}
    
    Args:
        {{ args_description }}
    
    Returns:
        {{ return_description }}
    """
    pass
''',
            'javascript_class': '''class {{ class_name }} {
    constructor() {
        // Initialize {{ class_name }}
    }
    
    {{ method_name }}() {
        // {{ method_description }}
    }
}
''',
            'javascript_function': '''function {{ function_name }}({{ parameters }}) {
    /**
     * {{ description }}
     * @param {{ param_description }}
     * @returns {{ return_description }}
     */
}
'''
        }
        
        return templates.get(template_name, "Template not found")
    
    def _rename_function(self, content: str, old_name: str, new_name: str) -> str:
        """Rename a function in code content."""
        import re
        
        # Simple function renaming (can be enhanced)
        patterns = [
            (rf'\bdef\s+{old_name}\b', f'def {new_name}'),
            (rf'\b{old_name}\s*\(', f'{new_name}('),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def _extract_method(self, content: str, params: Dict[str, Any]) -> str:
        """Extract a method from existing code."""
        # This is a placeholder for method extraction logic
        # In a real implementation, this would parse the AST and extract code blocks
        return content
    
    def _add_docstring(self, content: str, params: Dict[str, Any]) -> str:
        """Add docstrings to functions/classes."""
        # This is a placeholder for docstring addition logic
        # In a real implementation, this would parse the AST and add docstrings
        return content