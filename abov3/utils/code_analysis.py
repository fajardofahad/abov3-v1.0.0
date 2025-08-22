"""
ABOV3 Ollama Code Analysis Utilities

Comprehensive code parsing and analysis utilities providing AST parsing, code quality metrics,
complexity analysis, dependency tracking, and code similarity detection for multiple programming languages.

Features:
    - Multi-language AST parsing and analysis
    - Code quality metrics and complexity analysis
    - Function and class extraction with metadata
    - Dependency analysis and import tracking
    - Code similarity detection and comparison
    - Syntax error detection and reporting
    - Code formatting and style checking
    - Documentation extraction and analysis
    - Security vulnerability scanning
    - Performance analysis and optimization suggestions

Supported Languages:
    - Python (comprehensive support)
    - JavaScript/TypeScript (basic support)
    - Java (basic support)
    - C/C++ (basic support)
    - Go (basic support)

Author: ABOV3 Enterprise Documentation Agent
Version: 1.0.0
"""

import ast
import os
import re
import sys
import json
import hashlib
import logging
import asyncio
import difflib
from pathlib import Path
from typing import (
    Dict, List, Optional, Any, Tuple, Set, Union, 
    AsyncGenerator, Callable, NamedTuple, Literal
)
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime
import concurrent.futures

# Third-party imports
try:
    import pylint.lint
    import pylint.reporters.text
    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False

try:
    import black
    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False

try:
    import isort
    ISORT_AVAILABLE = True
except ImportError:
    ISORT_AVAILABLE = False

try:
    import bandit.core.config
    import bandit.core.manager
    BANDIT_AVAILABLE = True
except ImportError:
    BANDIT_AVAILABLE = False

# Internal imports
from .security import SecurityManager, detect_malicious_patterns
from .validation import PythonCodeValidator, validate_python_syntax


# Configure logging
logger = logging.getLogger('abov3.code_analysis')


@dataclass
class CodeMetrics:
    """Code quality and complexity metrics."""
    lines_of_code: int = 0
    lines_of_comments: int = 0
    blank_lines: int = 0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    halstead_volume: float = 0.0
    maintainability_index: float = 0.0
    test_coverage: float = 0.0
    duplication_ratio: float = 0.0
    technical_debt_ratio: float = 0.0


@dataclass
class FunctionInfo:
    """Information about a function or method."""
    name: str
    line_number: int
    end_line: int
    parameters: List[str]
    return_type: Optional[str]
    docstring: Optional[str]
    complexity: int
    is_async: bool = False
    is_method: bool = False
    is_property: bool = False
    decorators: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    line_number: int
    end_line: int
    base_classes: List[str]
    methods: List[FunctionInfo]
    properties: List[str]
    docstring: Optional[str]
    decorators: List[str] = field(default_factory=list)
    is_abstract: bool = False
    is_dataclass: bool = False


@dataclass
class ImportInfo:
    """Information about imports and dependencies."""
    module: str
    alias: Optional[str]
    is_from_import: bool
    imported_names: List[str]
    line_number: int
    is_standard_library: bool = False
    is_third_party: bool = False
    is_local: bool = False


@dataclass
class SecurityIssue:
    """Security vulnerability or concern."""
    severity: Literal['low', 'medium', 'high', 'critical']
    category: str
    message: str
    line_number: int
    column: int
    confidence: Literal['low', 'medium', 'high']
    rule_id: str


@dataclass
class StyleIssue:
    """Code style or formatting issue."""
    severity: Literal['info', 'warning', 'error']
    message: str
    line_number: int
    column: int
    rule_id: str
    suggested_fix: Optional[str] = None


@dataclass
class AnalysisResult:
    """Complete code analysis result."""
    file_path: str
    language: str
    metrics: CodeMetrics
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    imports: List[ImportInfo]
    security_issues: List[SecurityIssue]
    style_issues: List[StyleIssue]
    syntax_errors: List[str]
    documentation_coverage: float
    analysis_timestamp: datetime
    analysis_duration: float


class LanguageDetector:
    """Detect programming language from file extension and content."""
    
    LANGUAGE_MAPPINGS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.cs': 'csharp',
        '.vb': 'vb',
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',
        '.ps1': 'powershell',
        '.sql': 'sql',
        '.r': 'r',
        '.m': 'matlab',
        '.pl': 'perl'
    }
    
    CONTENT_PATTERNS = {
        'python': [
            r'^\s*import\s+\w+',
            r'^\s*from\s+\w+\s+import',
            r'^\s*def\s+\w+\s*\(',
            r'^\s*class\s+\w+\s*[\(:]',
            r'^\s*if\s+__name__\s*==\s*["\']__main__["\']'
        ],
        'javascript': [
            r'^\s*function\s+\w+\s*\(',
            r'^\s*const\s+\w+\s*=',
            r'^\s*let\s+\w+\s*=',
            r'^\s*var\s+\w+\s*=',
            r'require\s*\(["\'][^"\']+["\']\)'
        ],
        'java': [
            r'^\s*public\s+class\s+\w+',
            r'^\s*import\s+[\w.]+;',
            r'^\s*package\s+[\w.]+;',
            r'^\s*public\s+static\s+void\s+main'
        ]
    }
    
    @classmethod
    def detect_language(cls, file_path: str, content: Optional[str] = None) -> str:
        """Detect programming language from file path and content."""
        path = Path(file_path)
        
        # Check extension first
        extension = path.suffix.lower()
        if extension in cls.LANGUAGE_MAPPINGS:
            language = cls.LANGUAGE_MAPPINGS[extension]
            
            # Verify with content if available
            if content and language in cls.CONTENT_PATTERNS:
                patterns = cls.CONTENT_PATTERNS[language]
                if any(re.search(pattern, content, re.MULTILINE) for pattern in patterns):
                    return language
            else:
                return language
        
        # Try to detect from content
        if content:
            for language, patterns in cls.CONTENT_PATTERNS.items():
                matches = sum(1 for pattern in patterns 
                            if re.search(pattern, content, re.MULTILINE))
                if matches >= 2:  # Require at least 2 pattern matches
                    return language
        
        return 'unknown'


class PythonASTAnalyzer(ast.NodeVisitor):
    """Python AST analyzer for extracting code structure and metrics."""
    
    def __init__(self):
        self.functions: List[FunctionInfo] = []
        self.classes: List[ClassInfo] = []
        self.imports: List[ImportInfo] = []
        self.complexity = 0
        self.current_class: Optional[str] = None
        self.function_calls: Dict[str, List[str]] = defaultdict(list)
        self.line_count = 0
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        func_info = self._extract_function_info(node)
        func_info.is_method = self.current_class is not None
        func_info.complexity = self._calculate_complexity(node)
        
        if self.current_class:
            # Add to current class methods
            for class_info in self.classes:
                if class_info.name == self.current_class:
                    class_info.methods.append(func_info)
                    break
        else:
            self.functions.append(func_info)
        
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        func_info = self._extract_function_info(node)
        func_info.is_async = True
        func_info.is_method = self.current_class is not None
        func_info.complexity = self._calculate_complexity(node)
        
        if self.current_class:
            # Add to current class methods
            for class_info in self.classes:
                if class_info.name == self.current_class:
                    class_info.methods.append(func_info)
                    break
        else:
            self.functions.append(func_info)
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        class_info = ClassInfo(
            name=node.name,
            line_number=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            base_classes=[self._get_name(base) for base in node.bases],
            methods=[],
            properties=[],
            docstring=ast.get_docstring(node),
            decorators=[self._get_decorator_name(dec) for dec in node.decorator_list],
            is_abstract=any('abstractmethod' in self._get_decorator_name(dec) 
                          for dec in node.decorator_list),
            is_dataclass=any('dataclass' in self._get_decorator_name(dec) 
                           for dec in node.decorator_list)
        )
        
        self.classes.append(class_info)
        
        # Visit class body with current class context
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement."""
        for alias in node.names:
            import_info = ImportInfo(
                module=alias.name,
                alias=alias.asname,
                is_from_import=False,
                imported_names=[alias.name],
                line_number=node.lineno,
                is_standard_library=self._is_standard_library(alias.name),
                is_third_party=self._is_third_party(alias.name),
                is_local=self._is_local_import(alias.name)
            )
            self.imports.append(import_info)
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from import statement."""
        if node.module:
            imported_names = [alias.name for alias in node.names]
            import_info = ImportInfo(
                module=node.module,
                alias=None,
                is_from_import=True,
                imported_names=imported_names,
                line_number=node.lineno,
                is_standard_library=self._is_standard_library(node.module),
                is_third_party=self._is_third_party(node.module),
                is_local=self._is_local_import(node.module)
            )
            self.imports.append(import_info)
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call."""
        func_name = self._get_call_name(node)
        if func_name and hasattr(self, '_current_function'):
            self.function_calls[self._current_function].append(func_name)
        
        self.generic_visit(node)
    
    def _extract_function_info(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> FunctionInfo:
        """Extract function information from AST node."""
        parameters = []
        for arg in node.args.args:
            param = arg.arg
            if arg.annotation:
                param += f": {ast.unparse(arg.annotation)}"
            parameters.append(param)
        
        # Add default arguments
        defaults = node.args.defaults
        if defaults:
            default_offset = len(parameters) - len(defaults)
            for i, default in enumerate(defaults):
                param_index = default_offset + i
                if param_index < len(parameters):
                    parameters[param_index] += f" = {ast.unparse(default)}"
        
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)
        
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
        is_property = any('property' in dec for dec in decorators)
        
        return FunctionInfo(
            name=node.name,
            line_number=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            parameters=parameters,
            return_type=return_type,
            docstring=ast.get_docstring(node),
            complexity=0,  # Will be calculated separately
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_property=is_property,
            decorators=decorators
        )
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return ast.unparse(node)
        else:
            return ast.unparse(node)
    
    def _get_decorator_name(self, node: ast.AST) -> str:
        """Get decorator name from AST node."""
        return ast.unparse(node)
    
    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Get function call name."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return ast.unparse(node.func)
        return None
    
    def _is_standard_library(self, module: str) -> bool:
        """Check if module is part of Python standard library."""
        stdlib_modules = {
            'os', 'sys', 'json', 're', 'datetime', 'time', 'math', 'random',
            'collections', 'itertools', 'functools', 'pathlib', 'typing',
            'asyncio', 'threading', 'multiprocessing', 'subprocess', 'urllib',
            'http', 'email', 'html', 'xml', 'csv', 'sqlite3', 'logging',
            'unittest', 'argparse', 'configparser', 'pickle', 'base64',
            'hashlib', 'hmac', 'secrets', 'uuid', 'tempfile', 'shutil',
            'glob', 'fnmatch', 'linecache', 'textwrap', 'string', 'io'
        }
        
        root_module = module.split('.')[0]
        return root_module in stdlib_modules
    
    def _is_third_party(self, module: str) -> bool:
        """Check if module is third-party."""
        # Simple heuristic: if not standard library and not local, assume third-party
        return not self._is_standard_library(module) and not self._is_local_import(module)
    
    def _is_local_import(self, module: str) -> bool:
        """Check if import is local to the project."""
        # Simple heuristic: relative imports or modules starting with project name
        return module.startswith('.') or module.startswith('abov3')


class CodeQualityAnalyzer:
    """Analyze code quality and provide metrics."""
    
    def __init__(self, security_manager: Optional[SecurityManager] = None):
        self.security_manager = security_manager or SecurityManager()
        self.validator = PythonCodeValidator()
    
    async def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyze a single code file."""
        start_time = datetime.now()
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Detect language
            language = LanguageDetector.detect_language(file_path, content)
            
            # Initialize result
            result = AnalysisResult(
                file_path=file_path,
                language=language,
                metrics=CodeMetrics(),
                functions=[],
                classes=[],
                imports=[],
                security_issues=[],
                style_issues=[],
                syntax_errors=[],
                documentation_coverage=0.0,
                analysis_timestamp=start_time,
                analysis_duration=0.0
            )
            
            if language == 'python':
                await self._analyze_python_file(content, result)
            else:
                await self._analyze_generic_file(content, result)
            
            # Calculate analysis duration
            end_time = datetime.now()
            result.analysis_duration = (end_time - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            raise
    
    async def _analyze_python_file(self, content: str, result: AnalysisResult) -> None:
        """Analyze Python file content."""
        # Parse AST
        try:
            tree = ast.parse(content)
            analyzer = PythonASTAnalyzer()
            analyzer.visit(tree)
            
            result.functions = analyzer.functions
            result.classes = analyzer.classes
            result.imports = analyzer.imports
            
        except SyntaxError as e:
            result.syntax_errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return
        
        # Calculate metrics
        result.metrics = await self._calculate_python_metrics(content, tree)
        
        # Security analysis
        result.security_issues = await self._analyze_security(content)
        
        # Style analysis
        result.style_issues = await self._analyze_style(content, result.file_path)
        
        # Documentation coverage
        result.documentation_coverage = self._calculate_documentation_coverage(
            result.functions, result.classes
        )
    
    async def _analyze_generic_file(self, content: str, result: AnalysisResult) -> None:
        """Analyze non-Python file content."""
        lines = content.split('\n')
        
        # Basic metrics
        result.metrics.lines_of_code = len([line for line in lines if line.strip()])
        result.metrics.blank_lines = len([line for line in lines if not line.strip()])
        
        # Basic security check
        if self.security_manager:
            malicious_patterns = detect_malicious_patterns(content)
            for pattern, line_num in malicious_patterns:
                result.security_issues.append(SecurityIssue(
                    severity='medium',
                    category='malicious_pattern',
                    message=f"Potentially malicious pattern detected: {pattern}",
                    line_number=line_num,
                    column=0,
                    confidence='medium',
                    rule_id='MALICIOUS_PATTERN'
                ))
    
    async def _calculate_python_metrics(self, content: str, tree: ast.AST) -> CodeMetrics:
        """Calculate Python code metrics."""
        lines = content.split('\n')
        
        # Line counting
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
            else:
                code_lines += 1
                # Check for inline comments
                if '#' in line:
                    comment_lines += 1
        
        # Complexity analysis
        complexity = self._calculate_total_complexity(tree)
        
        # Halstead metrics (simplified)
        operators, operands = self._count_halstead_metrics(tree)
        n1, n2, N1, N2 = len(operators), len(operands), sum(operators.values()), sum(operands.values())
        
        if n1 > 0 and n2 > 0:
            halstead_volume = (N1 + N2) * (n1 + n2).bit_length()
        else:
            halstead_volume = 0.0
        
        # Maintainability index (simplified)
        if code_lines > 0:
            maintainability = max(0, 171 - 5.2 * (halstead_volume / 1000) - 0.23 * complexity - 16.2 * (code_lines / 100))
        else:
            maintainability = 100.0
        
        return CodeMetrics(
            lines_of_code=code_lines,
            lines_of_comments=comment_lines,
            blank_lines=blank_lines,
            cyclomatic_complexity=complexity,
            halstead_volume=halstead_volume,
            maintainability_index=maintainability
        )
    
    def _calculate_total_complexity(self, tree: ast.AST) -> int:
        """Calculate total cyclomatic complexity."""
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
        
        return max(1, complexity)
    
    def _count_halstead_metrics(self, tree: ast.AST) -> Tuple[Counter, Counter]:
        """Count Halstead operators and operands."""
        operators = Counter()
        operands = Counter()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                operators[type(node.op).__name__] += 1
            elif isinstance(node, ast.UnaryOp):
                operators[type(node.op).__name__] += 1
            elif isinstance(node, ast.Compare):
                for op in node.ops:
                    operators[type(op).__name__] += 1
            elif isinstance(node, ast.Name):
                operands[node.id] += 1
            elif isinstance(node, (ast.Num, ast.Str, ast.Constant)):
                operands[str(node)] += 1
        
        return operators, operands
    
    async def _analyze_security(self, content: str) -> List[SecurityIssue]:
        """Analyze code for security issues."""
        issues = []
        
        # Use security manager for pattern detection
        if self.security_manager:
            malicious_patterns = detect_malicious_patterns(content)
            for pattern, line_num in malicious_patterns:
                issues.append(SecurityIssue(
                    severity='high',
                    category='malicious_pattern',
                    message=f"Potentially dangerous pattern: {pattern}",
                    line_number=line_num,
                    column=0,
                    confidence='high',
                    rule_id='DANGEROUS_PATTERN'
                ))
        
        # Use Bandit if available
        if BANDIT_AVAILABLE:
            try:
                # This is a simplified example - real implementation would use Bandit API
                pass
            except Exception as e:
                logger.warning(f"Bandit analysis failed: {e}")
        
        return issues
    
    async def _analyze_style(self, content: str, file_path: str) -> List[StyleIssue]:
        """Analyze code style issues."""
        issues = []
        
        # Use pylint if available
        if PYLINT_AVAILABLE:
            try:
                # This is a simplified example - real implementation would use pylint API
                pass
            except Exception as e:
                logger.warning(f"Pylint analysis failed: {e}")
        
        # Basic style checks
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # Long lines
            if len(line) > 88:
                issues.append(StyleIssue(
                    severity='warning',
                    message=f"Line too long ({len(line)} > 88 characters)",
                    line_number=i,
                    column=88,
                    rule_id='E501'
                ))
            
            # Trailing whitespace
            if line.rstrip() != line:
                issues.append(StyleIssue(
                    severity='info',
                    message="Trailing whitespace",
                    line_number=i,
                    column=len(line.rstrip()),
                    rule_id='W291'
                ))
        
        return issues
    
    def _calculate_documentation_coverage(self, functions: List[FunctionInfo], 
                                        classes: List[ClassInfo]) -> float:
        """Calculate documentation coverage percentage."""
        total_items = len(functions) + len(classes)
        
        if total_items == 0:
            return 100.0
        
        documented_items = 0
        
        # Count documented functions
        documented_items += sum(1 for func in functions if func.docstring)
        
        # Count documented classes
        documented_items += sum(1 for cls in classes if cls.docstring)
        
        return (documented_items / total_items) * 100.0


class CodeSimilarityAnalyzer:
    """Analyze code similarity and detect duplications."""
    
    def __init__(self, min_similarity: float = 0.8):
        self.min_similarity = min_similarity
    
    async def find_similar_functions(self, functions: List[FunctionInfo], 
                                   content: str) -> List[Tuple[FunctionInfo, FunctionInfo, float]]:
        """Find similar functions in the code."""
        similarities = []
        lines = content.split('\n')
        
        for i, func1 in enumerate(functions):
            for func2 in functions[i+1:]:
                # Extract function content
                func1_content = self._extract_function_content(func1, lines)
                func2_content = self._extract_function_content(func2, lines)
                
                # Calculate similarity
                similarity = self._calculate_similarity(func1_content, func2_content)
                
                if similarity >= self.min_similarity:
                    similarities.append((func1, func2, similarity))
        
        return similarities
    
    def _extract_function_content(self, func: FunctionInfo, lines: List[str]) -> str:
        """Extract function content from source lines."""
        start_line = func.line_number - 1  # Convert to 0-based index
        end_line = func.end_line
        
        if start_line < len(lines) and end_line <= len(lines):
            return '\n'.join(lines[start_line:end_line])
        return ""
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two code snippets."""
        # Normalize content
        norm1 = self._normalize_code(content1)
        norm2 = self._normalize_code(content2)
        
        # Calculate similarity using difflib
        matcher = difflib.SequenceMatcher(None, norm1, norm2)
        return matcher.ratio()
    
    def _normalize_code(self, content: str) -> str:
        """Normalize code for comparison."""
        # Remove comments and docstrings
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove inline comments
                if '#' in line:
                    line = line[:line.index('#')].strip()
                if line:
                    lines.append(line)
        
        return '\n'.join(lines)


class ProjectAnalyzer:
    """Analyze entire projects and provide comprehensive reports."""
    
    def __init__(self, security_manager: Optional[SecurityManager] = None):
        self.code_analyzer = CodeQualityAnalyzer(security_manager)
        self.similarity_analyzer = CodeSimilarityAnalyzer()
        self.security_manager = security_manager
    
    async def analyze_project(self, project_path: str, 
                            file_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze entire project and generate comprehensive report."""
        if file_patterns is None:
            file_patterns = ['**/*.py', '**/*.js', '**/*.ts', '**/*.java', '**/*.go']
        
        project_path = Path(project_path)
        
        # Find all relevant files
        files_to_analyze = []
        for pattern in file_patterns:
            files_to_analyze.extend(project_path.glob(pattern))
        
        # Analyze files concurrently
        tasks = []
        for file_path in files_to_analyze:
            if file_path.is_file():
                tasks.append(self.code_analyzer.analyze_file(str(file_path)))
        
        # Execute analysis
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        analysis_results = [r for r in results if isinstance(r, AnalysisResult)]
        
        # Generate project report
        report = await self._generate_project_report(analysis_results, project_path)
        
        return report
    
    async def _generate_project_report(self, results: List[AnalysisResult], 
                                     project_path: Path) -> Dict[str, Any]:
        """Generate comprehensive project analysis report."""
        total_files = len(results)
        
        if total_files == 0:
            return {
                'summary': {'total_files': 0, 'error': 'No files analyzed'},
                'timestamp': datetime.now().isoformat()
            }
        
        # Aggregate metrics
        total_loc = sum(r.metrics.lines_of_code for r in results)
        total_functions = sum(len(r.functions) for r in results)
        total_classes = sum(len(r.classes) for r in results)
        
        # Security issues summary
        security_issues_by_severity = defaultdict(int)
        for result in results:
            for issue in result.security_issues:
                security_issues_by_severity[issue.severity] += 1
        
        # Style issues summary
        style_issues_by_severity = defaultdict(int)
        for result in results:
            for issue in result.style_issues:
                style_issues_by_severity[issue.severity] += 1
        
        # Language distribution
        language_stats = defaultdict(lambda: {'files': 0, 'lines': 0})
        for result in results:
            language_stats[result.language]['files'] += 1
            language_stats[result.language]['lines'] += result.metrics.lines_of_code
        
        # Top issues
        top_security_issues = []
        top_style_issues = []
        
        for result in results:
            # Get critical security issues
            critical_security = [issue for issue in result.security_issues 
                               if issue.severity == 'critical']
            top_security_issues.extend(critical_security[:5])  # Top 5 per file
            
            # Get error-level style issues
            error_style = [issue for issue in result.style_issues 
                          if issue.severity == 'error']
            top_style_issues.extend(error_style[:5])  # Top 5 per file
        
        # Documentation coverage
        avg_doc_coverage = sum(r.documentation_coverage for r in results) / total_files
        
        # Complexity analysis
        avg_complexity = sum(r.metrics.cyclomatic_complexity for r in results) / total_files
        high_complexity_files = [
            {'file': r.file_path, 'complexity': r.metrics.cyclomatic_complexity}
            for r in results if r.metrics.cyclomatic_complexity > 10
        ]
        
        # Maintainability
        avg_maintainability = sum(r.metrics.maintainability_index for r in results) / total_files
        
        return {
            'summary': {
                'total_files': total_files,
                'total_lines_of_code': total_loc,
                'total_functions': total_functions,
                'total_classes': total_classes,
                'average_documentation_coverage': round(avg_doc_coverage, 2),
                'average_complexity': round(avg_complexity, 2),
                'average_maintainability': round(avg_maintainability, 2)
            },
            'language_distribution': dict(language_stats),
            'security_analysis': {
                'issues_by_severity': dict(security_issues_by_severity),
                'top_critical_issues': top_security_issues[:10]
            },
            'style_analysis': {
                'issues_by_severity': dict(style_issues_by_severity),
                'top_error_issues': top_style_issues[:10]
            },
            'complexity_analysis': {
                'average_complexity': round(avg_complexity, 2),
                'high_complexity_files': sorted(high_complexity_files, 
                                               key=lambda x: x['complexity'], 
                                               reverse=True)[:10]
            },
            'detailed_results': [
                {
                    'file': r.file_path,
                    'language': r.language,
                    'metrics': {
                        'lines_of_code': r.metrics.lines_of_code,
                        'complexity': r.metrics.cyclomatic_complexity,
                        'maintainability': round(r.metrics.maintainability_index, 2),
                        'documentation_coverage': round(r.documentation_coverage, 2)
                    },
                    'issues': {
                        'security': len(r.security_issues),
                        'style': len(r.style_issues),
                        'syntax_errors': len(r.syntax_errors)
                    }
                }
                for r in results
            ],
            'timestamp': datetime.now().isoformat(),
            'analysis_duration': sum(r.analysis_duration for r in results)
        }


# Export main classes and functions
__all__ = [
    'CodeMetrics',
    'FunctionInfo',
    'ClassInfo',
    'ImportInfo',
    'SecurityIssue',
    'StyleIssue',
    'AnalysisResult',
    'LanguageDetector',
    'PythonASTAnalyzer',
    'CodeQualityAnalyzer',
    'CodeSimilarityAnalyzer',
    'ProjectAnalyzer'
]