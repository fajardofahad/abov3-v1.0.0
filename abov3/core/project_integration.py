"""
Project Integration for ABOV3.

This module provides the integration layer between project management,
context management, and AI assistance, enabling context-aware coding support.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, AsyncIterator, Union
from dataclasses import dataclass
from datetime import datetime

from .config import get_config
from .context.manager import ContextManager
from .project.manager import ProjectManager, ProjectState
from .api.ollama_client import OllamaClient, ChatMessage
from ..utils.project_file_ops import ProjectFileOperations


logger = logging.getLogger(__name__)


@dataclass
class CodingContext:
    """Coding context for AI assistance."""
    project_name: str
    project_type: str
    current_files: Dict[str, str]
    recent_changes: List[str]
    user_intent: str
    context_summary: str


class ProjectAwareOllamaClient:
    """
    Project-aware Ollama client that provides context-aware coding assistance.
    
    This client enhances the base OllamaClient with project context awareness,
    automatically including relevant project information in AI conversations.
    """
    
    def __init__(self, 
                 project_manager: Optional[ProjectManager] = None,
                 context_manager: Optional[ContextManager] = None,
                 ollama_client: Optional[OllamaClient] = None):
        """Initialize the project-aware Ollama client."""
        self.config = get_config()
        self.project_manager = project_manager
        self.context_manager = context_manager
        self.ollama_client = ollama_client or OllamaClient()
        self.project_file_ops = None
        
        # Configure context manager with project manager
        if self.context_manager and self.project_manager:
            self.context_manager.set_project_manager(self.project_manager)
        
        logger.info("ProjectAwareOllamaClient initialized")
    
    def set_project_manager(self, project_manager: ProjectManager) -> None:
        """Set or update the project manager."""
        self.project_manager = project_manager
        
        if self.context_manager:
            self.context_manager.set_project_manager(project_manager)
        
        # Initialize project file operations
        if project_manager.current_project:
            self.project_file_ops = ProjectFileOperations(
                security_manager=project_manager.security_manager,
                project_root=project_manager.current_project.root_path
            )
    
    def set_context_manager(self, context_manager: ContextManager) -> None:
        """Set or update the context manager."""
        self.context_manager = context_manager
        
        if self.project_manager:
            self.context_manager.set_project_manager(self.project_manager)
    
    async def chat_with_context(self, 
                               message: str,
                               model: Optional[str] = None,
                               stream: bool = True,
                               include_project_context: bool = True,
                               code_assistance: bool = False) -> Union[str, AsyncIterator[str]]:
        """
        Chat with AI including project context.
        
        Args:
            message: User message
            model: Model to use (uses default if not specified)
            stream: Whether to stream the response
            include_project_context: Whether to include project context
            code_assistance: Whether to optimize for code assistance
            
        Returns:
            AI response (string or async iterator for streaming)
        """
        try:
            # Prepare context
            messages = await self._prepare_context_messages(
                message, 
                include_project_context,
                code_assistance
            )
            
            # Use the underlying Ollama client
            if stream:
                return self._stream_chat_response(messages, model)
            else:
                response = await self.ollama_client.chat(
                    model=model or self.config.model.default_model,
                    messages=[msg.dict() if hasattr(msg, 'dict') else msg for msg in messages],
                    stream=False
                )
                return response['message']['content']
        
        except Exception as e:
            logger.error(f"Error in chat_with_context: {e}")
            raise
    
    async def get_code_suggestions(self, 
                                  file_path: str,
                                  cursor_position: Optional[Dict[str, int]] = None,
                                  suggestion_type: str = "completion") -> Dict[str, Any]:
        """
        Get code suggestions for a specific file.
        
        Args:
            file_path: Path to the file
            cursor_position: Cursor position {"line": int, "column": int}
            suggestion_type: Type of suggestion ("completion", "refactor", "optimize")
            
        Returns:
            Code suggestions with metadata
        """
        if not self.project_manager or not self.project_manager.current_project:
            raise RuntimeError("No project selected")
        
        try:
            # Read file content
            file_content = await self.project_manager.read_file(file_path)
            
            # Analyze file if project file ops available
            analysis = None
            if self.project_file_ops:
                analysis = await self.project_file_ops.analyze_code(
                    self.project_manager.current_project.root_path + "/" + file_path,
                    file_content
                )
            
            # Prepare context for code suggestions
            context_prompt = self._create_code_suggestion_prompt(
                file_path, 
                file_content, 
                cursor_position, 
                suggestion_type,
                analysis
            )
            
            # Get AI response
            response = await self.chat_with_context(
                context_prompt,
                stream=False,
                include_project_context=True,
                code_assistance=True
            )
            
            return {
                'suggestions': self._parse_code_suggestions(response),
                'file_path': file_path,
                'suggestion_type': suggestion_type,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting code suggestions: {e}")
            raise
    
    async def explain_code(self, 
                          file_path: str,
                          code_selection: Optional[str] = None) -> str:
        """
        Get explanation for code in a file.
        
        Args:
            file_path: Path to the file
            code_selection: Specific code selection to explain (optional)
            
        Returns:
            Code explanation
        """
        if not self.project_manager or not self.project_manager.current_project:
            raise RuntimeError("No project selected")
        
        try:
            # Read file content
            file_content = await self.project_manager.read_file(file_path)
            
            # Use selection or entire file
            code_to_explain = code_selection or file_content
            
            # Create explanation prompt
            prompt = f"""Please explain the following code from file '{file_path}':

```
{code_to_explain}
```

Provide a clear explanation that covers:
1. What this code does
2. How it works
3. Key components and their roles
4. Any design patterns or best practices used
5. Potential improvements or issues (if any)

Keep the explanation clear and focused on helping understand the code."""
            
            # Get AI response
            response = await self.chat_with_context(
                prompt,
                stream=False,
                include_project_context=True,
                code_assistance=True
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error explaining code: {e}")
            raise
    
    async def suggest_refactoring(self, 
                                 file_path: str,
                                 refactor_type: str = "improve") -> Dict[str, Any]:
        """
        Suggest refactoring for a file.
        
        Args:
            file_path: Path to the file
            refactor_type: Type of refactoring ("improve", "optimize", "simplify", "modernize")
            
        Returns:
            Refactoring suggestions
        """
        if not self.project_manager or not self.project_manager.current_project:
            raise RuntimeError("No project selected")
        
        try:
            # Read and analyze file
            file_content = await self.project_manager.read_file(file_path)
            analysis = None
            
            if self.project_file_ops:
                analysis = await self.project_file_ops.analyze_code(
                    self.project_manager.current_project.root_path + "/" + file_path,
                    file_content
                )
            
            # Create refactoring prompt
            prompt = self._create_refactoring_prompt(file_path, file_content, refactor_type, analysis)
            
            # Get AI response
            response = await self.chat_with_context(
                prompt,
                stream=False,
                include_project_context=True,
                code_assistance=True
            )
            
            return {
                'suggestions': self._parse_refactoring_suggestions(response),
                'file_path': file_path,
                'refactor_type': refactor_type,
                'analysis': analysis.dict() if analysis and hasattr(analysis, 'dict') else None,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error suggesting refactoring: {e}")
            raise
    
    async def generate_tests(self, 
                            file_path: str,
                            test_framework: Optional[str] = None) -> str:
        """
        Generate tests for a file.
        
        Args:
            file_path: Path to the file to test
            test_framework: Preferred test framework (auto-detected if not specified)
            
        Returns:
            Generated test code
        """
        if not self.project_manager or not self.project_manager.current_project:
            raise RuntimeError("No project selected")
        
        try:
            # Read file content
            file_content = await self.project_manager.read_file(file_path)
            
            # Detect language and framework
            language = self._detect_language(file_path)
            if not test_framework:
                test_framework = self._suggest_test_framework(language)
            
            # Create test generation prompt
            prompt = f"""Generate comprehensive tests for the following code from '{file_path}':

```{language}
{file_content}
```

Requirements:
- Use {test_framework} testing framework
- Include unit tests for all functions/methods
- Test both positive and negative cases
- Include edge cases and error conditions
- Follow {language} testing best practices
- Provide clear test names and comments

Generate complete, runnable test code."""
            
            # Get AI response
            response = await self.chat_with_context(
                prompt,
                stream=False,
                include_project_context=True,
                code_assistance=True
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating tests: {e}")
            raise
    
    async def _prepare_context_messages(self, 
                                      user_message: str,
                                      include_project_context: bool,
                                      code_assistance: bool) -> List[Dict[str, str]]:
        """Prepare messages with context for AI chat."""
        messages = []
        
        # Add system message for code assistance
        if code_assistance:
            system_msg = """You are an expert software engineer and coding assistant. You have access to the user's project context and files. Provide helpful, accurate, and practical coding advice. When suggesting code changes:

1. Consider the existing codebase and project structure
2. Follow the project's coding style and conventions
3. Provide complete, runnable code examples
4. Explain your reasoning and approach
5. Consider performance, maintainability, and best practices
6. Flag potential issues or improvements

Be concise but thorough in your responses."""
            
            messages.append({"role": "system", "content": system_msg})
        
        # Get context from context manager
        if self.context_manager:
            if include_project_context:
                # Sync project context first
                await self.context_manager.sync_project_context()
            
            context_messages = await self.context_manager.get_context_for_model(
                include_project=include_project_context
            )
            messages.extend(context_messages)
        
        # Add user message
        messages.append({"role": "user", "content": user_message})
        
        # Add to context manager
        if self.context_manager:
            self.context_manager.add_message(
                "user", 
                user_message, 
                project_related=code_assistance
            )
        
        return messages
    
    async def _stream_chat_response(self, 
                                  messages: List[Dict[str, str]], 
                                  model: Optional[str]) -> AsyncIterator[str]:
        """Stream chat response."""
        model_name = model or self.config.model.default_model
        
        full_response = ""
        async for chunk in self.ollama_client.stream_chat(model_name, messages):
            if chunk:
                full_response += chunk
                yield chunk
        
        # Add response to context manager
        if self.context_manager and full_response:
            self.context_manager.add_message(
                "assistant",
                full_response,
                project_related=True
            )
    
    def _create_code_suggestion_prompt(self, 
                                     file_path: str,
                                     file_content: str,
                                     cursor_position: Optional[Dict[str, int]],
                                     suggestion_type: str,
                                     analysis: Any) -> str:
        """Create prompt for code suggestions."""
        prompt_parts = []
        
        prompt_parts.append(f"File: {file_path}")
        
        if suggestion_type == "completion":
            prompt_parts.append("Please provide code completion suggestions for this file:")
        elif suggestion_type == "refactor":
            prompt_parts.append("Please suggest refactoring improvements for this code:")
        elif suggestion_type == "optimize":
            prompt_parts.append("Please suggest optimizations for this code:")
        
        prompt_parts.append(f"```\n{file_content}\n```")
        
        if cursor_position:
            prompt_parts.append(f"Cursor is at line {cursor_position['line']}, column {cursor_position['column']}")
        
        if analysis:
            prompt_parts.append(f"Code analysis: {analysis.lines_of_code} lines, complexity score: {analysis.complexity_score:.1f}")
        
        return "\n\n".join(prompt_parts)
    
    def _create_refactoring_prompt(self, 
                                 file_path: str,
                                 file_content: str,
                                 refactor_type: str,
                                 analysis: Any) -> str:
        """Create prompt for refactoring suggestions."""
        prompts = {
            "improve": "Please analyze this code and suggest improvements for readability, maintainability, and best practices:",
            "optimize": "Please analyze this code and suggest performance optimizations:",
            "simplify": "Please analyze this code and suggest ways to simplify and reduce complexity:",
            "modernize": "Please analyze this code and suggest modern language features and patterns to adopt:"
        }
        
        prompt_parts = []
        prompt_parts.append(f"File: {file_path}")
        prompt_parts.append(prompts.get(refactor_type, prompts["improve"]))
        prompt_parts.append(f"```\n{file_content}\n```")
        
        if analysis:
            issues = analysis.issues if hasattr(analysis, 'issues') else []
            if issues:
                prompt_parts.append(f"Known issues: {', '.join(issues)}")
        
        prompt_parts.append("Provide specific, actionable suggestions with code examples.")
        
        return "\n\n".join(prompt_parts)
    
    def _parse_code_suggestions(self, response: str) -> List[Dict[str, Any]]:
        """Parse code suggestions from AI response."""
        # Simple parsing - can be enhanced with more sophisticated parsing
        suggestions = []
        
        lines = response.split('\n')
        current_suggestion = None
        
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*')):
                if current_suggestion:
                    suggestions.append(current_suggestion)
                current_suggestion = {"text": line, "code": ""}
            elif current_suggestion and line.startswith('```'):
                # Start/end of code block
                continue
            elif current_suggestion:
                if line:
                    current_suggestion["code"] += line + "\n"
        
        if current_suggestion:
            suggestions.append(current_suggestion)
        
        return suggestions
    
    def _parse_refactoring_suggestions(self, response: str) -> List[Dict[str, Any]]:
        """Parse refactoring suggestions from AI response."""
        return self._parse_code_suggestions(response)  # Same parsing logic for now
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file path."""
        extension = file_path.lower().split('.')[-1] if '.' in file_path else ''
        
        language_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'go': 'go',
            'rs': 'rust',
            'php': 'php',
            'rb': 'ruby',
        }
        
        return language_map.get(extension, 'text')
    
    def _suggest_test_framework(self, language: str) -> str:
        """Suggest test framework based on language."""
        frameworks = {
            'python': 'pytest',
            'javascript': 'jest',
            'typescript': 'jest',
            'java': 'junit',
            'cpp': 'gtest',
            'c': 'unity',
            'go': 'testing',
            'rust': 'rust test',
            'php': 'phpunit',
            'ruby': 'rspec',
        }
        
        return frameworks.get(language, 'unit testing framework')


class ProjectIntegrationService:
    """
    Service that coordinates project management, context management, and AI assistance.
    """
    
    def __init__(self):
        """Initialize the integration service."""
        self.config = get_config()
        self.project_manager: Optional[ProjectManager] = None
        self.context_manager: Optional[ContextManager] = None
        self.ollama_client: Optional[ProjectAwareOllamaClient] = None
        
        logger.info("ProjectIntegrationService initialized")
    
    async def initialize(self) -> None:
        """Initialize all components."""
        try:
            # Initialize context manager
            self.context_manager = ContextManager()
            
            # Initialize project manager
            self.project_manager = ProjectManager()
            
            # Initialize project-aware Ollama client
            self.ollama_client = ProjectAwareOllamaClient(
                project_manager=self.project_manager,
                context_manager=self.context_manager
            )
            
            logger.info("ProjectIntegrationService components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ProjectIntegrationService: {e}")
            raise
    
    async def select_project(self, project_path: str) -> bool:
        """Select a project and set up integration."""
        if not self.project_manager:
            raise RuntimeError("Project manager not initialized")
        
        try:
            success = await self.project_manager.select_project(project_path)
            if success and self.ollama_client:
                self.ollama_client.set_project_manager(self.project_manager)
            
            return success
        
        except Exception as e:
            logger.error(f"Error selecting project: {e}")
            raise
    
    def get_project_manager(self) -> Optional[ProjectManager]:
        """Get the project manager instance."""
        return self.project_manager
    
    def get_context_manager(self) -> Optional[ContextManager]:
        """Get the context manager instance."""
        return self.context_manager
    
    def get_ollama_client(self) -> Optional[ProjectAwareOllamaClient]:
        """Get the project-aware Ollama client."""
        return self.ollama_client
    
    async def shutdown(self) -> None:
        """Shutdown the service and clean up resources."""
        try:
            if self.project_manager:
                await self.project_manager.close_project()
            
            logger.info("ProjectIntegrationService shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")