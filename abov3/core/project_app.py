"""
Project-aware ABOV3 Application.

This module provides the main application class that integrates project management,
context-aware AI assistance, and the enhanced REPL interface.
"""

import logging
import asyncio
from typing import Optional, Any, Dict, List
from pathlib import Path

from .config import get_config
from .project_integration import ProjectIntegrationService
from .app import ABOV3App
from ..ui.console.repl import ABOV3REPL, REPLConfig


logger = logging.getLogger(__name__)


class ProjectAwareABOV3App(ABOV3App):
    """
    Enhanced ABOV3 Application with comprehensive project management capabilities.
    
    This application extends the base ABOV3App with:
    - Project directory selection and management
    - Context-aware AI coding assistance
    - Enhanced REPL with project commands
    - File operations and analysis
    - Real-time project monitoring
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the project-aware ABOV3 application."""
        super().__init__(config_path)
        
        # Project integration service
        self.project_service: Optional[ProjectIntegrationService] = None
        
        # Enhanced REPL
        self.enhanced_repl: Optional[ABOV3REPL] = None
        
        logger.info("ProjectAwareABOV3App initialized")
    
    async def initialize(self) -> None:
        """Initialize the application and all components."""
        try:
            # Initialize base app
            await super().startup()
            
            # Initialize project integration service
            self.project_service = ProjectIntegrationService()
            await self.project_service.initialize()
            
            # Initialize enhanced REPL with project manager
            await self._initialize_enhanced_repl()
            
            logger.info("ProjectAwareABOV3App initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize ProjectAwareABOV3App: {e}")
            raise
    
    async def run_interactive(self, project_path: Optional[str] = None) -> None:
        """
        Run the interactive REPL with project support.
        
        Args:
            project_path: Optional initial project path to load
        """
        try:
            if not self.enhanced_repl:
                raise RuntimeError("Enhanced REPL not initialized")
            
            # Load project if specified
            if project_path:
                await self.select_project(project_path)
            
            # Run the enhanced REPL
            await self.enhanced_repl.run()
            
        except Exception as e:
            logger.error(f"Error running interactive mode: {e}")
            raise
    
    async def select_project(self, project_path: str) -> bool:
        """
        Select a project directory.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            bool: True if successful
        """
        if not self.project_service:
            raise RuntimeError("Project service not initialized")
        
        try:
            success = await self.project_service.select_project(project_path)
            
            if success:
                # Update REPL prompt to show current project
                project_manager = self.project_service.get_project_manager()
                if project_manager:
                    project_info = await project_manager.get_project_info()
                    if project_info and self.enhanced_repl:
                        project_name = project_info.get('name', 'Unknown')
                        self.enhanced_repl.config.prompt_text = f"ABOV3({project_name})> "
                
                logger.info(f"Project selected: {project_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error selecting project: {e}")
            raise
    
    async def chat_with_context(self, 
                               message: str,
                               stream: bool = True,
                               code_assistance: bool = False) -> Any:
        """
        Chat with AI including project context.
        
        Args:
            message: User message
            stream: Whether to stream response
            code_assistance: Whether to optimize for code assistance
            
        Returns:
            AI response
        """
        if not self.project_service:
            raise RuntimeError("Project service not initialized")
        
        ollama_client = self.project_service.get_ollama_client()
        if not ollama_client:
            raise RuntimeError("Ollama client not available")
        
        return await ollama_client.chat_with_context(
            message=message,
            stream=stream,
            code_assistance=code_assistance
        )
    
    async def get_project_info(self) -> Optional[Dict[str, Any]]:
        """Get current project information."""
        if not self.project_service:
            return None
        
        project_manager = self.project_service.get_project_manager()
        if not project_manager:
            return None
        
        return await project_manager.get_project_info()
    
    async def list_project_files(self, 
                                pattern: str = "*",
                                max_results: int = 50) -> Optional[List[Dict[str, Any]]]:
        """List files in the current project."""
        if not self.project_service:
            return None
        
        project_manager = self.project_service.get_project_manager()
        if not project_manager:
            return None
        
        return await project_manager.list_files(
            pattern=pattern,
            max_results=max_results
        )
    
    async def read_project_file(self, file_path: str) -> Optional[str]:
        """Read a file from the current project."""
        if not self.project_service:
            return None
        
        project_manager = self.project_service.get_project_manager()
        if not project_manager:
            return None
        
        return await project_manager.read_file(file_path)
    
    async def write_project_file(self, file_path: str, content: str) -> bool:
        """Write content to a file in the current project."""
        if not self.project_service:
            return False
        
        project_manager = self.project_service.get_project_manager()
        if not project_manager:
            return False
        
        try:
            await project_manager.write_file(file_path, content)
            return True
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return False
    
    async def search_project_files(self, 
                                  query: str,
                                  file_pattern: str = "*",
                                  max_results: int = 20) -> Optional[List[Dict[str, Any]]]:
        """Search for text within project files."""
        if not self.project_service:
            return None
        
        project_manager = self.project_service.get_project_manager()
        if not project_manager:
            return None
        
        return await project_manager.search_files(
            query=query,
            file_pattern=file_pattern,
            max_results=max_results
        )
    
    async def get_project_tree(self, max_depth: int = 3) -> Optional[Dict[str, Any]]:
        """Get project directory tree."""
        if not self.project_service:
            return None
        
        project_manager = self.project_service.get_project_manager()
        if not project_manager:
            return None
        
        return await project_manager.get_project_tree(max_depth=max_depth)
    
    async def get_code_suggestions(self, 
                                  file_path: str,
                                  cursor_position: Optional[Dict[str, int]] = None) -> Optional[Dict[str, Any]]:
        """Get AI code suggestions for a file."""
        if not self.project_service:
            return None
        
        ollama_client = self.project_service.get_ollama_client()
        if not ollama_client:
            return None
        
        return await ollama_client.get_code_suggestions(
            file_path=file_path,
            cursor_position=cursor_position
        )
    
    async def explain_code(self, 
                          file_path: str,
                          code_selection: Optional[str] = None) -> Optional[str]:
        """Get AI explanation for code."""
        if not self.project_service:
            return None
        
        ollama_client = self.project_service.get_ollama_client()
        if not ollama_client:
            return None
        
        return await ollama_client.explain_code(
            file_path=file_path,
            code_selection=code_selection
        )
    
    async def suggest_refactoring(self, 
                                 file_path: str,
                                 refactor_type: str = "improve") -> Optional[Dict[str, Any]]:
        """Get AI refactoring suggestions."""
        if not self.project_service:
            return None
        
        ollama_client = self.project_service.get_ollama_client()
        if not ollama_client:
            return None
        
        return await ollama_client.suggest_refactoring(
            file_path=file_path,
            refactor_type=refactor_type
        )
    
    async def generate_tests(self, 
                            file_path: str,
                            test_framework: Optional[str] = None) -> Optional[str]:
        """Generate tests for a file."""
        if not self.project_service:
            return None
        
        ollama_client = self.project_service.get_ollama_client()
        if not ollama_client:
            return None
        
        return await ollama_client.generate_tests(
            file_path=file_path,
            test_framework=test_framework
        )
    
    async def _initialize_enhanced_repl(self) -> None:
        """Initialize the enhanced REPL with project support."""
        try:
            # Create REPL config
            config = REPLConfig(
                prompt_text="ABOV3> ",
                theme=self.config.ui.theme if hasattr(self.config.ui, 'theme') else "monokai",
                enable_vim_mode=self.config.ui.vim_mode if hasattr(self.config.ui, 'vim_mode') else False,
                history_file=self.config.get_data_dir() / "repl_history.txt",
                session_file=self.config.get_data_dir() / "repl_session.json",
                streaming_output=self.config.ui.streaming_output if hasattr(self.config.ui, 'streaming_output') else True
            )
            
            # Create process callback that integrates with our services
            async def process_callback(text: str) -> Any:
                """Process user input with project context awareness."""
                try:
                    # Check if it's a regular chat message (not a command)
                    if not text.strip().startswith('/'):
                        # Use our chat_with_context method
                        return await self.chat_with_context(
                            message=text,
                            stream=True,
                            code_assistance=True  # Always enable for better assistance
                        )
                    
                    # For commands, let the REPL handle them
                    return None
                
                except Exception as e:
                    logger.error(f"Error processing input: {e}")
                    return f"Error: {e}"
            
            # Get project manager for REPL
            project_manager = None
            if self.project_service:
                project_manager = self.project_service.get_project_manager()
            
            # Create enhanced REPL
            self.enhanced_repl = ABOV3REPL(
                config=config,
                process_callback=process_callback,
                project_manager=project_manager
            )
            
            # Set app instance for status commands
            self.enhanced_repl.app_instance = self
            
            logger.info("Enhanced REPL initialized")
            
        except Exception as e:
            logger.error(f"Error initializing enhanced REPL: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the application and clean up resources."""
        try:
            # Shutdown project service
            if self.project_service:
                await self.project_service.shutdown()
            
            # Shutdown base app
            await super().shutdown()
            
            logger.info("ProjectAwareABOV3App shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def create_project_aware_app(config_path: Optional[str] = None) -> ProjectAwareABOV3App:
    """
    Create a project-aware ABOV3 application instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ProjectAwareABOV3App instance
    """
    return ProjectAwareABOV3App(config_path)


async def run_project_aware_app(project_path: Optional[str] = None,
                               config_path: Optional[str] = None) -> None:
    """
    Run the project-aware ABOV3 application.
    
    Args:
        project_path: Optional initial project path
        config_path: Optional configuration file path
    """
    app = None
    try:
        # Create and initialize app
        app = create_project_aware_app(config_path)
        await app.initialize()
        
        # Run interactive mode
        await app.run_interactive(project_path)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
    finally:
        if app:
            await app.shutdown()


if __name__ == "__main__":
    import sys
    
    # Simple command line handling
    project_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run the application
    asyncio.run(run_project_aware_app(project_path))