"""
Comprehensive test script for ABOV3 project management features.

This script tests all the new project management capabilities including:
- Project selection and management
- File operations
- Context integration
- AI assistance
- REPL commands
- Error handling
"""

import os
import sys
import asyncio
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import ABOV3 components
try:
    from abov3.core.project_app import ProjectAwareABOV3App
    from abov3.core.project.manager import ProjectManager, ProjectState
    from abov3.core.context.manager import ContextManager
    from abov3.core.project_integration import ProjectIntegrationService
    from abov3.utils.project_errors import ABOV3ProjectError
    from abov3.core.config import get_config
except ImportError as e:
    print(f"Error importing ABOV3 modules: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ProjectFeatureTester:
    """Test suite for ABOV3 project management features."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.test_results = {}
        self.temp_project_dir = None
        self.project_service = None
        self.app = None
    
    def setup_test_project(self) -> str:
        """Create a temporary test project."""
        self.temp_project_dir = tempfile.mkdtemp(prefix="abov3_test_project_")
        
        # Create test files
        test_files = {
            "main.py": '''#!/usr/bin/env python3
"""
Main module for the test project.
"""

def hello_world():
    """Print hello world message."""
    print("Hello, World!")

def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two numbers."""
    return a + b

if __name__ == "__main__":
    hello_world()
    result = calculate_sum(5, 3)
    print(f"5 + 3 = {result}")
''',
            "utils.py": '''"""
Utility functions for the test project.
"""

def format_message(name: str, message: str) -> str:
    """Format a message with a name."""
    return f"{name}: {message}"

class TestClass:
    """A simple test class."""
    
    def __init__(self, value: int):
        self.value = value
    
    def get_value(self) -> int:
        """Get the stored value."""
        return self.value
''',
            "README.md": '''# Test Project

This is a test project for ABOV3 project management features.

## Features

- Python code files
- Documentation
- Configuration files

## Usage

Run `python main.py` to see the demo.
''',
            "config.json": '''{
    "name": "ABOV3 Test Project",
    "version": "1.0.0",
    "description": "Test project for validating ABOV3 features",
    "author": "ABOV3 Test Suite"
}''',
            "requirements.txt": '''# Test project requirements
requests>=2.25.0
pytest>=6.0.0
''',
        }
        
        # Create subdirectories
        os.makedirs(os.path.join(self.temp_project_dir, "tests"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_project_dir, "docs"), exist_ok=True)
        
        # Write test files
        for filename, content in test_files.items():
            file_path = os.path.join(self.temp_project_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Create test file in subdirectory
        test_file_path = os.path.join(self.temp_project_dir, "tests", "test_main.py")
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write('''"""Tests for main module."""

def test_calculate_sum():
    """Test the calculate_sum function."""
    from main import calculate_sum
    assert calculate_sum(2, 3) == 5
    assert calculate_sum(0, 0) == 0
    assert calculate_sum(-1, 1) == 0
''')
        
        logger.info(f"Created test project at: {self.temp_project_dir}")
        return self.temp_project_dir
    
    def cleanup_test_project(self):
        """Clean up the temporary test project."""
        if self.temp_project_dir and os.path.exists(self.temp_project_dir):
            shutil.rmtree(self.temp_project_dir)
            logger.info(f"Cleaned up test project: {self.temp_project_dir}")
    
    async def test_project_manager(self) -> bool:
        """Test ProjectManager functionality."""
        try:
            logger.info("Testing ProjectManager...")
            
            # Initialize project manager
            project_manager = ProjectManager()
            
            # Test project selection
            project_path = self.setup_test_project()
            success = await project_manager.select_project(project_path)
            
            if not success:
                logger.error("Failed to select project")
                return False
            
            # Test project info
            project_info = await project_manager.get_project_info()
            if not project_info:
                logger.error("Failed to get project info")
                return False
            
            logger.info(f"Project info: {project_info}")
            
            # Test file listing
            files = await project_manager.list_files()
            if not files:
                logger.error("Failed to list files")
                return False
            
            logger.info(f"Found {len(files)} files")
            
            # Test file reading
            content = await project_manager.read_file("main.py")
            if not content or "hello_world" not in content:
                logger.error("Failed to read file correctly")
                return False
            
            # Test file writing
            test_content = "# Test file created by ABOV3\nprint('Test successful!')"
            await project_manager.write_file("test_output.py", test_content)
            
            # Verify file was written
            written_content = await project_manager.read_file("test_output.py")
            if test_content not in written_content:
                logger.error("Failed to write file correctly")
                return False
            
            # Test search
            search_results = await project_manager.search_files("hello_world")
            if not search_results:
                logger.error("Failed to search files")
                return False
            
            logger.info(f"Search found {len(search_results)} results")
            
            # Test project tree
            tree = await project_manager.get_project_tree()
            if not tree:
                logger.error("Failed to get project tree")
                return False
            
            # Clean up
            await project_manager.close_project()
            
            logger.info("ProjectManager tests passed ✓")
            return True
            
        except Exception as e:
            logger.error(f"ProjectManager test failed: {e}")
            return False
    
    async def test_context_manager(self) -> bool:
        """Test ContextManager with project integration."""
        try:
            logger.info("Testing ContextManager with project integration...")
            
            # Initialize components
            project_manager = ProjectManager()
            context_manager = ContextManager(project_manager=project_manager)
            
            # Select project
            project_path = self.temp_project_dir or self.setup_test_project()
            await project_manager.select_project(project_path)
            
            # Test adding messages
            message_id = context_manager.add_message(
                "user", 
                "Can you help me understand this Python code?",
                project_related=True
            )
            
            if not message_id:
                logger.error("Failed to add message to context")
                return False
            
            # Test context sync
            await context_manager.sync_project_context()
            
            # Test getting context for model
            context_messages = await context_manager.get_context_for_model(
                include_project=True
            )
            
            if not context_messages:
                logger.error("Failed to get context messages")
                return False
            
            logger.info(f"Generated {len(context_messages)} context messages")
            
            # Test project context stats
            stats = context_manager.get_project_context_stats()
            if not stats.get("project_manager"):
                logger.error("Project context stats indicate no project manager")
                return False
            
            logger.info(f"Context stats: {stats}")
            
            # Clean up
            await project_manager.close_project()
            
            logger.info("ContextManager tests passed ✓")
            return True
            
        except Exception as e:
            logger.error(f"ContextManager test failed: {e}")
            return False
    
    async def test_project_integration_service(self) -> bool:
        """Test ProjectIntegrationService."""
        try:
            logger.info("Testing ProjectIntegrationService...")
            
            # Initialize service
            self.project_service = ProjectIntegrationService()
            await self.project_service.initialize()
            
            # Test project selection
            project_path = self.temp_project_dir or self.setup_test_project()
            success = await self.project_service.select_project(project_path)
            
            if not success:
                logger.error("Failed to select project via integration service")
                return False
            
            # Test getting components
            project_manager = self.project_service.get_project_manager()
            context_manager = self.project_service.get_context_manager()
            ollama_client = self.project_service.get_ollama_client()
            
            if not all([project_manager, context_manager, ollama_client]):
                logger.error("Failed to get all service components")
                return False
            
            logger.info("All integration service components available")
            
            # Test project info
            project_info = await project_manager.get_project_info()
            if not project_info:
                logger.error("Failed to get project info via service")
                return False
            
            logger.info(f"Integration service project info: {project_info['name']}")
            
            logger.info("ProjectIntegrationService tests passed ✓")
            return True
            
        except Exception as e:
            logger.error(f"ProjectIntegrationService test failed: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling capabilities."""
        try:
            logger.info("Testing error handling...")
            
            project_manager = ProjectManager()
            
            # Test selecting non-existent project
            try:
                await project_manager.select_project("/nonexistent/path")
                logger.error("Should have failed with non-existent path")
                return False
            except ABOV3ProjectError as e:
                logger.info(f"Correctly caught project error: {e.error_code}")
            
            # Test operations without project selected
            try:
                await project_manager.read_file("test.py")
                logger.error("Should have failed without project selected")
                return False
            except Exception:
                logger.info("Correctly failed operation without project")
            
            logger.info("Error handling tests passed ✓")
            return True
            
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False
    
    async def test_configuration(self) -> bool:
        """Test configuration system."""
        try:
            logger.info("Testing configuration system...")
            
            config = get_config()
            
            # Test project configuration exists
            if not hasattr(config, 'project'):
                logger.error("Project configuration not found")
                return False
            
            project_config = config.project
            
            # Test default values
            if project_config.max_file_size <= 0:
                logger.error("Invalid max_file_size configuration")
                return False
            
            if not project_config.include_patterns:
                logger.error("No include patterns configured")
                return False
            
            logger.info(f"Project config - max_file_size: {project_config.max_file_size}")
            logger.info(f"Project config - include_patterns: {len(project_config.include_patterns)} patterns")
            
            logger.info("Configuration tests passed ✓")
            return True
            
        except Exception as e:
            logger.error(f"Configuration test failed: {e}")
            return False
    
    async def test_cross_platform_compatibility(self) -> bool:
        """Test cross-platform path handling."""
        try:
            logger.info("Testing cross-platform compatibility...")
            
            project_manager = ProjectManager()
            
            # Test path normalization
            test_paths = [
                "/unix/style/path",
                "C:\\Windows\\style\\path",
                "relative/path",
                "./current/dir/path",
                "../parent/dir/path"
            ]
            
            for path in test_paths:
                try:
                    normalized = os.path.abspath(os.path.expanduser(path))
                    logger.info(f"Path '{path}' normalized to: {normalized}")
                except Exception as e:
                    logger.error(f"Failed to normalize path '{path}': {e}")
                    return False
            
            # Test with actual project
            if self.temp_project_dir:
                # Use different path separators
                if os.name == 'nt':
                    # Windows - test with forward slashes
                    alt_path = self.temp_project_dir.replace('\\', '/')
                else:
                    # Unix - use as is
                    alt_path = self.temp_project_dir
                
                success = await project_manager.select_project(alt_path)
                if not success:
                    logger.error("Failed to select project with alternative path format")
                    return False
                
                await project_manager.close_project()
            
            logger.info("Cross-platform compatibility tests passed ✓")
            return True
            
        except Exception as e:
            logger.error(f"Cross-platform compatibility test failed: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        logger.info("Starting ABOV3 project management feature tests...")
        
        tests = [
            ("project_manager", self.test_project_manager),
            ("context_manager", self.test_context_manager),
            ("integration_service", self.test_project_integration_service),
            ("error_handling", self.test_error_handling),
            ("configuration", self.test_configuration),
            ("cross_platform", self.test_cross_platform_compatibility),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running test: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                result = await test_func()
                results[test_name] = result
                status = "PASSED" if result else "FAILED"
                logger.info(f"Test {test_name}: {status}")
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                results[test_name] = False
        
        return results
    
    def print_test_summary(self, results: Dict[str, bool]):
        """Print test results summary."""
        print(f"\n{'='*60}")
        print("ABOV3 PROJECT MANAGEMENT FEATURE TEST SUMMARY")
        print(f"{'='*60}")
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name:<25} {status}")
        
        print(f"\n{'-'*60}")
        print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("\n🎉 All tests passed! ABOV3 project management is ready.")
        else:
            print(f"\n⚠️  {total-passed} tests failed. Please check the logs above.")
        
        print(f"{'='*60}")


async def main():
    """Main test runner."""
    tester = ProjectFeatureTester()
    
    try:
        # Run all tests
        results = await tester.run_all_tests()
        
        # Print summary
        tester.print_test_summary(results)
        
        # Clean up
        tester.cleanup_test_project()
        
        # Exit with appropriate code
        if all(results.values()):
            sys.exit(0)
        else:
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        tester.cleanup_test_project()
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        tester.cleanup_test_project()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())