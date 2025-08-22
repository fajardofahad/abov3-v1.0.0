# Contributing to ABOV3 4 Ollama

Thank you for your interest in contributing to ABOV3 4 Ollama! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Contribution Types](#contribution-types)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, gender identity, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, or nationality.

### Expected Behavior

- **Be respectful**: Treat all community members with respect and kindness
- **Be collaborative**: Work together constructively and help others learn
- **Be inclusive**: Welcome newcomers and help them get started
- **Be patient**: Understand that people have different skill levels and backgrounds
- **Be constructive**: Provide helpful feedback and suggestions

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing private information without permission
- Spam or inappropriate promotional content
- Any behavior that would be inappropriate in a professional setting

### Enforcement

Code of conduct violations can be reported to contact@abov3.dev. All reports will be handled confidentially and fairly.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.8+ (3.10+ recommended)
- Git
- Ollama installed and running
- Basic knowledge of Python and async programming
- Familiarity with Git workflow

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/abov3-ollama.git
   cd abov3-ollama
   ```

3. **Set up the development environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

4. **Verify setup**:
   ```bash
   # Run tests
   pytest
   
   # Start ABOV3 in debug mode
   python -m abov3.cli chat --debug
   ```

5. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/abov3/abov3-ollama.git
   ```

## Development Process

### Workflow Overview

1. **Issue Discussion**: Discuss changes in issues before implementation
2. **Branch Creation**: Create feature branches from `main`
3. **Development**: Implement changes following guidelines
4. **Testing**: Add/update tests for your changes
5. **Documentation**: Update documentation as needed
6. **Pull Request**: Submit PR for review
7. **Review Process**: Address feedback and iterate
8. **Merge**: Maintainers merge approved PRs

### Staying Updated

```bash
# Keep your fork updated
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

### Branch Strategy

- **main**: Stable release branch
- **feature/**: New features (`feature/add-plugin-system`)
- **fix/**: Bug fixes (`fix/memory-leak`)
- **docs/**: Documentation updates (`docs/update-api-guide`)
- **refactor/**: Code refactoring (`refactor/cleanup-models`)

## Contribution Types

### 🐛 Bug Reports

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the bug
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, Ollama version)
- **Log output** or error messages
- **Screenshots** if applicable

**Bug Report Template**:
```markdown
## Bug Description
Brief description of what went wrong.

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should have happened.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Windows 11, macOS 13, Ubuntu 22.04]
- Python: [e.g., 3.11.0]
- ABOV3: [e.g., 1.0.0]
- Ollama: [e.g., 0.3.1]

## Additional Context
Any other relevant information.
```

### 💡 Feature Requests

For feature requests, provide:

- **Clear use case** and motivation
- **Detailed description** of the feature
- **Examples** of how it would work
- **Alternatives considered**
- **Implementation ideas** (if any)

**Feature Request Template**:
```markdown
## Feature Description
What feature would you like to see added?

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Solution
How should this feature work?

## Examples
Provide examples of the feature in action.

## Alternatives
What alternatives have you considered?

## Additional Context
Any other relevant information.
```

### 🔧 Code Contributions

#### Types of Code Contributions

1. **Bug Fixes**: Fix reported issues
2. **Feature Implementation**: Add new functionality
3. **Performance Improvements**: Optimize existing code
4. **Refactoring**: Improve code structure without changing functionality
5. **Plugin Development**: Create new plugins

#### Contribution Areas

- **Core**: Application logic, API clients, configuration
- **UI**: Terminal interface, formatting, user experience
- **Models**: Model management, fine-tuning, evaluation
- **Plugins**: Built-in and community plugins
- **Utils**: Utilities, security, file operations
- **Tests**: Unit tests, integration tests, fixtures
- **Documentation**: API docs, guides, examples

### 📚 Documentation

Documentation contributions are highly valued:

- **API Documentation**: Improve docstrings and API references
- **User Guides**: Enhance tutorials and how-to guides
- **Examples**: Add practical examples and use cases
- **README Updates**: Keep project information current
- **Blog Posts**: Share experiences and tutorials

### 🧪 Testing

Help improve test coverage:

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark and optimization tests
- **Plugin Tests**: Test plugin functionality
- **Documentation Tests**: Verify examples work

## Pull Request Process

### Before Submitting

1. **Search existing PRs** to avoid duplicates
2. **Discuss in issues** for significant changes
3. **Create feature branch** from latest `main`
4. **Follow coding standards** and guidelines
5. **Add tests** for new functionality
6. **Update documentation** as needed
7. **Test thoroughly** in your environment

### PR Requirements

- [ ] **Descriptive title** and detailed description
- [ ] **Reference issues** being addressed
- [ ] **Tests added/updated** with good coverage
- [ ] **Documentation updated** if needed
- [ ] **CHANGELOG.md updated** for user-facing changes
- [ ] **All tests pass** locally
- [ ] **Code follows style guidelines**
- [ ] **No merge conflicts** with main branch

### PR Template

```markdown
## Description
Brief description of the changes and their purpose.

## Related Issues
Fixes #123
Relates to #456

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Screenshots/Examples
If applicable, add screenshots or code examples.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] CHANGELOG.md updated
- [ ] Breaking changes documented

## Notes for Reviewers
Any specific areas you'd like reviewers to focus on.
```

### Review Process

1. **Automated Checks**: CI runs tests and checks
2. **Maintainer Review**: Core team reviews code
3. **Community Feedback**: Other contributors may comment
4. **Iteration**: Address feedback and update PR
5. **Approval**: Maintainer approves when ready
6. **Merge**: Maintainer merges the PR

## Style Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with specific configurations:

#### Formatting

```python
# Use Black formatter (configured in pyproject.toml)
# Line length: 88 characters
# Automatic formatting on save recommended

# Example of properly formatted code
async def send_message(
    self,
    message: str,
    session_id: Optional[str] = None,
    stream: bool = True,
) -> AsyncIterator[str]:
    """Send message with proper formatting."""
    if not message.strip():
        raise ValueError("Message cannot be empty")
    
    async for chunk in self._process_message(message, session_id):
        yield chunk
```

#### Import Organization

```python
# Standard library imports
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import aiohttp
import click
from rich.console import Console

# Local imports
from .config import Config
from ..models.manager import ModelManager
```

#### Type Hints

```python
# Always use type hints for public APIs
from typing import Any, Dict, List, Optional, Union, AsyncIterator

async def process_data(
    data: List[Dict[str, Any]],
    config: Optional[Config] = None
) -> Dict[str, Union[str, int]]:
    """Process data with comprehensive type hints."""
    result: Dict[str, Union[str, int]] = {}
    # Implementation here
    return result
```

#### Naming Conventions

- **Classes**: `PascalCase` (ModelManager, ChatResponse)
- **Functions/Methods**: `snake_case` (send_message, get_config)
- **Variables**: `snake_case` (session_id, max_retries)
- **Constants**: `UPPER_SNAKE_CASE` (DEFAULT_MODEL, MAX_CONTEXT_LENGTH)
- **Private members**: `_leading_underscore` (_cache, _validate_input)

#### Error Handling

```python
# Use specific exception types
try:
    result = await self.api_client.request(data)
except ConnectionError as e:
    logger.error(f"Connection failed: {e}")
    raise APIError(f"Failed to connect: {e}") from e
except TimeoutError as e:
    logger.warning(f"Request timed out: {e}")
    raise APIError(f"Request timeout: {e}") from e
except Exception as e:
    logger.exception("Unexpected error")
    raise APIError(f"Unexpected error: {e}") from e
```

### Documentation Style

#### Docstrings

Use Google-style docstrings:

```python
async def analyze_code(
    self,
    code: str,
    language: str = "python",
    include_suggestions: bool = True
) -> Dict[str, Any]:
    """
    Analyze code for issues and improvements.
    
    This method performs static analysis on the provided code,
    identifying potential bugs, performance issues, and style violations.
    
    Args:
        code: The source code to analyze.
        language: Programming language of the code.
        include_suggestions: Whether to include improvement suggestions.
        
    Returns:
        Dictionary containing analysis results with keys:
        - 'issues': List of identified issues
        - 'suggestions': List of improvement suggestions (if enabled)
        - 'metrics': Code quality metrics
        
    Raises:
        ValidationError: If the code is invalid or empty.
        UnsupportedLanguageError: If the language is not supported.
        
    Example:
        >>> analyzer = CodeAnalyzer()
        >>> result = await analyzer.analyze_code("def hello(): pass")
        >>> print(f"Found {len(result['issues'])} issues")
        Found 0 issues
        
    Note:
        Analysis quality depends on the complexity of the code.
        Large files may take longer to process.
        
    .. versionadded:: 1.0.0
    .. versionchanged:: 1.1.0
        Added support for TypeScript analysis.
    """
```

#### Comments

```python
# Focus on "why" not "what"
# Use exponential backoff to handle rate limiting gracefully
delay = self.base_delay * (2 ** attempt)

# TODO: Consider adding caching for frequently requested models
# FIXME: Handle edge case where model name contains special characters
# NOTE: This optimization provides 2x performance improvement
```

### Commit Messages

Follow [Conventional Commits](https://conventionalcommits.org/):

```bash
# Format: type(scope): description

# Types: feat, fix, docs, style, refactor, test, chore
feat(models): add support for model performance benchmarking
fix(chat): resolve memory leak in long conversations
docs(api): update ModelManager documentation
test(core): add integration tests for Ollama client
refactor(ui): simplify command parsing logic
chore(deps): update dependencies to latest versions

# For breaking changes
feat(api)!: redesign plugin interface for better extensibility

# Longer description and footer
feat(plugins): add plugin hot-reloading support

Add ability to reload plugins without restarting the application.
This enables faster development iteration and better user experience.

Closes #123
Refs #456
```

## Testing Requirements

### Test Types

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Benchmark critical paths
5. **Security Tests**: Validate security measures

### Test Structure

```python
# tests/unit/test_model_manager.py
import pytest
from unittest.mock import Mock, AsyncMock, patch
from abov3.models.manager import ModelManager
from abov3.core.config import Config

class TestModelManager:
    @pytest.fixture
    def config(self):
        return Config(model=ModelConfig(default_model="test-model"))
    
    @pytest.fixture
    def manager(self, config):
        return ModelManager(config)
    
    @pytest.mark.asyncio
    async def test_list_models_success(self, manager):
        # Arrange
        expected_models = ["model1", "model2"]
        
        with patch.object(manager.client, 'list_models') as mock_list:
            mock_list.return_value = expected_models
            
            # Act
            models = await manager.list_models()
            
            # Assert
            assert models == expected_models
            mock_list.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_install_model_with_progress(self, manager):
        # Test with progress callback
        progress_calls = []
        
        def progress_callback(status):
            progress_calls.append(status)
        
        with patch.object(manager.client, 'pull_model') as mock_pull:
            await manager.install_model("test-model", progress_callback)
            
            mock_pull.assert_called_once_with("test-model", progress_callback)
```

### Test Requirements

- **Coverage**: Aim for >90% code coverage
- **Isolation**: Tests should not depend on external services
- **Mocking**: Mock external dependencies appropriately
- **Fixtures**: Use pytest fixtures for common test data
- **Markers**: Use markers to categorize tests (unit, integration, slow)

### Running Tests

```bash
# Run all tests
pytest

# Run specific test types
pytest tests/unit/
pytest tests/integration/ -m integration

# Run with coverage
pytest --cov=abov3 --cov-report=html

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest tests/unit/test_config.py

# Run with verbose output
pytest -v -s
```

## Documentation

### Types of Documentation

1. **API Documentation**: Docstrings and API references
2. **User Documentation**: Guides and tutorials
3. **Developer Documentation**: Contributing and architecture
4. **Examples**: Code examples and use cases

### Documentation Standards

- **Clarity**: Write for the intended audience
- **Completeness**: Cover all important aspects
- **Examples**: Include practical examples
- **Accuracy**: Keep documentation up-to-date
- **Formatting**: Use consistent formatting

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation locally
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Community discussions and Q&A
- **Pull Requests**: Code review and collaboration
- **Email**: contact@abov3.dev for private matters

### Getting Help

- **Documentation**: Check existing documentation first
- **Search**: Search issues and discussions
- **Ask Questions**: Create discussion topics for questions
- **Be Specific**: Provide context and details

### Recognition

We appreciate all contributions! Contributors are recognized in:

- **CHANGELOG.md**: Major contributions noted in releases
- **README.md**: Contributors section
- **GitHub**: Contributor statistics and badges
- **Community**: Shout-outs in discussions and social media

### Maintainers

Current maintainers:

- **ABOV3 Team**: @abov3-team
- **Core Contributors**: Listed in README.md

### Becoming a Maintainer

Regular contributors may be invited to become maintainers based on:

- **Quality**: Consistent high-quality contributions
- **Expertise**: Deep understanding of the codebase
- **Collaboration**: Positive community interactions
- **Commitment**: Long-term commitment to the project

## Thank You!

We're grateful for your interest in contributing to ABOV3 4 Ollama. Every contribution, no matter how small, helps make the project better for everyone.

**Questions?** Don't hesitate to ask in [GitHub Discussions](https://github.com/abov3/abov3-ollama/discussions) or reach out to contact@abov3.dev.

Happy coding! 🚀