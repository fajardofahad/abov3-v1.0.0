"""
Integration test suite for ABOV3 4 Ollama.

This module tests complete workflows, end-to-end scenarios,
and integration between different components.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from abov3.cli import CLI, CommandProcessor
from abov3.core.api.ollama_client import OllamaClient, ChatMessage, ChatResponse
from abov3.core.app import Application
from abov3.core.config import Config
from abov3.core.context.manager import ContextManager
from abov3.models.manager import ModelManager
from abov3.models.registry import ModelRegistry
from abov3.plugins.base.manager import PluginManager
from abov3.ui.console.repl import REPL
from abov3.utils.security import SecurityValidator


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_conversation_flow(self, test_config, mock_ollama_client):
        """Test a complete conversation workflow."""
        # Initialize application
        app = Application(config=test_config)
        app.ollama_client = mock_ollama_client
        
        # Setup mock responses
        mock_ollama_client.health_check.return_value = True
        mock_ollama_client.list_models.return_value = [
            {"name": "llama3.2:latest", "size": 4000000000}
        ]
        mock_ollama_client.chat.return_value = ChatResponse(
            message=ChatMessage(role="assistant", content="Hello! I can help you."),
            done=True
        )
        
        # Start application
        await app.start()
        
        # Create conversation
        context = app.context_manager.new_context()
        
        # Send message
        response = await app.send_message("Hello, can you help me?")
        
        assert response is not None
        assert "help" in response.lower()
        
        # Check context was updated
        messages = app.context_manager.get_messages()
        assert len(messages) >= 2
        
        # Shutdown
        await app.shutdown()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_model_management_workflow(self, test_config, mock_ollama_client):
        """Test complete model management workflow."""
        # Initialize components
        model_manager = ModelManager(client=mock_ollama_client)
        registry = ModelRegistry()
        
        # Mock model list
        mock_ollama_client.list_models.return_value = [
            {"name": "model1:latest", "size": 1000000000},
            {"name": "model2:latest", "size": 2000000000}
        ]
        
        # Discover models
        models = await model_manager.discover()
        assert len(models) == 2
        
        # Register models
        for model in models:
            registry.register(model)
        
        # Set default model
        registry.set_default("model1:latest")
        assert registry.get_default().name == "model1:latest"
        
        # Pull new model
        mock_ollama_client.pull_model.return_value = True
        success = await model_manager.pull("new-model:latest")
        assert success is True
        
        # Delete model
        mock_ollama_client.delete_model.return_value = True
        deleted = await model_manager.delete("model2:latest")
        assert deleted is True
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_plugin_system_workflow(self, test_config):
        """Test plugin system integration."""
        # Initialize plugin manager
        plugin_manager = PluginManager(config=test_config)
        
        # Create test plugin
        class TestPlugin:
            def __init__(self):
                self.name = "test_plugin"
                self.enabled = True
            
            def initialize(self):
                return True
            
            def execute(self, *args, **kwargs):
                return "Plugin executed"
        
        # Register plugin
        test_plugin = TestPlugin()
        plugin_manager.register(test_plugin)
        
        # Load plugin
        loaded = plugin_manager.load("test_plugin")
        assert loaded is True
        
        # Execute plugin
        result = plugin_manager.execute("test_plugin", "execute")
        assert result == "Plugin executed"
        
        # Unload plugin
        unloaded = plugin_manager.unload("test_plugin")
        assert unloaded is True
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_persistence_workflow(self, test_config, temp_dir):
        """Test session persistence across restarts."""
        session_file = temp_dir / "session.json"
        
        # First session
        app1 = Application(config=test_config)
        context1 = app1.context_manager.new_context()
        
        # Add messages
        app1.context_manager.add_message("user", "First message")
        app1.context_manager.add_message("assistant", "First response")
        
        # Save session
        app1.save_session(session_file)
        
        # New application instance
        app2 = Application(config=test_config)
        
        # Load session
        app2.load_session(session_file)
        
        # Verify messages persisted
        messages = app2.context_manager.get_messages()
        assert len(messages) == 2
        assert messages[0]["content"] == "First message"
        assert messages[1]["content"] == "First response"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, test_config, mock_ollama_client):
        """Test error recovery and retry mechanisms."""
        app = Application(config=test_config)
        app.ollama_client = mock_ollama_client
        
        # Simulate connection failures then recovery
        call_count = 0
        
        async def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return ChatResponse(
                message=ChatMessage(role="assistant", content="Success after retry"),
                done=True
            )
        
        mock_ollama_client.chat.side_effect = failing_then_success
        
        # Should retry and eventually succeed
        response = await app.send_message_with_retry("Test message")
        
        assert response is not None
        assert "Success" in response
        assert call_count == 3


class TestCLIIntegration:
    """Test CLI integration with core components."""
    
    @pytest.mark.integration
    def test_cli_command_processing(self, test_config):
        """Test CLI command processing."""
        cli = CLI(config=test_config)
        processor = CommandProcessor()
        
        # Test various commands
        commands = [
            "/help",
            "/models",
            "/config show",
            "/history",
            "/clear"
        ]
        
        for cmd in commands:
            result = processor.process(cmd)
            assert result is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cli_chat_interaction(self, test_config, mock_ollama_client):
        """Test CLI chat interaction."""
        cli = CLI(config=test_config)
        cli.ollama_client = mock_ollama_client
        
        mock_ollama_client.chat.return_value = ChatResponse(
            message=ChatMessage(role="assistant", content="CLI response"),
            done=True
        )
        
        # Simulate user input
        response = await cli.process_input("Hello from CLI")
        
        assert response is not None
        assert "CLI response" in response
    
    @pytest.mark.integration
    def test_cli_configuration_management(self, test_config, temp_dir):
        """Test CLI configuration management."""
        cli = CLI(config=test_config)
        config_file = temp_dir / "cli_config.toml"
        
        # Save configuration
        cli.save_config(config_file)
        assert config_file.exists()
        
        # Modify configuration
        cli.update_config({"model": {"temperature": 0.5}})
        
        # Load configuration
        new_cli = CLI()
        new_cli.load_config(config_file)
        
        # Original config should be loaded
        assert new_cli.config.model.temperature == test_config.model.temperature


class TestREPLIntegration:
    """Test REPL integration."""
    
    @pytest.mark.integration
    def test_repl_initialization(self, test_config):
        """Test REPL initialization."""
        repl = REPL(config=test_config)
        
        assert repl.config == test_config
        assert repl.prompt is not None
        assert repl.completer is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_repl_command_execution(self, test_config, mock_ollama_client):
        """Test REPL command execution."""
        repl = REPL(config=test_config)
        repl.ollama_client = mock_ollama_client
        
        # Test command execution
        commands = [
            ("/help", "help"),
            ("/models", "models"),
            ("/clear", "clear")
        ]
        
        for cmd, expected in commands:
            result = await repl.execute_command(cmd)
            assert expected in result.lower() or result is not None
    
    @pytest.mark.integration
    def test_repl_autocomplete(self, test_config):
        """Test REPL autocomplete functionality."""
        repl = REPL(config=test_config)
        
        # Test command completion
        completions = repl.completer.get_completions("/mo", 3)
        assert "/models" in completions
        
        # Test file path completion
        completions = repl.completer.get_completions("/load ", 6)
        assert isinstance(completions, list)


class TestSecurityIntegration:
    """Test security integration across components."""
    
    @pytest.mark.integration
    @pytest.mark.security
    def test_input_sanitization_flow(self, test_config):
        """Test input sanitization through the application."""
        app = Application(config=test_config)
        validator = SecurityValidator()
        
        # Test malicious inputs
        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "'; DROP TABLE users; --",
            "../../etc/passwd"
        ]
        
        for malicious in malicious_inputs:
            # Input should be sanitized
            sanitized = app.sanitize_input(malicious)
            assert sanitized != malicious
            
            # Validate sanitized input
            is_safe = validator.validate_input(sanitized)
            assert is_safe is True
    
    @pytest.mark.integration
    @pytest.mark.security
    def test_authentication_flow(self, test_config):
        """Test authentication flow."""
        app = Application(config=test_config)
        
        # Test user authentication
        credentials = {"username": "testuser", "password": "testpass123"}
        
        # Register user
        registered = app.register_user(credentials)
        assert registered is True
        
        # Login
        token = app.login(credentials)
        assert token is not None
        
        # Validate token
        is_valid = app.validate_token(token)
        assert is_valid is True
        
        # Logout
        logged_out = app.logout(token)
        assert logged_out is True
        
        # Token should be invalid after logout
        is_valid = app.validate_token(token)
        assert is_valid is False
    
    @pytest.mark.integration
    @pytest.mark.security
    def test_rate_limiting_integration(self, test_config):
        """Test rate limiting across API calls."""
        app = Application(config=test_config)
        
        # Configure rate limiting
        app.configure_rate_limit(max_requests=5, window=60)
        
        client_id = "test_client"
        
        # Should allow first 5 requests
        for i in range(5):
            allowed = app.check_rate_limit(client_id)
            assert allowed is True
        
        # Should block 6th request
        blocked = app.check_rate_limit(client_id)
        assert blocked is False


class TestPerformanceIntegration:
    """Test performance across integrated components."""
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_chat_sessions(self, test_config, mock_ollama_client):
        """Test handling concurrent chat sessions."""
        mock_ollama_client.chat.return_value = ChatResponse(
            message=ChatMessage(role="assistant", content="Concurrent response"),
            done=True
        )
        
        # Create multiple applications
        apps = [Application(config=test_config) for _ in range(10)]
        for app in apps:
            app.ollama_client = mock_ollama_client
        
        # Send concurrent messages
        tasks = []
        for i, app in enumerate(apps):
            task = app.send_message(f"Message from session {i}")
            tasks.append(task)
        
        import time
        start = time.perf_counter()
        responses = await asyncio.gather(*tasks)
        end = time.perf_counter()
        
        assert len(responses) == 10
        assert all(r is not None for r in responses)
        
        # Should handle 10 concurrent sessions quickly
        assert (end - start) < 2.0
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_large_context_handling(self, test_config):
        """Test handling large conversation contexts."""
        app = Application(config=test_config)
        
        # Add many messages
        import time
        start = time.perf_counter()
        
        for i in range(1000):
            app.context_manager.add_message("user", f"Message {i}")
            app.context_manager.add_message("assistant", f"Response {i}")
        
        end = time.perf_counter()
        
        # Should handle 2000 messages efficiently
        assert (end - start) < 1.0
        
        # Retrieval should be fast
        start = time.perf_counter()
        messages = app.context_manager.get_messages(last_n=100)
        end = time.perf_counter()
        
        assert len(messages) == 100
        assert (end - start) < 0.01
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_model_switching_performance(self, test_config, mock_ollama_client):
        """Test performance of switching between models."""
        app = Application(config=test_config)
        app.ollama_client = mock_ollama_client
        
        models = ["model1", "model2", "model3"]
        mock_ollama_client.list_models.return_value = [
            {"name": m, "size": 1000000000} for m in models
        ]
        
        import time
        start = time.perf_counter()
        
        # Switch models multiple times
        for _ in range(10):
            for model in models:
                app.set_model(model)
                await app.send_message("Test")
        
        end = time.perf_counter()
        
        # Should handle 30 model switches quickly
        assert (end - start) < 5.0


class TestDataPersistenceIntegration:
    """Test data persistence across components."""
    
    @pytest.mark.integration
    def test_complete_state_persistence(self, test_config, temp_dir):
        """Test persisting complete application state."""
        state_dir = temp_dir / "state"
        state_dir.mkdir()
        
        # Create application with state
        app1 = Application(config=test_config)
        
        # Add various data
        app1.context_manager.add_message("user", "Message 1")
        app1.set_model("llama3.2:latest")
        app1.add_plugin("test_plugin")
        
        # Save complete state
        app1.save_state(state_dir)
        
        # Create new application
        app2 = Application(config=test_config)
        
        # Load state
        app2.load_state(state_dir)
        
        # Verify state restored
        messages = app2.context_manager.get_messages()
        assert len(messages) == 1
        assert app2.current_model == "llama3.2:latest"
        assert "test_plugin" in app2.plugins
    
    @pytest.mark.integration
    def test_history_persistence(self, test_config, temp_dir):
        """Test conversation history persistence."""
        history_file = temp_dir / "history.json"
        
        app = Application(config=test_config)
        
        # Create conversations
        for i in range(5):
            app.context_manager.new_context()
            app.context_manager.add_message("user", f"Conversation {i}")
            app.context_manager.add_message("assistant", f"Response {i}")
            app.context_manager.save_current_context()
        
        # Save history
        app.save_history(history_file)
        
        # Load in new application
        new_app = Application(config=test_config)
        new_app.load_history(history_file)
        
        # Verify history loaded
        history = new_app.get_conversation_history()
        assert len(history) == 5
    
    @pytest.mark.integration
    def test_configuration_hot_reload(self, test_config, temp_dir):
        """Test configuration hot reload."""
        config_file = temp_dir / "config.toml"
        
        app = Application(config=test_config)
        
        # Save initial config
        app.save_config(config_file)
        
        # Modify config file
        import toml
        config_data = toml.load(config_file)
        config_data["model"]["temperature"] = 0.3
        toml.dump(config_data, open(config_file, "w"))
        
        # Hot reload
        app.reload_config(config_file)
        
        # Verify config updated
        assert app.config.model.temperature == 0.3


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, test_config):
        """Test graceful degradation when components fail."""
        app = Application(config=test_config)
        
        # Simulate Ollama connection failure
        app.ollama_client = None
        
        # Should fall back to offline mode
        response = await app.send_message("Test message")
        assert response is not None
        assert "offline" in response.lower() or "not available" in response.lower()
        
        # Other features should still work
        app.context_manager.add_message("user", "Offline message")
        messages = app.context_manager.get_messages()
        assert len(messages) > 0
    
    @pytest.mark.integration
    def test_error_propagation(self, test_config):
        """Test proper error propagation through layers."""
        app = Application(config=test_config)
        
        # Test various error scenarios
        error_scenarios = [
            ("invalid_model", ModelNotFoundError),
            ("connection_error", ConnectionError),
            ("validation_error", ValidationError)
        ]
        
        for scenario, expected_error in error_scenarios:
            with pytest.raises(expected_error):
                app.trigger_error(scenario)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cleanup_on_error(self, test_config):
        """Test proper cleanup when errors occur."""
        app = Application(config=test_config)
        
        # Simulate error during operation
        with pytest.raises(Exception):
            async with app:
                raise Exception("Simulated error")
        
        # Resources should be cleaned up
        assert app.is_running is False
        assert app.ollama_client is None
        assert app.session is None


class TestDockerIntegration:
    """Test Docker container integration."""
    
    @pytest.mark.integration
    @pytest.mark.skipif(not os.path.exists("/.dockerenv"), reason="Not in Docker")
    def test_docker_environment(self):
        """Test running in Docker environment."""
        # Check Docker-specific configurations
        assert os.path.exists("/.dockerenv")
        
        # Verify environment variables
        assert os.environ.get("ABOV3_DOCKER") == "true"
        
        # Check volume mounts
        assert Path("/data").exists()
        assert Path("/config").exists()
    
    @pytest.mark.integration
    @pytest.mark.skipif(not os.path.exists("/.dockerenv"), reason="Not in Docker")
    @pytest.mark.asyncio
    async def test_ollama_container_connection(self):
        """Test connection to Ollama container."""
        config = Config()
        config.ollama.host = "http://ollama:11434"
        
        client = OllamaClient(config=config)
        
        async with client:
            # Should connect to Ollama container
            health = await client.health_check()
            assert health is True


class TestKubernetesIntegration:
    """Test Kubernetes deployment integration."""
    
    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("KUBERNETES_SERVICE_HOST"), reason="Not in K8s")
    def test_kubernetes_environment(self):
        """Test running in Kubernetes environment."""
        # Check K8s environment
        assert os.environ.get("KUBERNETES_SERVICE_HOST") is not None
        
        # Verify service account
        assert Path("/var/run/secrets/kubernetes.io/serviceaccount").exists()
    
    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("KUBERNETES_SERVICE_HOST"), reason="Not in K8s")
    def test_configmap_loading(self):
        """Test loading configuration from ConfigMap."""
        config_path = Path("/config/abov3-config.yaml")
        
        if config_path.exists():
            config = Config.from_yaml(config_path)
            assert config is not None