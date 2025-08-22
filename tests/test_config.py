"""
Test suite for configuration management.

This module tests all aspects of configuration loading, validation,
environment variable handling, and configuration persistence.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, mock_open, patch

import pytest
import toml
import yaml
from pydantic import ValidationError

from abov3.core.config import (
    Config,
    ConfigLoader,
    ConfigValidator,
    HistoryConfig,
    LoggingConfig,
    ModelConfig,
    OllamaConfig,
    PluginConfig,
    UIConfig,
    get_config,
    load_config,
    save_config,
    validate_config,
)


class TestOllamaConfig:
    """Test cases for OllamaConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = OllamaConfig()
        assert config.host == "http://localhost:11434"
        assert config.timeout == 120
        assert config.verify_ssl is True
        assert config.max_retries == 3
    
    def test_host_validation(self):
        """Test host URL validation and formatting."""
        # Test adding http:// prefix
        config = OllamaConfig(host="localhost:11434")
        assert config.host == "http://localhost:11434"
        
        # Test removing trailing slash
        config = OllamaConfig(host="http://localhost:11434/")
        assert config.host == "http://localhost:11434"
        
        # Test https:// preservation
        config = OllamaConfig(host="https://secure.ollama.com")
        assert config.host == "https://secure.ollama.com"
    
    def test_invalid_timeout(self):
        """Test invalid timeout values."""
        with pytest.raises(ValidationError):
            OllamaConfig(timeout=-1)
        
        with pytest.raises(ValidationError):
            OllamaConfig(timeout="not_a_number")
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = OllamaConfig(
            host="http://custom:8080",
            timeout=60,
            verify_ssl=False,
            max_retries=5
        )
        assert config.host == "http://custom:8080"
        assert config.timeout == 60
        assert config.verify_ssl is False
        assert config.max_retries == 5


class TestModelConfig:
    """Test cases for ModelConfig."""
    
    def test_default_values(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert config.default_model == "llama3.2:latest"
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.max_tokens == 4096
        assert config.context_length == 8192
        assert config.repeat_penalty == 1.1
        assert config.seed is None
    
    def test_temperature_validation(self):
        """Test temperature parameter validation."""
        # Valid temperatures
        config = ModelConfig(temperature=0.0)
        assert config.temperature == 0.0
        
        config = ModelConfig(temperature=2.0)
        assert config.temperature == 2.0
        
        # Invalid temperatures
        with pytest.raises(ValidationError):
            ModelConfig(temperature=-0.1)
        
        with pytest.raises(ValidationError):
            ModelConfig(temperature=2.1)
    
    def test_top_p_validation(self):
        """Test top_p parameter validation."""
        # Valid values
        config = ModelConfig(top_p=0.0)
        assert config.top_p == 0.0
        
        config = ModelConfig(top_p=1.0)
        assert config.top_p == 1.0
        
        # Invalid values
        with pytest.raises(ValidationError):
            ModelConfig(top_p=-0.1)
        
        with pytest.raises(ValidationError):
            ModelConfig(top_p=1.1)
    
    def test_top_k_validation(self):
        """Test top_k parameter validation."""
        # Valid values
        config = ModelConfig(top_k=1)
        assert config.top_k == 1
        
        config = ModelConfig(top_k=100)
        assert config.top_k == 100
        
        # Invalid values
        with pytest.raises(ValidationError):
            ModelConfig(top_k=0)
        
        with pytest.raises(ValidationError):
            ModelConfig(top_k=-1)
    
    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid values
        config = ModelConfig(max_tokens=1)
        assert config.max_tokens == 1
        
        config = ModelConfig(max_tokens=100000)
        assert config.max_tokens == 100000
        
        # Invalid values
        with pytest.raises(ValidationError):
            ModelConfig(max_tokens=0)
        
        with pytest.raises(ValidationError):
            ModelConfig(max_tokens=-100)
    
    def test_seed_reproducibility(self):
        """Test seed parameter for reproducibility."""
        config = ModelConfig(seed=42)
        assert config.seed == 42
        
        config = ModelConfig(seed=None)
        assert config.seed is None


class TestUIConfig:
    """Test cases for UIConfig."""
    
    def test_default_values(self):
        """Test default UI configuration."""
        config = UIConfig()
        assert config.theme == "dark"
        assert config.syntax_highlighting is True
        assert config.line_numbers is True
        assert config.word_wrap is True
        assert config.auto_complete is True
        assert config.vim_mode is False
        assert config.show_model_info is True
        assert config.max_history_display == 50
    
    def test_theme_options(self):
        """Test theme configuration options."""
        for theme in ["dark", "light", "auto"]:
            config = UIConfig(theme=theme)
            assert config.theme == theme
    
    def test_boolean_flags(self):
        """Test boolean configuration flags."""
        config = UIConfig(
            syntax_highlighting=False,
            line_numbers=False,
            word_wrap=False,
            auto_complete=False,
            vim_mode=True,
            show_model_info=False
        )
        assert config.syntax_highlighting is False
        assert config.line_numbers is False
        assert config.word_wrap is False
        assert config.auto_complete is False
        assert config.vim_mode is True
        assert config.show_model_info is False


class TestHistoryConfig:
    """Test cases for HistoryConfig."""
    
    def test_default_values(self):
        """Test default history configuration."""
        config = HistoryConfig()
        assert config.max_conversations == 100
        assert config.auto_save is True
        assert config.compression is True
        assert config.search_index is True
    
    def test_max_conversations_validation(self):
        """Test max_conversations validation."""
        config = HistoryConfig(max_conversations=1)
        assert config.max_conversations == 1
        
        config = HistoryConfig(max_conversations=10000)
        assert config.max_conversations == 10000
    
    def test_feature_flags(self):
        """Test history feature flags."""
        config = HistoryConfig(
            auto_save=False,
            compression=False,
            search_index=False
        )
        assert config.auto_save is False
        assert config.compression is False
        assert config.search_index is False


class TestPluginConfig:
    """Test cases for PluginConfig."""
    
    def test_default_values(self):
        """Test default plugin configuration."""
        config = PluginConfig()
        assert config.enabled == []
        assert config.directories == []
        assert config.auto_load is True
    
    def test_plugin_lists(self):
        """Test plugin list configuration."""
        config = PluginConfig(
            enabled=["plugin1", "plugin2"],
            directories=["/path/to/plugins", "/another/path"]
        )
        assert "plugin1" in config.enabled
        assert "plugin2" in config.enabled
        assert "/path/to/plugins" in config.directories
        assert "/another/path" in config.directories
    
    def test_auto_load_flag(self):
        """Test auto_load configuration."""
        config = PluginConfig(auto_load=False)
        assert config.auto_load is False


class TestLoggingConfig:
    """Test cases for LoggingConfig."""
    
    def test_default_values(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.file_path is None
        assert config.max_file_size == 10485760  # 10MB
        assert config.backup_count == 5
        assert "%(asctime)s" in config.format
    
    def test_log_levels(self):
        """Test log level configuration."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LoggingConfig(level=level)
            assert config.level == level
    
    def test_file_configuration(self):
        """Test file logging configuration."""
        config = LoggingConfig(
            file_path="/var/log/abov3.log",
            max_file_size=5242880,  # 5MB
            backup_count=10
        )
        assert config.file_path == "/var/log/abov3.log"
        assert config.max_file_size == 5242880
        assert config.backup_count == 10
    
    def test_custom_format(self):
        """Test custom log format."""
        custom_format = "%(levelname)s: %(message)s"
        config = LoggingConfig(format=custom_format)
        assert config.format == custom_format


class TestMainConfig:
    """Test cases for main Config class."""
    
    def test_default_config(self):
        """Test default configuration initialization."""
        config = Config()
        assert isinstance(config.ollama, OllamaConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.ui, UIConfig)
        assert isinstance(config.history, HistoryConfig)
        assert isinstance(config.plugins, PluginConfig)
        assert isinstance(config.logging, LoggingConfig)
    
    def test_nested_config(self):
        """Test nested configuration initialization."""
        config = Config(
            ollama=OllamaConfig(host="http://custom:11434"),
            model=ModelConfig(temperature=0.5),
            ui=UIConfig(theme="light")
        )
        assert config.ollama.host == "http://custom:11434"
        assert config.model.temperature == 0.5
        assert config.ui.theme == "light"
    
    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = Config()
        config_dict = config.dict()
        
        assert isinstance(config_dict, dict)
        assert "ollama" in config_dict
        assert "model" in config_dict
        assert "ui" in config_dict
        assert config_dict["ollama"]["host"] == "http://localhost:11434"
    
    def test_config_to_json(self):
        """Test configuration serialization to JSON."""
        config = Config()
        config_json = config.json()
        
        assert isinstance(config_json, str)
        parsed = json.loads(config_json)
        assert parsed["ollama"]["host"] == "http://localhost:11434"
    
    def test_config_copy(self):
        """Test configuration deep copy."""
        config1 = Config()
        config2 = config1.copy(deep=True)
        
        # Modify config2
        config2.ollama.host = "http://modified:11434"
        
        # Original should be unchanged
        assert config1.ollama.host == "http://localhost:11434"
        assert config2.ollama.host == "http://modified:11434"


class TestConfigEnvironmentVariables:
    """Test cases for environment variable configuration."""
    
    def test_env_var_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {
            "ABOV3_OLLAMA_HOST": "http://env:11434",
            "ABOV3_MODEL_TEMPERATURE": "0.5",
            "ABOV3_UI_THEME": "light"
        }):
            # Assuming load_config respects env vars
            config = Config()
            # This would need actual implementation
            # assert config.ollama.host == "http://env:11434"
            # assert config.model.temperature == 0.5
            # assert config.ui.theme == "light"
    
    def test_env_var_type_conversion(self):
        """Test environment variable type conversion."""
        with patch.dict(os.environ, {
            "ABOV3_OLLAMA_TIMEOUT": "60",
            "ABOV3_OLLAMA_VERIFY_SSL": "false",
            "ABOV3_MODEL_MAX_TOKENS": "2048"
        }):
            # Test integer conversion
            # Test boolean conversion
            # This would need actual implementation
            pass


class TestConfigFileLoading:
    """Test cases for configuration file loading."""
    
    def test_load_toml_config(self, temp_dir):
        """Test loading configuration from TOML file."""
        config_file = temp_dir / "config.toml"
        config_data = {
            "ollama": {"host": "http://toml:11434"},
            "model": {"temperature": 0.8}
        }
        
        with open(config_file, "w") as f:
            toml.dump(config_data, f)
        
        # This would need actual implementation
        # config = load_config(config_file)
        # assert config.ollama.host == "http://toml:11434"
        # assert config.model.temperature == 0.8
    
    def test_load_json_config(self, temp_dir):
        """Test loading configuration from JSON file."""
        config_file = temp_dir / "config.json"
        config_data = {
            "ollama": {"host": "http://json:11434"},
            "model": {"temperature": 0.6}
        }
        
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        
        # This would need actual implementation
        # config = load_config(config_file)
        # assert config.ollama.host == "http://json:11434"
        # assert config.model.temperature == 0.6
    
    def test_load_yaml_config(self, temp_dir):
        """Test loading configuration from YAML file."""
        config_file = temp_dir / "config.yaml"
        config_data = {
            "ollama": {"host": "http://yaml:11434"},
            "model": {"temperature": 0.9}
        }
        
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        # This would need actual implementation
        # config = load_config(config_file)
        # assert config.ollama.host == "http://yaml:11434"
        # assert config.model.temperature == 0.9
    
    def test_config_file_not_found(self):
        """Test handling of missing configuration file."""
        # This would need actual implementation
        # with pytest.raises(FileNotFoundError):
        #     load_config("/nonexistent/config.toml")
        pass
    
    def test_invalid_config_format(self, temp_dir):
        """Test handling of invalid configuration format."""
        config_file = temp_dir / "invalid.txt"
        config_file.write_text("This is not valid config")
        
        # This would need actual implementation
        # with pytest.raises(ValidationError):
        #     load_config(config_file)


class TestConfigValidation:
    """Test cases for configuration validation."""
    
    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = Config()
        # This would need actual implementation
        # assert validate_config(config) is True
    
    def test_invalid_model_params(self):
        """Test validation of invalid model parameters."""
        with pytest.raises(ValidationError):
            Config(model=ModelConfig(temperature=3.0))
    
    def test_invalid_url_format(self):
        """Test validation of invalid URL formats."""
        # This would catch malformed URLs
        config = OllamaConfig(host="not_a_url")
        # Should be corrected to http://not_a_url
        assert config.host.startswith("http://")
    
    def test_mutually_exclusive_options(self):
        """Test validation of mutually exclusive options."""
        # Example: can't have both vim_mode and custom keybindings
        # This would need actual implementation
        pass


class TestConfigMerging:
    """Test cases for configuration merging."""
    
    def test_merge_configs(self):
        """Test merging multiple configuration sources."""
        default_config = Config()
        file_config = Config(
            ollama=OllamaConfig(host="http://file:11434")
        )
        env_config = Config(
            model=ModelConfig(temperature=0.5)
        )
        
        # Merge order: default -> file -> env
        # This would need actual implementation
        # merged = merge_configs(default_config, file_config, env_config)
        # assert merged.ollama.host == "http://file:11434"
        # assert merged.model.temperature == 0.5
    
    def test_partial_config_merge(self):
        """Test merging partial configurations."""
        base = Config()
        partial = {"model": {"temperature": 0.3}}
        
        # This would need actual implementation
        # merged = base.merge(partial)
        # assert merged.model.temperature == 0.3
        # assert merged.ollama.host == base.ollama.host


class TestConfigPersistence:
    """Test cases for configuration persistence."""
    
    def test_save_config_toml(self, temp_dir):
        """Test saving configuration to TOML file."""
        config = Config(
            ollama=OllamaConfig(host="http://save:11434"),
            model=ModelConfig(temperature=0.4)
        )
        
        config_file = temp_dir / "saved.toml"
        # This would need actual implementation
        # save_config(config, config_file)
        
        # assert config_file.exists()
        # loaded = toml.load(config_file)
        # assert loaded["ollama"]["host"] == "http://save:11434"
    
    def test_save_config_json(self, temp_dir):
        """Test saving configuration to JSON file."""
        config = Config()
        config_file = temp_dir / "saved.json"
        
        # This would need actual implementation
        # save_config(config, config_file)
        
        # assert config_file.exists()
        # loaded = json.load(open(config_file))
        # assert loaded["ollama"]["host"] == "http://localhost:11434"
    
    def test_config_backup(self, temp_dir):
        """Test configuration backup creation."""
        config_file = temp_dir / "config.toml"
        config_file.write_text("original content")
        
        # This would need actual implementation
        # backup_file = create_config_backup(config_file)
        # assert backup_file.exists()
        # assert "backup" in backup_file.name


class TestConfigSecurity:
    """Test cases for configuration security."""
    
    def test_sensitive_data_masking(self):
        """Test masking of sensitive configuration data."""
        config = Config()
        # Add some sensitive data
        config.ollama.host = "http://user:password@host:11434"
        
        # This would need actual implementation
        # masked = config.mask_sensitive()
        # assert "password" not in str(masked)
        # assert "***" in str(masked)
    
    def test_secure_file_permissions(self, temp_dir):
        """Test secure file permissions for config files."""
        config_file = temp_dir / "secure.toml"
        config = Config()
        
        # This would need actual implementation
        # save_config(config, config_file, secure=True)
        # Check file permissions (Unix only)
        # if os.name != 'nt':
        #     stat_info = os.stat(config_file)
        #     assert stat_info.st_mode & 0o077 == 0
    
    def test_config_encryption(self, temp_dir):
        """Test configuration encryption/decryption."""
        # This would need actual implementation
        # config = Config()
        # encrypted = encrypt_config(config, password="secret")
        # decrypted = decrypt_config(encrypted, password="secret")
        # assert decrypted == config
        pass


class TestConfigDefaults:
    """Test cases for configuration defaults."""
    
    def test_development_defaults(self):
        """Test development environment defaults."""
        with patch.dict(os.environ, {"ABOV3_ENV": "development"}):
            # This would need actual implementation
            # config = load_config()
            # assert config.logging.level == "DEBUG"
            # assert config.ollama.verify_ssl is False
            pass
    
    def test_production_defaults(self):
        """Test production environment defaults."""
        with patch.dict(os.environ, {"ABOV3_ENV": "production"}):
            # This would need actual implementation
            # config = load_config()
            # assert config.logging.level == "WARNING"
            # assert config.ollama.verify_ssl is True
            pass
    
    def test_testing_defaults(self):
        """Test testing environment defaults."""
        with patch.dict(os.environ, {"ABOV3_ENV": "testing"}):
            # This would need actual implementation
            # config = load_config()
            # assert config.logging.level == "DEBUG"
            # assert config.history.auto_save is False
            pass


class TestConfigPerformance:
    """Performance tests for configuration management."""
    
    @pytest.mark.performance
    def test_config_load_time(self, benchmark_timer):
        """Test configuration loading performance."""
        with benchmark_timer:
            for _ in range(100):
                config = Config()
        
        assert benchmark_timer.average < 0.01  # Should load in < 10ms
    
    @pytest.mark.performance
    def test_config_validation_time(self, benchmark_timer):
        """Test configuration validation performance."""
        config = Config()
        
        with benchmark_timer:
            for _ in range(1000):
                config.dict()  # Force validation
        
        assert benchmark_timer.average < 0.001  # Should validate in < 1ms
    
    @pytest.mark.performance
    def test_large_config_handling(self):
        """Test handling of large configuration objects."""
        # Create a config with many plugins
        config = Config(
            plugins=PluginConfig(
                enabled=[f"plugin_{i}" for i in range(1000)],
                directories=[f"/path/{i}" for i in range(100)]
            )
        )
        
        # Should handle large lists efficiently
        assert len(config.plugins.enabled) == 1000
        assert len(config.plugins.directories) == 100