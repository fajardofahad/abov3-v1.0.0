"""
Configuration management for ABOV3 4 Ollama.

This module handles all configuration settings, including user preferences,
model configurations, API settings, and application behavior.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import toml
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv


class OllamaConfig(BaseModel):
    """Configuration for Ollama API connection."""
    
    host: str = Field(default="http://localhost:11434", description="Ollama server host")
    timeout: int = Field(default=120, description="Request timeout in seconds")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    required_for_startup: bool = Field(default=False, description="Whether Ollama is required for application startup")
    
    @validator("host")
    def validate_host(cls, v: str) -> str:
        """Ensure host is properly formatted."""
        if not v.startswith(("http://", "https://")):
            v = f"http://{v}"
        return v.rstrip("/")


class ModelConfig(BaseModel):
    """Configuration for AI models."""
    
    default_model: str = Field(default="llama3.2:latest", description="Default model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=40, ge=1, description="Top-k sampling")
    max_tokens: int = Field(default=4096, ge=1, description="Maximum tokens in response")
    context_length: int = Field(default=8192, ge=1, description="Context window size")
    repeat_penalty: float = Field(default=1.1, ge=0.0, description="Repetition penalty")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class UIConfig(BaseModel):
    """Configuration for user interface."""
    
    theme: str = Field(default="dark", description="UI theme (dark/light/auto)")
    syntax_highlighting: bool = Field(default=True, description="Enable syntax highlighting")
    line_numbers: bool = Field(default=True, description="Show line numbers in code")
    word_wrap: bool = Field(default=True, description="Enable word wrapping")
    auto_complete: bool = Field(default=True, description="Enable auto-completion")
    vim_mode: bool = Field(default=False, description="Enable vim key bindings")
    show_model_info: bool = Field(default=True, description="Show current model in prompt")
    max_history_display: int = Field(default=50, description="Max history items to display")


class HistoryConfig(BaseModel):
    """Configuration for conversation history."""
    
    max_conversations: int = Field(default=100, description="Maximum conversations to keep")
    auto_save: bool = Field(default=True, description="Auto-save conversations")
    compression: bool = Field(default=True, description="Compress stored conversations")
    search_index: bool = Field(default=True, description="Index conversations for search")


class PluginConfig(BaseModel):
    """Configuration for plugins."""
    
    enabled: List[str] = Field(default_factory=list, description="List of enabled plugins")
    directories: List[str] = Field(default_factory=list, description="Plugin search directories")
    auto_load: bool = Field(default=True, description="Auto-load plugins on startup")


class LoggingConfig(BaseModel):
    """Enhanced configuration for logging system."""
    
    # Basic settings
    level: str = Field(default="INFO", description="Default logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # File logging
    enable_file_logging: bool = Field(default=True, description="Enable file logging")
    log_dir: str = Field(default="logs", description="Log directory path")
    log_filename: str = Field(default="abov3.log", description="Main log filename")
    max_file_size: int = Field(default=50 * 1024 * 1024, description="Max log file size (50MB)")
    backup_count: int = Field(default=10, description="Number of backup files")
    compress_backups: bool = Field(default=True, description="Compress rotated logs")
    
    # Console logging
    enable_console_logging: bool = Field(default=True, description="Enable console logging")
    console_level: str = Field(default="INFO", description="Console log level")
    colored_output: bool = Field(default=True, description="Enable colored console output")
    
    # Structured logging
    enable_json_logging: bool = Field(default=True, description="Enable JSON structured logging")
    json_filename: str = Field(default="abov3.jsonl", description="JSON log filename")
    
    # Performance logging
    enable_performance_logging: bool = Field(default=True, description="Enable performance logging")
    performance_filename: str = Field(default="performance.log", description="Performance log filename")
    slow_query_threshold: float = Field(default=1.0, description="Slow query threshold in seconds")
    
    # Security logging
    enable_security_logging: bool = Field(default=True, description="Enable security event logging")
    security_filename: str = Field(default="security.log", description="Security log filename")
    
    # Remote logging
    enable_remote_logging: bool = Field(default=False, description="Enable remote log shipping")
    remote_host: Optional[str] = Field(default=None, description="Remote log server host")
    remote_port: int = Field(default=514, description="Remote log server port")
    remote_protocol: str = Field(default="UDP", description="Remote protocol (UDP/TCP)")
    
    # Advanced features
    enable_correlation_ids: bool = Field(default=True, description="Enable correlation ID tracking")
    enable_context_logging: bool = Field(default=True, description="Enable context-aware logging")
    buffer_size: int = Field(default=1000, description="Log buffer size for async operations")
    flush_interval: float = Field(default=5.0, description="Buffer flush interval in seconds")
    
    # PII filtering
    enable_pii_filtering: bool = Field(default=True, description="Enable PII data filtering")
    
    # Legacy compatibility
    file_path: Optional[str] = Field(default=None, description="Legacy log file path")
    
    @validator("log_dir")
    def ensure_log_dir_is_relative_to_data_dir(cls, v: str) -> str:
        """Ensure log directory is properly configured."""
        if not v or v == "logs":
            # Will be resolved relative to data directory
            return "logs"
        return v


class Config(BaseModel):
    """Main configuration class for ABOV3 4 Ollama."""
    
    # Core configurations
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Application settings
    debug: bool = Field(default=False, description="Enable debug mode")
    first_run: bool = Field(default=True, description="First time running the application")
    check_updates: bool = Field(default=True, description="Check for updates on startup")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"

    @classmethod
    def get_config_dir(cls) -> Path:
        """Get the configuration directory path."""
        if os.name == "nt":  # Windows
            config_dir = Path(os.environ.get("APPDATA", "")) / "abov3"
        else:  # Unix-like systems
            config_dir = Path.home() / ".config" / "abov3"
        
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    @classmethod
    def get_data_dir(cls) -> Path:
        """Get the data directory path."""
        if os.name == "nt":  # Windows
            data_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "abov3"
        else:  # Unix-like systems
            data_dir = Path.home() / ".local" / "share" / "abov3"
        
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @classmethod
    def get_cache_dir(cls) -> Path:
        """Get the cache directory path."""
        if os.name == "nt":  # Windows
            cache_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "abov3" / "cache"
        else:  # Unix-like systems
            cache_dir = Path.home() / ".cache" / "abov3"
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @classmethod
    def load_from_file(cls, config_path: Optional[Union[str, Path]] = None) -> "Config":
        """Load configuration from a file."""
        if config_path is None:
            config_path = cls.get_config_dir() / "config.toml"
        else:
            config_path = Path(config_path)

        # Load environment variables
        load_dotenv()

        if not config_path.exists():
            # Create default config file
            config = cls()
            config.save_to_file(config_path)
            return config

        try:
            if config_path.suffix.lower() == ".json":
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:  # Assume TOML
                with open(config_path, "r", encoding="utf-8") as f:
                    data = toml.load(f)
            
            # Override with environment variables
            cls._apply_env_overrides(data)
            
            return cls(**data)
        
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using default configuration.")
            return cls()

    def save_to_file(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to a file."""
        if config_path is None:
            config_path = self.get_config_dir() / "config.toml"
        else:
            config_path = Path(config_path)

        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = self.model_dump()
            
            if config_path.suffix.lower() == ".json":
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:  # TOML
                with open(config_path, "w", encoding="utf-8") as f:
                    toml.dump(data, f)
                    
        except Exception as e:
            raise RuntimeError(f"Failed to save config to {config_path}: {e}")

    @staticmethod
    def _apply_env_overrides(data: Dict[str, Any]) -> None:
        """Apply environment variable overrides to configuration data."""
        env_mappings = {
            "ABOV3_OLLAMA_HOST": ("ollama", "host"),
            "ABOV3_OLLAMA_TIMEOUT": ("ollama", "timeout"),
            "ABOV3_DEFAULT_MODEL": ("model", "default_model"),
            "ABOV3_TEMPERATURE": ("model", "temperature"),
            "ABOV3_MAX_TOKENS": ("model", "max_tokens"),
            "ABOV3_DEBUG": ("debug",),
            "ABOV3_LOG_LEVEL": ("logging", "level"),
            "ABOV3_LOG_FILE": ("logging", "file_path"),
        }

        for env_var, key_path in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Navigate to the nested key and set the value
                current = data
                for key in key_path[:-1]:
                    current = current.setdefault(key, {})
                
                # Convert value to appropriate type
                final_key = key_path[-1]
                if final_key in ["timeout", "max_tokens", "max_conversations"]:
                    current[final_key] = int(env_value)
                elif final_key in ["temperature", "top_p", "repeat_penalty"]:
                    current[final_key] = float(env_value)
                elif final_key in ["debug", "auto_save", "syntax_highlighting"]:
                    current[final_key] = env_value.lower() in ("true", "1", "yes", "on")
                else:
                    current[final_key] = env_value

    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from a dictionary."""
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), BaseModel):
                    # Update nested model
                    current_obj = getattr(self, key)
                    for nested_key, nested_value in value.items():
                        setattr(current_obj, nested_key, nested_value)
                else:
                    setattr(self, key, value)

    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for Ollama API calls."""
        return {
            "temperature": self.model.temperature,
            "top_p": self.model.top_p,
            "top_k": self.model.top_k,
            "repeat_penalty": self.model.repeat_penalty,
            "seed": self.model.seed,
        }


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config.load_from_file()
    return _config_instance


def reload_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Reload the global configuration instance."""
    global _config_instance
    _config_instance = Config.load_from_file(config_path)
    return _config_instance


def save_config(config: Optional[Config] = None) -> None:
    """Save the global configuration instance."""
    if config is None:
        config = get_config()
    config.save_to_file()