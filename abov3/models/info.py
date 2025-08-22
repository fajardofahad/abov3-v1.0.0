"""
Model information and metadata classes.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class ModelSize(str, Enum):
    """Model size categories."""
    TINY = "tiny"      # < 1B parameters
    SMALL = "small"    # 1B - 7B parameters  
    MEDIUM = "medium"  # 7B - 13B parameters
    LARGE = "large"    # 13B - 30B parameters
    XLARGE = "xlarge"  # 30B+ parameters


class ModelType(str, Enum):
    """Model type categories."""
    CHAT = "chat"
    CODE = "code"
    INSTRUCT = "instruct"
    EMBEDDING = "embedding"
    VISION = "vision"
    MULTIMODAL = "multimodal"


class ModelCapability(str, Enum):
    """Model capabilities."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    CODE_COMPLETION = "code_completion"
    CHAT = "chat"
    INSTRUCTION_FOLLOWING = "instruction_following"
    REASONING = "reasoning"
    MATHEMATICS = "mathematics"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
    FUNCTION_CALLING = "function_calling"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"


class ModelInfo(BaseModel):
    """Information about an Ollama model."""
    
    name: str = Field(..., description="Model name")
    tag: str = Field(default="latest", description="Model tag/version")
    full_name: str = Field(..., description="Full model name with tag")
    size: int = Field(..., description="Model size in bytes")
    digest: str = Field(..., description="Model digest/hash")
    modified_at: datetime = Field(..., description="Last modified timestamp")
    
    # Optional metadata
    parameter_count: Optional[str] = Field(None, description="Number of parameters")
    quantization: Optional[str] = Field(None, description="Quantization method")
    architecture: Optional[str] = Field(None, description="Model architecture")
    
    @property
    def size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size / (1024 * 1024)
    
    @property
    def size_gb(self) -> float:
        """Get size in gigabytes."""
        return self.size / (1024 * 1024 * 1024)
    
    @property
    def size_category(self) -> ModelSize:
        """Estimate model size category based on file size."""
        gb_size = self.size_gb
        if gb_size < 1:
            return ModelSize.TINY
        elif gb_size < 5:
            return ModelSize.SMALL
        elif gb_size < 10:
            return ModelSize.MEDIUM
        elif gb_size < 20:
            return ModelSize.LARGE
        else:
            return ModelSize.XLARGE


class ModelMetadata(BaseModel):
    """Extended metadata for models."""
    
    # Basic info
    name: str = Field(..., description="Model name")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Model description")
    version: str = Field(default="latest", description="Model version")
    
    # Model characteristics
    model_type: ModelType = Field(..., description="Type of model")
    capabilities: List[ModelCapability] = Field(default_factory=list, description="Model capabilities")
    size_category: ModelSize = Field(..., description="Model size category")
    parameter_count: Optional[str] = Field(None, description="Number of parameters")
    context_length: Optional[int] = Field(None, description="Maximum context length")
    
    # Technical details
    architecture: Optional[str] = Field(None, description="Model architecture")
    quantization: Optional[str] = Field(None, description="Quantization method")
    training_data: Optional[str] = Field(None, description="Training data information")
    license: Optional[str] = Field(None, description="Model license")
    
    # Performance metrics
    performance_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Performance score (0-10)")
    speed_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Speed score (0-10)")
    quality_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Quality score (0-10)")
    
    # Usage statistics
    download_count: int = Field(default=0, description="Number of downloads")
    usage_count: int = Field(default=0, description="Number of times used")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    
    # Configuration
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Default temperature")
    recommended_params: Dict[str, Any] = Field(default_factory=dict, description="Recommended parameters")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    tags: List[str] = Field(default_factory=list, description="Model tags")
    
    def increment_usage(self) -> None:
        """Increment usage count and update last used timestamp."""
        self.usage_count += 1
        self.last_used = datetime.now()
        self.updated_at = datetime.now()
    
    def update_performance_metrics(
        self, 
        performance: Optional[float] = None,
        speed: Optional[float] = None,
        quality: Optional[float] = None
    ) -> None:
        """Update performance metrics."""
        if performance is not None:
            self.performance_score = max(0.0, min(10.0, performance))
        if speed is not None:
            self.speed_score = max(0.0, min(10.0, speed))
        if quality is not None:
            self.quality_score = max(0.0, min(10.0, quality))
        self.updated_at = datetime.now()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag if present."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now()
    
    def has_capability(self, capability: ModelCapability) -> bool:
        """Check if model has a specific capability."""
        return capability in self.capabilities
    
    def is_suitable_for(self, task_type: str) -> bool:
        """Check if model is suitable for a specific task type."""
        task_mapping = {
            "coding": [ModelCapability.CODE_GENERATION, ModelCapability.CODE_COMPLETION],
            "chat": [ModelCapability.CHAT, ModelCapability.INSTRUCTION_FOLLOWING],
            "analysis": [ModelCapability.REASONING, ModelCapability.TEXT_GENERATION],
            "math": [ModelCapability.MATHEMATICS, ModelCapability.REASONING],
            "vision": [ModelCapability.VISION],
            "embedding": [ModelCapability.EMBEDDINGS],
        }
        
        required_caps = task_mapping.get(task_type.lower(), [])
        return any(cap in self.capabilities for cap in required_caps)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.dict()
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.json(indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create instance from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ModelMetadata":
        """Create instance from JSON string."""
        return cls.parse_raw(json_str)