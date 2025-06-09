"""Configuration management for Documatic RAG application.

This module provides a centralized configuration system using Pydantic settings
that supports multiple configuration sources with proper validation and type safety.
"""

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    """Configuration for LLM settings."""

    model: str = Field(default="gpt-4o-mini", description="LLM model name")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Temperature for response generation"
    )
    max_tokens: int = Field(
        default=2000, gt=0, description="Maximum tokens for responses"
    )
    system_prompt: str = Field(
        default="You are a helpful assistant that answers questions about AppPack.io documentation. "
                "Provide accurate, concise answers based on the provided context. "
                "Always cite your sources using [source] format.",
        description="System prompt for the LLM"
    )


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model settings."""

    model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )
    dimensions: int | None = Field(
        default=None, description="Embedding dimensions (None for model default)"
    )
    batch_size: int = Field(
        default=50, gt=0, description="Batch size for embedding generation"
    )
    max_retries: int = Field(
        default=3, ge=0, description="Maximum retries for failed embeddings"
    )
    retry_delay: float = Field(
        default=1.0, ge=0.0, description="Delay between retries in seconds"
    )


class LanceDBConfig(BaseModel):
    """Configuration for LanceDB vector database."""

    db_path: str = Field(default="data/embeddings", description="Path to LanceDB database")
    table_name: str = Field(default="documents", description="Table name for document embeddings")
    create_index: bool = Field(default=True, description="Whether to create vector index")
    index_metric: str = Field(default="cosine", description="Distance metric for vector index")
    nprobes: int = Field(default=20, gt=0, description="Number of probes for vector search")


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""

    chunk_size: int = Field(default=512, gt=0, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=77, ge=0, description="Overlap between chunks in tokens")
    max_chunk_size: int = Field(default=1024, gt=0, description="Maximum chunk size in tokens")
    preserve_sections: bool = Field(default=True, description="Whether to preserve section boundaries")
    min_chunk_size: int = Field(default=100, gt=0, description="Minimum chunk size in tokens")


class SearchConfig(BaseModel):
    """Configuration for search functionality."""

    default_limit: int = Field(default=5, gt=0, description="Default number of search results")
    max_limit: int = Field(default=50, gt=0, description="Maximum number of search results")
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for vector search in hybrid mode")
    fulltext_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for full-text search in hybrid mode")
    rerank_enabled: bool = Field(default=False, description="Whether to enable LLM-based reranking")
    rerank_top_k: int = Field(default=10, gt=0, description="Number of results to rerank")


class ChatConfig(BaseModel):
    """Configuration for chat interface."""

    max_conversation_length: int = Field(default=50, gt=0, description="Maximum conversation turns to keep")
    context_limit: int = Field(default=5, gt=0, description="Number of search results to include in context")
    stream_responses: bool = Field(default=True, description="Whether to stream chat responses")
    save_conversations: bool = Field(default=True, description="Whether to save conversation history")
    conversation_dir: str = Field(default="data/conversations", description="Directory for conversation files")


class EvaluationConfig(BaseModel):
    """Configuration for quality evaluation."""

    questions_per_doc: int = Field(default=3, gt=0, description="Questions to generate per document")
    max_documents: int = Field(default=10, gt=0, description="Maximum documents for evaluation")
    pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Pass threshold for evaluation")
    metrics: list[str] = Field(
        default=["mrr", "recall_at_5", "relevance", "citation_accuracy"],
        description="Evaluation metrics to compute"
    )


class DocumenticConfig(BaseSettings):
    """Main configuration class for Documatic application."""

    model_config = SettingsConfigDict(
        env_prefix="DOCUMATIC_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Data directory settings
    data_dir: str = Field(default="data", description="Base data directory")
    raw_data_dir: str = Field(default="data/raw", description="Raw data directory")

    # API keys
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")

    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    lancedb: LanceDBConfig = Field(default_factory=LanceDBConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # Debug and logging
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize configuration with environment variable loading."""
        # Load OpenAI API key from environment if not provided
        if "openai_api_key" not in kwargs:
            kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY")

        super().__init__(**kwargs)

        # Ensure data directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        directories = [
            self.data_dir,
            self.raw_data_dir,
            self.lancedb.db_path,
            self.chat.conversation_dir,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load_from_file(cls, config_path: str | Path) -> "DocumenticConfig":
        """Load configuration from a file.
        
        Args:
            config_path: Path to configuration file (YAML or TOML)
            
        Returns:
            Loaded configuration instance
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If config file doesn't exist
        """
        import tomllib
        from pathlib import Path

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_path.suffix.lower() in (".toml", ".tml"):
            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)
        elif config_path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped,unused-ignore]
                with open(config_path, encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
            except ImportError as e:
                raise ImportError("PyYAML is required for YAML configuration files") from e
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

        return cls(**config_data)

    def save_to_file(self, config_path: str | Path, format: str = "toml") -> None:
        """Save configuration to a file.
        
        Args:
            config_path: Path to save configuration file
            format: File format ('toml' or 'yaml')
            
        Raises:
            ValueError: If format is not supported
        """
        from pathlib import Path

        import tomli_w

        config_path = Path(config_path)
        config_dict = self.model_dump(exclude={"openai_api_key"}, exclude_none=True)  # Don't save API key or None values

        if format.lower() == "toml":
            with open(config_path, "wb") as f:
                tomli_w.dump(config_dict, f)
        elif format.lower() == "yaml":
            try:
                import yaml  # type: ignore[import-untyped,unused-ignore]
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
            except ImportError as e:
                raise ImportError("PyYAML is required for YAML configuration files") from e
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_openai_api_key(self) -> str:
        """Get OpenAI API key with validation.
        
        Returns:
            OpenAI API key
            
        Raises:
            ValueError: If API key is not configured
        """
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment variable "
                "or provide it in configuration."
            )
        return self.openai_api_key


# Global configuration instance
_config: DocumenticConfig | None = None


def get_config() -> DocumenticConfig:
    """Get the global configuration instance.
    
    Returns:
        Global configuration instance
    """
    global _config
    if _config is None:
        _config = DocumenticConfig()
    return _config


def set_config(config: DocumenticConfig) -> None:
    """Set the global configuration instance.
    
    Args:
        config: Configuration instance to set as global
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to None."""
    global _config
    _config = None
