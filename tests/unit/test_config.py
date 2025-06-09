"""Tests for configuration system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from documatic.config import (
    ChunkingConfig,
    DocumenticConfig,
    EmbeddingConfig,
    LLMConfig,
    get_config,
    reset_config,
    set_config,
)


class TestLLMConfig:
    """Test LLM configuration."""

    def test_default_values(self) -> None:
        """Test default LLM configuration values."""
        config = LLMConfig()
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.1
        assert config.max_tokens == 2000
        assert "helpful assistant" in config.system_prompt

    def test_validation(self) -> None:
        """Test LLM configuration validation."""
        # Valid temperature
        config = LLMConfig(temperature=1.5)
        assert config.temperature == 1.5

        # Invalid temperature - too high
        with pytest.raises(ValueError):
            LLMConfig(temperature=3.0)

        # Invalid temperature - negative
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)

        # Invalid max_tokens
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)


class TestEmbeddingConfig:
    """Test embedding configuration."""

    def test_default_values(self) -> None:
        """Test default embedding configuration values."""
        config = EmbeddingConfig()
        assert config.model == "text-embedding-3-small"
        assert config.dimensions is None
        assert config.batch_size == 50
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_validation(self) -> None:
        """Test embedding configuration validation."""
        # Valid batch size
        config = EmbeddingConfig(batch_size=100)
        assert config.batch_size == 100

        # Invalid batch size
        with pytest.raises(ValueError):
            EmbeddingConfig(batch_size=0)

        # Invalid max_retries
        with pytest.raises(ValueError):
            EmbeddingConfig(max_retries=-1)


class TestChunkingConfig:
    """Test chunking configuration."""

    def test_default_values(self) -> None:
        """Test default chunking configuration values."""
        config = ChunkingConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 77
        assert config.max_chunk_size == 1024
        assert config.preserve_sections is True
        assert config.min_chunk_size == 100

    def test_validation(self) -> None:
        """Test chunking configuration validation."""
        # Valid sizes
        config = ChunkingConfig(chunk_size=256, max_chunk_size=512)
        assert config.chunk_size == 256
        assert config.max_chunk_size == 512

        # Invalid chunk size
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=0)

        # Invalid overlap
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_overlap=-1)


class TestDocumenticConfig:
    """Test main configuration class."""

    def setup_method(self) -> None:
        """Reset config before each test."""
        reset_config()

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DocumenticConfig()
        assert config.data_dir == "data"
        assert config.raw_data_dir == "data/raw"
        assert config.debug is False
        assert config.log_level == "INFO"

        # Test nested configs
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.chunking, ChunkingConfig)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_environment_variable_loading(self) -> None:
        """Test loading from environment variables."""
        config = DocumenticConfig()
        assert config.openai_api_key == "test-key"

    @patch.dict(os.environ, {"DOCUMATIC_DEBUG": "true", "DOCUMATIC_LOG_LEVEL": "DEBUG"})
    def test_env_prefix_loading(self) -> None:
        """Test loading with DOCUMATIC_ prefix."""
        config = DocumenticConfig()
        assert config.debug is True
        assert config.log_level == "DEBUG"

    def test_api_key_validation(self) -> None:
        """Test API key validation."""
        config = DocumenticConfig(openai_api_key=None)

        with pytest.raises(ValueError, match="OpenAI API key not configured"):
            config.get_openai_api_key()

        config.openai_api_key = "test-key"
        assert config.get_openai_api_key() == "test-key"

    def test_directory_creation(self) -> None:
        """Test that required directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "test_data"
            config = DocumenticConfig(data_dir=str(data_dir))

            # Directories should be created
            assert Path(config.data_dir).exists()
            assert Path(config.raw_data_dir).exists()
            assert Path(config.lancedb.db_path).exists()
            assert Path(config.chat.conversation_dir).exists()

    @patch.dict(os.environ, {}, clear=True)  # Clear environment for this test
    def test_save_and_load_toml(self) -> None:
        """Test saving and loading TOML configuration."""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            config_path = Path(f.name)

        try:
            # Create config with custom values
            config = DocumenticConfig(
                debug=True,
                log_level="DEBUG",
                openai_api_key="test-key"
            )
            config.llm.temperature = 0.5
            config.embedding.batch_size = 25

            # Save to file (should exclude API key)
            config.save_to_file(config_path, format="toml")

            # Load from file (no environment variables set)
            loaded_config = DocumenticConfig.load_from_file(config_path)

            assert loaded_config.debug is True
            assert loaded_config.log_level == "DEBUG"
            assert loaded_config.llm.temperature == 0.5
            assert loaded_config.embedding.batch_size == 25
            # API key should not be saved and no env var to load from
            assert loaded_config.openai_api_key is None

        finally:
            config_path.unlink(missing_ok=True)

    def test_load_nonexistent_file(self) -> None:
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            DocumenticConfig.load_from_file("nonexistent.toml")

    def test_unsupported_file_format(self) -> None:
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unsupported configuration file format"):
                DocumenticConfig.load_from_file(config_path)
        finally:
            config_path.unlink(missing_ok=True)


class TestGlobalConfig:
    """Test global configuration management."""

    def setup_method(self) -> None:
        """Reset config before each test."""
        reset_config()

    def test_get_config_creates_default(self) -> None:
        """Test that get_config creates default config."""
        config = get_config()
        assert isinstance(config, DocumenticConfig)
        assert config.data_dir == "data"

    def test_set_and_get_config(self) -> None:
        """Test setting and getting global config."""
        custom_config = DocumenticConfig(debug=True, log_level="DEBUG")
        set_config(custom_config)

        retrieved_config = get_config()
        assert retrieved_config is custom_config
        assert retrieved_config.debug is True
        assert retrieved_config.log_level == "DEBUG"

    def test_reset_config(self) -> None:
        """Test resetting global config."""
        # Set custom config
        custom_config = DocumenticConfig(debug=True)
        set_config(custom_config)
        assert get_config().debug is True

        # Reset and get new default
        reset_config()
        new_config = get_config()
        assert new_config is not custom_config
        assert new_config.debug is False


class TestConfigIntegration:
    """Test configuration integration scenarios."""

    def test_cli_override_behavior(self) -> None:
        """Test how CLI parameters should override config values."""
        # Base config
        config = DocumenticConfig()
        config.embedding.batch_size = 25
        config.search.default_limit = 3

        # Simulate CLI overrides
        cli_batch_size = 100
        cli_limit = 10

        # CLI should override when different from defaults
        if cli_batch_size != 50:  # 50 is default
            config.embedding.batch_size = cli_batch_size
        if cli_limit != 5:  # 5 is default
            config.search.default_limit = cli_limit

        assert config.embedding.batch_size == 100
        assert config.search.default_limit == 10

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"})
    def test_api_key_precedence(self) -> None:
        """Test API key precedence: explicit > env > config."""
        # Environment variable should be picked up
        config = DocumenticConfig()
        assert config.openai_api_key == "env-key"

        # Explicit setting should override
        config = DocumenticConfig(openai_api_key="explicit-key")
        assert config.openai_api_key == "explicit-key"

    def test_nested_config_updates(self) -> None:
        """Test updating nested configuration values."""
        config = DocumenticConfig()

        # Update nested values
        config.llm.model = "gpt-4"
        config.llm.temperature = 0.8
        config.embedding.model = "text-embedding-3-large"
        config.chunking.chunk_size = 1024

        assert config.llm.model == "gpt-4"
        assert config.llm.temperature == 0.8
        assert config.embedding.model == "text-embedding-3-large"
        assert config.chunking.chunk_size == 1024
