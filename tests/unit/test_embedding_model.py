"""Unit tests for embedding model functionality."""

import os
from unittest.mock import patch

import pytest

from src.documatic.embeddings import EmbeddingConfig, EmbeddingPipeline
from tests.fixtures.mocks import MockEmbeddingAPI, RateLimitError


class TestEmbeddingModel:
    """Test embedding model initialization and configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = EmbeddingConfig()

        assert config.model_name == "text-embedding-3-small"
        assert config.batch_size == 50
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.dimension == 1536
        assert config.openai_api_key is None

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = EmbeddingConfig(
            model_name="text-embedding-ada-002",
            batch_size=100,
            max_retries=5,
            retry_delay=2.0,
            dimension=512,
            openai_api_key="test-key"
        )

        assert config.model_name == "text-embedding-ada-002"
        assert config.batch_size == 100
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.dimension == 512
        assert config.openai_api_key == "test-key"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"})
    def test_pipeline_initialization_with_env_key(self, tmp_path):
        """Test pipeline initialization with environment API key."""
        config = EmbeddingConfig()

        # Should not raise exception with env key set
        pipeline = EmbeddingPipeline(config, tmp_path / "test_db")
        assert pipeline.config == config
        assert pipeline.db_path == tmp_path / "test_db"

    def test_pipeline_initialization_without_key(self, tmp_path):
        """Test pipeline initialization fails without API key."""
        config = EmbeddingConfig()

        # Clear any existing env key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key required"):
                EmbeddingPipeline(config, tmp_path / "test_db")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_pipeline_initialization_with_config_key(self, tmp_path):
        """Test pipeline initialization with config API key."""
        config = EmbeddingConfig(openai_api_key="config-key")

        # Should use config key over env key
        pipeline = EmbeddingPipeline(config, tmp_path / "test_db")
        assert pipeline.config.openai_api_key == "config-key"


class TestEmbeddingGeneration:
    """Test embedding generation functionality."""

    @pytest.fixture
    def mock_api(self):
        """Create mock embedding API."""
        return MockEmbeddingAPI(dimension=1536)

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create embedding pipeline for testing."""
        config = EmbeddingConfig(openai_api_key="test-key")
        with patch("src.documatic.embeddings.lancedb"):
            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    return EmbeddingPipeline(config, tmp_path / "test_db")

    def test_single_text_embedding(self, mock_api):
        """Test generating embedding for single text."""
        text = "Test document content"
        embeddings = mock_api.embed([text])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536
        assert all(isinstance(x, float) for x in embeddings[0])

    def test_embedding_dimension_validation(self, mock_api):
        """Test embedding dimension is correct."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = mock_api.embed(texts)

        assert len(embeddings) == 3
        for embedding in embeddings:
            assert len(embedding) == 1536

    def test_deterministic_results(self, mock_api):
        """Test that same input produces same embedding."""
        text = "Consistent input text"

        embedding1 = mock_api.embed([text])[0]
        embedding2 = mock_api.embed([text])[0]

        assert embedding1 == embedding2

    def test_empty_string_handling(self, mock_api):
        """Test handling of empty strings."""
        embeddings = mock_api.embed([""])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536

    def test_unicode_handling(self, mock_api):
        """Test handling of unicode characters."""
        unicode_text = "Unicode: 中文, العربية, Ελληνικά, русский 🚀"
        embeddings = mock_api.embed([unicode_text])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536

    def test_special_characters(self, mock_api):
        """Test handling of special characters."""
        special_text = "Special chars: @#$%^&*()_+-=[]{}|;':\".,<>?/~`"
        embeddings = mock_api.embed([special_text])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536


class TestTokenLimits:
    """Test token limit handling."""

    @pytest.fixture
    def mock_api(self):
        """Create mock embedding API."""
        return MockEmbeddingAPI(dimension=1536)

    def test_normal_length_text(self, mock_api):
        """Test processing normal length text."""
        # Normal text under token limits
        text = "This is a normal length text that should process fine."
        embeddings = mock_api.embed([text])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536

    def test_long_text_handling(self, mock_api):
        """Test handling of very long text."""
        # Very long text that would exceed token limits
        long_text = "Very long text " * 1000
        embeddings = mock_api.embed([long_text])

        # Should still process (truncation would happen in real implementation)
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536

    def test_multiple_long_texts(self, mock_api):
        """Test processing multiple long texts."""
        long_texts = ["Long text " * 500 for _ in range(5)]
        embeddings = mock_api.embed(long_texts)

        assert len(embeddings) == 5
        for embedding in embeddings:
            assert len(embedding) == 1536


class TestErrorHandling:
    """Test error handling in embedding generation."""

    def test_rate_limit_handling(self):
        """Test handling of rate limit errors."""
        mock_api = MockEmbeddingAPI()
        mock_api.rate_limit_after = 2

        # First two calls should work
        mock_api.embed(["text1"])
        mock_api.embed(["text2"])

        # Third call should raise rate limit error
        with pytest.raises(RateLimitError):
            mock_api.embed(["text3"])

    def test_api_failure_handling(self):
        """Test handling of general API failures."""
        mock_api = MockEmbeddingAPI()
        mock_api.fail_after = 1

        # First call should work
        mock_api.embed(["text1"])

        # Second call should fail
        with pytest.raises(Exception, match="API failure"):
            mock_api.embed(["text2"])

    def test_network_timeout_simulation(self):
        """Test handling of network timeouts."""
        mock_api = MockEmbeddingAPI()
        mock_api.response_delay = 0.1  # Small delay for testing

        # Should still work with delay
        embeddings = mock_api.embed(["test text"])
        assert len(embeddings) == 1


class TestContentHashing:
    """Test content hashing for change detection."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create embedding pipeline for testing."""
        config = EmbeddingConfig(openai_api_key="test-key")
        with patch("src.documatic.embeddings.lancedb"):
            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    return EmbeddingPipeline(config, tmp_path / "test_db")

    def test_consistent_hashing(self, pipeline):
        """Test that same content produces same hash."""
        content = "Test content for hashing"

        hash1 = pipeline._compute_content_hash(content)
        hash2 = pipeline._compute_content_hash(content)

        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated to 16 chars

    def test_different_content_different_hash(self, pipeline):
        """Test that different content produces different hashes."""
        content1 = "First content"
        content2 = "Second content"

        hash1 = pipeline._compute_content_hash(content1)
        hash2 = pipeline._compute_content_hash(content2)

        assert hash1 != hash2

    def test_unicode_content_hashing(self, pipeline):
        """Test hashing of unicode content."""
        unicode_content = "Unicode content: 中文, العربية, Ελληνικά"

        hash_value = pipeline._compute_content_hash(unicode_content)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 16

    def test_empty_content_hashing(self, pipeline):
        """Test hashing of empty content."""
        empty_content = ""

        hash_value = pipeline._compute_content_hash(empty_content)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 16
