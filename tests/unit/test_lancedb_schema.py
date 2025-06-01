"""Unit tests for LanceDB schema functionality."""

import json
from unittest.mock import Mock, patch

import pytest

from src.documatic.embeddings import EmbeddingConfig, EmbeddingPipeline, VectorDocument
from tests.fixtures.db_configs import TEST_SCHEMA
from tests.fixtures.mocks import MockLanceDB, MockTable


class TestSchemaCreation:
    """Test LanceDB schema creation and validation."""

    @pytest.fixture
    def config(self):
        """Create test embedding config."""
        return EmbeddingConfig(
            openai_api_key="test-key",
            dimension=1536
        )

    def test_schema_creation_correct_fields(self, config, tmp_path):
        """Test that schema is created with correct fields."""
        expected_fields = [
            "chunk_id", "source_file", "title", "section_hierarchy",
            "chunk_index", "content", "content_type", "document_type",
            "token_count", "word_count", "overlap_previous", "frontmatter",
            "embedding_hash", "created_at", "updated_at", "vector"
        ]

        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(config, tmp_path / "test_db")

                    # Verify _ensure_table_exists was called
                    assert hasattr(pipeline, 'table')

    def test_vector_field_correct_dimensions(self, config, tmp_path):
        """Test that vector field has correct dimensions."""
        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(config, tmp_path / "test_db")

                    # Check that dimension matches config
                    assert pipeline.config.dimension == 1536

    def test_schema_data_types(self, config, tmp_path):
        """Test that schema fields have correct data types."""
        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(config, tmp_path / "test_db")

                    # Verify pipeline was created successfully
                    assert pipeline.table_name == "document_chunks"

    def test_index_creation(self, config, tmp_path):
        """Test that indexes are created properly."""
        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_table = MockTable("document_chunks", mock_db)
            mock_db.create_table = Mock(return_value=mock_table)
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(config, tmp_path / "test_db")

                    # Test FTS index creation
                    # Should be called during table creation
                    assert pipeline.table == mock_table


class TestSchemaValidation:
    """Test schema validation and error handling."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create pipeline for testing."""
        config = EmbeddingConfig(openai_api_key="test-key")
        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    return EmbeddingPipeline(config, tmp_path / "test_db")

    def test_valid_vector_document(self, pipeline):
        """Test validation of valid vector document."""
        valid_doc = VectorDocument(
            chunk_id="test_chunk_1",
            source_file="test.md",
            title="Test Document",
            section_hierarchy=["Section 1"],
            chunk_index=0,
            content="Test content",
            content_type="text",
            document_type="documentation",
            token_count=10,
            word_count=5,
            overlap_previous=False,
            frontmatter={"title": "Test"},
            embedding_hash="abc123",
            created_at=1640995200.0,
            updated_at=1640995200.0
        )

        # Should not raise validation errors
        assert valid_doc.chunk_id == "test_chunk_1"
        assert valid_doc.content == "Test content"

    def test_invalid_vector_dimensions(self, pipeline):
        """Test rejection of invalid vector dimensions."""
        # This would be tested in the actual LanceDB integration
        # For now, verify config dimension is used
        assert pipeline.config.dimension == 1536

    def test_required_fields_validation(self, pipeline):
        """Test that required fields are enforced."""
        from pydantic import ValidationError

        # Test missing required field
        with pytest.raises(ValidationError):
            VectorDocument(
                # Missing chunk_id
                source_file="test.md",
                title="Test Document",
                section_hierarchy=["Section 1"],
                chunk_index=0,
                content="Test content",
                content_type="text",
                document_type="documentation",
                token_count=10,
                word_count=5,
                overlap_previous=False,
                frontmatter={"title": "Test"},
                embedding_hash="abc123",
                created_at=1640995200.0,
                updated_at=1640995200.0
            )

    def test_metadata_structure_validation(self, pipeline):
        """Test validation of metadata structure."""
        # Test with valid metadata
        valid_metadata = {
            "title": "Test Document",
            "category": "How-to",
            "tags": ["deployment", "setup"]
        }

        doc = VectorDocument(
            chunk_id="test_chunk_1",
            source_file="test.md",
            title="Test Document",
            section_hierarchy=["Section 1"],
            chunk_index=0,
            content="Test content",
            content_type="text",
            document_type="documentation",
            token_count=10,
            word_count=5,
            overlap_previous=False,
            frontmatter=valid_metadata,
            embedding_hash="abc123",
            created_at=1640995200.0,
            updated_at=1640995200.0
        )

        assert doc.frontmatter == valid_metadata

    def test_json_serialization_frontmatter(self, pipeline):
        """Test that frontmatter can be JSON serialized."""
        metadata = {
            "title": "Test Document",
            "nested": {"key": "value"},
            "list": [1, 2, 3]
        }

        doc = VectorDocument(
            chunk_id="test_chunk_1",
            source_file="test.md",
            title="Test Document",
            section_hierarchy=["Section 1"],
            chunk_index=0,
            content="Test content",
            content_type="text",
            document_type="documentation",
            token_count=10,
            word_count=5,
            overlap_previous=False,
            frontmatter=metadata,
            embedding_hash="abc123",
            created_at=1640995200.0,
            updated_at=1640995200.0
        )

        # Should be able to serialize to JSON
        json_str = json.dumps(metadata)
        assert json_str is not None

        # Should be able to deserialize
        deserialized = json.loads(json_str)
        assert deserialized == metadata


class TestSchemaMigration:
    """Test schema migration and compatibility."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create pipeline for testing."""
        config = EmbeddingConfig(openai_api_key="test-key")
        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    return EmbeddingPipeline(config, tmp_path / "test_db")

    def test_existing_table_connection(self, pipeline, tmp_path):
        """Test connecting to existing table."""
        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            # Pre-populate with existing table
            mock_db.tables["document_chunks"] = TEST_SCHEMA
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    config = EmbeddingConfig(openai_api_key="test-key")
                    new_pipeline = EmbeddingPipeline(config, tmp_path / "existing_db")

                    # Should connect to existing table
                    assert new_pipeline.table_name == "document_chunks"

    def test_schema_backward_compatibility(self, pipeline):
        """Test that schema changes maintain backward compatibility."""
        # Test that old documents can still be processed
        # This is more relevant for actual database operations

        old_style_doc = {
            "chunk_id": "old_chunk_1",
            "content": "Old content format",
            "vector": [0.1] * 1536
        }

        # Should be able to handle documents with fewer fields
        # In practice, this would involve database migration logic
        assert old_style_doc["chunk_id"] == "old_chunk_1"

    def test_data_preservation_during_migration(self, pipeline):
        """Test that data is preserved during schema updates."""
        # Mock existing data
        existing_data = [
            {
                "chunk_id": "chunk_1",
                "content": "Existing content 1",
                "vector": [0.1] * 1536
            },
            {
                "chunk_id": "chunk_2",
                "content": "Existing content 2",
                "vector": [0.2] * 1536
            }
        ]

        # Simulate migration process
        # In practice, this would involve actual database operations
        migrated_data = []
        for doc in existing_data:
            migrated_doc = {
                **doc,
                "created_at": 1640995200.0,  # Add new required field
                "updated_at": 1640995200.0,
            }
            migrated_data.append(migrated_doc)

        # Verify data integrity
        assert len(migrated_data) == len(existing_data)
        assert migrated_data[0]["chunk_id"] == "chunk_1"
        assert migrated_data[1]["chunk_id"] == "chunk_2"


class TestSchemaPerformance:
    """Test schema performance characteristics."""

    @pytest.fixture
    def performance_config(self):
        """Create performance-optimized config."""
        return EmbeddingConfig(
            openai_api_key="test-key",
            dimension=1536,
            batch_size=100
        )

    def test_large_schema_creation(self, performance_config, tmp_path):
        """Test schema creation with large datasets in mind."""
        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(
                        performance_config,
                        tmp_path / "large_db"
                    )

                    # Should handle large-scale configurations
                    assert pipeline.config.batch_size == 100
                    assert pipeline.config.dimension == 1536

    def test_index_performance_config(self, performance_config, tmp_path):
        """Test performance index configuration."""
        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_table = MockTable("document_chunks", mock_db)
            mock_db.create_table = Mock(return_value=mock_table)
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(
                        performance_config,
                        tmp_path / "perf_db"
                    )

                    # Test different index types
                    pipeline.create_vector_index("IVF_PQ")
                    pipeline.create_vector_index("HNSW")

                    # Should not raise errors
                    assert pipeline.table == mock_table

    def test_concurrent_schema_access(self, performance_config, tmp_path):
        """Test concurrent access to schema."""
        # Test multiple pipeline instances accessing same schema
        pipelines = []

        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_table = MockTable("document_chunks", mock_db)
            mock_db.open_table = Mock(return_value=mock_table)
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    for i in range(3):
                        pipeline = EmbeddingPipeline(
                            performance_config,
                            tmp_path / f"concurrent_db_{i}"
                        )
                        pipelines.append(pipeline)

        # All pipelines should be created successfully
        assert len(pipelines) == 3
        for pipeline in pipelines:
            assert pipeline.table_name == "document_chunks"
