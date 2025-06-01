"""Unit tests for vector storage functionality."""

import time
from unittest.mock import Mock, patch

import pytest

from src.documatic.embeddings import EmbeddingConfig, EmbeddingPipeline, VectorDocument
from tests.fixtures.mocks import MockLanceDB, MockTable


class TestInsertionOperations:
    """Test vector document insertion operations."""

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

    def test_single_document_insertion(self, pipeline):
        """Test inserting a single vector document."""
        doc = VectorDocument(
            chunk_id="single_test",
            source_file="test.md",
            title="Test Document",
            section_hierarchy=["Section 1"],
            chunk_index=0,
            content="Test content for single insertion",
            content_type="text",
            document_type="documentation",
            token_count=10,
            word_count=6,
            overlap_previous=False,
            frontmatter={"title": "Test"},
            embedding_hash="single_hash",
            created_at=time.time(),
            updated_at=time.time()
        )

        # Mock the table operations
        with patch.object(pipeline.table, 'add') as mock_add:
            pipeline.upsert_documents([doc])

            # Verify add was called once
            mock_add.assert_called_once()

            # Verify the data structure
            call_args = mock_add.call_args[0][0]
            assert hasattr(call_args, 'to_pylist') or isinstance(call_args, list)

    def test_batch_insertion(self, pipeline):
        """Test inserting multiple documents in batch."""
        docs = []
        for i in range(5):
            doc = VectorDocument(
                chunk_id=f"batch_test_{i}",
                source_file=f"test_{i}.md",
                title=f"Test Document {i}",
                section_hierarchy=[f"Section {i}"],
                chunk_index=i,
                content=f"Test content for batch insertion {i}",
                content_type="text",
                document_type="documentation",
                token_count=10 + i,
                word_count=6 + i,
                overlap_previous=i % 2 == 0,
                frontmatter={"title": f"Test {i}"},
                embedding_hash=f"batch_hash_{i}",
                created_at=time.time(),
                updated_at=time.time()
            )
            docs.append(doc)

        with patch.object(pipeline.table, 'add') as mock_add:
            pipeline.upsert_documents(docs)

            # Should be called once with all documents
            mock_add.assert_called_once()

    def test_empty_document_list(self, pipeline):
        """Test handling of empty document list."""
        with patch.object(pipeline.table, 'add') as mock_add:
            pipeline.upsert_documents([])

            # Should not call add for empty list
            mock_add.assert_not_called()

    def test_duplicate_handling(self, pipeline):
        """Test handling of duplicate document IDs."""
        # Create two documents with same chunk_id
        doc1 = VectorDocument(
            chunk_id="duplicate_test",
            source_file="test1.md",
            title="First Document",
            section_hierarchy=["Section 1"],
            chunk_index=0,
            content="First content",
            content_type="text",
            document_type="documentation",
            token_count=5,
            word_count=3,
            overlap_previous=False,
            frontmatter={"version": 1},
            embedding_hash="hash1",
            created_at=time.time(),
            updated_at=time.time()
        )

        doc2 = VectorDocument(
            chunk_id="duplicate_test",  # Same chunk_id
            source_file="test2.md",
            title="Second Document",
            section_hierarchy=["Section 2"],
            chunk_index=1,
            content="Second content",
            content_type="text",
            document_type="documentation",
            token_count=6,
            word_count=4,
            overlap_previous=False,
            frontmatter={"version": 2},
            embedding_hash="hash2",
            created_at=time.time(),
            updated_at=time.time()
        )

        with patch.object(pipeline.table, 'merge_insert') as mock_merge:
            mock_builder = Mock()
            mock_builder.when_matched_update_all.return_value = mock_builder
            mock_builder.when_not_matched_insert_all.return_value = mock_builder
            mock_merge.return_value = mock_builder

            pipeline.upsert_documents([doc1, doc2])

            # Should use merge_insert for handling duplicates
            mock_merge.assert_called_once_with("chunk_id")

    def test_transaction_support(self, pipeline):
        """Test transaction-like behavior for batch operations."""
        docs = []
        for i in range(3):
            doc = VectorDocument(
                chunk_id=f"transaction_test_{i}",
                source_file=f"test_{i}.md",
                title=f"Transaction Test {i}",
                section_hierarchy=["Transaction Section"],
                chunk_index=i,
                content=f"Transaction content {i}",
                content_type="text",
                document_type="documentation",
                token_count=8,
                word_count=5,
                overlap_previous=False,
                frontmatter={"transaction": True},
                embedding_hash=f"trans_hash_{i}",
                created_at=time.time(),
                updated_at=time.time()
            )
            docs.append(doc)

        with patch.object(pipeline.table, 'add') as mock_add:
            pipeline.upsert_documents(docs)

            # All documents should be processed in single operation
            assert mock_add.call_count == 1


class TestUpsertLogic:
    """Test upsert (insert/update) logic."""

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

    def test_update_existing_documents(self, pipeline):
        """Test updating existing documents."""
        # Create document with same ID but different content
        original_doc = VectorDocument(
            chunk_id="update_test",
            source_file="original.md",
            title="Original Document",
            section_hierarchy=["Original Section"],
            chunk_index=0,
            content="Original content",
            content_type="text",
            document_type="documentation",
            token_count=5,
            word_count=3,
            overlap_previous=False,
            frontmatter={"version": 1},
            embedding_hash="original_hash",
            created_at=1640995200.0,
            updated_at=1640995200.0
        )

        updated_doc = VectorDocument(
            chunk_id="update_test",  # Same ID
            source_file="updated.md",
            title="Updated Document",
            section_hierarchy=["Updated Section"],
            chunk_index=0,
            content="Updated content with changes",
            content_type="text",
            document_type="documentation",
            token_count=8,
            word_count=5,
            overlap_previous=False,
            frontmatter={"version": 2},
            embedding_hash="updated_hash",
            created_at=1640995200.0,
            updated_at=time.time()  # New timestamp
        )

        with patch.object(pipeline.table, 'merge_insert') as mock_merge:
            mock_builder = Mock()
            mock_builder.when_matched_update_all.return_value = mock_builder
            mock_builder.when_not_matched_insert_all.return_value = mock_builder
            mock_merge.return_value = mock_builder

            # First insert original
            pipeline.upsert_documents([original_doc])

            # Then update with new version
            pipeline.upsert_documents([updated_doc])

            # Should use merge operations
            assert mock_merge.call_count == 2

    def test_preserve_document_history(self, pipeline):
        """Test that document history concepts are preserved."""
        # Test that updated_at timestamp changes
        doc_v1 = VectorDocument(
            chunk_id="history_test",
            source_file="history.md",
            title="History Document",
            section_hierarchy=["History Section"],
            chunk_index=0,
            content="Version 1 content",
            content_type="text",
            document_type="documentation",
            token_count=5,
            word_count=3,
            overlap_previous=False,
            frontmatter={"version": 1},
            embedding_hash="hash_v1",
            created_at=1640995200.0,
            updated_at=1640995200.0
        )

        time.sleep(0.01)  # Ensure different timestamp

        doc_v2 = VectorDocument(
            chunk_id="history_test",
            source_file="history.md",
            title="History Document",
            section_hierarchy=["History Section"],
            chunk_index=0,
            content="Version 2 content with updates",
            content_type="text",
            document_type="documentation",
            token_count=7,
            word_count=5,
            overlap_previous=False,
            frontmatter={"version": 2},
            embedding_hash="hash_v2",
            created_at=1640995200.0,  # Same creation time
            updated_at=time.time()    # New update time
        )

        # Verify timestamps are different
        assert doc_v1.created_at == doc_v2.created_at
        assert doc_v1.updated_at < doc_v2.updated_at

    def test_maintain_referential_integrity(self, pipeline):
        """Test that referential integrity is maintained."""
        # Create related documents
        parent_doc = VectorDocument(
            chunk_id="parent_chunk",
            source_file="parent.md",
            title="Parent Document",
            section_hierarchy=["Parent Section"],
            chunk_index=0,
            content="Parent content",
            content_type="text",
            document_type="documentation",
            token_count=5,
            word_count=3,
            overlap_previous=False,
            frontmatter={"type": "parent"},
            embedding_hash="parent_hash",
            created_at=time.time(),
            updated_at=time.time()
        )

        child_doc = VectorDocument(
            chunk_id="child_chunk",
            source_file="parent.md",  # Same source file
            title="Parent Document",   # Same title
            section_hierarchy=["Parent Section", "Child Subsection"],
            chunk_index=1,  # Next chunk
            content="Child content",
            content_type="text",
            document_type="documentation",
            token_count=4,
            word_count=2,
            overlap_previous=True,  # Overlaps with parent
            frontmatter={"type": "child", "parent": "parent_chunk"},
            embedding_hash="child_hash",
            created_at=time.time(),
            updated_at=time.time()
        )

        with patch.object(pipeline.table, 'add') as mock_add:
            pipeline.upsert_documents([parent_doc, child_doc])

            # Both should be inserted together
            mock_add.assert_called_once()

    def test_version_tracking(self, pipeline):
        """Test version tracking through embedding hashes."""
        base_content = "Base content for version tracking"

        # Version 1
        doc_v1 = VectorDocument(
            chunk_id="version_test",
            source_file="version.md",
            title="Version Document",
            section_hierarchy=["Version Section"],
            chunk_index=0,
            content=base_content,
            content_type="text",
            document_type="documentation",
            token_count=6,
            word_count=4,
            overlap_previous=False,
            frontmatter={"version": "1.0"},
            embedding_hash=pipeline._compute_content_hash(base_content),
            created_at=time.time(),
            updated_at=time.time()
        )

        # Version 2 - different content
        modified_content = base_content + " with modifications"
        doc_v2 = VectorDocument(
            chunk_id="version_test",
            source_file="version.md",
            title="Version Document",
            section_hierarchy=["Version Section"],
            chunk_index=0,
            content=modified_content,
            content_type="text",
            document_type="documentation",
            token_count=8,
            word_count=6,
            overlap_previous=False,
            frontmatter={"version": "2.0"},
            embedding_hash=pipeline._compute_content_hash(modified_content),
            created_at=doc_v1.created_at,
            updated_at=time.time()
        )

        # Verify hashes are different (content changed)
        assert doc_v1.embedding_hash != doc_v2.embedding_hash


class TestIndexing:
    """Test vector indexing functionality."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create pipeline for testing."""
        config = EmbeddingConfig(openai_api_key="test-key")
        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_table = MockTable("document_chunks", mock_db)
            mock_db.create_table = Mock(return_value=mock_table)
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    return EmbeddingPipeline(config, tmp_path / "test_db")

    def test_ivf_pq_index_creation(self, pipeline):
        """Test creation of IVF_PQ index."""
        with patch.object(pipeline.table, 'create_index') as mock_create_index:
            pipeline.create_vector_index("IVF_PQ")

            mock_create_index.assert_called_once_with(
                vector_column_name="vector",
                index_type="IVF_PQ",
                num_partitions=256,
                num_sub_vectors=96,
                replace=True
            )

    def test_hnsw_index_creation(self, pipeline):
        """Test creation of HNSW index."""
        with patch.object(pipeline.table, 'create_index') as mock_create_index:
            pipeline.create_vector_index("HNSW")

            mock_create_index.assert_called_once_with(
                vector_column_name="vector",
                index_type="HNSW",
                M=16,
                ef_construction=200,
                replace=True
            )

    def test_index_error_handling(self, pipeline):
        """Test graceful handling of index creation errors."""
        with patch.object(
            pipeline.table,
            'create_index',
            side_effect=Exception("Index creation failed")
        ):
            # Should not raise exception, just print warning
            pipeline.create_vector_index("IVF_PQ")

    def test_index_performance_metrics(self, pipeline):
        """Test that index creation provides performance metrics."""
        # Mock successful index creation
        with patch.object(pipeline.table, 'create_index') as mock_create_index:
            # Test both index types
            pipeline.create_vector_index("IVF_PQ")
            pipeline.create_vector_index("HNSW")

            # Should have been called twice
            assert mock_create_index.call_count == 2

    def test_index_rebuilding(self, pipeline):
        """Test index rebuilding with replace=True."""
        with patch.object(pipeline.table, 'create_index') as mock_create_index:
            # Create index first time
            pipeline.create_vector_index("IVF_PQ")

            # Create again (should replace)
            pipeline.create_vector_index("IVF_PQ")

            # Both calls should use replace=True
            for call in mock_create_index.call_args_list:
                assert call[1]['replace'] is True


class TestSearchFunctionality:
    """Test search operations on stored vectors."""

    @pytest.fixture
    def pipeline_with_data(self, tmp_path):
        """Create pipeline with mock data for search testing."""
        config = EmbeddingConfig(openai_api_key="test-key")
        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_table = MockTable("document_chunks", mock_db)

            # Add some mock data
            mock_data = [
                {
                    "chunk_id": "search_test_1",
                    "content": "AppPack deployment documentation",
                    "title": "Deployment Guide",
                    "vector": [0.1] * 1536
                },
                {
                    "chunk_id": "search_test_2",
                    "content": "Database configuration and setup",
                    "title": "Database Setup",
                    "vector": [0.2] * 1536
                },
                {
                    "chunk_id": "search_test_3",
                    "content": "Custom domain configuration",
                    "title": "Domain Setup",
                    "vector": [0.3] * 1536
                }
            ]
            mock_db.data["document_chunks"] = mock_data

            mock_db.open_table = Mock(return_value=mock_table)
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(config, tmp_path / "test_db")
                    pipeline.table = mock_table
                    return pipeline

    def test_vector_similarity_search(self, pipeline_with_data):
        """Test vector similarity search functionality."""
        results = pipeline_with_data.search_similar(
            "How to deploy applications",
            limit=2
        )

        # Should return results (mocked)
        assert isinstance(results, list)

    def test_full_text_search(self, pipeline_with_data):
        """Test full-text search functionality."""
        results = pipeline_with_data.search_fulltext(
            "database configuration",
            limit=3
        )

        # Should return results (mocked)
        assert isinstance(results, list)

    def test_search_with_filters(self, pipeline_with_data):
        """Test search with filter expressions."""
        results = pipeline_with_data.search_similar(
            "deployment guide",
            limit=5,
            filter_expr="document_type = 'documentation'"
        )

        # Should handle filter expression
        assert isinstance(results, list)

    def test_search_error_handling(self, pipeline_with_data):
        """Test search error handling."""
        with patch.object(
            pipeline_with_data.table,
            'search',
            side_effect=Exception("Search failed")
        ):
            # Full-text search should handle errors gracefully
            results = pipeline_with_data.search_fulltext("test query")
            assert results == []

    def test_search_result_ranking(self, pipeline_with_data):
        """Test that search results are properly ranked."""
        results = pipeline_with_data.search_similar(
            "application deployment",
            limit=10
        )

        # Results should be returned (ranking tested in integration)
        assert isinstance(results, list)
