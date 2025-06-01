"""Integration tests for the complete embedding pipeline."""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from src.documatic.chunking import DocumentChunk
from src.documatic.embeddings import (
    EmbeddingConfig,
    EmbeddingPipeline,
)
from tests.fixtures.batch_data import (
    generate_test_chunks,
)
from tests.fixtures.mocks import MockLanceDB, MockTable


class TestEndToEndPipeline:
    """Test full document processing from chunks to stored vectors."""

    @pytest.fixture
    def integration_config(self):
        """Create config for integration testing."""
        return EmbeddingConfig(
            openai_api_key="test-key",
            batch_size=10,
            max_retries=2,
            dimension=1536
        )

    @pytest.fixture
    def mock_db_setup(self):
        """Set up mock database for integration testing."""
        mock_db = MockLanceDB()
        mock_table = MockTable("document_chunks", mock_db)
        # Initialize the data structure for the table
        mock_db.data["document_chunks"] = []
        mock_db.tables["document_chunks"] = {}
        mock_db.create_table = Mock(return_value=mock_table)
        mock_db.open_table = Mock(return_value=mock_table)
        return mock_db, mock_table

    def test_full_document_processing(self, integration_config, mock_db_setup, tmp_path):
        """Test complete pipeline from chunks to stored vectors."""
        mock_db, mock_table = mock_db_setup

        # Create test chunks
        chunk_data = generate_test_chunks(15)
        chunks = [DocumentChunk(**chunk) for chunk in chunk_data]

        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(integration_config, tmp_path / "integration_db")

                    # Mock embedding generation
                    async def mock_embed_batch(texts):
                        return [[0.1 + i * 0.01] * 1536 for i in range(len(texts))]

                    with patch.object(
                        pipeline,
                        '_generate_embeddings_batch',
                        side_effect=mock_embed_batch
                    ):
                        async def run_pipeline():
                            # Generate embeddings
                            vector_docs = await pipeline.embed_chunks(chunks)

                            # Store in database
                            pipeline.upsert_documents(vector_docs)

                            return vector_docs

                        vector_docs = asyncio.run(run_pipeline())

                        # Verify processing
                        assert len(vector_docs) == 15

                        # Verify document structure
                        for doc in vector_docs:
                            assert doc.chunk_id.startswith("test_chunk_")
                            assert doc.content.startswith("Test content")
                            assert doc.embedding_hash is not None
                            assert doc.created_at > 0
                            assert doc.updated_at > 0

    def test_metadata_preservation(self, integration_config, mock_db_setup, tmp_path):
        """Test that metadata is preserved through the pipeline."""
        mock_db, mock_table = mock_db_setup

        # Create chunk with rich metadata
        chunk_data = {
            "chunk_id": "metadata_test",
            "source_file": "docs/advanced/configuration.md",
            "title": "Advanced Configuration Guide",
            "section_hierarchy": ["Configuration", "Advanced", "Environment Variables"],
            "chunk_index": 5,
            "content": "Advanced configuration options for production deployments",
            "content_type": "text",
            "document_type": "documentation",
            "token_count": 25,
            "word_count": 15,
            "start_position": 1500,
            "end_position": 1560,
            "overlap_previous": True,
            "frontmatter": {
                "title": "Advanced Configuration",
                "category": "Configuration",
                "tags": ["advanced", "production", "environment"],
                "difficulty": "expert",
                "last_updated": "2024-01-15"
            }
        }

        chunk = DocumentChunk(**chunk_data)

        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(integration_config, tmp_path / "metadata_db")

                    async def mock_embed_batch(texts):
                        return [[0.5] * 1536 for _ in texts]

                    with patch.object(
                        pipeline,
                        '_generate_embeddings_batch',
                        side_effect=mock_embed_batch
                    ):
                        async def run_test():
                            vector_docs = await pipeline.embed_chunks([chunk])
                            pipeline.upsert_documents(vector_docs)
                            return vector_docs

                        vector_docs = asyncio.run(run_test())

                        # Verify metadata preservation
                        doc = vector_docs[0]
                        assert doc.source_file == "docs/advanced/configuration.md"
                        assert doc.title == "Advanced Configuration Guide"
                        assert doc.section_hierarchy == [
                            "Configuration", "Advanced", "Environment Variables"
                        ]
                        assert doc.chunk_index == 5
                        assert doc.content_type == "text"
                        assert doc.document_type == "documentation"
                        assert doc.token_count == 25
                        assert doc.word_count == 15
                        assert doc.overlap_previous is True

                        # Verify frontmatter preservation
                        frontmatter = doc.frontmatter
                        assert frontmatter["title"] == "Advanced Configuration"
                        assert frontmatter["category"] == "Configuration"
                        assert "advanced" in frontmatter["tags"]
                        assert frontmatter["difficulty"] == "expert"

    def test_batch_processing_multiple_documents(
        self, integration_config, mock_db_setup, tmp_path
    ):
        """Test processing multiple documents in batches."""
        mock_db, mock_table = mock_db_setup

        # Create chunks from multiple documents
        chunks = []
        for doc_id in range(5):  # 5 documents
            for chunk_idx in range(8):  # 8 chunks per document
                chunk_data = {
                    "chunk_id": f"doc_{doc_id}_chunk_{chunk_idx}",
                    "source_file": f"docs/document_{doc_id}.md",
                    "title": f"Document {doc_id}",
                    "section_hierarchy": [f"Document {doc_id}", f"Section {chunk_idx // 3}"],
                    "chunk_index": chunk_idx,
                    "content": f"Content for document {doc_id}, chunk {chunk_idx} " * 20,
                    "content_type": "text",
                    "document_type": "documentation",
                    "token_count": 60,
                    "word_count": 40,
                    "start_position": chunk_idx * 1200,
                    "end_position": (chunk_idx + 1) * 1200,
                    "overlap_previous": chunk_idx > 0,
                    "frontmatter": {
                        "document_id": doc_id,
                        "total_chunks": 8
                    }
                }
                chunks.append(DocumentChunk(**chunk_data))

        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(integration_config, tmp_path / "batch_db")

                    call_count = 0
                    async def mock_embed_batch(texts):
                        nonlocal call_count
                        call_count += 1
                        # Return unique embeddings for each batch
                        return [[call_count * 0.1 + i * 0.01] * 1536 for i in range(len(texts))]

                    with patch.object(
                        pipeline,
                        '_generate_embeddings_batch',
                        side_effect=mock_embed_batch
                    ):
                        async def run_test():
                            vector_docs = await pipeline.embed_chunks(chunks)
                            pipeline.upsert_documents(vector_docs)
                            return vector_docs

                        vector_docs = asyncio.run(run_test())

                        # Verify all chunks processed
                        assert len(vector_docs) == 40  # 5 docs * 8 chunks

                        # Verify batch processing occurred
                        expected_batches = (40 + integration_config.batch_size - 1) // integration_config.batch_size
                        assert call_count == expected_batches

                        # Verify document grouping preserved
                        doc_0_chunks = [doc for doc in vector_docs if doc.source_file == "docs/document_0.md"]
                        assert len(doc_0_chunks) == 8
                        assert all(doc.title == "Document 0" for doc in doc_0_chunks)


class TestIncrementalUpdates:
    """Test incremental update functionality."""

    @pytest.fixture
    def update_config(self):
        """Create config for update testing."""
        return EmbeddingConfig(
            openai_api_key="test-key",
            batch_size=5,
            dimension=1536
        )

    def test_add_new_documents(self, update_config, tmp_path):
        """Test adding new documents to existing collection."""
        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_table = MockTable("document_chunks", mock_db)
            # Initialize the data structure for the table
            mock_db.data["document_chunks"] = []
            mock_db.tables["document_chunks"] = {}
            mock_db.create_table = Mock(return_value=mock_table)
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(update_config, tmp_path / "update_db")

                    # First batch of documents
                    initial_chunks = [
                        DocumentChunk(**chunk) for chunk in generate_test_chunks(3)
                    ]

                    # Second batch of documents (new)
                    new_chunk_data = generate_test_chunks(2)
                    for i, chunk in enumerate(new_chunk_data):
                        chunk["chunk_id"] = f"new_chunk_{i}"
                        chunk["source_file"] = f"new_doc_{i}.md"

                    new_chunks = [DocumentChunk(**chunk) for chunk in new_chunk_data]

                    async def mock_embed_batch(texts):
                        return [[0.1] * 1536 for _ in texts]

                    with patch.object(
                        pipeline,
                        '_generate_embeddings_batch',
                        side_effect=mock_embed_batch
                    ):
                        async def run_test():
                            # Process initial documents
                            initial_docs = await pipeline.embed_chunks(initial_chunks)
                            pipeline.upsert_documents(initial_docs)

                            # Add new documents
                            new_docs = await pipeline.embed_chunks(new_chunks)
                            pipeline.upsert_documents(new_docs)

                            return initial_docs, new_docs

                        initial_docs, new_docs = asyncio.run(run_test())

                        # Verify both sets processed
                        assert len(initial_docs) == 3
                        assert len(new_docs) == 2

                        # Verify new documents have correct IDs
                        assert new_docs[0].chunk_id == "new_chunk_0"
                        assert new_docs[1].chunk_id == "new_chunk_1"

    def test_update_existing_documents(self, update_config, tmp_path):
        """Test updating existing documents."""
        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_table = MockTable("document_chunks", mock_db)
            mock_db.create_table = Mock(return_value=mock_table)
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(update_config, tmp_path / "update_db")

                    # Original document
                    original_data = {
                        "chunk_id": "update_target",
                        "source_file": "updatable.md",
                        "title": "Updatable Document",
                        "section_hierarchy": ["Original Section"],
                        "chunk_index": 0,
                        "content": "Original content that will be updated",
                        "content_type": "text",
                        "document_type": "documentation",
                        "token_count": 20,
                        "word_count": 12,
                        "start_position": 0,
                        "end_position": 40,
                        "overlap_previous": False,
                        "frontmatter": {"version": "1.0"}
                    }

                    # Updated document (same chunk_id)
                    updated_data = {
                        **original_data,
                        "content": "Updated content with new information and changes",
                        "token_count": 30,
                        "word_count": 18,
                        "end_position": 50,
                        "frontmatter": {"version": "2.0"}
                    }

                    original_chunk = DocumentChunk(**original_data)
                    updated_chunk = DocumentChunk(**updated_data)

                    async def mock_embed_batch(texts):
                        # Return different embeddings for different content
                        return [[hash(text) % 1000 / 1000] * 1536 for text in texts]

                    with patch.object(
                        pipeline,
                        '_generate_embeddings_batch',
                        side_effect=mock_embed_batch
                    ):
                        with patch.object(pipeline.table, 'merge_insert') as mock_merge:
                            mock_builder = Mock()
                            mock_builder.when_matched_update_all.return_value = mock_builder
                            mock_builder.when_not_matched_insert_all.return_value = mock_builder
                            mock_merge.return_value = mock_builder

                            async def run_test():
                                # Process original
                                original_docs = await pipeline.embed_chunks([original_chunk])
                                pipeline.upsert_documents(original_docs)

                                # Process update
                                updated_docs = await pipeline.embed_chunks([updated_chunk])
                                pipeline.upsert_documents(updated_docs)

                                return original_docs, updated_docs

                            original_docs, updated_docs = asyncio.run(run_test())

                            # Verify updates
                            assert len(updated_docs) == 1
                            updated_doc = updated_docs[0]

                            assert updated_doc.chunk_id == "update_target"
                            assert updated_doc.content == "Updated content with new information and changes"
                            assert updated_doc.frontmatter["version"] == "2.0"
                            assert updated_doc.token_count == 30

                            # Should have used merge operation
                            assert mock_merge.call_count == 2

    def test_delete_outdated_documents(self, update_config, tmp_path):
        """Test handling of document deletion concept."""
        # Note: Actual deletion would require additional pipeline methods
        # For now, test that we can identify outdated documents

        current_time = time.time()
        old_time = current_time - 86400  # 24 hours ago

        # Old document that should be considered for deletion
        old_chunk_data = {
            "chunk_id": "old_chunk",
            "source_file": "old_doc.md",
            "title": "Old Document",
            "section_hierarchy": ["Old Section"],
            "chunk_index": 0,
            "content": "This content is outdated",
            "content_type": "text",
            "document_type": "documentation",
            "token_count": 15,
            "word_count": 10,
            "start_position": 0,
            "end_position": 25,
            "overlap_previous": False,
            "frontmatter": {"status": "outdated"}
        }

        # Current document that should be kept
        current_chunk_data = {
            "chunk_id": "current_chunk",
            "source_file": "current_doc.md",
            "title": "Current Document",
            "section_hierarchy": ["Current Section"],
            "chunk_index": 0,
            "content": "This content is current and relevant",
            "content_type": "text",
            "document_type": "documentation",
            "token_count": 18,
            "word_count": 12,
            "start_position": 0,
            "end_position": 37,
            "overlap_previous": False,
            "frontmatter": {"status": "current"}
        }

        old_chunk = DocumentChunk(**old_chunk_data)
        current_chunk = DocumentChunk(**current_chunk_data)

        # Test timestamp-based filtering logic
        chunks_to_process = [old_chunk, current_chunk]

        # In a real implementation, you might filter based on source timestamps
        # or other criteria before processing
        active_chunks = [
            chunk for chunk in chunks_to_process
            if chunk.frontmatter.get("status") != "outdated"
        ]

        assert len(active_chunks) == 1
        assert active_chunks[0].chunk_id == "current_chunk"


class TestSearchFunctionality:
    """Test search functionality in the complete pipeline."""

    @pytest.fixture
    def search_pipeline(self, tmp_path):
        """Create pipeline with search data."""
        config = EmbeddingConfig(openai_api_key="test-key")

        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_table = MockTable("document_chunks", mock_db)

            # Add search test data
            search_data = [
                {
                    "chunk_id": "deploy_1",
                    "content": "How to deploy applications using AppPack CLI commands",
                    "title": "Deployment Guide",
                    "document_type": "tutorial",
                    "vector": [0.1] * 1536
                },
                {
                    "chunk_id": "config_1",
                    "content": "Database configuration and connection settings",
                    "title": "Database Configuration",
                    "document_type": "reference",
                    "vector": [0.2] * 1536
                },
                {
                    "chunk_id": "domain_1",
                    "content": "Setting up custom domains and SSL certificates",
                    "title": "Domain Setup",
                    "document_type": "how-to",
                    "vector": [0.3] * 1536
                }
            ]
            mock_db.data["document_chunks"] = search_data

            mock_db.create_table = Mock(return_value=mock_table)
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(config, tmp_path / "search_db")
                    pipeline.table = mock_table
                    return pipeline

    def test_vector_similarity_search(self, search_pipeline):
        """Test vector similarity search integration."""
        results = search_pipeline.search_similar(
            "How to deploy my application",
            limit=2
        )

        assert isinstance(results, list)
        # In a real test with actual embeddings, we'd verify relevance ranking

    def test_full_text_search(self, search_pipeline):
        """Test full-text search integration."""
        results = search_pipeline.search_fulltext(
            "database configuration",
            limit=3
        )

        assert isinstance(results, list)

    def test_hybrid_search_concept(self, search_pipeline):
        """Test concept of combining vector and full-text search."""
        # Vector search results
        vector_results = search_pipeline.search_similar(
            "application deployment",
            limit=5
        )

        # Full-text search results
        text_results = search_pipeline.search_fulltext(
            "deployment CLI",
            limit=5
        )

        # In a real implementation, these would be combined and ranked
        assert isinstance(vector_results, list)
        assert isinstance(text_results, list)

    def test_search_result_ranking(self, search_pipeline):
        """Test that search results can be ranked by relevance."""
        results = search_pipeline.search_similar(
            "database setup and configuration",
            limit=10
        )

        # Results should be returned (specific ranking logic would be tested separately)
        assert isinstance(results, list)
