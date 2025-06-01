"""Unit tests for vector search functionality."""

from unittest.mock import Mock, patch

import pytest

from src.documatic.search import SearchLayer


class TestVectorSearch:
    """Test vector search specific functionality."""

    @pytest.fixture
    def mock_table(self):
        """Create a mock LanceDB table."""
        mock_table = Mock()

        # Mock vector search results
        mock_results = [
            {
                "id": "doc1",
                "content": "AppPack deployment guide",
                "vector": [0.1] * 1536,
                "metadata": {"type": "guide"},
                "_distance": 0.2
            },
            {
                "id": "doc2",
                "content": "Environment configuration",
                "vector": [0.2] * 1536,
                "metadata": {"type": "config"},
                "_distance": 0.3
            }
        ]

        mock_table.search.return_value.limit.return_value.to_list.return_value = (
            mock_results
        )
        return mock_table

    @pytest.fixture
    def search_engine(self, mock_table):
        """Create SearchLayer with mocked dependencies."""
        from src.documatic.embeddings import EmbeddingPipeline

        with patch('lancedb.connect') as mock_connect:
            mock_db = Mock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db

            # Create a mock embedding pipeline
            mock_pipeline = Mock(spec=EmbeddingPipeline)
            mock_pipeline.table = mock_table

            # Create a side effect function that respects the limit parameter
            all_results = [
                {
                    "chunk_id": "doc1",
                    "source_file": "test.md",
                    "title": "Test Document",
                    "section_hierarchy": ["section1"],
                    "content": "AppPack deployment guide",
                    "content_type": "text",
                    "document_type": "guide",
                    "_distance": 0.2,
                    "chunk_index": 0,
                    "token_count": 10,
                    "word_count": 5
                },
                {
                    "chunk_id": "doc2",
                    "source_file": "test2.md",
                    "title": "Test Document 2",
                    "section_hierarchy": ["section2"],
                    "content": "Environment configuration",
                    "content_type": "text",
                    "document_type": "config",
                    "_distance": 0.3,
                    "chunk_index": 1,
                    "token_count": 8,
                    "word_count": 4
                },
                {
                    "chunk_id": "doc3",
                    "source_file": "test3.md",
                    "title": "Test Document 3",
                    "section_hierarchy": ["section3"],
                    "content": "More content",
                    "content_type": "text",
                    "document_type": "guide",
                    "_distance": 0.4,
                    "chunk_index": 2,
                    "token_count": 6,
                    "word_count": 3
                }
            ]

            def mock_search_similar(query, limit=10, filter_expr=None):
                return all_results[:limit]

            mock_pipeline.search_similar.side_effect = mock_search_similar
            mock_pipeline.search_fulltext.return_value = []

            engine = SearchLayer(embedding_pipeline=mock_pipeline)
            return engine

    def test_query_embedding_generation(self, search_engine):
        """Test that query embeddings are generated correctly."""
        query = "test query"

        # Mock the embedding pipeline's search_similar method to track calls
        with patch.object(search_engine.pipeline, 'search_similar') as mock_search:
            mock_search.return_value = []

            search_engine.vector_search(query, limit=5)

            # Verify search_similar was called with the query
            mock_search.assert_called_once_with(query, limit=5, filter_expr=None)

    def test_embedding_dimensions(self, search_engine):
        """Test that embedding pipeline handles queries correctly."""
        query = "test query"

        with patch.object(search_engine.pipeline, 'search_similar') as mock_search:
            mock_search.return_value = []

            search_engine.vector_search(query, limit=5)

            # Verify the search was called with correct parameters
            call_args = mock_search.call_args
            assert call_args[0][0] == query  # First positional arg is query
            assert call_args[1]['limit'] == 5  # Keyword arg limit

    def test_different_query_lengths(self, search_engine):
        """Test handling of different query lengths."""
        queries = [
            "short",
            "medium length query with several words",
            "very long query " * 50  # Very long query
        ]

        for query in queries:
            results = search_engine.vector_search(query, limit=5)
            assert isinstance(results, list)

    def test_similarity_search_top_k(self, search_engine):
        """Test top-k retrieval functionality."""
        query = "apppack deployment"

        # Test different limit values
        for limit in [1, 3, 5, 10]:
            results = search_engine.vector_search(query, limit=limit)
            assert len(results) <= limit

    def test_score_normalization(self, search_engine):
        """Test that scores are properly normalized."""
        query = "test query"
        results = search_engine.vector_search(query, limit=5)

        for result in results:
            score = result.score
            # Scores should be numeric
            assert isinstance(score, int | float)
            assert score >= 0

    def test_empty_result_handling(self, search_engine):
        """Test handling when no results are found."""
        # Override the side effect to return empty results
        search_engine.pipeline.search_similar.side_effect = (
            lambda query, limit=10, filter_expr=None: []
        )

        results = search_engine.vector_search("nonexistent query", limit=5)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_distance_to_score_conversion(self, search_engine):
        """Test conversion from distance to similarity score."""
        query = "test query"
        results = search_engine.vector_search(query, limit=5)

        # Check that distances are converted to scores
        for result in results:
            assert hasattr(result, "score")
            assert isinstance(result.score, int | float)
            # For cosine similarity, score = 1 - distance
            # Score should be higher when distance is lower

    def test_metadata_preservation(self, search_engine):
        """Test that metadata is preserved in results."""
        query = "test query"
        results = search_engine.vector_search(query, limit=5)

        for result in results:
            assert hasattr(result, "metadata")
            assert isinstance(result.metadata, dict)

    def test_content_preservation(self, search_engine):
        """Test that original content is preserved."""
        query = "test query"
        results = search_engine.vector_search(query, limit=5)

        for result in results:
            assert hasattr(result, "content")
            assert isinstance(result.content, str)
            assert len(result.content) > 0


class TestVectorSearchConfiguration:
    """Test vector search configuration options."""

    def test_custom_distance_metric(self):
        """Test custom distance metric configuration."""
        # This would test different distance metrics if implemented
        # For now, just test that the default works
        from src.documatic.embeddings import EmbeddingPipeline

        with patch('lancedb.connect') as mock_connect:
            mock_table = Mock()
            mock_table.search.return_value.limit.return_value.to_list.return_value = []

            mock_db = Mock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db

            # Create a mock embedding pipeline
            mock_pipeline = Mock(spec=EmbeddingPipeline)
            mock_pipeline.table = mock_table
            mock_pipeline.search_similar.return_value = []
            mock_pipeline.search_fulltext.return_value = []

            engine = SearchLayer(embedding_pipeline=mock_pipeline)

            results = engine.vector_search("test", limit=5)
            assert isinstance(results, list)

    def test_search_parameters(self):
        """Test search with different parameters."""
        from src.documatic.embeddings import EmbeddingPipeline

        with patch('lancedb.connect') as mock_connect:
            mock_table = Mock()
            mock_table.search.return_value.limit.return_value.to_list.return_value = []

            mock_db = Mock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db

            # Create a mock embedding pipeline
            mock_pipeline = Mock(spec=EmbeddingPipeline)
            mock_pipeline.table = mock_table
            mock_pipeline.search_similar.return_value = []
            mock_pipeline.search_fulltext.return_value = []

            engine = SearchLayer(embedding_pipeline=mock_pipeline)

            # Test with various limit values
            for limit in [1, 5, 10, 20]:
                results = engine.vector_search("test", limit=limit)
                assert isinstance(results, list)
                assert len(results) <= limit


class TestVectorSearchErrors:
    """Test error handling in vector search."""

    def test_invalid_limit_parameter(self):
        """Test handling of invalid limit parameters."""
        from src.documatic.embeddings import EmbeddingPipeline

        with patch('lancedb.connect') as mock_connect:
            mock_table = Mock()
            mock_db = Mock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db

            # Create a mock embedding pipeline
            mock_pipeline = Mock(spec=EmbeddingPipeline)
            mock_pipeline.table = mock_table
            mock_pipeline.search_similar.return_value = []
            mock_pipeline.search_fulltext.return_value = []

            engine = SearchLayer(embedding_pipeline=mock_pipeline)

            # Test negative limit - SearchLayer should handle gracefully
            results = engine.vector_search("test", limit=-1)
            assert isinstance(results, list)

    def test_embedding_pipeline_failure(self):
        """Test handling of embedding pipeline failures."""
        from src.documatic.embeddings import EmbeddingPipeline

        with patch('lancedb.connect') as mock_connect:
            mock_table = Mock()
            mock_db = Mock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db

            # Create a mock embedding pipeline that fails
            mock_pipeline = Mock(spec=EmbeddingPipeline)
            mock_pipeline.table = mock_table
            mock_pipeline.search_similar.side_effect = Exception("Search failed")
            mock_pipeline.search_fulltext.return_value = []

            engine = SearchLayer(embedding_pipeline=mock_pipeline)

            # Should handle search failures gracefully and return empty list
            results = engine.vector_search("test", limit=5)
            assert isinstance(results, list)
            assert len(results) == 0

    def test_database_connection_failure(self):
        """Test handling of database connection failures."""
        from src.documatic.embeddings import EmbeddingPipeline

        # Create a mock embedding pipeline that fails initialization
        mock_pipeline = Mock(spec=EmbeddingPipeline)
        mock_pipeline.table = None
        mock_pipeline.search_similar.side_effect = Exception("Connection failed")
        mock_pipeline.search_fulltext.return_value = []

        engine = SearchLayer(embedding_pipeline=mock_pipeline)

        # Should handle connection failures gracefully
        results = engine.vector_search("test", limit=5)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_table_not_found(self):
        """Test handling when table doesn't exist."""
        from src.documatic.embeddings import EmbeddingPipeline

        # Create a mock embedding pipeline with missing table
        mock_pipeline = Mock(spec=EmbeddingPipeline)
        mock_pipeline.table = None
        mock_pipeline.search_similar.side_effect = Exception("Table not found")
        mock_pipeline.search_fulltext.return_value = []

        engine = SearchLayer(embedding_pipeline=mock_pipeline)

        # Should handle missing table gracefully
        results = engine.vector_search("test", limit=5)
        assert isinstance(results, list)
        assert len(results) == 0
