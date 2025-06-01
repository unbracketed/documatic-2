"""Integration tests for the search layer."""

from unittest.mock import Mock, patch

import pytest

from src.documatic.search import SearchLayer


class TestSearchIntegration:
    """Integration tests for the complete search pipeline."""

    @pytest.fixture
    def mock_table(self):
        """Create a mock LanceDB table with test data."""
        mock_table = Mock()

        # Mock search results
        mock_results = [
            {
                "id": "doc1",
                "content": "AppPack deployment guide for Python applications",
                "vector": [0.1] * 1536,
                "metadata": {"type": "guide", "section": "deployment"},
                "_distance": 0.2
            },
            {
                "id": "doc2",
                "content": "How to configure environment variables in AppPack",
                "vector": [0.2] * 1536,
                "metadata": {"type": "howto", "section": "configuration"},
                "_distance": 0.3
            }
        ]

        mock_table.search.return_value.limit.return_value.to_list.return_value = mock_results
        mock_table.search.return_value.limit.return_value = Mock(to_list=Mock(return_value=mock_results))

        return mock_table

    @pytest.fixture
    def search_engine(self, mock_table):
        """Create a SearchLayer with mocked dependencies."""
        from src.documatic.embeddings import EmbeddingPipeline

        with patch('lancedb.connect') as mock_connect:
            mock_db = Mock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db

            # Create a mock embedding pipeline
            mock_pipeline = Mock(spec=EmbeddingPipeline)
            mock_pipeline.table = mock_table
            mock_pipeline.search_similar.return_value = [
                {
                    "chunk_id": "doc1",
                    "source_file": "test.md",
                    "title": "Test Document",
                    "section_hierarchy": ["section1"],
                    "content": "Test content",
                    "content_type": "text",
                    "document_type": "guide",
                    "_distance": 0.2,
                    "chunk_index": 0,
                    "token_count": 10,
                    "word_count": 5
                }
            ]
            mock_pipeline.search_fulltext.return_value = []

            engine = SearchLayer(embedding_pipeline=mock_pipeline)
            return engine

    def test_vector_search_basic(self, search_engine):
        """Test basic vector search functionality."""
        results = search_engine.vector_search("apppack deployment", limit=5)

        assert len(results) <= 5
        assert all(hasattr(result, "content") for result in results)
        assert all(hasattr(result, "score") for result in results)
        assert all(hasattr(result, "metadata") for result in results)

    def test_vector_search_empty_query(self, search_engine):
        """Test vector search with empty query."""
        results = search_engine.vector_search("", limit=5)
        assert isinstance(results, list)

    def test_vector_search_long_query(self, search_engine):
        """Test vector search with very long query."""
        long_query = "apppack deployment " * 100  # Very long query
        results = search_engine.vector_search(long_query, limit=5)
        assert isinstance(results, list)

    def test_search_result_format(self, search_engine):
        """Test that search results have the expected format."""
        results = search_engine.vector_search("test query", limit=3)

        for result in results:
            assert hasattr(result, "content")
            assert hasattr(result, "score")
            assert hasattr(result, "metadata")
            assert isinstance(result.score, (int, float))
            assert isinstance(result.metadata, dict)

    def test_search_k_parameter(self, search_engine):
        """Test that limit parameter limits results correctly."""
        results_k3 = search_engine.vector_search("test", limit=3)
        results_k1 = search_engine.vector_search("test", limit=1)

        assert len(results_k3) <= 3
        assert len(results_k1) <= 1

    def test_search_score_ordering(self, search_engine):
        """Test that results are ordered by relevance score."""
        results = search_engine.vector_search("apppack", limit=5)

        if len(results) > 1:
            scores = [result.score for result in results]
            # Scores should be in descending order (higher = more relevant)
            assert scores == sorted(scores, reverse=True)


class TestSearchQueries:
    """Test different types of search queries."""

    @pytest.fixture
    def search_engine(self):
        """Create a SearchLayer for query testing."""
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
            return engine

    def test_keyword_queries(self, search_engine):
        """Test keyword-based queries."""
        queries = [
            "apppack deployment",
            "docker configuration",
            "environment variables"
        ]

        for query in queries:
            results = search_engine.vector_search(query, limit=5)
            assert isinstance(results, list)

    def test_natural_language_queries(self, search_engine):
        """Test natural language queries."""
        queries = [
            "How do I deploy a Python application?",
            "What are the best practices for scaling?",
            "Why is my deployment failing?"
        ]

        for query in queries:
            results = search_engine.vector_search(query, limit=5)
            assert isinstance(results, list)

    def test_code_queries(self, search_engine):
        """Test code-related queries."""
        queries = [
            "flask app.py example",
            "dockerfile FROM python:3.9",
            "yaml configuration syntax"
        ]

        for query in queries:
            results = search_engine.vector_search(query, limit=5)
            assert isinstance(results, list)

    def test_complex_queries(self, search_engine):
        """Test complex multi-part queries."""
        queries = [
            "deploy python flask app with postgresql database and redis cache to production",
            "troubleshoot 502 bad gateway error nginx ingress controller kubernetes"
        ]

        for query in queries:
            results = search_engine.vector_search(query, limit=5)
            assert isinstance(results, list)


class TestSearchEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def search_engine(self):
        """Create a SearchLayer for edge case testing."""
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
            return engine

    def test_empty_query(self, search_engine):
        """Test handling of empty queries."""
        results = search_engine.vector_search("", limit=5)
        assert isinstance(results, list)

    def test_single_character_query(self, search_engine):
        """Test handling of single character queries."""
        results = search_engine.vector_search("a", limit=5)
        assert isinstance(results, list)

    def test_very_long_query(self, search_engine):
        """Test handling of very long queries."""
        long_query = "a" * 1000
        results = search_engine.vector_search(long_query, limit=5)
        assert isinstance(results, list)

    def test_special_characters_query(self, search_engine):
        """Test handling of special characters."""
        special_queries = [
            "!@#$%^&*()",
            "SELECT * FROM users;",
            "<script>alert('xss')</script>",
            "null",
            "undefined"
        ]

        for query in special_queries:
            results = search_engine.vector_search(query, limit=5)
            assert isinstance(results, list)

    def test_unicode_query(self, search_engine):
        """Test handling of unicode characters."""
        unicode_queries = [
            "café deployment",
            "Kubernetes 配置",
            "🚀 deployment guide",
            "résumé parsing"
        ]

        for query in unicode_queries:
            results = search_engine.vector_search(query, limit=5)
            assert isinstance(results, list)

    def test_zero_limit_parameter(self, search_engine):
        """Test handling of limit=0."""
        results = search_engine.vector_search("test", limit=0)
        assert len(results) == 0

    def test_negative_limit_parameter(self, search_engine):
        """Test handling of negative limit."""
        # SearchLayer should handle negative limits gracefully
        results = search_engine.vector_search("test", limit=-1)
        assert isinstance(results, list)


class TestSearchPerformance:
    """Performance-related tests for search functionality."""

    @pytest.fixture
    def search_engine(self):
        """Create a SearchLayer for performance testing."""
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
            return engine

    def test_search_latency(self, search_engine):
        """Test that search completes within reasonable time."""
        import time

        start_time = time.time()
        results = search_engine.vector_search("apppack deployment", limit=10)
        end_time = time.time()

        # Should complete within 1 second (very generous for mocked test)
        assert (end_time - start_time) < 1.0
        assert isinstance(results, list)

    def test_concurrent_searches(self, search_engine):
        """Test handling of concurrent search requests."""
        import concurrent.futures

        def perform_search(query_id):
            return search_engine.vector_search(f"query {query_id}", limit=5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(perform_search, i) for i in range(10)]
            results = [future.result() for future in futures]

        assert len(results) == 10
        assert all(isinstance(result, list) for result in results)
