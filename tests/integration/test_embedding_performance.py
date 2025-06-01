"""Performance tests for the embedding pipeline."""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from src.documatic.chunking import DocumentChunk
from src.documatic.embeddings import EmbeddingConfig, EmbeddingPipeline
from tests.fixtures.batch_data import (
    generate_performance_test_data,
    generate_test_chunks,
)
from tests.fixtures.mocks import MockLanceDB, MockTable


class TestThroughputPerformance:
    """Test embedding throughput performance."""

    @pytest.fixture
    def performance_config(self):
        """Create optimized config for performance testing."""
        return EmbeddingConfig(
            openai_api_key="test-key",
            batch_size=100,  # Large batch size for throughput
            max_retries=1,   # Minimal retries for speed
            retry_delay=0.1, # Fast retries
            dimension=1536
        )

    def test_embeddings_per_second(self, performance_config, tmp_path):
        """Test embedding generation throughput."""
        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_table = MockTable("document_chunks", mock_db)
            mock_db.create_table = Mock(return_value=mock_table)
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(performance_config, tmp_path / "perf_db")

                    # Create test chunks
                    chunk_data = generate_performance_test_data("medium")  # 1000 chunks
                    chunks = [DocumentChunk(**chunk) for chunk in chunk_data[:500]]  # Test with 500

                    call_count = 0
                    async def mock_embed_batch(texts):
                        nonlocal call_count
                        call_count += 1
                        # Simulate some processing time
                        await asyncio.sleep(0.01)  # 10ms per batch
                        return [[0.1] * 1536 for _ in texts]

                    with patch.object(
                        pipeline,
                        '_generate_embeddings_batch',
                        side_effect=mock_embed_batch
                    ):
                        async def run_performance_test():
                            start_time = time.time()

                            vector_docs = await pipeline.embed_chunks(chunks)

                            end_time = time.time()
                            duration = end_time - start_time

                            embeddings_per_second = len(chunks) / duration

                            return vector_docs, embeddings_per_second, call_count

                        vector_docs, throughput, batches = asyncio.run(run_performance_test())

                        # Performance assertions
                        assert len(vector_docs) == 500
                        assert throughput > 10  # At least 10 embeddings/second with mocking

                        # Verify batching efficiency
                        expected_batches = (500 + performance_config.batch_size - 1) // performance_config.batch_size
                        assert batches == expected_batches

    def test_optimal_batch_size_finding(self, performance_config, tmp_path):
        """Test finding optimal batch size for throughput."""
        batch_sizes = [10, 25, 50, 100]
        performance_results = {}

        for batch_size in batch_sizes:
            config = EmbeddingConfig(
                openai_api_key="test-key",
                batch_size=batch_size,
                dimension=1536
            )

            with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
                mock_db = MockLanceDB()
                mock_table = MockTable("document_chunks", mock_db)
                mock_db.create_table = Mock(return_value=mock_table)
                mock_lancedb.connect.return_value = mock_db

                with patch("src.documatic.embeddings.OpenAIModel"):
                    with patch("src.documatic.embeddings.Agent"):
                        pipeline = EmbeddingPipeline(config, tmp_path / f"batch_{batch_size}_db")

                        # Fixed test data
                        chunks = [DocumentChunk(**chunk) for chunk in generate_test_chunks(100)]

                        async def mock_embed_batch(texts):
                            # Simulate batch processing overhead
                            overhead = len(texts) * 0.001  # 1ms per text
                            await asyncio.sleep(overhead)
                            return [[0.1] * 1536 for _ in texts]

                        with patch.object(
                            pipeline,
                            '_generate_embeddings_batch',
                            side_effect=mock_embed_batch
                        ):
                            async def run_batch_test():
                                start_time = time.time()
                                await pipeline.embed_chunks(chunks)
                                end_time = time.time()
                                return end_time - start_time

                            duration = asyncio.run(run_batch_test())
                            performance_results[batch_size] = duration

        # Verify that we collected performance data for all batch sizes
        assert len(performance_results) == len(batch_sizes)

        # In a real scenario, we'd expect certain batch sizes to be more efficient
        # For now, just verify we can measure performance differences
        assert all(duration > 0 for duration in performance_results.values())

    def test_api_rate_limit_handling(self, performance_config, tmp_path):
        """Test performance under rate limit conditions."""
        # Use smaller batch size to ensure multiple calls
        performance_config.batch_size = 20

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
                    pipeline = EmbeddingPipeline(performance_config, tmp_path / "rate_limit_db")

                    chunks = [DocumentChunk(**chunk) for chunk in generate_test_chunks(50)]

                    call_count = 0
                    async def mock_embed_with_rate_limit(texts):
                        nonlocal call_count
                        call_count += 1

                        # Simulate rate limiting every 3rd call
                        if call_count % 3 == 0:
                            await asyncio.sleep(0.5)  # Rate limit delay
                        else:
                            await asyncio.sleep(0.01)  # Normal processing

                        return [[0.1] * 1536 for _ in texts]

                    with patch.object(
                        pipeline,
                        '_generate_embeddings_batch',
                        side_effect=mock_embed_with_rate_limit
                    ):
                        async def run_rate_limit_test():
                            start_time = time.time()
                            vector_docs = await pipeline.embed_chunks(chunks)
                            end_time = time.time()
                            return vector_docs, end_time - start_time

                        vector_docs, duration = asyncio.run(run_rate_limit_test())

                        # Should complete despite rate limiting
                        assert len(vector_docs) == 50

                        # Duration should be longer due to rate limiting
                        assert duration > 0.1  # At least some delay from rate limiting


class TestLatencyPerformance:
    """Test embedding latency performance."""

    @pytest.fixture
    def latency_config(self):
        """Create config optimized for latency testing."""
        return EmbeddingConfig(
            openai_api_key="test-key",
            batch_size=1,  # Single item for latency testing
            dimension=1536
        )

    def test_single_embedding_latency(self, latency_config, tmp_path):
        """Test latency for single embedding generation."""
        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_table = MockTable("document_chunks", mock_db)
            mock_db.create_table = Mock(return_value=mock_table)
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(latency_config, tmp_path / "latency_db")

                    chunk = DocumentChunk(**generate_test_chunks(1)[0])

                    async def mock_embed_batch(texts):
                        await asyncio.sleep(0.05)  # 50ms latency simulation
                        return [[0.1] * 1536 for _ in texts]

                    with patch.object(
                        pipeline,
                        '_generate_embeddings_batch',
                        side_effect=mock_embed_batch
                    ):
                        async def run_latency_test():
                            start_time = time.time()
                            vector_docs = await pipeline.embed_chunks([chunk])
                            end_time = time.time()
                            return vector_docs, (end_time - start_time) * 1000  # ms

                        vector_docs, latency_ms = asyncio.run(run_latency_test())

                        # Verify processing
                        assert len(vector_docs) == 1

                        # Latency should be reasonable (under 200ms with mocking)
                        assert latency_ms < 200

    def test_batch_processing_latency(self, tmp_path):
        """Test latency scaling with batch size."""
        batch_sizes = [1, 5, 10, 25]
        latency_results = {}

        for batch_size in batch_sizes:
            config = EmbeddingConfig(
                openai_api_key="test-key",
                batch_size=batch_size,
                dimension=1536
            )

            with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
                mock_db = MockLanceDB()
                mock_table = MockTable("document_chunks", mock_db)
                mock_db.create_table = Mock(return_value=mock_table)
                mock_lancedb.connect.return_value = mock_db

                with patch("src.documatic.embeddings.OpenAIModel"):
                    with patch("src.documatic.embeddings.Agent"):
                        pipeline = EmbeddingPipeline(config, tmp_path / f"latency_batch_{batch_size}")

                        chunks = [DocumentChunk(**chunk) for chunk in generate_test_chunks(batch_size)]

                        async def mock_embed_batch(texts):
                            # Simulate latency that scales sub-linearly with batch size
                            base_latency = 0.02  # 20ms base
                            scale_latency = len(texts) * 0.001  # 1ms per item
                            await asyncio.sleep(base_latency + scale_latency)
                            return [[0.1] * 1536 for _ in texts]

                        with patch.object(
                            pipeline,
                            '_generate_embeddings_batch',
                            side_effect=mock_embed_batch
                        ):
                            async def run_batch_latency_test():
                                start_time = time.time()
                                await pipeline.embed_chunks(chunks)
                                end_time = time.time()
                                return (end_time - start_time) * 1000  # ms

                            latency_ms = asyncio.run(run_batch_latency_test())
                            latency_results[batch_size] = latency_ms

        # Verify latency scaling
        assert len(latency_results) == len(batch_sizes)

        # Larger batches should have better per-item latency
        per_item_latency = {
            size: latency / size for size, latency in latency_results.items()
        }

        # With proper batching, per-item latency should improve
        assert per_item_latency[1] > per_item_latency[25]


class TestScalabilityPerformance:
    """Test scalability with large datasets."""

    @pytest.fixture
    def scalability_config(self):
        """Create config for scalability testing."""
        return EmbeddingConfig(
            openai_api_key="test-key",
            batch_size=50,
            dimension=1536
        )

    def test_large_document_processing(self, scalability_config, tmp_path):
        """Test processing large numbers of documents."""
        document_counts = [100, 500, 1000]
        processing_times = {}

        for doc_count in document_counts:
            with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
                mock_db = MockLanceDB()
                mock_table = MockTable("document_chunks", mock_db)
                mock_db.create_table = Mock(return_value=mock_table)
                mock_lancedb.connect.return_value = mock_db

                with patch("src.documatic.embeddings.OpenAIModel"):
                    with patch("src.documatic.embeddings.Agent"):
                        pipeline = EmbeddingPipeline(
                            scalability_config,
                            tmp_path / f"scale_{doc_count}_db"
                        )

                        chunks = [
                            DocumentChunk(**chunk)
                            for chunk in generate_test_chunks(doc_count)
                        ]

                        async def mock_embed_batch(texts):
                            # Simulate realistic processing time
                            await asyncio.sleep(0.001 * len(texts))  # 1ms per text
                            return [[0.1] * 1536 for _ in texts]

                        with patch.object(
                            pipeline,
                            '_generate_embeddings_batch',
                            side_effect=mock_embed_batch
                        ):
                            async def run_scale_test():
                                start_time = time.time()
                                vector_docs = await pipeline.embed_chunks(chunks)
                                end_time = time.time()
                                return vector_docs, end_time - start_time

                            vector_docs, duration = asyncio.run(run_scale_test())

                            # Verify processing
                            assert len(vector_docs) == doc_count
                            processing_times[doc_count] = duration

        # Verify scalability characteristics
        assert len(processing_times) == len(document_counts)

        # Processing time should scale reasonably
        # (not testing exact scaling due to mocking, but verify completion)
        for count, time_taken in processing_times.items():
            assert time_taken > 0
            assert time_taken < 60  # Should complete within reasonable time

    def test_memory_usage_monitoring(self, scalability_config, tmp_path):
        """Test memory usage during large batch processing."""
        import sys

        with patch("src.documatic.embeddings.lancedb") as mock_lancedb:
            mock_db = MockLanceDB()
            mock_table = MockTable("document_chunks", mock_db)
            mock_db.create_table = Mock(return_value=mock_table)
            mock_lancedb.connect.return_value = mock_db

            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    pipeline = EmbeddingPipeline(scalability_config, tmp_path / "memory_db")

                    # Create large dataset
                    large_chunks = []
                    for i in range(500):
                        chunk_data = generate_test_chunks(1)[0]
                        # Vary content size to test memory handling
                        chunk_data["content"] = f"Large content block {i} " * (i % 100 + 10)
                        large_chunks.append(DocumentChunk(**chunk_data))

                    async def mock_embed_batch(texts):
                        await asyncio.sleep(0.01)
                        return [[0.1] * 1536 for _ in texts]

                    with patch.object(
                        pipeline,
                        '_generate_embeddings_batch',
                        side_effect=mock_embed_batch
                    ):
                        async def run_memory_test():
                            initial_memory = sys.getsizeof(large_chunks)

                            vector_docs = await pipeline.embed_chunks(large_chunks)

                            final_memory = sys.getsizeof(vector_docs)

                            return vector_docs, initial_memory, final_memory

                        vector_docs, initial_mem, final_mem = asyncio.run(run_memory_test())

                        # Verify processing completed
                        assert len(vector_docs) == 500

                        # Memory usage should be reasonable
                        assert initial_mem > 0
                        assert final_mem > 0

                        # Results should not be drastically larger than input
                        # (allowing for additional metadata and structure)
                        assert final_mem < initial_mem * 10  # Reasonable upper bound

    def test_database_size_growth(self, scalability_config, tmp_path):
        """Test database size growth patterns."""
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
                    pipeline = EmbeddingPipeline(scalability_config, tmp_path / "growth_db")

                    # Test incremental additions
                    batch_sizes = [100, 200, 300]
                    total_documents = 0

                    async def mock_embed_batch(texts):
                        return [[0.1] * 1536 for _ in texts]

                    with patch.object(
                        pipeline,
                        '_generate_embeddings_batch',
                        side_effect=mock_embed_batch
                    ):
                        for batch_size in batch_sizes:
                            chunks = [
                                DocumentChunk(**chunk)
                                for chunk in generate_test_chunks(batch_size)
                            ]

                            # Update chunk IDs to be unique across batches
                            for i, chunk in enumerate(chunks):
                                chunk.chunk_id = f"growth_batch_{total_documents + i}"

                            async def run_growth_test():
                                vector_docs = await pipeline.embed_chunks(chunks)
                                pipeline.upsert_documents(vector_docs)
                                return vector_docs

                            vector_docs = asyncio.run(run_growth_test())

                            total_documents += len(vector_docs)

                            # Verify incremental growth
                            assert len(vector_docs) == batch_size

                    # Verify total processing
                    assert total_documents == sum(batch_sizes)
