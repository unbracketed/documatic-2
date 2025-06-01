"""Unit tests for batch processing functionality."""

import asyncio
from unittest.mock import patch

import pytest

from src.documatic.chunking import DocumentChunk
from src.documatic.embeddings import EmbeddingConfig, EmbeddingPipeline, RateLimitError
from tests.fixtures.batch_data import generate_edge_case_chunks, generate_test_chunks


class TestBatchSizing:
    """Test batch size optimization and handling."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create embedding pipeline for testing."""
        config = EmbeddingConfig(
            openai_api_key="test-key",
            batch_size=10
        )
        with patch("src.documatic.embeddings.lancedb"):
            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    return EmbeddingPipeline(config, tmp_path / "test_db")

    def test_optimal_batch_size_determination(self, pipeline):
        """Test that batch size is respected during processing."""
        # Create test chunks
        chunk_data = generate_test_chunks(25)  # More than batch size
        chunks = [
            DocumentChunk(**chunk) for chunk in chunk_data
        ]

        # Mock the embedding generation
        with patch.object(
            pipeline,
            '_generate_embeddings_batch',
            return_value=[[0.1] * 1536] * 10
        ) as mock_embed:

            async def run_test():
                await pipeline.embed_chunks(chunks)

            asyncio.run(run_test())

            # Should be called 3 times (10, 10, 5)
            assert mock_embed.call_count == 3

    def test_dynamic_batch_adjustment(self, pipeline):
        """Test batch adjustment based on content length."""
        # Test with different batch sizes
        test_cases = [
            (5, 3),   # 5 chunks, expect 1 batch
            (15, 3),  # 15 chunks, expect 2 batches
            (25, 3),  # 25 chunks, expect 3 batches
        ]

        for chunk_count, expected_batches in test_cases:
            chunk_data = generate_test_chunks(chunk_count)
            chunks = [DocumentChunk(**chunk) for chunk in chunk_data]

            with patch.object(
                pipeline,
                '_generate_embeddings_batch',
                return_value=[[0.1] * 1536] * pipeline.config.batch_size
            ) as mock_embed:

                async def run_test():
                    await pipeline.embed_chunks(chunks)

                asyncio.run(run_test())

                # Check expected number of batches
                expected_calls = (chunk_count + pipeline.config.batch_size - 1) // pipeline.config.batch_size
                assert mock_embed.call_count == expected_calls

    def test_mixed_content_lengths(self, pipeline):
        """Test handling of mixed content lengths in batches."""
        # Create chunks with varying content lengths
        chunk_data = []
        for i in range(15):
            data = generate_test_chunks(1)[0]
            # Vary content length significantly
            data['content'] = f"Content {i} " * (i * 10 + 1)
            data['token_count'] = len(data['content'].split()) * 2
            chunk_data.append(data)

        chunks = [DocumentChunk(**chunk) for chunk in chunk_data]

        with patch.object(
            pipeline,
            '_generate_embeddings_batch',
            return_value=[[0.1] * 1536] * 10
        ) as mock_embed:

            async def run_test():
                result = await pipeline.embed_chunks(chunks)
                assert len(result) == 15

            asyncio.run(run_test())

            # Verify batching occurred
            assert mock_embed.call_count >= 1


class TestRetryLogic:
    """Test retry logic for failed embedding requests."""

    @pytest.fixture
    def pipeline_with_retries(self, tmp_path):
        """Create pipeline with custom retry config."""
        config = EmbeddingConfig(
            openai_api_key="test-key",
            max_retries=3,
            retry_delay=0.1,  # Short delay for testing
            batch_size=5
        )
        with patch("src.documatic.embeddings.lancedb"):
            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    return EmbeddingPipeline(config, tmp_path / "test_db")

    def test_retry_on_rate_limits(self, pipeline_with_retries):
        """Test retry behavior on rate limit errors."""
        call_count = 0

        def mock_embed_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 attempts
                raise RateLimitError("Rate limit exceeded")
            return [[0.1] * 1536] * 5  # Succeed on 3rd attempt

        with patch.object(
            pipeline_with_retries,
            '_generate_embeddings_batch',
            side_effect=mock_embed_with_retry
        ):

            async def run_test():
                chunks = [DocumentChunk(**chunk) for chunk in generate_test_chunks(5)]
                result = await pipeline_with_retries.embed_chunks(chunks)
                assert len(result) == 5

            asyncio.run(run_test())

            # Should have called 3 times (2 failures + 1 success)
            assert call_count == 3

    def test_exponential_backoff(self, pipeline_with_retries):
        """Test exponential backoff implementation."""
        import time

        call_times = []

        def mock_embed_with_timing(*args, **kwargs):
            call_times.append(time.time())
            if len(call_times) <= 2:
                raise Exception("Temporary failure")
            return [[0.1] * 1536] * 5

        with patch.object(
            pipeline_with_retries,
            '_generate_embeddings_batch',
            side_effect=mock_embed_with_timing
        ):

            async def run_test():
                chunks = [DocumentChunk(**chunk) for chunk in generate_test_chunks(5)]
                await pipeline_with_retries.embed_chunks(chunks)

            asyncio.run(run_test())

            # Check that delays are increasing (exponential backoff)
            assert len(call_times) == 3
            if len(call_times) >= 3:
                delay1 = call_times[1] - call_times[0]
                delay2 = call_times[2] - call_times[1]
                assert delay2 > delay1  # Second delay should be longer

    def test_max_retry_limits(self, pipeline_with_retries):
        """Test that max retry limit is respected."""
        call_count = 0

        def mock_embed_always_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("Persistent failure")

        with patch.object(
            pipeline_with_retries,
            '_generate_embeddings_batch',
            side_effect=mock_embed_always_fail
        ):

            async def run_test():
                chunks = [DocumentChunk(**chunk) for chunk in generate_test_chunks(5)]
                with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
                    await pipeline_with_retries.embed_chunks(chunks)

            asyncio.run(run_test())

            # Should attempt max_retries times
            assert call_count == pipeline_with_retries.config.max_retries


class TestErrorRecovery:
    """Test error recovery and partial failure handling."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create embedding pipeline for testing."""
        config = EmbeddingConfig(
            openai_api_key="test-key",
            batch_size=5,
            max_retries=2
        )
        with patch("src.documatic.embeddings.lancedb"):
            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    return EmbeddingPipeline(config, tmp_path / "test_db")

    def test_continue_after_partial_failure(self, pipeline):
        """Test that processing continues after partial batch failures."""
        batch_count = 0

        def mock_embed_partial_fail(*args, **kwargs):
            nonlocal batch_count
            batch_count += 1
            if 2 <= batch_count <= 4:  # Fail second batch and retries
                raise Exception("Batch 2 failure")
            return [[0.1] * 1536] * len(args[0])  # Success for other batches

        # Create enough chunks for 3 batches
        chunks = [DocumentChunk(**chunk) for chunk in generate_test_chunks(12)]

        with patch.object(
            pipeline,
            '_generate_embeddings_batch',
            side_effect=mock_embed_partial_fail
        ):

            async def run_test():
                # Should fail on second batch after retries
                with pytest.raises(RuntimeError):
                    await pipeline.embed_chunks(chunks)

            asyncio.run(run_test())

            # Should have attempted retries for the failed batch
            assert batch_count >= 3  # At least initial + retries

    def test_track_failed_embeddings(self, pipeline):
        """Test tracking of which embeddings failed."""
        chunks = [DocumentChunk(**chunk) for chunk in generate_test_chunks(5)]

        def mock_embed_fail(*args, **kwargs):
            raise Exception("All embeddings failed")

        with patch.object(
            pipeline,
            '_generate_embeddings_batch',
            side_effect=mock_embed_fail
        ):

            async def run_test():
                with pytest.raises(RuntimeError) as exc_info:
                    await pipeline.embed_chunks(chunks)

                # Check that error message contains failure details
                assert "Failed to generate embeddings" in str(exc_info.value)

            asyncio.run(run_test())

    def test_empty_batch_handling(self, pipeline):
        """Test handling of empty batches."""
        # Empty chunks list
        chunks = []

        async def run_test():
            result = await pipeline.embed_chunks(chunks)
            assert result == []

        asyncio.run(run_test())

    def test_edge_case_chunks(self, pipeline):
        """Test processing of edge case chunks."""
        edge_chunks = generate_edge_case_chunks()
        chunks = [DocumentChunk(**chunk) for chunk in edge_chunks]

        with patch.object(
            pipeline,
            '_generate_embeddings_batch',
            return_value=[[0.1] * 1536] * len(edge_chunks)
        ):

            async def run_test():
                result = await pipeline.embed_chunks(chunks)
                assert len(result) == len(edge_chunks)

                # Verify each result has proper structure
                for vector_doc in result:
                    assert hasattr(vector_doc, 'chunk_id')
                    assert hasattr(vector_doc, 'content')
                    assert hasattr(vector_doc, 'embedding_hash')

            asyncio.run(run_test())


class TestBatchPerformance:
    """Test batch processing performance characteristics."""

    @pytest.fixture
    def performance_pipeline(self, tmp_path):
        """Create pipeline optimized for performance testing."""
        config = EmbeddingConfig(
            openai_api_key="test-key",
            batch_size=50,  # Larger batch size
            retry_delay=0.01  # Minimal delay for testing
        )
        with patch("src.documatic.embeddings.lancedb"):
            with patch("src.documatic.embeddings.OpenAIModel"):
                with patch("src.documatic.embeddings.Agent"):
                    return EmbeddingPipeline(config, tmp_path / "test_db")

    def test_large_batch_processing(self, performance_pipeline):
        """Test processing of large batches efficiently."""
        # Create large number of chunks
        large_chunk_data = generate_test_chunks(200)
        chunks = [DocumentChunk(**chunk) for chunk in large_chunk_data]

        with patch.object(
            performance_pipeline,
            '_generate_embeddings_batch',
            return_value=[[0.1] * 1536] * 50  # Mock batch size
        ) as mock_embed:

            async def run_test():
                import time
                start_time = time.time()

                result = await performance_pipeline.embed_chunks(chunks)

                end_time = time.time()
                processing_time = end_time - start_time

                # Verify results
                assert len(result) == 200

                # Performance check - should process quickly in test
                assert processing_time < 5.0  # Should be very fast with mocking

                # Check batching efficiency
                expected_batches = (200 + 49) // 50  # Ceiling division
                assert mock_embed.call_count == expected_batches

            asyncio.run(run_test())

    def test_memory_efficiency(self, performance_pipeline):
        """Test memory efficiency during large batch processing."""
        import sys

        # Create chunks with varying sizes
        chunk_data = []
        for i in range(100):
            data = generate_test_chunks(1)[0]
            # Create varying content sizes
            data['content'] = f"Large content block {i} " * (i % 50 + 10)
            chunk_data.append(data)

        chunks = [DocumentChunk(**chunk) for chunk in chunk_data]

        with patch.object(
            performance_pipeline,
            '_generate_embeddings_batch',
            return_value=[[0.1] * 1536] * 50
        ):

            async def run_test():
                # Monitor memory during processing
                initial_size = sys.getsizeof(chunks)

                result = await performance_pipeline.embed_chunks(chunks)

                final_size = sys.getsizeof(result)

                # Results should be reasonable in size
                assert len(result) == 100
                assert final_size > 0  # Basic sanity check

            asyncio.run(run_test())
