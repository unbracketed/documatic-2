"""Integration tests for streaming responses."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.documatic.chat import ChatConfig, RAGChatInterface
from tests.fixtures.chat_mocks import MockSearchLayer


class TestStreamInitialization:
    """Test streaming response initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig(stream_responses=True)

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_first_token_latency(self, mock_agent, mock_openai_model):
        """Test time to first token in streaming."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Quick streaming response for latency test"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        start_time = time.time()
        first_chunk_time = None

        # Collect first chunk timing
        async for chunk in chat.chat_stream("Quick question?"):
            if first_chunk_time is None:
                first_chunk_time = time.time()
            break  # Only need first chunk for timing

        # First token should be relatively fast (< 1 second for mock)
        if first_chunk_time:
            latency = first_chunk_time - start_time
            assert latency < 1.0  # Mock should be very fast

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_stream_setup_time(self, mock_agent, mock_openai_model):
        """Test stream setup overhead."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Stream setup test response"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        start_time = time.time()

        # Start streaming
        stream = chat.chat_stream("Test setup time")

        # First chunk should be available quickly
        first_chunk = await stream.__anext__()
        setup_time = time.time() - start_time

        assert isinstance(first_chunk, str)
        assert setup_time < 0.5  # Setup should be fast

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_stream_error_handling(self, mock_agent, mock_openai_model):
        """Test error handling during stream initialization."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.run.side_effect = Exception("Stream init error")

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Should handle initialization errors gracefully
        chunks = []
        async for chunk in chat.chat_stream("Error test"):
            chunks.append(chunk)

        # Should receive error message
        assert len(chunks) > 0
        error_response = "".join(chunks)
        assert "error" in error_response.lower()


class TestChunkProcessing:
    """Test streaming chunk processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig(stream_responses=True)

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_partial_response_handling(self, mock_agent, mock_openai_model):
        """Test handling of partial responses during streaming."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "This is a longer response that will be chunked into multiple parts for streaming"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        chunks = []
        async for chunk in chat.chat_stream("Long response test"):
            chunks.append(chunk)
            # Each chunk should be a string
            assert isinstance(chunk, str)
            # Chunks should not be empty (except possibly last)
            if chunk != chunks[-1]:  # Allow last chunk to potentially be empty
                assert len(chunk) > 0

        # Should have multiple chunks
        assert len(chunks) > 1

        # All chunks combined should form complete response
        complete_response = "".join(chunks)
        assert "longer response" in complete_response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_buffer_management(self, mock_agent, mock_openai_model):
        """Test buffer management during streaming."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        # Very long response to test buffering
        long_response = "This is a very long response. " * 100
        mock_result.data = long_response
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        total_chunks = 0
        total_chars = 0

        async for chunk in chat.chat_stream("Buffer test"):
            total_chunks += 1
            total_chars += len(chunk)

            # Individual chunks should be reasonable size
            assert len(chunk) <= 100  # Current implementation uses chunk_size=50

        # Should have processed all content
        assert total_chars >= len(long_response)
        assert total_chunks > 10  # Should be many chunks for long response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_unicode_handling(self, mock_agent, mock_openai_model):
        """Test handling of Unicode content in streaming."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        unicode_response = "AppPack supports émojis 🚀, Chinese 中文, and special chars ñáéíóú"
        mock_result.data = unicode_response
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        chunks = []
        async for chunk in chat.chat_stream("Unicode test"):
            chunks.append(chunk)
            # Ensure chunk is valid UTF-8
            assert isinstance(chunk, str)
            chunk.encode('utf-8')  # Should not raise exception

        complete_response = "".join(chunks)

        # Should preserve Unicode characters
        assert "émojis" in complete_response
        assert "🚀" in complete_response
        assert "中文" in complete_response
        assert "ñáéíóú" in complete_response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_markdown_preservation(self, mock_agent, mock_openai_model):
        """Test preservation of Markdown formatting during streaming."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        markdown_response = """# AppPack Deployment

To deploy your app:

1. Configure `apppack.toml`
2. Run `apppack deploy`

**Important**: Check logs for errors.

```bash
apppack logs --tail
```
"""
        mock_result.data = markdown_response
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        chunks = []
        async for chunk in chat.chat_stream("Markdown test"):
            chunks.append(chunk)

        complete_response = "".join(chunks)

        # Should preserve Markdown formatting
        assert "# AppPack" in complete_response
        assert "1." in complete_response
        assert "**Important**" in complete_response
        assert "```bash" in complete_response
        assert "`apppack.toml`" in complete_response


class TestStreamCompletion:
    """Test stream completion and cleanup."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig(stream_responses=True)

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_final_response_assembly(self, mock_agent, mock_openai_model):
        """Test assembly of final complete response."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Complete response for assembly test"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Collect all chunks
        all_chunks = []
        async for chunk in chat.chat_stream("Assembly test"):
            all_chunks.append(chunk)

        final_response = "".join(all_chunks)

        # Should include main response
        assert "Complete response for assembly test" in final_response

        # Should include sources
        assert "Sources:" in final_response

        # Verify conversation was recorded correctly
        assert len(chat.context.turns) == 1
        recorded_response = chat.context.turns[0].assistant_response
        assert recorded_response == final_response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_citation_appendix(self, mock_agent, mock_openai_model):
        """Test citation appendix at end of streaming."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Response with citations needed"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        chunks = []
        async for chunk in chat.chat_stream("Citation test"):
            chunks.append(chunk)

        # Last chunks should include citation information
        final_chunks = "".join(chunks[-5:])  # Check last few chunks
        assert "Sources:" in final_chunks

        # Should list source files
        complete_response = "".join(chunks)
        assert "overview.md" in complete_response or "deployment.md" in complete_response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_cleanup_operations(self, mock_agent, mock_openai_model):
        """Test cleanup operations after streaming completes."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Cleanup test response"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Complete streaming
        chunks = []
        async for chunk in chat.chat_stream("Cleanup test"):
            chunks.append(chunk)

        # After streaming, conversation should be properly recorded
        assert len(chat.context.turns) == 1
        turn = chat.context.turns[0]
        assert turn.user_query == "Cleanup test"
        assert "Cleanup test response" in turn.assistant_response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_connection_closing(self, mock_agent, mock_openai_model):
        """Test proper connection closing after streaming."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Connection test response"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Stream should complete cleanly
        chunks = []
        async for chunk in chat.chat_stream("Connection test"):
            chunks.append(chunk)

        # No exceptions should be raised
        assert len(chunks) > 0

        # Agent should have been called once
        mock_agent_instance.run.assert_called_once()


class TestStreamingPerformance:
    """Test streaming performance characteristics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig(stream_responses=True)

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_streaming_throughput(self, mock_agent, mock_openai_model):
        """Test streaming throughput metrics."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        # Large response for throughput testing
        large_response = "Throughput test content. " * 200  # ~5000 chars
        mock_result.data = large_response
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        start_time = time.time()
        total_chars = 0
        chunk_count = 0

        async for chunk in chat.chat_stream("Throughput test"):
            total_chars += len(chunk)
            chunk_count += 1

        total_time = time.time() - start_time

        # Calculate throughput
        chars_per_second = total_chars / total_time if total_time > 0 else 0

        # Should have reasonable throughput (mock should be very fast)
        assert chars_per_second > 1000  # At least 1000 chars/sec for mock
        assert chunk_count > 10  # Should be multiple chunks

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_concurrent_streams(self, mock_agent, mock_openai_model):
        """Test handling of concurrent streaming requests."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Concurrent stream response"
        mock_agent_instance.run.return_value = mock_result

        # Create multiple chat instances for concurrent streams
        chats = [
            RAGChatInterface(self.mock_search_layer, self.config)
            for _ in range(3)
        ]

        async def stream_question(chat, question):
            chunks = []
            async for chunk in chat.chat_stream(question):
                chunks.append(chunk)
            return "".join(chunks)

        # Run streams concurrently
        tasks = [
            stream_question(chat, f"Concurrent question {i}")
            for i, chat in enumerate(chats)
        ]

        responses = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(responses) == 3
        for response in responses:
            assert "Concurrent stream response" in response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_memory_usage_during_streaming(self, mock_agent, mock_openai_model):
        """Test memory usage patterns during streaming."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        # Very large response to test memory usage
        huge_response = "Memory test content. " * 1000  # ~22KB
        mock_result.data = huge_response
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        chunks_processed = 0
        max_chunk_size = 0

        async for chunk in chat.chat_stream("Memory test"):
            chunks_processed += 1
            max_chunk_size = max(max_chunk_size, len(chunk))

        # Should process in reasonable chunks
        assert chunks_processed > 20  # Many small chunks
        assert max_chunk_size < 1000  # No single chunk too large

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_streaming_vs_blocking_performance(self, mock_agent, mock_openai_model):
        """Test performance comparison between streaming and blocking."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        test_response = "Performance comparison response content"
        mock_result.data = test_response
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Test blocking response time
        start_time = time.time()
        blocking_response = await chat.chat("Performance test")
        blocking_time = time.time() - start_time

        # Test streaming response time to completion
        start_time = time.time()
        chunks = []
        async for chunk in chat.chat_stream("Performance test streaming"):
            chunks.append(chunk)
        streaming_time = time.time() - start_time

        # Both should complete in reasonable time
        assert blocking_time < 1.0  # Mock should be fast
        assert streaming_time < 1.0  # Mock should be fast

        # Streaming might have slight overhead but should be comparable
        assert abs(streaming_time - blocking_time) < 0.5


class TestStreamingEdgeCases:
    """Test edge cases in streaming responses."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig(stream_responses=True)

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_empty_response_streaming(self, mock_agent, mock_openai_model):
        """Test streaming with empty response."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = ""  # Empty response
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        chunks = []
        async for chunk in chat.chat_stream("Empty response test"):
            chunks.append(chunk)

        # Should handle empty response gracefully
        complete_response = "".join(chunks)
        # Should at least have sources section
        assert "Sources:" in complete_response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_single_character_response(self, mock_agent, mock_openai_model):
        """Test streaming with very short response."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Yes"  # Very short response
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        chunks = []
        async for chunk in chat.chat_stream("Short response test"):
            chunks.append(chunk)

        complete_response = "".join(chunks)
        assert "Yes" in complete_response
        assert "Sources:" in complete_response

    async def test_empty_question_streaming(self):
        """Test streaming with empty question."""
        chat = RAGChatInterface(self.mock_search_layer, self.config)

        chunks = []
        async for chunk in chat.chat_stream(""):
            chunks.append(chunk)

        complete_response = "".join(chunks)
        assert "Please provide a question" in complete_response


if __name__ == "__main__":
    pytest.main([__file__])
