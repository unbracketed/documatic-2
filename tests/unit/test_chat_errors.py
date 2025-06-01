"""Unit tests for system errors and failure handling in chat interface."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.documatic.chat import ChatConfig, RAGChatInterface
from tests.fixtures.chat_mocks import MockSearchLayer


class TestLLMFailures:
    """Test handling of LLM service failures."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_api_timeouts(self, mock_agent, mock_openai_model):
        """Test handling of API timeouts."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.run.side_effect = TimeoutError("API request timed out")

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("Test timeout handling")

        # Should return error message, not crash
        assert "error" in response.lower()
        assert "timeout" in response.lower() or "timed out" in response.lower()

        # Should record the error turn
        assert len(chat.context.turns) == 1
        turn = chat.context.turns[0]
        assert turn.user_query == "Test timeout handling"
        assert "error" in turn.assistant_response.lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_rate_limiting(self, mock_agent, mock_openai_model):
        """Test handling of rate limiting errors."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance

        # Simulate rate limiting error
        rate_limit_error = Exception("Rate limit exceeded. Please try again later.")
        mock_agent_instance.run.side_effect = rate_limit_error

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("Test rate limiting")

        # Should handle rate limiting gracefully
        assert "error" in response.lower()
        assert "rate limit" in response.lower() or "try again" in response.lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_model_errors(self, mock_agent, mock_openai_model):
        """Test handling of model-specific errors."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance

        model_errors = [
            Exception("Model is currently overloaded"),
            Exception("Invalid request format"),
            Exception("Model not found"),
            Exception("Authentication failed"),
            Exception("Service temporarily unavailable")
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        for error in model_errors:
            mock_agent_instance.run.side_effect = error
            response = await chat.chat("Test model error")

            # Should handle each error type gracefully
            assert "error" in response.lower()
            assert len(response) > 20  # Should be informative

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_fallback_responses(self, mock_agent, mock_openai_model):
        """Test fallback responses when LLM fails."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.run.side_effect = Exception("LLM service unavailable")

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("How do I deploy an app?")

        # Should provide fallback response
        assert "error" in response.lower()
        assert "try" in response.lower() or "rephras" in response.lower()

        # Should still include available information (search results)
        assert len(response) > 30

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_streaming_llm_failures(self, mock_agent, mock_openai_model):
        """Test LLM failures during streaming."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.run.side_effect = Exception("Streaming failed")

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        chunks = []
        async for chunk in chat.chat_stream("Test streaming failure"):
            chunks.append(chunk)

        # Should handle streaming failure gracefully
        response = "".join(chunks)
        assert "error" in response.lower()
        assert len(chunks) > 0  # Should yield at least error message


class TestSearchFailures:
    """Test handling of search service failures."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_no_results_found(self, mock_agent, mock_openai_model):
        """Test handling when search returns no results."""
        # Setup search to return empty results
        self.mock_search_layer.set_results_for_query("nonexistent", [])

        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "I don't have specific documentation about that topic."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("Tell me about nonexistent feature")

        # Should handle no results gracefully
        assert len(response) > 0

        # Should have called LLM with "No relevant documentation found"
        agent_call = mock_agent_instance.run.call_args[0][0]
        assert "No relevant documentation found" in agent_call

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_search_timeout(self, mock_agent, mock_openai_model):
        """Test handling of search timeouts."""
        # Setup search to timeout
        self.mock_search_layer.set_failure(True, "Search timeout")

        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "I'm having trouble accessing the documentation right now."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("Test search timeout")

        # Should handle search timeout gracefully
        assert len(response) > 0
        # Should still try to provide some response
        assert "trouble" in response.lower() or "error" in response.lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_database_errors(self, mock_agent, mock_openai_model):
        """Test handling of database connectivity errors."""
        # Setup search to fail with database error
        self.mock_search_layer.set_failure(True, "Database connection failed")

        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "I can't access the documentation database right now."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("Test database error")

        # Should handle database errors
        assert len(response) > 0
        assert "database" in response.lower() or "access" in response.lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_graceful_degradation(self, mock_agent, mock_openai_model):
        """Test graceful degradation when search fails."""
        # Setup search to fail
        self.mock_search_layer.set_failure(True, "Search service unavailable")

        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Based on general AppPack knowledge, here's what I can tell you..."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("How do I deploy an app?")

        # Should still provide some response even without search
        assert len(response) > 50
        assert "AppPack" in response

        # Should have called LLM with indication of search failure
        agent_call = mock_agent_instance.run.call_args[0][0]
        assert "Error retrieving context" in agent_call or "No relevant documentation" in agent_call


class TestResourceLimits:
    """Test handling of resource limit errors."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_context_too_large(self, mock_agent, mock_openai_model):
        """Test handling when context exceeds size limits."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance

        # Simulate context too large error
        context_error = Exception("Context length exceeds maximum allowed")
        mock_agent_instance.run.side_effect = context_error

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("Test large context")

        # Should handle context size errors
        assert "error" in response.lower()
        assert "context" in response.lower() or "large" in response.lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_memory_constraints(self, mock_agent, mock_openai_model):
        """Test handling of memory constraint errors."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.run.side_effect = MemoryError("Not enough memory")

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("Test memory constraints")

        # Should handle memory errors gracefully
        assert "error" in response.lower()
        assert len(response) > 20

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_response_size_limits(self, mock_agent, mock_openai_model):
        """Test handling when response exceeds size limits."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance

        # Simulate very large response that might cause issues
        mock_result = Mock()
        huge_response = "Very long response content. " * 10000  # ~270KB
        mock_result.data = huge_response
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("Test large response")

        # Should handle large responses
        assert len(response) > 0
        # May be truncated or handled differently
        # At minimum, should not crash

    async def test_concurrent_sessions(self):
        """Test handling of many concurrent chat sessions."""
        # Create many chat interfaces
        chats = [
            RAGChatInterface(MockSearchLayer(), self.config)
            for _ in range(50)
        ]

        # This test mainly ensures no resource conflicts
        # In a real system, this might test connection pooling, etc.
        for i, chat in enumerate(chats):
            assert chat.context.conversation_id is not None
            assert "chat_" in chat.context.conversation_id


class TestNetworkErrors:
    """Test handling of network-related errors."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_connection_errors(self, mock_agent, mock_openai_model):
        """Test handling of network connection errors."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance

        connection_errors = [
            ConnectionError("Connection refused"),
            ConnectionResetError("Connection reset by peer"),
            OSError("Network is unreachable"),
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        for error in connection_errors:
            mock_agent_instance.run.side_effect = error
            response = await chat.chat("Test connection error")

            # Should handle connection errors
            assert "error" in response.lower()
            assert "connection" in response.lower() or "network" in response.lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_dns_resolution_errors(self, mock_agent, mock_openai_model):
        """Test handling of DNS resolution errors."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.run.side_effect = OSError("Name resolution failed")

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("Test DNS error")

        # Should handle DNS errors
        assert "error" in response.lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_ssl_certificate_errors(self, mock_agent, mock_openai_model):
        """Test handling of SSL certificate errors."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance

        import ssl
        ssl_error = ssl.SSLError("Certificate verification failed")
        mock_agent_instance.run.side_effect = ssl_error

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("Test SSL error")

        # Should handle SSL errors
        assert "error" in response.lower()


class TestCascadingFailures:
    """Test handling of multiple simultaneous failures."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_search_and_llm_failure(self, mock_agent, mock_openai_model):
        """Test handling when both search and LLM fail."""
        # Setup search to fail
        self.mock_search_layer.set_failure(True, "Search service down")

        # Setup LLM to fail
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.run.side_effect = Exception("LLM service down")

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("Test cascading failure")

        # Should still provide some response, not crash
        assert len(response) > 0
        assert "error" in response.lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_multiple_retry_failures(self, mock_agent, mock_openai_model):
        """Test handling of repeated failures."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.run.side_effect = Exception("Persistent failure")

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Multiple calls should all fail gracefully
        for i in range(5):
            response = await chat.chat(f"Test persistent failure {i}")
            assert "error" in response.lower()
            # Should record error turns
            assert len(chat.context.turns) == i + 1

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_partial_service_degradation(self, mock_agent, mock_openai_model):
        """Test handling of partial service degradation."""
        # Search works but returns limited results
        limited_results = self.mock_search_layer.default_results[:1]  # Only 1 result
        self.mock_search_layer.set_results_for_query("limited", limited_results)

        # LLM works but slowly (simulated)
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Limited response based on available information."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("Tell me about limited topic")

        # Should work with degraded performance
        assert len(response) > 0
        assert "Limited response" in response


class TestErrorRecovery:
    """Test error recovery and resilience."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_recovery_after_failure(self, mock_agent, mock_openai_model):
        """Test recovery after a failure."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance

        # First call fails, second succeeds
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            else:
                result = Mock()
                result.data = "Recovery successful"
                return result

        mock_agent_instance.run.side_effect = side_effect

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # First call should fail
        response1 = await chat.chat("First call")
        assert "error" in response1.lower()

        # Second call should succeed
        response2 = await chat.chat("Second call")
        assert "Recovery successful" in response2

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_error_state_isolation(self, mock_agent, mock_openai_model):
        """Test that errors don't affect other conversations."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance

        # First chat instance fails
        mock_agent_instance.run.side_effect = Exception("Chat 1 failure")
        chat1 = RAGChatInterface(self.mock_search_layer, self.config)
        response1 = await chat1.chat("Failing chat")
        assert "error" in response1.lower()

        # Second chat instance should work independently
        mock_result = Mock()
        mock_result.data = "Chat 2 works fine"
        mock_agent_instance.run.side_effect = None
        mock_agent_instance.run.return_value = mock_result

        chat2 = RAGChatInterface(self.mock_search_layer, self.config)
        response2 = await chat2.chat("Working chat")
        assert "Chat 2 works fine" in response2

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_graceful_shutdown_on_critical_error(self, mock_agent, mock_openai_model):
        """Test graceful handling of critical system errors."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance

        # Simulate critical system error
        mock_agent_instance.run.side_effect = SystemError("Critical system failure")

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("Critical error test")

        # Should handle even critical errors gracefully
        assert len(response) > 0
        assert "error" in response.lower()
        # Should not crash the application


if __name__ == "__main__":
    pytest.main([__file__])
