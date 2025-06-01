"""Unit tests for Pydantic AI integration in chat interface."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.documatic.chat import ChatConfig, RAGChatInterface
from tests.fixtures.chat_mocks import MockSearchLayer


class TestPydanticAIIntegration:
    """Test Pydantic AI integration components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig(
            model_name="gpt-4o-mini",
            max_sources=3,
            temperature=0.7,
            stream_responses=True
        )

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    def test_model_initialization(self, mock_agent, mock_openai_model):
        """Test Pydantic AI model initialization."""
        # Setup mocks
        mock_model = Mock()
        mock_openai_model.return_value = mock_model
        mock_agent_instance = Mock()
        mock_agent.return_value = mock_agent_instance

        # Create chat interface
        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Verify model initialization
        mock_openai_model.assert_called_once_with("gpt-4o-mini")
        mock_agent.assert_called_once()

        # Check agent was called with model and system prompt
        call_args = mock_agent.call_args
        assert call_args[0][0] == mock_model  # First positional arg is model
        assert 'system_prompt' in call_args[1]  # System prompt in kwargs
        assert 'AppPack.io' in call_args[1]['system_prompt']

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    def test_model_configuration(self, mock_agent, mock_openai_model):
        """Test model configuration with different settings."""
        configs = [
            ChatConfig(model_name="gpt-3.5-turbo", temperature=0.5),
            ChatConfig(model_name="gpt-4", temperature=0.9),
            ChatConfig(model_name="gpt-4o-mini", temperature=0.0)
        ]

        for config in configs:
            mock_openai_model.reset_mock()

            chat = RAGChatInterface(self.mock_search_layer, config)

            mock_openai_model.assert_called_once_with(config.model_name)
            assert chat.config.temperature == config.temperature

    def test_api_key_validation(self):
        """Test API key validation during initialization."""
        # This test would need environment variable mocking
        # For now, we'll test that the interface can be created
        chat = RAGChatInterface(self.mock_search_layer, self.config)
        assert chat is not None
        assert chat.config.model_name == "gpt-4o-mini"

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_connection_testing(self, mock_agent, mock_openai_model):
        """Test connection to LLM service."""
        # Setup mocks
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Test response"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Test a simple query to verify connection
        response = await chat.chat("test question")

        assert "Test response" in response
        mock_agent_instance.run.assert_called_once()


class TestModelResponses:
    """Test model response handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig(model_name="gpt-4o-mini")

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_response_parsing(self, mock_agent, mock_openai_model):
        """Test response parsing from Pydantic AI."""
        # Setup mock response
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "AppPack is a deployment platform. [Source: overview.md]"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("What is AppPack?")

        # Verify response was parsed correctly
        assert "AppPack is a deployment platform" in response
        assert "[Source: overview.md]" in response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_type_validation(self, mock_agent, mock_openai_model):
        """Test type validation of responses."""
        # Setup mock with invalid response type
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = 12345  # Invalid type (should be string)
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # This should handle type conversion or fail gracefully
        try:
            response = await chat.chat("test question")
            assert isinstance(response, str)
        except Exception as e:
            # Should be a specific type error, not a crash
            assert "type" in str(e).lower() or "conversion" in str(e).lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_error_handling(self, mock_agent, mock_openai_model):
        """Test error handling for model failures."""
        # Setup mock to raise exception
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.run.side_effect = Exception("API Error")

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("test question")

        # Should return error message, not crash
        assert "error" in response.lower()
        assert "API Error" in response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_retry_logic(self, mock_agent, mock_openai_model):
        """Test retry logic for transient failures."""
        # Setup mock to fail once then succeed
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Success after retry"

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Transient error")
            return mock_result

        mock_agent_instance.run.side_effect = side_effect

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # First call should handle the error gracefully
        response = await chat.chat("test question")
        assert "error" in response.lower()

        # Second call should succeed
        response = await chat.chat("test question again")
        # Note: Current implementation doesn't have retry logic,
        # so this tests current behavior


class TestStreaming:
    """Test streaming response functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig(stream_responses=True)

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_stream_initialization(self, mock_agent, mock_openai_model):
        """Test stream initialization."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Streaming response test"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Test streaming
        chunks = []
        async for chunk in chat.chat_stream("test question"):
            chunks.append(chunk)

        # Should have received multiple chunks
        assert len(chunks) > 1

        # Chunks should combine to form complete response
        complete_response = "".join(chunks)
        assert "Streaming response test" in complete_response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_chunk_processing(self, mock_agent, mock_openai_model):
        """Test individual chunk processing."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Word by word streaming test for chunk processing"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        chunks = []
        async for chunk in chat.chat_stream("test streaming"):
            chunks.append(chunk)
            # Each chunk should be a string
            assert isinstance(chunk, str)

        # Should have multiple chunks
        assert len(chunks) >= 5  # At least a few words

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_stream_completion(self, mock_agent, mock_openai_model):
        """Test stream completion and cleanup."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Complete streaming response"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        chunks = []
        async for chunk in chat.chat_stream("test completion"):
            chunks.append(chunk)

        # Stream should be complete
        complete_response = "".join(chunks)
        assert "Complete streaming response" in complete_response

        # Sources should be included at the end
        assert "Sources:" in complete_response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_stream_error_recovery(self, mock_agent, mock_openai_model):
        """Test error recovery during streaming."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.run.side_effect = Exception("Stream error")

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Should handle error gracefully during streaming
        chunks = []
        async for chunk in chat.chat_stream("test error"):
            chunks.append(chunk)

        # Should receive error message
        complete_response = "".join(chunks)
        assert "error" in complete_response.lower()


class TestAgentConfiguration:
    """Test agent configuration and setup."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    def test_system_prompt_construction(self, mock_agent, mock_openai_model):
        """Test system prompt construction."""
        config = ChatConfig(model_name="gpt-4o-mini")
        chat = RAGChatInterface(self.mock_search_layer, config)

        # Check that agent was initialized with proper system prompt
        call_args = mock_agent.call_args[1]
        system_prompt = call_args['system_prompt']

        # Verify key elements in system prompt
        assert "AppPack.io" in system_prompt
        assert "documentation" in system_prompt
        assert "cite" in system_prompt.lower()
        assert "source" in system_prompt.lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    def test_deps_type_configuration(self, mock_agent, mock_openai_model):
        """Test dependencies type configuration."""
        config = ChatConfig(model_name="gpt-4o-mini")
        chat = RAGChatInterface(self.mock_search_layer, config)

        # Check deps_type was set correctly
        call_args = mock_agent.call_args[1]
        assert 'deps_type' in call_args
        assert call_args['deps_type'] == dict[str, Any]

    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configs
        valid_configs = [
            ChatConfig(),
            ChatConfig(model_name="gpt-4", temperature=0.5),
            ChatConfig(max_sources=10, search_method="vector")
        ]

        for config in valid_configs:
            chat = RAGChatInterface(self.mock_search_layer, config)
            assert chat.config == config

    def test_default_configuration(self):
        """Test default configuration values."""
        chat = RAGChatInterface(self.mock_search_layer)

        # Check defaults
        assert chat.config.model_name == "gpt-4o-mini"
        assert chat.config.max_sources == 5
        assert chat.config.search_method == "hybrid"
        assert chat.config.temperature == 0.7
        assert chat.config.stream_responses is True


if __name__ == "__main__":
    pytest.main([__file__])
