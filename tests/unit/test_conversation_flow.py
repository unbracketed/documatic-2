"""Unit tests for conversation flow and patterns."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.documatic.chat import (
    ChatConfig,
    ConversationContext,
    RAGChatInterface,
)
from src.documatic.search import SearchResult
from tests.fixtures.chat_mocks import MockSearchLayer


class TestSingleTurnQA:
    """Test single-turn question and answer flows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_question_answer_flow(self, mock_agent, mock_openai_model):
        """Test basic question to answer flow."""
        # Setup mock response
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "AppPack is a platform for deploying applications easily."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("What is AppPack?")

        # Should provide answer
        assert "AppPack" in response
        assert "platform" in response
        assert len(response) > 20  # Non-trivial response

    async def test_context_retrieval(self):
        """Test context retrieval for questions."""
        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Test specific query
        results = await chat._retrieve_context("How do I deploy?")

        # Should retrieve relevant context
        assert len(results) > 0

        # Should have called search layer
        search_calls = self.mock_search_layer.search_history
        assert len(search_calls) == 1
        assert "deploy" in search_calls[0]["query"].lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_response_generation(self, mock_agent, mock_openai_model):
        """Test response generation process."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "To deploy an application, use 'apppack deploy' command."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("How do I deploy an app?")

        # Should generate appropriate response
        assert "deploy" in response.lower()
        assert "apppack" in response.lower()

        # Should have conversation turn recorded
        assert len(chat.context.turns) == 1
        turn = chat.context.turns[0]
        assert turn.user_query == "How do I deploy an app?"
        assert "deploy" in turn.assistant_response.lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_citation_inclusion(self, mock_agent, mock_openai_model):
        """Test inclusion of citations in single-turn responses."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Environment variables are configured using apppack config."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("How do I set environment variables?")

        # Should include source citations
        assert "Sources:" in response

        # Should have at least one source listed
        lines = response.split('\n')
        source_lines = [line for line in lines if line.startswith('- ')]
        assert len(source_lines) > 0


class TestMultiTurnConversations:
    """Test multi-turn conversation handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()
        self.context = ConversationContext(conversation_id="multi_turn_test")

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_context_preservation(self, mock_agent, mock_openai_model):
        """Test preservation of context across turns."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()

        # Setup different responses for different calls
        responses = [
            "AppPack is a deployment platform.",
            "For Python apps, create an apppack.toml file.",
            "Yes, Python 3.8+ is supported."
        ]
        response_iter = iter(responses)
        mock_result.data = next(response_iter)

        def side_effect(*args, **kwargs):
            try:
                mock_result.data = next(response_iter)
            except StopIteration:
                mock_result.data = "No more responses"
            return mock_result

        mock_agent_instance.run.side_effect = side_effect
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config, self.context)

        # First turn
        response1 = await chat.chat("What is AppPack?")
        assert "platform" in response1

        # Second turn - should have context from first
        response2 = await chat.chat("How do I deploy Python apps?")
        assert "Python" in response2

        # Third turn - should have context from previous turns
        response3 = await chat.chat("What Python versions are supported?")
        assert "Python" in response3

        # Check conversation context was preserved
        assert len(chat.context.turns) == 3

        # Check that conversation history was included in later calls
        final_call_args = mock_agent_instance.run.call_args[0][0]
        assert "CONVERSATION HISTORY:" in final_call_args
        assert "What is AppPack?" in final_call_args

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_follow_up_questions(self, mock_agent, mock_openai_model):
        """Test handling of follow-up questions."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Follow-up response"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config, self.context)

        # Initial question about deployment
        await chat.chat("How do I deploy my application?")

        # Follow-up questions
        follow_ups = [
            "What about environment variables?",
            "How do I check the deployment status?",
            "Can I rollback if something goes wrong?"
        ]

        for follow_up in follow_ups:
            response = await chat.chat(follow_up)
            assert len(response) > 0

        # Should have all turns recorded
        assert len(chat.context.turns) == 4  # Initial + 3 follow-ups

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_clarification_requests(self, mock_agent, mock_openai_model):
        """Test handling of clarification requests."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Could you clarify what specific deployment issue you're facing?"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config, self.context)

        # Vague initial question
        response1 = await chat.chat("It's not working")
        assert len(response1) > 0

        # More specific follow-up
        mock_result.data = "For deployment failures, check the build logs first."
        response2 = await chat.chat("My deployment is failing")
        assert "deployment" in response2.lower()

        # Should maintain conversation context
        assert len(chat.context.turns) == 2

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_topic_switching(self, mock_agent, mock_openai_model):
        """Test handling of topic switches in conversation."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()

        topics_and_responses = [
            ("How do I deploy?", "Use apppack deploy command"),
            ("What about databases?", "Create databases with apppack create database"),
            ("How do I check logs?", "Use apppack logs command"),
            ("What are the pricing plans?", "AppPack offers various pricing tiers")
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config, self.context)

        for question, expected_content in topics_and_responses:
            mock_result.data = expected_content
            response = await chat.chat(question)
            assert len(response) > 0

        # Should handle topic switches gracefully
        assert len(chat.context.turns) == 4

        # Each turn should be distinct
        topics = [turn.user_query for turn in chat.context.turns]
        assert "deploy" in topics[0]
        assert "database" in topics[1]
        assert "logs" in topics[2]
        assert "pricing" in topics[3]


class TestConversationPatterns:
    """Test specific conversation patterns and interactions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_greeting_handling(self, mock_agent, mock_openai_model):
        """Test handling of greetings and social interactions."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Hello! I'm here to help with AppPack documentation questions."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        greetings = ["Hello", "Hi there", "Good morning", "Hey"]

        for greeting in greetings:
            response = await chat.chat(greeting)
            # Should respond appropriately to greetings
            assert len(response) > 0
            # May or may not mention AppPack depending on implementation

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_goodbye_detection(self, mock_agent, mock_openai_model):
        """Test detection and handling of goodbye messages."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Goodbye! Feel free to ask more AppPack questions anytime."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        goodbyes = ["Goodbye", "Thanks, bye", "See you later", "That's all"]

        for goodbye in goodbyes:
            response = await chat.chat(goodbye)
            assert len(response) > 0
            # Should handle goodbyes gracefully

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_command_recognition(self, mock_agent, mock_openai_model):
        """Test recognition of command-like queries."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "The 'apppack deploy' command deploys your application."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        command_queries = [
            "apppack deploy",
            "apppack create app",
            "apppack logs",
            "apppack config set"
        ]

        for command in command_queries:
            response = await chat.chat(command)
            # Should recognize and explain commands
            assert "apppack" in response.lower()
            assert len(response) > 20

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_meta_questions(self, mock_agent, mock_openai_model):
        """Test handling of meta-questions about the system itself."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "I'm an AI assistant that helps with AppPack documentation questions."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        meta_questions = [
            "What can you help me with?",
            "Who are you?",
            "What do you know about?",
            "How do you work?"
        ]

        for question in meta_questions:
            response = await chat.chat(question)
            assert len(response) > 0
            # Should explain capabilities related to AppPack


class TestConversationState:
    """Test conversation state management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    def test_conversation_turn_creation(self):
        """Test creation and storage of conversation turns."""
        context = ConversationContext(conversation_id="turn_test")

        # Create sample search results
        sources = [
            SearchResult(
                content="Sample content",
                chunk_id="sample_1",
                source_file="sample.md",
                title="Test Document",
        section_hierarchy=["Sample"],
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={}
            ,
        search_method="hybrid")
        ]

        # Add turn with sources
        context.add_turn(
            "Test question",
            "Test response",
            sources
        )

        # Verify turn was created correctly
        assert len(context.turns) == 1
        turn = context.turns[0]
        assert turn.user_query == "Test question"
        assert turn.assistant_response == "Test response"
        assert len(turn.sources) == 1
        assert turn.sources[0].content == "Sample content"

    def test_conversation_title_generation(self):
        """Test automatic title generation."""
        context = ConversationContext(conversation_id="title_test")

        # Short query - no title
        context.add_turn("Hi", "Hello")
        assert context.title == ""

        # Clear and try longer query
        context.turns = []
        long_query = "How do I deploy a Python application using AppPack?"
        context.add_turn(long_query, "Response")

        # Should generate title from first meaningful query
        assert context.title != ""
        assert len(context.title) <= 53  # 50 chars + "..."
        assert "Python" in context.title or "deploy" in context.title

    def test_conversation_context_window(self):
        """Test context window size management."""
        context = ConversationContext(
            conversation_id="window_test",
            context_window=3
        )

        # Add more turns than window size
        for i in range(6):
            context.add_turn(f"Question {i}", f"Answer {i}")

        # Get recent context
        recent = context.get_recent_context()

        # Should respect window size (excluding current turn)
        lines = recent.split('\n') if recent else []
        human_lines = [line for line in lines if line.startswith('Human:')]

        # With window=3 and 6 total turns, should show turns 3,4 (excluding latest turn 5)
        assert len(human_lines) <= 2  # window_size - 1

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_conversation_summary_generation(self, mock_agent, mock_openai_model):
        """Test conversation summary statistics."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Test response"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Add some conversation turns
        await chat.chat("First question")
        await chat.chat("Second question")
        await chat.chat("Third question")

        # Get conversation summary
        summary = chat.get_conversation_summary()

        # Verify summary contents
        assert summary["turn_count"] == 3
        assert summary["conversation_id"] is not None
        assert summary["start_time"] is not None
        assert summary["last_activity"] is not None
        assert summary["total_sources_used"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
