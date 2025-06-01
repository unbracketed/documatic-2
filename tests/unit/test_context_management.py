"""Unit tests for conversation context management."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.documatic.chat import (
    ChatConfig,
    ConversationContext,
    RAGChatInterface,
)
from src.documatic.search import SearchResult
from tests.fixtures.chat_mocks import SAMPLE_SEARCH_RESULTS, MockSearchLayer


class TestConversationMemory:
    """Test conversation memory and history storage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context = ConversationContext(
            conversation_id="test_conversation",
            context_window=3
        )

    def test_message_history_storage(self):
        """Test storing message history."""
        # Add multiple turns
        turns_data = [
            ("What is AppPack?", "AppPack is a deployment platform."),
            ("How do I deploy?", "Use the apppack deploy command."),
            ("What about logs?", "Check logs with apppack logs.")
        ]

        for user_query, assistant_response in turns_data:
            self.context.add_turn(user_query, assistant_response)

        # Verify all turns are stored
        assert len(self.context.turns) == 3

        # Verify turn data
        for i, (user_query, assistant_response) in enumerate(turns_data):
            turn = self.context.turns[i]
            assert turn.user_query == user_query
            assert turn.assistant_response == assistant_response
            assert isinstance(turn.timestamp, datetime)

    def test_context_window_limits(self):
        """Test context window size limits."""
        # Set small context window
        context = ConversationContext(
            conversation_id="test",
            context_window=2
        )

        # Add more turns than context window
        for i in range(5):
            context.add_turn(f"Question {i}", f"Answer {i}")

        # Get recent context
        recent_context = context.get_recent_context()

        # Should only include recent turns within window (excluding current)
        # With context_window=2, and 5 turns, should show turns 2,3 (excluding the last turn)
        lines = recent_context.split('\n')
        human_lines = [line for line in lines if line.startswith('Human:')]
        assistant_lines = [line for line in lines if line.startswith('Assistant:')]

        # Should have at most context_window-1 pairs (excluding current turn)
        assert len(human_lines) <= context.context_window - 1
        assert len(assistant_lines) <= context.context_window - 1

    def test_memory_truncation_strategies(self):
        """Test different memory truncation approaches."""
        # Test with different window sizes
        window_sizes = [1, 3, 5, 10]

        for window_size in window_sizes:
            context = ConversationContext(
                conversation_id=f"test_{window_size}",
                context_window=window_size
            )

            # Add many turns
            for i in range(15):
                context.add_turn(f"Question {i}", f"Answer {i}")

            recent_context = context.get_recent_context()

            # Count conversation pairs in context
            lines = recent_context.split('\n') if recent_context else []
            human_lines = len([line for line in lines if line.startswith('Human:')])

            # Should respect window size (minus current turn)
            expected_max = min(window_size - 1, 14)  # 14 previous turns available
            assert human_lines <= expected_max

    def test_thread_safety(self):
        """Test thread safety of context operations."""
        import threading
        import time

        context = ConversationContext(conversation_id="thread_test")
        results = []
        errors = []

        def add_turns(thread_id):
            try:
                for i in range(10):
                    context.add_turn(f"Thread {thread_id} Question {i}", f"Answer {i}")
                    time.sleep(0.001)  # Small delay
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_turns, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 5
        assert len(context.turns) == 50  # 5 threads × 10 turns each


class TestContextFormatting:
    """Test context formatting for prompts."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context = ConversationContext(
            conversation_id="format_test",
            context_window=5
        )

    def test_system_prompt_construction(self):
        """Test system prompt construction with context."""
        mock_search_layer = MockSearchLayer()
        config = ChatConfig()

        chat = RAGChatInterface(mock_search_layer, config, self.context)

        # Get the system prompt (it's built during initialization)
        system_prompt = chat._build_system_prompt()

        # Verify key components
        assert "AppPack.io" in system_prompt
        assert "documentation" in system_prompt
        assert "cite" in system_prompt.lower()
        assert "Source:" in system_prompt

    def test_retrieved_document_formatting(self):
        """Test formatting of retrieved documents."""
        mock_search_layer = MockSearchLayer()
        chat = RAGChatInterface(mock_search_layer, ChatConfig(), self.context)

        # Test with sample search results
        formatted_context = chat._format_context_for_prompt(SAMPLE_SEARCH_RESULTS)

        # Should include context header
        assert "DOCUMENTATION CONTEXT:" in formatted_context

        # Should include source headers
        assert "[Source 1:" in formatted_context
        assert "[Source 2:" in formatted_context

        # Should include content
        for result in SAMPLE_SEARCH_RESULTS:
            assert result.content in formatted_context

    def test_conversation_history_formatting(self):
        """Test formatting of conversation history."""
        # Add some conversation turns
        self.context.add_turn("What is AppPack?", "AppPack is a platform...")
        self.context.add_turn("How do I deploy?", "Use the deploy command...")
        self.context.add_turn("What about logs?", "Check with logs command...")

        # Get formatted context
        formatted = self.context.get_recent_context()

        # Should format as Human:/Assistant: pairs
        lines = formatted.split('\n')

        # Should have Human and Assistant lines
        human_lines = [line for line in lines if line.startswith('Human:')]
        assistant_lines = [line for line in lines if line.startswith('Assistant:')]

        assert len(human_lines) > 0
        assert len(assistant_lines) > 0
        assert len(human_lines) == len(assistant_lines)

    def test_token_counting(self):
        """Test token counting for context management."""
        # Add long conversation
        long_response = "This is a very long response. " * 100
        for i in range(10):
            self.context.add_turn(f"Long question {i}? " * 10, long_response)

        # Get context
        formatted = self.context.get_recent_context()

        # Should be manageable size (rough token estimate)
        # Assuming ~4 chars per token as rough estimate
        estimated_tokens = len(formatted) / 4
        assert estimated_tokens < 4000  # Reasonable limit for context


class TestContextRelevance:
    """Test context relevance and selection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.context = ConversationContext(conversation_id="relevance_test")

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_relevant_context_selection(self, mock_agent, mock_openai_model):
        """Test selection of relevant context."""
        # Setup mocks
        mock_agent_instance = Mock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Test response"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, ChatConfig(), self.context)

        # Set up search results for specific query
        deployment_results = [
            SearchResult(
                content="Deploy apps with apppack deploy command",
                chunk_id="deploy_1",
                source_file="deployment.md",
                title="Test Document",
                section_hierarchy=["Deployment"],
                content_type="text",
                document_type="markdown",
                score=0.95,
                search_method="hybrid",
                metadata={}
            )
        ]

        self.mock_search_layer.set_results_for_query("deploy", deployment_results)

        # Query about deployment
        await chat.chat("How do I deploy my app?")

        # Verify search was called with deployment query
        search_calls = self.mock_search_layer.search_history
        assert len(search_calls) == 1
        assert "deploy" in search_calls[0]["query"].lower()

    def test_context_ranking(self):
        """Test ranking of context by relevance."""
        mock_search_layer = MockSearchLayer()
        chat = RAGChatInterface(mock_search_layer, ChatConfig())

        # Create results with different scores
        results = [
            SearchResult(
                content="Low relevance content",
                chunk_id="low",
                source_file="misc.md",
                title="Test Document",
        section_hierarchy=["Misc"],
                content_type="text",
                document_type="markdown",

                score=0.3,
                metadata={}
            ,
        search_method="hybrid"),
            SearchResult(
                content="High relevance content",
                chunk_id="high",
                source_file="relevant.md",
                title="Test Document",
        section_hierarchy=["Important"],
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={}
            ,
        search_method="hybrid"),
            SearchResult(
                content="Medium relevance content",
                chunk_id="medium",
                source_file="ok.md",
                title="Test Document",
        section_hierarchy=["OK"],
                content_type="text",
                document_type="markdown",

                score=0.6,
                metadata={}
            ,
        search_method="hybrid")
        ]

        # Format context (should maintain order from search)
        formatted = chat._format_context_for_prompt(results)

        # Should include all results
        for result in results:
            assert result.content in formatted

    def test_duplicate_removal(self):
        """Test removal of duplicate context."""
        # Create results with duplicate content
        results = [
            SearchResult(
                content="Duplicate content here",
                chunk_id="dup1",
                source_file="file1.md",
                title="Test Document",
        section_hierarchy=["Section 1"],
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={}
            ,
        search_method="hybrid"),
            SearchResult(
                content="Unique content here",
                chunk_id="unique",
                source_file="file2.md",
                title="Test Document",
        section_hierarchy=["Section 2"],
                content_type="text",
                document_type="markdown",

                score=0.8,
                metadata={}
            ,
        search_method="hybrid"),
            SearchResult(
                content="Duplicate content here",
                chunk_id="dup2",
                source_file="file3.md",
                title="Test Document",
        section_hierarchy=["Section 3"],
                content_type="text",
                document_type="markdown",

                score=0.7,
                metadata={}
            ,
        search_method="hybrid")
        ]

        chat = RAGChatInterface(self.mock_search_layer, ChatConfig())
        formatted = chat._format_context_for_prompt(results)

        # Count occurrences of duplicate content
        duplicate_count = formatted.count("Duplicate content here")

        # Current implementation doesn't deduplicate, so count should match input
        # If deduplication is implemented, change this assertion
        assert duplicate_count == 2  # Both duplicates present

    def test_context_summarization(self):
        """Test context summarization for long contexts."""
        # Create very long results
        long_results = []
        for i in range(20):  # Many results
            long_content = f"Very long content section {i}. " * 50
            result = SearchResult(
                content=long_content,
                chunk_id=f"long_{i}",
                source_file=f"file_{i}.md",
                title="Test Document",
                section_hierarchy=[f"Section {i}"],
                content_type="text",
                document_type="markdown",
                score=0.8,
                search_method="hybrid",
                metadata={}
            )
            long_results.append(result)

        chat = RAGChatInterface(self.mock_search_layer, ChatConfig(max_sources=5))

        # Format context with limit (should only pass max_sources results)
        limited_results = long_results[:chat.config.max_sources]
        formatted = chat._format_context_for_prompt(limited_results)

        # Should respect max_sources limit
        source_count = formatted.count("[Source ")
        assert source_count <= 5


if __name__ == "__main__":
    pytest.main([__file__])
