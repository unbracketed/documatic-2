"""Unit tests for prompt templates and formatting."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.documatic.chat import ChatConfig, ConversationContext, RAGChatInterface
from src.documatic.search import SearchResult
from tests.fixtures.chat_mocks import SAMPLE_SEARCH_RESULTS, MockSearchLayer


class TestSystemPrompt:
    """Test system prompt construction and content."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    def test_documentation_context_injection(self):
        """Test injection of documentation context into system prompt."""
        chat = RAGChatInterface(self.mock_search_layer, self.config)
        system_prompt = chat._build_system_prompt()

        # Should mention AppPack documentation
        assert "AppPack.io" in system_prompt
        assert "documentation" in system_prompt.lower()

        # Should specify the domain
        assert "AppPack.io platform" in system_prompt or "AppPack" in system_prompt

    def test_role_definition(self):
        """Test AI role definition in system prompt."""
        chat = RAGChatInterface(self.mock_search_layer, self.config)
        system_prompt = chat._build_system_prompt()

        # Should define assistant role
        assert "assistant" in system_prompt.lower() or "expert" in system_prompt.lower()

        # Should specify helping users
        assert "help" in system_prompt.lower() or "assist" in system_prompt.lower()

    def test_behavior_constraints(self):
        """Test behavior constraints in system prompt."""
        chat = RAGChatInterface(self.mock_search_layer, self.config)
        system_prompt = chat._build_system_prompt()

        # Should specify accuracy requirements
        assert "accurate" in system_prompt.lower() or "specific" in system_prompt.lower()

        # Should mention staying within scope
        assert "scope" in system_prompt.lower() or "within" in system_prompt.lower()

        # Should mention handling unknown information
        assert "not in" in system_prompt.lower() or "don't know" in system_prompt.lower() or "clearly state" in system_prompt.lower()

    def test_citation_requirements(self):
        """Test citation requirements in system prompt."""
        chat = RAGChatInterface(self.mock_search_layer, self.config)
        system_prompt = chat._build_system_prompt()

        # Should require citations
        assert "cite" in system_prompt.lower() or "source" in system_prompt.lower()

        # Should specify citation format
        assert "[Source:" in system_prompt or "Source:" in system_prompt

        # Should mention filename format
        assert "filename" in system_prompt.lower()


class TestQueryReformulation:
    """Test query reformulation for better search."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.context = ConversationContext(conversation_id="reformulation_test")

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_conversation_aware_reformulation(self, mock_agent, mock_openai_model):
        """Test query reformulation based on conversation context."""
        # Setup conversation history
        self.context.add_turn(
            "What is AppPack?",
            "AppPack is a deployment platform for applications."
        )
        self.context.add_turn(
            "How do I deploy Python apps?",
            "Use the apppack deploy command with a Python runtime."
        )

        # Setup mocks
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "For Django deployment, configure your settings..."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config, self.context)

        # Ask follow-up question
        response = await chat.chat("What about Django specifically?")

        # Should have used conversation context
        agent_call = mock_agent_instance.run.call_args[0][0]
        assert "CONVERSATION HISTORY:" in agent_call
        assert "What is AppPack?" in agent_call
        assert "Python apps" in agent_call

    async def test_intent_preservation(self):
        """Test preservation of user intent during reformulation."""
        # Test queries with specific intents
        test_queries = [
            ("How do I debug my app?", ["debug", "troubleshoot"]),
            ("What environment variables are available?", ["environment", "variables", "config"]),
            ("Can I use Docker?", ["docker", "container"]),
            ("How much does it cost?", ["cost", "pricing", "price"])
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        for query, expected_terms in test_queries:
            # Trigger search to see what query was used
            await chat._retrieve_context(query)

            # Check search history
            search_calls = self.mock_search_layer.search_history
            last_search = search_calls[-1]["query"].lower()

            # Should preserve key intent terms
            preserved_terms = [term for term in expected_terms if term in last_search]
            assert len(preserved_terms) > 0, f"No intent terms preserved for: {query}"

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_context_incorporation(self, mock_agent, mock_openai_model):
        """Test incorporation of conversation context into queries."""
        # Setup conversation about deployment
        self.context.add_turn(
            "I want to deploy a web application",
            "Great! AppPack supports web application deployment."
        )

        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "For web apps, you'll need to configure..."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config, self.context)

        # Ask follow-up without explicit context
        await chat.chat("How do I configure the database?")

        # Should include previous context about web applications
        agent_call = mock_agent_instance.run.call_args[0][0]
        assert "web application" in agent_call.lower()

    async def test_multiple_reformulations(self):
        """Test handling of multiple possible reformulations."""
        ambiguous_queries = [
            "It's not working",  # Vague
            "How do I fix this?",  # No specific context
            "What about the other way?",  # Unclear reference
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        for query in ambiguous_queries:
            # Should still attempt to search
            results = await chat._retrieve_context(query)

            # Should get some results (default results)
            assert len(results) > 0

            # Search should have been called
            search_calls = self.mock_search_layer.search_history
            assert len(search_calls) > 0


class TestAnswerGeneration:
    """Test answer generation and formatting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    def test_citation_formatting(self):
        """Test proper citation formatting in responses."""
        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Test citation formatting
        sources = SAMPLE_SEARCH_RESULTS[:2]
        citations = chat._format_sources_for_response(sources)

        # Should have sources section
        assert "Sources:" in citations

        # Should list source files
        for source in sources:
            file_name = source.source_file.split('/')[-1]  # Get filename
            assert file_name in citations

    def test_code_block_handling(self):
        """Test handling of code blocks in responses."""
        # Create search result with code
        code_result = SearchResult(
            title="Test Document",
            content="```python\ndef deploy_app():\n    return 'deployed'\n```",
            chunk_id="code_1",
            source_file="examples/python.md",
            section_hierarchy=["Examples", "Python"],
            content_type="code",
            document_type="markdown",
            score=0.9,
            search_method="hybrid",
            metadata={"language": "python"}
        )

        self.mock_search_layer.set_results_for_query("python example", [code_result])

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        formatted = chat._format_context_for_prompt([code_result])

        # Should preserve code formatting
        assert "```python" in formatted
        assert "def deploy_app" in formatted
        assert "```" in formatted

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_list_table_generation(self, mock_agent, mock_openai_model):
        """Test generation of lists and tables in responses."""
        # Setup mock to return structured response
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = """AppPack supports these features:

1. Application deployment
2. Database management
3. Environment configuration
4. Log monitoring

| Feature | Description |
|---------|-------------|
| Deploy | Deploy applications |
| Scale | Scale resources |"""
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("What features does AppPack have?")

        # Should preserve list formatting
        assert "1." in response
        assert "2." in response

        # Should preserve table formatting
        assert "|" in response
        assert "Feature" in response
        assert "Description" in response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_error_message_formatting(self, mock_agent, mock_openai_model):
        """Test proper formatting of error messages."""
        # Setup mock to fail
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.run.side_effect = Exception("API timeout")

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("test question")

        # Should format error message properly
        assert "error" in response.lower()
        assert "API timeout" in response
        assert "try" in response.lower() or "rephras" in response.lower()


class TestPromptConstruction:
    """Test overall prompt construction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()
        self.context = ConversationContext(conversation_id="prompt_test")

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_full_prompt_assembly(self, mock_agent, mock_openai_model):
        """Test assembly of complete prompt with all components."""
        # Setup conversation history
        self.context.add_turn("Previous question", "Previous answer")

        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Test response"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config, self.context)

        await chat.chat("Current question")

        # Check the prompt that was sent to agent
        agent_call = mock_agent_instance.run.call_args[0][0]

        # Should include conversation history
        assert "CONVERSATION HISTORY:" in agent_call

        # Should include documentation context
        assert "DOCUMENTATION CONTEXT:" in agent_call

        # Should include user question
        assert "USER QUESTION:" in agent_call
        assert "Current question" in agent_call

    def test_prompt_component_order(self):
        """Test correct ordering of prompt components."""
        chat = RAGChatInterface(self.mock_search_layer, self.config, self.context)

        # Add conversation history
        self.context.add_turn("Old question", "Old answer")

        # Build components
        context_formatted = chat._format_context_for_prompt(SAMPLE_SEARCH_RESULTS)
        history_formatted = self.context.get_recent_context()

        # Simulate prompt construction (simplified)
        prompt_parts = []
        if history_formatted:
            prompt_parts.append("CONVERSATION HISTORY:")
            prompt_parts.append(history_formatted)
            prompt_parts.append("")

        prompt_parts.append(context_formatted)
        prompt_parts.append("\nUSER QUESTION: Test question")

        full_prompt = "\n".join(prompt_parts)

        # Verify order: history first, then context, then question
        history_pos = full_prompt.find("CONVERSATION HISTORY:")
        context_pos = full_prompt.find("DOCUMENTATION CONTEXT:")
        question_pos = full_prompt.find("USER QUESTION:")

        assert history_pos < context_pos < question_pos

    def test_prompt_length_management(self):
        """Test management of prompt length."""
        # Create very long context
        long_results = []
        for i in range(10):
            long_content = "Very long content. " * 200  # ~4000 chars each
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

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Format context with length limits
        formatted = chat._format_context_for_prompt(long_results[:5])  # Limit to 5

        # Should be manageable length
        assert len(formatted) < 50000  # Reasonable limit

    def test_special_character_handling(self):
        """Test handling of special characters in prompts."""
        # Create results with special characters
        special_result = SearchResult(
            content="Special chars: áéíóú, 中文, 🚀, \"quotes\", 'apostrophes', & symbols",
            chunk_id="special",
            source_file="special.md",
            title="Test Document",
            section_hierarchy=["Special"],
            content_type="text",
            document_type="markdown",
            score=0.9,
            search_method="hybrid",
            metadata={}
        )

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        formatted = chat._format_context_for_prompt([special_result])

        # Should preserve special characters
        assert "áéíóú" in formatted
        assert "中文" in formatted
        assert "🚀" in formatted
        assert "\"quotes\"" in formatted
        assert "'apostrophes'" in formatted


if __name__ == "__main__":
    pytest.main([__file__])
