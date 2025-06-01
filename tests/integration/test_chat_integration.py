"""Integration tests for complete chat workflows."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.documatic.chat import (
    ChatConfig,
    RAGChatInterface,
    ask_question,
    create_chat_interface,
)
from src.documatic.search import SearchResult
from tests.fixtures.chat_mocks import MockSearchLayer


class TestEndToEndChatFlow:
    """Test complete end-to-end chat flows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig(
            max_sources=3,
            temperature=0.7,
            include_metadata=True
        )

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_complete_chat_flow(self, mock_agent, mock_openai_model):
        """Test complete user input to final response flow."""
        # Setup search results for deployment question
        deployment_results = [
            SearchResult(
                content="To deploy an application, use 'apppack deploy' command.",
                chunk_id="deploy_1",
                source_file="cli/commands.md",
                title="Test Document",
        section_hierarchy=["CLI", "Deploy"],
                content_type="text",
                document_type="markdown",

                score=0.95,
                metadata={"category": "deployment"}
            ,
        search_method="hybrid"),
            SearchResult(
                content="Before deploying, ensure your apppack.toml is configured.",
                chunk_id="deploy_2",
                source_file="config/apppack-toml.md",
                title="Test Document",
        section_hierarchy=["Configuration", "apppack.toml"],
                content_type="text",
                document_type="markdown",

                score=0.88,
                metadata={"category": "configuration"}
            ,
        search_method="hybrid")
        ]

        self.mock_search_layer.set_results_for_query("deploy", deployment_results)

        # Setup LLM mock
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = (
            "To deploy your application, first ensure your apppack.toml file is "
            "properly configured, then run the 'apppack deploy' command. "
            "[Source: commands.md] [Source: apppack-toml.md]"
        )
        mock_agent_instance.run.return_value = mock_result

        # Test complete flow
        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("How do I deploy my application?")

        # Verify search integration
        search_calls = self.mock_search_layer.search_history
        assert len(search_calls) == 1
        assert "deploy" in search_calls[0]["query"].lower()

        # Verify LLM interaction
        mock_agent_instance.run.assert_called_once()
        agent_call_args = mock_agent_instance.run.call_args[0][0]
        assert "DOCUMENTATION CONTEXT:" in agent_call_args
        assert "apppack deploy" in agent_call_args
        assert "apppack.toml" in agent_call_args

        # Verify response formatting
        assert "deploy" in response.lower()
        assert "apppack.toml" in response.lower()
        assert "Sources:" in response
        assert "commands.md" in response
        assert "apppack-toml.md" in response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_search_integration(self, mock_agent, mock_openai_model):
        """Test integration with search layer."""
        # Setup specific search results
        database_results = [
            SearchResult(
                content="Create a PostgreSQL database with 'apppack create database'.",
                chunk_id="db_1",
                source_file="databases/postgresql.md",
                title="Test Document",
        section_hierarchy=["Databases", "PostgreSQL", "Creation"],
                content_type="text",
                document_type="markdown",

                score=0.92,
                metadata={"database_type": "postgresql"}
            ,
        search_method="hybrid")
        ]

        self.mock_search_layer.set_results_for_query("database", database_results)

        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Create PostgreSQL databases using the apppack CLI."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Test search integration
        await chat.chat("How do I create a database?")

        # Verify search was called with correct parameters
        search_calls = self.mock_search_layer.search_history
        assert len(search_calls) == 1
        search_call = search_calls[0]
        assert search_call["method"] == "hybrid"
        assert search_call["limit"] == 3  # max_sources from config
        assert "database" in search_call["query"].lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_llm_interaction(self, mock_agent, mock_openai_model):
        """Test interaction with LLM service."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Environment variables are set using 'apppack config set'."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("How do I set environment variables?")

        # Verify LLM was called correctly
        mock_agent_instance.run.assert_called_once()

        # Check the prompt structure
        prompt = mock_agent_instance.run.call_args[0][0]
        assert "DOCUMENTATION CONTEXT:" in prompt
        assert "USER QUESTION:" in prompt
        assert "How do I set environment variables?" in prompt

        # Check dependencies were passed
        deps = mock_agent_instance.run.call_args[1]["deps"]
        assert "query" in deps
        assert "context" in deps

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_response_formatting(self, mock_agent, mock_openai_model):
        """Test final response formatting and assembly."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "AppPack logs can be viewed with 'apppack logs' command."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("How do I view logs?")

        # Verify response includes LLM output
        assert "apppack logs" in response.lower()

        # Verify sources are appended
        assert "Sources:" in response

        # Verify conversation turn was recorded
        assert len(chat.context.turns) == 1
        turn = chat.context.turns[0]
        assert turn.user_query == "How do I view logs?"
        assert "apppack logs" in turn.assistant_response.lower()


class TestDifferentQuestionTypes:
    """Test handling of different types of questions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_how_to_questions(self, mock_agent, mock_openai_model):
        """Test handling of how-to questions."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "To configure SSL, use 'apppack create custom-domain' with SSL options."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        how_to_questions = [
            "How do I configure SSL?",
            "How to set up a custom domain?",
            "How can I scale my application?",
            "How do I backup my database?"
        ]

        for question in how_to_questions:
            response = await chat.chat(question)

            # Should provide actionable instructions
            assert len(response) > 30
            # Should mention AppPack commands or processes
            assert "apppack" in response.lower() or "config" in response.lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_troubleshooting_questions(self, mock_agent, mock_openai_model):
        """Test handling of troubleshooting questions."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "For 502 errors, check your application health and logs."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        troubleshooting_questions = [
            "My app returns 502 errors",
            "Deployment is stuck at building",
            "Database connection is failing",
            "App is running out of memory"
        ]

        for question in troubleshooting_questions:
            response = await chat.chat(question)

            # Should provide diagnostic guidance
            assert len(response) > 20
            # Should suggest checking logs, status, or configuration
            assert any(word in response.lower() for word in ["check", "logs", "status", "config"])

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_conceptual_queries(self, mock_agent, mock_openai_model):
        """Test handling of conceptual questions."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "AppPack uses containerization to isolate and deploy applications."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        conceptual_questions = [
            "What is AppPack's architecture?",
            "How does AppPack work?",
            "What are the benefits of using AppPack?",
            "What is the difference between staging and production?"
        ]

        for question in conceptual_questions:
            response = await chat.chat(question)

            # Should provide explanatory content
            assert len(response) > 40
            # Should be informative rather than just procedural
            assert not response.lower().startswith("to ")

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_code_examples(self, mock_agent, mock_openai_model):
        """Test handling of requests for code examples."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = """Here's a sample apppack.toml:
```toml
name = "my-app"
build.dockerfile = "Dockerfile"
```"""
        mock_agent_instance.run.return_value = mock_result

        # Setup code search results
        code_result = SearchResult(
            content="```toml\nname = \"example\"\nbuild.dockerfile = \"Dockerfile\"\n```",
            chunk_id="code_1",
            source_file="examples/apppack-toml.md",
            title="Test Document",
        section_hierarchy=["Examples", "Configuration"],
            content_type="code",
            document_type="markdown",

            score=0.9,
            metadata={"language": "toml"}
        ,
        search_method="hybrid")

        self.mock_search_layer.set_results_for_query("example", [code_result])

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        code_questions = [
            "Show me an apppack.toml example",
            "Give me a Python deployment example",
            "What does a Dockerfile look like for AppPack?",
            "Can you show me environment variable configuration?"
        ]

        for question in code_questions:
            response = await chat.chat(question)

            # Should include code or examples
            assert len(response) > 30
            # May include code blocks
            if "```" in response:
                assert response.count("```") >= 2  # Opening and closing

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_configuration_help(self, mock_agent, mock_openai_model):
        """Test handling of configuration questions."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Configure environment variables using 'apppack config set KEY=value'."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        config_questions = [
            "How do I configure environment variables?",
            "What are the available configuration options?",
            "How do I set up different environments?",
            "What configuration is required for Python apps?"
        ]

        for question in config_questions:
            response = await chat.chat(question)

            # Should provide configuration guidance
            assert len(response) > 25
            # Should mention configuration methods or files
            assert any(word in response.lower() for word in ["config", "set", "toml", "environment"])


class TestConversationQuality:
    """Test overall conversation quality metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_answer_accuracy(self, mock_agent, mock_openai_model):
        """Test accuracy of answers based on documentation."""
        # Setup specific documentation content
        accurate_result = SearchResult(
            content="AppPack requires Docker for containerized deployments.",
            chunk_id="docker_req",
            source_file="requirements/docker.md",
            title="Test Document",
        section_hierarchy=["Requirements", "Docker"],
            content_type="text",
            document_type="markdown",

            score=0.95,
            metadata={}
        ,
        search_method="hybrid")

        self.mock_search_layer.set_results_for_query("docker", [accurate_result])

        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Yes, AppPack requires Docker for containerized deployments."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("Does AppPack require Docker?")

        # Should be accurate based on provided documentation
        assert "docker" in response.lower()
        assert "require" in response.lower()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_response_coherence(self, mock_agent, mock_openai_model):
        """Test coherence and logical flow of responses."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "AppPack supports Python applications. First install dependencies, then configure apppack.toml, and finally deploy."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("How do I deploy a Python app?")

        # Should have logical structure
        assert len(response) > 50
        # Should mention key steps in logical order
        python_pos = response.lower().find("python")
        deploy_pos = response.lower().rfind("deploy")
        assert python_pos < deploy_pos  # Should mention Python before deploy

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_citation_relevance(self, mock_agent, mock_openai_model):
        """Test relevance of citations to response content."""
        # Setup relevant source
        relevant_result = SearchResult(
            content="SSL certificates are automatically managed by AppPack.",
            chunk_id="ssl_auto",
            source_file="security/ssl.md",
            title="Test Document",
        section_hierarchy=["Security", "SSL"],
            content_type="text",
            document_type="markdown",

            score=0.9,
            metadata={}
        ,
        search_method="hybrid")

        self.mock_search_layer.set_results_for_query("ssl", [relevant_result])

        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "AppPack automatically manages SSL certificates for your domains."
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("How does SSL work with AppPack?")

        # Citations should be relevant to SSL topic
        assert "ssl" in response.lower()
        assert "Sources:" in response
        assert "ssl.md" in response

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_helpfulness_scoring(self, mock_agent, mock_openai_model):
        """Test overall helpfulness of responses."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = """To scale your application:
1. Use 'apppack update app --scale N' to change instance count
2. Monitor performance with 'apppack logs'
3. Consider upgrading your plan for more resources"""
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("How do I scale my application?")

        # Should be helpful with actionable steps
        assert len(response) > 80
        # Should include specific commands or steps
        assert "apppack" in response.lower()
        # Should provide multiple pieces of information
        response_lines = [line.strip() for line in response.split('\n') if line.strip()]
        assert len(response_lines) >= 3


class TestConvenienceFunctions:
    """Test convenience functions for chat interface."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()

    def test_create_chat_interface(self):
        """Test chat interface creation function."""
        config = ChatConfig(max_sources=5)
        chat = create_chat_interface(self.mock_search_layer, config)

        assert isinstance(chat, RAGChatInterface)
        assert chat.config.max_sources == 5
        assert chat.search_layer == self.mock_search_layer

    def test_create_chat_interface_defaults(self):
        """Test chat interface creation with defaults."""
        chat = create_chat_interface(self.mock_search_layer)

        assert isinstance(chat, RAGChatInterface)
        assert chat.config.model_name == "gpt-4o-mini"
        assert chat.config.max_sources == 5

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_ask_question_function(self, mock_agent, mock_openai_model):
        """Test simple ask_question convenience function."""
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Simple question response"
        mock_agent_instance.run.return_value = mock_result

        response = await ask_question(
            "Simple question?",
            self.mock_search_layer
        )

        assert "Simple question response" in response
        assert "Sources:" in response


if __name__ == "__main__":
    pytest.main([__file__])
