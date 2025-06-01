"""Mock objects for RAG chat interface testing."""

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from src.documatic.search import SearchResult


class MockLLM:
    """Mock LLM for testing Pydantic AI integration."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize mock LLM."""
        self.model_name = model_name
        self.call_history: list[dict[str, Any]] = []
        self.response_delay = 0.01  # Fast for tests
        self.responses: dict[str, str] = {}
        self.default_response = (
            "Based on the documentation, AppPack is a platform for deploying applications. "
            "[Source: overview.md]"
        )
        self.should_fail = False
        self.fail_message = "Mock LLM failure"

    def set_response(self, query_key: str, response: str) -> None:
        """Set custom response for specific query patterns."""
        self.responses[query_key] = response

    def set_failure(self, should_fail: bool = True, message: str = "Mock LLM failure") -> None:
        """Configure LLM to simulate failures."""
        self.should_fail = should_fail
        self.fail_message = message

    async def complete(self, messages: list[dict[str, Any]], stream: bool = False) -> Any:
        """Mock completion method."""
        self.call_history.append({
            "messages": messages,
            "stream": stream,
            "timestamp": datetime.now()
        })

        if self.should_fail:
            raise Exception(self.fail_message)

        if stream:
            return self._stream_response(messages)
        else:
            return self._complete_response(messages)

    async def _stream_response(self, messages: list[dict[str, Any]]) -> AsyncGenerator[str]:
        """Mock streaming response."""
        response = self._get_response_for_messages(messages)

        # Split into chunks for streaming
        words = response.split()
        for word in words:
            await asyncio.sleep(self.response_delay)
            yield word + " "

    def _complete_response(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Mock complete response."""
        response = self._get_response_for_messages(messages)
        return {
            "content": response,
            "role": "assistant",
            "metadata": {"model": self.model_name}
        }

    def _get_response_for_messages(self, messages: list[dict[str, Any]]) -> str:
        """Get appropriate response based on messages."""
        # Extract user query from messages
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "").lower()
                break

        # Check for custom responses
        for key, response in self.responses.items():
            if key.lower() in user_message:
                return response

        # Return default response
        return self.default_response


class MockSearchLayer:
    """Mock search layer for testing."""

    def __init__(self):
        """Initialize mock search layer."""
        self.search_history: list[dict[str, Any]] = []
        self.mock_results: dict[str, list[SearchResult]] = {}
        self.default_results = [
            SearchResult(
                content="AppPack is a platform for deploying applications easily.",
                chunk_id="chunk_1",
                source_file="overview.md",
                title="AppPack Overview",
                section_hierarchy=["Introduction"],
                content_type="text",
                document_type="markdown",
                score=0.9,
                search_method="hybrid",
                metadata={"section": "Overview"}
            ),
            SearchResult(
                content="To deploy an application, use the 'apppack create app' command.",
                chunk_id="chunk_2",
                source_file="deployment.md",
                title="Deployment Guide",
                section_hierarchy=["Getting Started", "Deploy App"],
                content_type="text",
                document_type="markdown",
                score=0.85,
                search_method="hybrid",
                metadata={"section": "Deployment"}
            )
        ]
        self.should_fail = False
        self.fail_message = "Mock search failure"

    def set_results_for_query(self, query_pattern: str, results: list[SearchResult]) -> None:
        """Set custom results for specific query patterns."""
        self.mock_results[query_pattern.lower()] = results

    def set_failure(self, should_fail: bool = True, message: str = "Mock search failure") -> None:
        """Configure search to simulate failures."""
        self.should_fail = should_fail
        self.fail_message = message

    async def search(
        self,
        query: str,
        method: str = "hybrid",
        limit: int = 5
    ) -> list[SearchResult]:
        """Mock search method."""
        self.search_history.append({
            "query": query,
            "method": method,
            "limit": limit,
            "timestamp": datetime.now()
        })

        if self.should_fail:
            raise Exception(self.fail_message)

        # Check for custom results
        query_lower = query.lower()
        for pattern, results in self.mock_results.items():
            if pattern in query_lower:
                return results[:limit]

        # Return default results
        return self.default_results[:limit]


class MockConversationStore:
    """Mock conversation storage for testing."""

    def __init__(self):
        """Initialize mock conversation store."""
        self.conversations: dict[str, list[dict[str, Any]]] = {}
        self.operation_history: list[dict[str, Any]] = []

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Save a message to conversation history."""
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }

        self.conversations[session_id].append(message)
        self.operation_history.append({
            "operation": "save_message",
            "session_id": session_id,
            "timestamp": datetime.now()
        })

    def get_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get conversation history for a session."""
        self.operation_history.append({
            "operation": "get_history",
            "session_id": session_id,
            "limit": limit,
            "timestamp": datetime.now()
        })

        return self.conversations.get(session_id, [])[-limit:]

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        if session_id in self.conversations:
            del self.conversations[session_id]

        self.operation_history.append({
            "operation": "clear_session",
            "session_id": session_id,
            "timestamp": datetime.now()
        })

    def get_all_sessions(self) -> list[str]:
        """Get all session IDs."""
        return list(self.conversations.keys())


# Sample test data
SAMPLE_QUESTIONS = {
    "basic": [
        "What is AppPack?",
        "How do I deploy an application?",
        "What are environment variables?",
        "How do I create a database?"
    ],
    "technical": [
        "How do I configure a custom domain with SSL?",
        "What's the difference between staging and production environments?",
        "How can I debug a failing deployment?",
        "How do I set up CI/CD with GitHub Actions?"
    ],
    "troubleshooting": [
        "My app won't start, what should I check?",
        "I'm getting a 502 error, how do I fix it?",
        "The deployment is stuck, what can I do?",
        "How do I check application logs?"
    ],
    "code_examples": [
        "Show me a Python deployment example",
        "How do I configure a Django app?",
        "What's the apppack.toml format?",
        "How do I use environment variables in my code?"
    ]
}

SAMPLE_CONVERSATIONS = [
    {
        "name": "deployment_flow",
        "turns": [
            {
                "user": "How do I deploy a Python app?",
                "assistant": (
                    "To deploy a Python app with AppPack, you need to create an apppack.toml "
                    "file and use the CLI. [Source: deployment.md]"
                ),
                "sources": ["deployment.md"]
            },
            {
                "user": "What version of Python is supported?",
                "assistant": (
                    "AppPack supports Python 3.8, 3.9, 3.10, and 3.11. "
                    "[Source: languages/python.md]"
                ),
                "sources": ["languages/python.md"]
            },
            {
                "user": "Can I use a requirements.txt file?",
                "assistant": (
                    "Yes, AppPack automatically detects and installs from requirements.txt, "
                    "poetry.lock, or Pipfile. [Source: languages/python.md]"
                ),
                "sources": ["languages/python.md"]
            }
        ]
    },
    {
        "name": "troubleshooting_flow",
        "turns": [
            {
                "user": "My deployment failed, what should I check?",
                "assistant": (
                    "Check the build logs, verify your apppack.toml configuration, "
                    "and ensure all dependencies are specified. [Source: troubleshooting.md]"
                ),
                "sources": ["troubleshooting.md"]
            },
            {
                "user": "Where do I find the build logs?",
                "assistant": (
                    "Use 'apppack logs' command or check the AppPack dashboard "
                    "under your app's deployment history. [Source: troubleshooting.md]"
                ),
                "sources": ["troubleshooting.md"]
            }
        ]
    }
]

SAMPLE_SEARCH_RESULTS = [
    SearchResult(
        content="AppPack supports deploying applications using Docker containers.",
        chunk_id="deployment_1",
        source_file="deployment/overview.md",
        title="Deployment Overview",
        section_hierarchy=["Deployment", "Docker Support"],
        content_type="text",
        document_type="markdown",
        score=0.95,
        search_method="hybrid",
        metadata={"category": "deployment"}
    ),
    SearchResult(
        content="Environment variables can be set using the apppack config set command.",
        chunk_id="config_1",
        source_file="configuration/environment.md",
        title="Environment Configuration",
        section_hierarchy=["Configuration", "Environment Variables"],
        content_type="text",
        document_type="markdown",
        score=0.88,
        search_method="hybrid",
        metadata={"category": "configuration"}
    ),
    SearchResult(
        content="```yaml\nname: my-app\nbuild:\n  dockerfile: Dockerfile\n```",
        chunk_id="config_2",
        source_file="configuration/apppack-toml.md",
        title="AppPack Configuration",
        section_hierarchy=["Configuration", "apppack.toml"],
        content_type="code",
        document_type="markdown",
        score=0.82,
        search_method="hybrid",
        metadata={"category": "configuration", "language": "yaml"}
    )
]

# Expected responses for testing
EXPECTED_RESPONSES = {
    "what_is_apppack": {
        "must_contain": ["AppPack", "platform", "deploy"],
        "must_have_citations": True,
        "min_length": 50,
        "should_not_contain": ["I don't know", "unclear"]
    },
    "deployment_help": {
        "must_contain": ["deploy", "apppack.toml", "command"],
        "must_have_citations": True,
        "min_length": 100,
        "should_not_contain": ["error", "failed"]
    },
    "troubleshooting": {
        "must_contain": ["check", "logs", "debug"],
        "must_have_citations": True,
        "min_length": 80,
        "should_not_contain": ["unknown", "impossible"]
    }
}

# Prompt templates for testing
SYSTEM_PROMPT_TEMPLATE = """You are an expert assistant helping users with AppPack.io documentation questions.

Your role:
- Answer questions about AppPack.io platform, deployment, configuration, and usage
- Use the provided documentation context to give accurate, specific answers
- Always cite your sources with [Source: filename] format
- If information isn't in the provided context, clearly state that
- Be concise but comprehensive in your responses
- Maintain conversation context for follow-up questions

Response format:
1. Direct answer to the user's question
2. Relevant details from the documentation
3. Source citations in [Source: filename] format
4. Helpful next steps or related information if applicable

Guidelines:
- Focus on practical, actionable information
- Use code examples from the docs when relevant
- Explain concepts clearly for different skill levels
- Stay within the scope of AppPack.io documentation
- Be helpful but honest about limitations of available information"""

QUERY_REFORMULATION_TEMPLATE = """Given the conversation history:
{history}

And the user's question: {question}

Reformulate this into a search query that captures the user's intent."""

CONTEXT_FORMATTING_TEMPLATE = """DOCUMENTATION CONTEXT:
{context}

USER QUESTION: {question}

Please answer based on the documentation context above. Include source citations and be specific about AppPack.io features."""
