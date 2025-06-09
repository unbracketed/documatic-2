"""RAG Chat Interface for Documatic.

Implements conversation-based Q&A interface using Pydantic AI with context management,
prompt engineering, source attribution, and streaming responses.
"""

import json
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from .config import get_config
from .search import SearchLayer, SearchResult


class ConversationTurn(BaseModel):
    """Represents a single turn in conversation."""

    timestamp: datetime = Field(default_factory=datetime.now)
    user_query: str = Field(description="User's question or input")
    assistant_response: str = Field(description="Assistant's response")
    sources: list[SearchResult] = Field(
        default_factory=list, description="Sources used for response"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional turn metadata"
    )


class ConversationContext(BaseModel):
    """Manages conversation state and context."""

    conversation_id: str = Field(description="Unique conversation identifier")
    title: str = Field(default="", description="Conversation title")
    turns: list[ConversationTurn] = Field(
        default_factory=list, description="Conversation history"
    )
    context_window: int = Field(
        default=5, description="Number of previous turns to include in context"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Conversation metadata"
    )

    def add_turn(
        self,
        user_query: str,
        assistant_response: str,
        sources: list[SearchResult] | None = None
    ) -> None:
        """Add a new conversation turn."""
        turn = ConversationTurn(
            user_query=user_query,
            assistant_response=assistant_response,
            sources=sources or []
        )
        self.turns.append(turn)

        # Auto-generate title from first meaningful query
        if not self.title and len(self.turns) == 1 and len(user_query) > 10:
            self.title = user_query[:50] + ("..." if len(user_query) > 50 else "")

    def get_recent_context(self) -> str:
        """Get recent conversation context for prompt."""
        if not self.turns:
            return ""

        context_parts = []
        recent_turns = self.turns[-self.context_window:]

        for turn in recent_turns[:-1]:  # Exclude current turn
            context_parts.append(f"Human: {turn.user_query}")
            context_parts.append(f"Assistant: {turn.assistant_response}")

        return "\n".join(context_parts)


class ChatConfig(BaseModel):
    """Configuration for chat interface."""

    model_name: str = Field(default="gpt-4o-mini", description="LLM model name")
    max_sources: int = Field(default=5, description="Maximum sources to retrieve")
    search_method: str = Field(default="hybrid", description="Search method to use")
    include_metadata: bool = Field(
        default=True, description="Include source metadata in responses"
    )
    temperature: float = Field(default=0.7, description="Response temperature")
    max_tokens: int = Field(default=1000, description="Maximum response tokens")
    stream_responses: bool = Field(
        default=True, description="Enable streaming responses"
    )


class RAGChatInterface:
    """Main RAG chat interface using Pydantic AI."""

    def __init__(
        self,
        search_layer: SearchLayer,
        config: ChatConfig | None = None,
        conversation_context: ConversationContext | None = None
    ):
        """Initialize chat interface.

        Args:
            search_layer: Configured search layer for retrieval
            config: Chat configuration
            conversation_context: Existing conversation context
        """
        self.search_layer = search_layer

        # Use provided config or create one with app config defaults
        if config:
            self.config = config
        else:
            app_config = get_config()
            self.config = ChatConfig(
                model_name=app_config.llm.model,
                max_sources=app_config.chat.context_limit,
                temperature=app_config.llm.temperature,
                max_tokens=app_config.llm.max_tokens,
                stream_responses=app_config.chat.stream_responses
            )

        self.context = conversation_context or ConversationContext(
            conversation_id=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            context_window=self.config.max_sources
        )

        # Initialize Pydantic AI agent
        model = OpenAIModel(self.config.model_name)

        self.agent = Agent(
            model,
            system_prompt=self._build_system_prompt(),
            deps_type=dict[str, Any]
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt for the RAG agent."""
        return (
            "You are an expert assistant helping users with AppPack.io "
            "documentation questions.\n\n"
            "Your role:\n"
            "- Answer questions about AppPack.io platform, deployment, "
            "configuration, and usage\n"
            "- Use the provided documentation context to give accurate, "
            "specific answers\n"
            "- Always cite your sources with [Source: filename] format\n"
            "- If information isn't in the provided context, clearly state that\n"
            "- Be concise but comprehensive in your responses\n"
            "- Maintain conversation context for follow-up questions\n\n"
            "Response format:\n"
            "1. Direct answer to the user's question\n"
            "2. Relevant details from the documentation\n"
            "3. Source citations in [Source: filename] format\n"
            "4. Helpful next steps or related information if applicable\n\n"
            "Guidelines:\n"
            "- Focus on practical, actionable information\n"
            "- Use code examples from the docs when relevant\n"
            "- Explain concepts clearly for different skill levels\n"
            "- Stay within the scope of AppPack.io documentation\n"
            "- Be helpful but honest about limitations of available information"
        )

    async def _retrieve_context(self, query: str) -> list[SearchResult]:
        """Retrieve relevant context for the query."""
        try:
            results = await self.search_layer.search(
                query,
                method=self.config.search_method,
                limit=self.config.max_sources
            )
            return results
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []

    def _format_context_for_prompt(self, sources: list[SearchResult]) -> str:
        """Format search results for inclusion in prompt."""
        if not sources:
            return "No relevant documentation found."

        context_parts = ["DOCUMENTATION CONTEXT:"]

        for i, source in enumerate(sources, 1):
            # Build source header
            source_header = f"\n[Source {i}: {source.source_file}"
            if source.section_hierarchy:
                hierarchy = " > ".join(source.section_hierarchy)
                source_header += f" - {hierarchy}"
            source_header += "]"

            context_parts.append(source_header)
            context_parts.append(source.content)

            # Add metadata if enabled
            if self.config.include_metadata:
                context_parts.append(
                    f"(Content type: {source.content_type}, "
                    f"Document type: {source.document_type})"
                )

        return "\n".join(context_parts)

    def _format_sources_for_response(self, sources: list[SearchResult]) -> str:
        """Format sources for final response citation."""
        if not sources:
            return ""

        citation_parts = ["\n\nSources:"]
        for source in sources:
            file_name = Path(source.source_file).name
            if source.section_hierarchy:
                hierarchy = " > ".join(source.section_hierarchy)
                citation_parts.append(f"- {file_name}: {hierarchy}")
            else:
                citation_parts.append(f"- {file_name}")

        return "\n".join(citation_parts)

    async def _run_agent(
        self,
        user_query: str,
        context: str,
        conversation_history: str
    ) -> str:
        """Run the Pydantic AI agent with context."""
        # Build the full prompt
        prompt_parts = []

        if conversation_history:
            prompt_parts.append("CONVERSATION HISTORY:")
            prompt_parts.append(conversation_history)
            prompt_parts.append("")

        prompt_parts.append(context)
        prompt_parts.append(f"\nUSER QUESTION: {user_query}")
        prompt_parts.append(
            "\nPlease answer based on the documentation context above. "
            "Include source citations and be specific about AppPack.io features."
        )

        full_prompt = "\n".join(prompt_parts)

        # Run agent
        deps = {"query": user_query, "context": context}
        result = await self.agent.run(full_prompt, deps=deps)

        return result.data

    async def chat(self, user_query: str) -> str:
        """Process a chat query and return response.

        Args:
            user_query: User's question or input

        Returns:
            Assistant's response with source citations
        """
        if not user_query.strip():
            return "Please provide a question about AppPack.io documentation."

        try:
            # Retrieve relevant context
            sources = await self._retrieve_context(user_query)

            # Format context for prompt
            formatted_context = self._format_context_for_prompt(sources)

            # Get conversation history
            conversation_history = self.context.get_recent_context()

            # Generate response using agent
            response = await self._run_agent(
                user_query, formatted_context, conversation_history
            )

            # Add source citations to response
            source_citations = self._format_sources_for_response(sources)
            final_response = response + source_citations

            # Add turn to conversation context
            self.context.add_turn(user_query, final_response, sources)

            return final_response

        except Exception as e:
            error_response = (
                f"I encountered an error while processing your question: {e}\n"
                "Please try rephrasing your question or check if the documentation "
                "database is available."
            )
            self.context.add_turn(user_query, error_response)
            return error_response

    async def chat_stream(self, user_query: str) -> AsyncGenerator[str]:
        """Process chat query with streaming response.

        Args:
            user_query: User's question or input

        Yields:
            Response chunks as they are generated
        """
        if not user_query.strip():
            yield "Please provide a question about AppPack.io documentation."
            return

        try:
            # Retrieve context (non-streaming)
            sources = await self._retrieve_context(user_query)
            formatted_context = self._format_context_for_prompt(sources)
            conversation_history = self.context.get_recent_context()

            # Build prompt
            prompt_parts = []
            if conversation_history:
                prompt_parts.append("CONVERSATION HISTORY:")
                prompt_parts.append(conversation_history)
                prompt_parts.append("")

            prompt_parts.append(formatted_context)
            prompt_parts.append(f"\nUSER QUESTION: {user_query}")
            prompt_parts.append(
                "\nPlease answer based on the documentation context above. "
                "Include source citations and be specific about AppPack.io features."
            )

            "\n".join(prompt_parts)

            # Stream response from agent

            # Note: This is a simplified streaming implementation
            # Pydantic AI may require different streaming setup
            response_parts = []

            # Simplified streaming - use regular chat for now
            response = await self._run_agent(
                user_query, formatted_context, conversation_history
            )

            # Yield response in chunks for streaming effect
            chunk_size = 50
            for i in range(0, len(response), chunk_size):
                chunk = response[i:i + chunk_size]
                response_parts.append(chunk)
                yield chunk

            # Add source citations at the end
            source_citations = self._format_sources_for_response(sources)
            if source_citations:
                yield source_citations
                response_parts.append(source_citations)

            # Save complete response to context
            final_response = "".join(response_parts)
            self.context.add_turn(user_query, final_response, sources)

        except Exception as e:
            error_msg = (
                f"I encountered an error while processing your question: {e}\n"
                "Please try rephrasing your question."
            )
            yield error_msg
            self.context.add_turn(user_query, error_msg)

    def save_conversation(self, file_path: Path) -> None:
        """Save conversation to file.

        Args:
            file_path: Path to save conversation
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.context.model_dump(), f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving conversation: {e}")

    @classmethod
    def load_conversation(
        cls,
        file_path: Path,
        search_layer: SearchLayer,
        config: ChatConfig | None = None
    ) -> "RAGChatInterface":
        """Load conversation from file.

        Args:
            file_path: Path to conversation file
            search_layer: Configured search layer
            config: Chat configuration

        Returns:
            Chat interface with loaded conversation
        """
        try:
            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)

            # Convert data back to ConversationContext
            context = ConversationContext.model_validate(data)

            return cls(search_layer, config, context)

        except Exception as e:
            print(f"Error loading conversation: {e}")
            return cls(search_layer, config)

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.context = ConversationContext(
            conversation_id=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    def get_conversation_summary(self) -> dict[str, Any]:
        """Get conversation summary statistics."""
        return {
            "conversation_id": self.context.conversation_id,
            "title": self.context.title,
            "turn_count": len(self.context.turns),
            "start_time": (
                self.context.turns[0].timestamp if self.context.turns else None
            ),
            "last_activity": (
                self.context.turns[-1].timestamp if self.context.turns else None
            ),
            "total_sources_used": sum(len(turn.sources) for turn in self.context.turns)
        }


# Convenience functions
def create_chat_interface(
    search_layer: SearchLayer,
    config: ChatConfig | None = None
) -> RAGChatInterface:
    """Create a configured chat interface.

    Args:
        search_layer: Configured search layer
        config: Chat configuration

    Returns:
        Configured chat interface
    """
    return RAGChatInterface(search_layer, config)


async def ask_question(
    question: str,
    search_layer: SearchLayer,
    config: ChatConfig | None = None
) -> str:
    """Simple function to ask a single question.

    Args:
        question: Question to ask
        search_layer: Configured search layer
        config: Chat configuration

    Returns:
        Answer with source citations
    """
    chat = create_chat_interface(search_layer, config)
    return await chat.chat(question)


if __name__ == "__main__":
    from pathlib import Path

    from .embeddings import EmbeddingPipeline
    from .search import create_search_layer

    async def main() -> None:
        """Example usage of the chat interface."""

        # Initialize components
        db_path = Path("data/embeddings")
        if not db_path.exists():
            print(f"Vector database not found at {db_path}")
            print("Run the embedding pipeline first to create the database.")
            return

        try:
            # Setup pipeline and search
            pipeline = EmbeddingPipeline(db_path=db_path)
            search_layer = create_search_layer(pipeline)

            # Create chat interface
            config = ChatConfig(
                max_sources=3,
                temperature=0.7,
                include_metadata=True
            )
            chat = create_chat_interface(search_layer, config)

            print("AppPack.io Documentation Chat")
            print("Ask questions about AppPack.io deployment and configuration.")
            print("Type 'quit' to exit, 'clear' to clear conversation.\n")

            while True:
                try:
                    question = input("\nYou: ").strip()

                    if question.lower() in ['quit', 'exit', 'q']:
                        break
                    elif question.lower() == 'clear':
                        chat.clear_conversation()
                        print("Conversation cleared.")
                        continue
                    elif not question:
                        continue

                    print("\nAssistant: ", end="", flush=True)

                    # Use streaming response
                    async for chunk in chat.chat_stream(question):
                        print(chunk, end="", flush=True)

                    print()  # New line after response

                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"\nError: {e}")

            # Show conversation summary
            summary = chat.get_conversation_summary()
            print("\nConversation summary:")
            print(f"- Turns: {summary['turn_count']}")
            print(f"- Sources used: {summary['total_sources_used']}")

        except Exception as e:
            print(f"Error initializing chat: {e}")

    # Note: Requires vector database and API key
    print("Chat module loaded. Set OPENAI_API_KEY and run embedding pipeline first.")
    # asyncio.run(main())
