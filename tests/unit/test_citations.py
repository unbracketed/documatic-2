"""Unit tests for citation management and source attribution."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.documatic.chat import ChatConfig, RAGChatInterface
from src.documatic.search import SearchResult
from tests.fixtures.chat_mocks import MockSearchLayer


class TestSourceAttribution:
    """Test source attribution and citation extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    def test_citation_extraction(self):
        """Test extraction of citations from search results."""
        # Create test search results
        results = [
            SearchResult(
                content="AppPack deployment guide content",
                chunk_id="deploy_1",
                source_file="guides/deployment.md",
                title="Test Document",
        section_hierarchy=["Deployment", "Getting Started"],
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={"author": "AppPack Team"}
            ,
        search_method="hybrid"),
            SearchResult(
                content="Configuration examples and settings",
                chunk_id="config_1",
                source_file="reference/configuration.md",
                title="Test Document",
        section_hierarchy=["Configuration", "Settings"],
                content_type="text",
                document_type="markdown",

                score=0.85,
                metadata={"version": "1.2"}
            ,
        search_method="hybrid")
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        citations = chat._format_sources_for_response(results)

        # Should include Sources header
        assert "Sources:" in citations

        # Should include file names (not full paths)
        assert "deployment.md" in citations
        assert "configuration.md" in citations

        # Should not include full paths
        assert "guides/" not in citations
        assert "reference/" not in citations

    def test_source_linking(self):
        """Test proper linking to source documents."""
        results = [
            SearchResult(
                content="Database configuration details",
                chunk_id="db_1",
                source_file="how-to/databases.md",
                title="Test Document",
        section_hierarchy=["How To", "Databases", "PostgreSQL"],
                content_type="text",
                document_type="markdown",

                score=0.95,
                metadata={"url": "https://docs.apppack.io/how-to/databases#postgresql"}
            ,
        search_method="hybrid")
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        citations = chat._format_sources_for_response(results)

        # Should include section hierarchy
        assert "How To > Databases > PostgreSQL" in citations

        # Should format as bullet points
        assert "- " in citations

    def test_reference_formatting(self):
        """Test formatting of references in different styles."""
        results = [
            SearchResult(
                content="Content 1",
                chunk_id="ref_1",
                source_file="doc1.md",
                title="Test Document",
        section_hierarchy=["Section A"],
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={}
            ,
        search_method="hybrid"),
            SearchResult(
                content="Content 2",
                chunk_id="ref_2",
                source_file="doc2.md",
                title="Test Document",
        section_hierarchy=["Section B", "Subsection"],
                content_type="text",
                document_type="markdown",

                score=0.8,
                metadata={}
            ,
        search_method="hybrid"),
            SearchResult(
                content="Content 3",
                chunk_id="ref_3",
                source_file="doc3.md",
                title="Test Document",
        section_hierarchy=[],  # No hierarchy
                content_type="text",
                document_type="markdown",

                score=0.7,
                metadata={}
            ,
        search_method="hybrid")
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        citations = chat._format_sources_for_response(results)

        # Should handle different hierarchy formats
        assert "doc1.md: Section A" in citations
        assert "doc2.md: Section B > Subsection" in citations
        assert "doc3.md" in citations  # No colon for empty hierarchy

    def test_uniqueness_checking(self):
        """Test removal of duplicate citations."""
        # Create results with duplicate files but different sections
        results = [
            SearchResult(
                content="Content from section 1",
                chunk_id="dup_1",
                source_file="same-file.md",
                title="Test Document",
        section_hierarchy=["Section 1"],
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={}
            ,
        search_method="hybrid"),
            SearchResult(
                content="Content from section 2",
                chunk_id="dup_2",
                source_file="same-file.md",
                title="Test Document",
        section_hierarchy=["Section 2"],
                content_type="text",
                document_type="markdown",

                score=0.8,
                metadata={}
            ,
        search_method="hybrid"),
            SearchResult(
                content="Content from different file",
                chunk_id="unique_1",
                source_file="different-file.md",
                title="Test Document",
        section_hierarchy=["Other Section"],
                content_type="text",
                document_type="markdown",

                score=0.7,
                metadata={}
            ,
        search_method="hybrid")
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        citations = chat._format_sources_for_response(results)

        # Should list each unique file/section combination
        assert "same-file.md: Section 1" in citations
        assert "same-file.md: Section 2" in citations
        assert "different-file.md: Other Section" in citations

        # Count occurrences
        citation_lines = [line for line in citations.split('\n') if line.startswith('- ')]
        assert len(citation_lines) == 3


class TestCitationAccuracy:
    """Test accuracy of citations and content matching."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_content_citation_matching(self, mock_agent, mock_openai_model):
        """Test that citations match the content used."""
        # Setup specific search results
        deployment_result = SearchResult(
            content="To deploy an app, run 'apppack deploy' command",
            chunk_id="deploy_cmd",
            source_file="cli/commands.md",
            title="Test Document",
        section_hierarchy=["CLI", "Deploy Command"],
            content_type="text",
            document_type="markdown",

            score=0.95,
            metadata={}
        ,
        search_method="hybrid")

        self.mock_search_layer.set_results_for_query("deploy", [deployment_result])

        # Setup mock to use the content
        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "To deploy your application, use the 'apppack deploy' command. [Source: commands.md]"
        mock_agent_instance.run.return_value = mock_result

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        response = await chat.chat("How do I deploy my app?")

        # Should include both inline citation and source list
        assert "[Source: commands.md]" in response or "commands.md" in response
        assert "Sources:" in response
        assert "commands.md" in response

    async def test_citation_validation(self):
        """Test validation of citation accuracy."""
        # Setup results with specific content
        results = [
            SearchResult(
                content="Database setup instructions here",
                chunk_id="db_setup",
                source_file="tutorials/database-setup.md",
                title="Test Document",
        section_hierarchy=["Tutorials", "Database Setup"],
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={}
            ,
        search_method="hybrid")
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)

        # Format context and check citation tracking
        formatted_context = chat._format_context_for_prompt(results)

        # Should include source marker in context
        assert "[Source 1: database-setup.md" in formatted_context
        assert "Database setup instructions here" in formatted_context

        # Citation should match the source
        citations = chat._format_sources_for_response(results)
        assert "database-setup.md" in citations

    def test_missing_citation_detection(self):
        """Test detection of missing citations."""
        # Test with empty results
        empty_results = []

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        citations = chat._format_sources_for_response(empty_results)

        # Should handle empty citations gracefully
        assert citations == ""

    @patch('src.documatic.chat.OpenAIModel')
    @patch('src.documatic.chat.Agent')
    async def test_over_citation_prevention(self, mock_agent, mock_openai_model):
        """Test prevention of excessive citations."""
        # Setup many similar results
        many_results = []
        for i in range(10):
            result = SearchResult(
                content=f"Similar content {i}",
                chunk_id=f"similar_{i}",
                source_file=f"file_{i}.md",
                title="Test Document",
        section_hierarchy=[f"Section {i}"],
                content_type="text",
                document_type="markdown",
                chunk_index=i,
                score=0.8,
                metadata={}
            ,
        search_method="hybrid")
            many_results.append(result)

        mock_agent_instance = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        mock_result = Mock()
        mock_result.data = "Response using similar content"
        mock_agent_instance.run.return_value = mock_result

        # Set max sources limit
        config = ChatConfig(max_sources=3)
        chat = RAGChatInterface(self.mock_search_layer, config)

        # Override search results
        self.mock_search_layer.set_results_for_query("similar", many_results)

        response = await chat.chat("Find similar content")

        # Should limit number of sources cited
        citation_count = response.count("file_")
        assert citation_count <= 3  # Respects max_sources limit


class TestCitationFormats:
    """Test different citation format styles."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    def test_inline_citations(self):
        """Test inline citation format [1], [2], etc."""
        results = [
            SearchResult(
                content="First source content",
                chunk_id="inline_1",
                source_file="source1.md",
                title="Test Document",
        section_hierarchy=["Section 1"],
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={}
            ,
        search_method="hybrid"),
            SearchResult(
                content="Second source content",
                chunk_id="inline_2",
                source_file="source2.md",
                title="Test Document",
        section_hierarchy=["Section 2"],
                content_type="text",
                document_type="markdown",

                score=0.8,
                metadata={}
            ,
        search_method="hybrid")
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        formatted_context = chat._format_context_for_prompt(results)

        # Should use numbered source format
        assert "[Source 1:" in formatted_context
        assert "[Source 2:" in formatted_context

    def test_footnote_style(self):
        """Test footnote-style citations."""
        chat = RAGChatInterface(self.mock_search_layer, self.config)

        results = [
            SearchResult(
                content="Footnote content",
                chunk_id="footnote_1",
                source_file="footnotes.md",
                title="Test Document",
        section_hierarchy=["Footnotes"],
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={}
            ,
        search_method="hybrid")
        ]

        citations = chat._format_sources_for_response(results)

        # Should format as list
        assert "Sources:" in citations
        assert "- footnotes.md" in citations

    def test_source_sections(self):
        """Test source sections at end of response."""
        results = [
            SearchResult(
                content="Source section content",
                chunk_id="section_1",
                source_file="sections.md",
                title="Test Document",
        section_hierarchy=["Main", "Sub"],
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={}
            ,
        search_method="hybrid")
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        citations = chat._format_sources_for_response(results)

        # Should have clear section header
        assert citations.startswith("\n\nSources:")

        # Should include hierarchy
        assert "sections.md: Main > Sub" in citations

    def test_url_generation(self):
        """Test URL generation for sources."""
        # Note: Current implementation doesn't generate URLs
        # This test documents expected behavior if implemented

        results = [
            SearchResult(
                content="URL content",
                chunk_id="url_1",
                source_file="docs/api.md",
                title="Test Document",
        section_hierarchy=["API", "Endpoints"],
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={"url": "https://docs.apppack.io/api/endpoints"}
            ,
        search_method="hybrid")
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        citations = chat._format_sources_for_response(results)

        # Current implementation just shows filename and hierarchy
        assert "api.md" in citations
        assert "API > Endpoints" in citations

        # Future enhancement could include URLs
        # assert "https://docs.apppack.io/api/endpoints" in citations


class TestCitationEdgeCases:
    """Test edge cases in citation handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_layer = MockSearchLayer()
        self.config = ChatConfig()

    def test_empty_source_files(self):
        """Test handling of empty or missing source files."""
        results = [
            SearchResult(
                content="Content without source",
                chunk_id="no_source",
                source_file="",  # Empty source file
                title="Test Document",
        section_hierarchy=[],
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={}
            ,
        search_method="hybrid")
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        citations = chat._format_sources_for_response(results)

        # Should handle empty source gracefully
        # Current implementation might show empty filename or skip
        assert "Sources:" in citations or citations == ""

    def test_special_characters_in_filenames(self):
        """Test handling of special characters in filenames."""
        results = [
            SearchResult(
                content="Special filename content",
                chunk_id="special_char",
                source_file="docs/file-with-dashes.md",
                title="Test Document",
        section_hierarchy=["Section with spaces", "Sub-section"],
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={}
            ,
        search_method="hybrid"),
            SearchResult(
                content="Unicode filename content",
                chunk_id="unicode_char",
                source_file="docs/файл-unicode.md",
                title="Test Document",
        section_hierarchy=["Раздел", "Подраздел"],
                content_type="text",
                document_type="markdown",

                score=0.8,
                metadata={}
            ,
        search_method="hybrid")
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        citations = chat._format_sources_for_response(results)

        # Should handle special characters
        assert "file-with-dashes.md" in citations
        assert "Section with spaces > Sub-section" in citations

        # Should handle Unicode
        assert "файл-unicode.md" in citations
        assert "Раздел > Подраздел" in citations

    def test_very_long_section_hierarchies(self):
        """Test handling of very long section hierarchies."""
        long_hierarchy = [f"Level {i}" for i in range(10)]

        results = [
            SearchResult(
                content="Deep hierarchy content",
                chunk_id="deep_hierarchy",
                source_file="deep.md",
                title="Test Document",
        section_hierarchy=long_hierarchy,
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={}
            ,
        search_method="hybrid")
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        citations = chat._format_sources_for_response(results)

        # Should handle long hierarchies (may truncate)
        assert "deep.md" in citations
        assert "Level 0" in citations
        # Should include some hierarchy
        assert ">" in citations

    def test_duplicate_filenames_different_paths(self):
        """Test handling of duplicate filenames in different directories."""
        results = [
            SearchResult(
                content="Content from first README",
                chunk_id="readme_1",
                source_file="project1/README.md",
                title="Test Document",
        section_hierarchy=["Project 1"],
                content_type="text",
                document_type="markdown",

                score=0.9,
                metadata={}
            ,
        search_method="hybrid"),
            SearchResult(
                content="Content from second README",
                chunk_id="readme_2",
                source_file="project2/README.md",
                title="Test Document",
        section_hierarchy=["Project 2"],
                content_type="text",
                document_type="markdown",

                score=0.8,
                metadata={}
            ,
        search_method="hybrid")
        ]

        chat = RAGChatInterface(self.mock_search_layer, self.config)
        citations = chat._format_sources_for_response(results)

        # Should distinguish between same filenames in different directories
        # Current implementation only shows filename, so both would show as README.md
        readme_count = citations.count("README.md")
        assert readme_count == 2

        # Should show different sections to distinguish
        assert "Project 1" in citations
        assert "Project 2" in citations


if __name__ == "__main__":
    pytest.main([__file__])
