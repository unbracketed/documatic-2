# Documatic

A powerful RAG (Retrieval-Augmented Generation) application that ingests AppPack.io documentation and provides an intelligent chat interface for Q&A. Built with LanceDB for vector storage and proper document chunking with metadata preservation.

## Features

- **Smart Document Processing**: Fetches and processes AppPack.io documentation with incremental updates
- **Intelligent Chunking**: Markdown-aware chunking that preserves semantic structure
- **Vector Search**: Advanced vector similarity search using OpenAI embeddings
- **Hybrid Search**: Combines vector search with full-text search using BM25
- **Interactive Chat**: RAG-powered chat interface with source citations
- **Quality Evaluation**: Automated evaluation system with multiple metrics
- **CLI Interface**: Easy-to-use command-line interface for all operations

## Tech Stack

- **Python 3.13+** with `uv` package manager
- **LanceDB** for vector database storage
- **Pydantic AI** for LLM interactions
- **OpenAI API** for embeddings and chat
- **Click** for CLI interface
- **GitPython** for repository operations
- **pytest** for comprehensive testing

## Quick Start

### Prerequisites

- Python 3.13+
- OpenAI API key
- `uv` package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/unbracketed/documatic-2.git
cd documatic-2

# Install dependencies
uv pip install -e .

# Set up your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key"
```

### Usage

```bash
# Fetch AppPack documentation
uv run documatic fetch

# Index documents (create embeddings)
uv run documatic index

# Search documents
uv run documatic search "deploy flask app"

# Start interactive chat
uv run documatic chat

# Run quality evaluation
uv run documatic evaluate
```

## Architecture

Documatic follows a 10-stage pipeline architecture:

1. **Document Acquisition** - Git-based incremental documentation fetching
2. **Document Chunking** - Markdown-aware semantic chunking
3. **Embedding Pipeline** - OpenAI embeddings with batch processing
4. **Search Layer** - Vector, full-text, and hybrid search strategies
5. **RAG Chat Interface** - Context-aware conversation with citations
6. **CLI Application** - User-friendly command-line interface
7. **Quality Evaluation** - Automated testing and metrics
8. **Configuration System** - (Planned)
9. **Testing Infrastructure** - Comprehensive test suite
10. **Documentation** - Complete technical documentation

## Data Architecture

```
data/
├── raw/
│   ├── apppack-docs/           # Source documentation
│   └── manifest.json           # Acquisition metadata
├── embeddings/                 # LanceDB vector database
└── conversations/              # Chat history (optional)
```

## Development

### Setup

```bash
# Install development dependencies
uv pip install -e .[dev]

# Run tests
uv run pytest

# Run linting
uv run ruff check . --fix

# Run type checking
uv run mypy src/documatic/
```

### Code Quality

This project maintains strict code quality standards:

- **Type Safety**: Complete type annotations with mypy strict mode
- **Linting**: Ruff linting with zero errors required
- **Testing**: Comprehensive unit and integration tests
- **Modern Python**: Python 3.13+ syntax and features

## Search Strategies

Documatic supports multiple search approaches:

- **Vector Search**: Semantic similarity using OpenAI embeddings
- **Full-text Search**: BM25-based keyword matching
- **Hybrid Search**: Combines vector and full-text with rank fusion
- **LLM Reranking**: Optional LLM-based result reranking

## Chat Features

- **Contextual Responses**: Maintains conversation context
- **Source Citations**: Provides citations for all claims
- **Streaming Support**: Real-time response streaming
- **Conversation Persistence**: Save and load chat history
- **Configurable Models**: Support for different LLM models

## Quality Evaluation

Built-in evaluation system with:

- **Automated Dataset Generation**: Creates test questions from documents
- **Multiple Metrics**: MRR, Recall@K, relevance scoring
- **Answer Quality Assessment**: LLM-based quality evaluation
- **Citation Validation**: Ensures accurate source attribution
- **Regression Testing**: Tracks system performance over time

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper type annotations
4. Ensure all tests pass: `uv run pytest`
5. Verify code quality: `uv run ruff check . --fix && uv run mypy src/documatic/`
6. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For questions or issues, please open an issue on GitHub or refer to the comprehensive documentation in the `_tasks/` directory.