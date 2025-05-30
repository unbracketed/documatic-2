# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Documatic is a RAG (Retrieval-Augmented Generation) application that ingests AppPack.io documentation and provides a chat interface for Q&A. The system uses LanceDB for vector storage and implements proper document chunking with metadata preservation.

## Tech Stack

- **Python** with `uv` package manager
- **LanceDB** for vector database
- **Pydantic AI** for LLM interactions
- **Click** for CLI interface
- **pytest** for testing
- Embeddings and chunking strategies optimized for technical documentation

## Development Commands

### Setup and Dependencies
```bash
# Initialize project with uv
uv init

# Install dependencies
uv pip install -r requirements.txt

# Run in development mode
uv run python -m documatic
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_<module>.py

# Run with coverage
uv run pytest --cov=documatic
```

### Linting and Type Checking
```bash
# Run ruff for linting
uv run ruff check .

# Run type checking with mypy
uv run mypy documatic/
```

## Architecture

The application is structured as a pipeline with distinct components:

1. **Document Acquisition**: Downloads/syncs documentation from https://github.com/apppackio/apppack-docs/
2. **Document Processing**: Chunks documents preserving structure, headings, and metadata
3. **Embedding Pipeline**: Converts chunks to embeddings and stores in LanceDB
4. **Search Layer**: Implements vector, full-text, and hybrid search
5. **Chat Interface**: CLI tool using Click for RAG-powered Q&A

## Key Implementation Notes

- Document boundaries and structure must be preserved during chunking
- Metadata should include: title, section, subsection, source URL, document type
- Quality checks should validate retrieval accuracy with sample questions
- The system should support incremental updates when documentation changes
- Favor simplicity - separate tools/commands for each pipeline stage is acceptable

## AppPack Documentation Sources

- Online docs: https://docs.apppack.io/
- Source repository: https://github.com/apppackio/apppack-docs/