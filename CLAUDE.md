# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Documatic is a RAG (Retrieval-Augmented Generation) application that ingests AppPack.io documentation and provides a chat interface for Q&A. The system uses LanceDB for vector storage and implements proper document chunking with metadata preservation.

## Tech Stack

- **Python** with `uv` package manager (Python 3.13+)
- **LanceDB** for vector database
- **Pydantic AI** for LLM interactions
- **Click** for CLI interface
- **GitPython** for repository operations
- **python-frontmatter** for markdown metadata extraction
- **pytest** for testing

## Development Commands

### Setup and Dependencies
```bash
# Install dependencies (dependencies are managed in pyproject.toml)
uv pip install -e .

# Install development dependencies
uv pip install -e .[dev]

# Test document acquisition
uv run python -c "from src.documatic.acquisition import acquire_apppack_docs; result = acquire_apppack_docs(); print('Success:', result['status'])"
```

### Testing
```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test categories
uv run pytest tests/unit/          # Unit tests only
uv run pytest tests/integration/   # Integration tests only

# Run specific test files
uv run pytest tests/unit/test_embedding_model.py
uv run pytest tests/unit/test_batch_processing.py
uv run pytest tests/unit/test_lancedb_schema.py
uv run pytest tests/unit/test_vector_storage.py
uv run pytest tests/integration/test_embedding_pipeline_integration.py
uv run pytest tests/integration/test_embedding_performance.py

# Run specific test classes or methods
uv run pytest tests/unit/test_embedding_model.py::TestEmbeddingGeneration
uv run pytest tests/unit/test_embedding_model.py::TestEmbeddingGeneration::test_single_text_embedding

# Run tests with coverage
uv run pytest --cov=documatic

# Run tests with coverage report
uv run pytest --cov=documatic --cov-report=html
uv run pytest --cov=documatic --cov-report=term-missing

# Run tests that match a pattern
uv run pytest -k "embedding"      # Run tests with 'embedding' in name
uv run pytest -k "batch"          # Run tests with 'batch' in name
uv run pytest -k "performance"    # Run performance tests

# Run tests and stop on first failure
uv run pytest -x

# Run tests in parallel (if pytest-xdist is installed)
uv run pytest -n auto
```

### Linting and Type Checking
```bash
# Run ruff for linting (with auto-fix)
uv run ruff check . --fix

# Run type checking with mypy
uv run mypy src/documatic/

# IMPORTANT: Always run both before committing code
# Code must pass both ruff and mypy checks with zero errors
```

## Architecture

### Pipeline Design
The application follows a 10-stage pipeline architecture with task-driven development. Each stage is defined in `_tasks/` with technical requirements and test specifications:

1. ✅ **Document Acquisition** (`src/documatic/acquisition.py`)
2. ✅ **Document Chunking** (`src/documatic/chunking.py`)
3. ✅ **Embedding Pipeline** (`src/documatic/embeddings.py`)
4. **Search Layer** - Planned
5. **RAG Chat Interface** - Planned
6. **CLI Application** - Planned
7. **Quality Evaluation** - Planned
8. **Configuration System** - Planned
9. **Testing Infrastructure** - Planned

### Data Architecture
```
data/
├── raw/
│   ├── apppack-docs/           # Git clone of source documentation
│   └── manifest.json           # Acquisition metadata and document tracking
├── embeddings/                 # LanceDB vector database (auto-created)
└── (future: processed/, etc.)
```

### Key Patterns

**Document Acquisition Pattern:**
- Git-based incremental updates using commit hashes
- Comprehensive manifest system tracking file hashes, metadata, and timestamps
- Frontmatter extraction preserving document structure
- Class-based modular design with proper error handling

**Document Chunking Pattern:**
- Markdown-aware chunking preserving semantic structure
- Configurable chunk sizes with intelligent overlap
- Content type detection (text, code, lists, tables)
- Position tracking and hierarchy preservation

**Embedding Pipeline Pattern:**
- OpenAI text-embedding-3-small model integration
- Batch processing with retry logic and rate limiting
- LanceDB vector storage with hybrid search capabilities
- Metadata preservation through the embedding process
- Change detection via content hashing

**Metadata Preservation:**
- YAML frontmatter extraction from markdown files
- Document hierarchy tracking (title, section, subsection)  
- Source URL and document type metadata
- Change detection via file hashing

**Code Quality Standards:**
- **Strict Type Annotations**: All functions and methods must have complete type hints
- **Mypy Compliance**: Code must pass `mypy src/documatic/` with strict=true (zero errors)
- **Ruff Linting**: Code must pass `ruff check . --fix` with zero errors
- **Modern Python**: Use Python 3.13+ syntax (e.g., `dict[str, Any]` not `Dict[str, Any]`)
- **Error Handling**: Comprehensive try/catch blocks with proper logging
- **Import Standards**: Imports must be sorted and formatted per ruff rules

## Code Standards and Configuration

### Ruff Configuration
The project uses ruff for linting with the following configuration:
- **Target**: Python 3.13
- **Line length**: 88 characters
- **Enabled rules**: E (pycodestyle errors), F (pyflakes), W (pycodestyle warnings), I (isort), N (pep8-naming), UP (pyupgrade), B (flake8-bugbear), A (flake8-builtins), C4 (flake8-comprehensions), SIM (flake8-simplify)

### Mypy Configuration
Strict type checking is enforced with:
- **Python version**: 3.13
- **Strict mode**: Enabled (all optional error codes enabled)
- **Type hints required**: For all functions, methods, and class attributes

### Development Workflow
1. Write code following type annotation requirements
2. Run `uv run ruff check . --fix` to auto-fix formatting issues
3. Run `uv run mypy src/documatic/` to verify type correctness
4. Both commands must pass with zero errors before committing
5. Use modern Python 3.13+ syntax (e.g., `dict[str, Any]`, `list[str]`, `X | Y` unions)

### Type Annotation Examples
```python
# Good - Modern Python 3.13+ syntax
def process_documents(docs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Process documents with proper type hints."""
    
# Bad - Old typing syntax
def process_documents(docs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Avoid legacy typing imports."""
```

## Current Implementation Status

The project has **Document Acquisition**, **Chunking**, and **Embedding Pipeline** fully implemented:

### Document Acquisition ✅
- `DocumentAcquisition` class handles clone/pull operations
- Processes 32+ AppPack documentation files  
- Manifest-based change tracking
- Frontmatter metadata extraction
- Incremental update support

### Document Chunking ✅
- `ChunkingStrategy` with markdown-aware processing
- Configurable chunk sizes (512-1024 tokens default)
- Content type detection and semantic preservation
- Hierarchical section tracking
- Intelligent overlap handling (15% default)

### Embedding Pipeline ✅
- `EmbeddingPipeline` with OpenAI integration
- Batch processing (50 chunks default) with retry logic
- LanceDB vector storage with full-text search indexing
- Hybrid search capabilities (vector + full-text)
- Change detection and incremental updates
- Comprehensive test suite with 50+ test cases

**Key Files:**
- `src/documatic/acquisition.py` - Document acquisition logic
- `src/documatic/chunking.py` - Document chunking logic  
- `src/documatic/embeddings.py` - Embedding and vector storage
- `tests/` - Comprehensive test suite (unit + integration)
- `_tasks/` - Technical specifications for each pipeline stage
- `data/raw/manifest.json` - Current acquisition state

## Test Infrastructure

The project includes a comprehensive test suite covering:

### Unit Tests (`tests/unit/`)
- **Embedding Model Tests** (21 tests) - Model initialization, configuration, embedding generation
- **Batch Processing Tests** (12 tests) - Batch sizing, retry logic, error recovery
- **LanceDB Schema Tests** (15 tests) - Schema creation, validation, migration
- **Vector Storage Tests** - Insertion, upsert, indexing, search operations

### Integration Tests (`tests/integration/`)
- **End-to-End Pipeline Tests** - Full document processing workflows
- **Performance Tests** - Throughput, latency, and scalability validation

### Test Features
- **Mock Infrastructure** - Complete mocking of external dependencies
- **Deterministic Testing** - Consistent, reproducible results
- **Performance Benchmarking** - Automated performance measurement
- **Edge Case Coverage** - Unicode, empty content, large text handling

## AppPack Documentation Sources

- Online docs: https://docs.apppack.io/
- Source repository: https://github.com/apppackio/apppack-docs/
- Local clone location: `data/raw/apppack-docs/`