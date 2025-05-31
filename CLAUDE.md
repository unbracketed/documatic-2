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

# Run specific test file
uv run pytest tests/test_<module>.py

# Run with coverage
uv run pytest --cov=documatic
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
2. **Document Chunking** - Planned
3. **Embedding Pipeline** - Planned  
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
└── (future: processed/, embeddings/, etc.)
```

### Key Patterns

**Document Acquisition Pattern:**
- Git-based incremental updates using commit hashes
- Comprehensive manifest system tracking file hashes, metadata, and timestamps
- Frontmatter extraction preserving document structure
- Class-based modular design with proper error handling

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

The project currently has **Document Acquisition** fully implemented:
- `DocumentAcquisition` class handles clone/pull operations
- Processes 32+ AppPack documentation files
- Manifest-based change tracking
- Frontmatter metadata extraction
- Incremental update support

**Key Files:**
- `src/documatic/acquisition.py` - Main acquisition logic
- `_tasks/` - Technical specifications for each pipeline stage
- `data/raw/manifest.json` - Current acquisition state

## AppPack Documentation Sources

- Online docs: https://docs.apppack.io/
- Source repository: https://github.com/apppackio/apppack-docs/
- Local clone location: `data/raw/apppack-docs/`